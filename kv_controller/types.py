from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, order=True)
class KVPageId:
    """Logical KV page identity.

    In a real paged-KV engine, a page is not just "page 17".
    We usually need to know *which layer* that page belongs to as well.

    Example:
    - layer 3, page 17
    - layer 11, page 17

    Those are different pages, so we store both pieces of information.

    Why ``frozen=True``:
    - It makes the object immutable, which means it can safely be used as a
      dictionary key or stored in a set.

    Why ``order=True``:
    - It gives Python a stable ordering for these objects, which can be handy
      when sorting pages for deterministic experiments and debugging.
    """

    layer_id: int
    page_id: int


class TransferKind(str, Enum):
    """Why a page is being moved into GPU HBM.

    ``DEMAND``:
    - The current decode step needs this page right now.
    - If it is not ready, the decode step may stall.

    ``PREFETCH``:
    - We think a future step will need this page soon.
    - Prefetches are proactive and may or may not pay off.
    """

    DEMAND = "demand"
    PREFETCH = "prefetch"


@dataclass(frozen=True)
class LayerBudget:
    """How much HBM capacity a layer is allowed to use.

    We are building toward layer-aware capacity management, so the simulator
    needs a simple way to represent "layer 5 may keep up to N pages hot".
    """

    layer_id: int
    max_resident_pages: int


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the memory hierarchy being simulated.

    ``hbm_capacity_pages``:
    - How many KV pages can fit in the GPU-resident hot tier.

    ``cpu_capacity_pages``:
    - Optional CPU-side limit. For now we mostly treat CPU as "large enough",
      but the field is here because a real multi-tier system may eventually
      want limits for colder tiers too.

    ``page_bytes``:
    - Size of one KV page in bytes. This is what turns "number of pages moved"
      into transfer time.

    ``layers``:
    - Number of transformer layers represented by the workload.

    ``hbm_kv_format``:
    - Records the intended format of HBM-resident KV pages.
    - For this project we keep the HBM format uniform, meaning every page that
      reaches HBM is presented in one consistent format before decode launch.
    """

    hbm_capacity_pages: int
    cpu_capacity_pages: int | None = None
    page_bytes: int = 1
    layers: int = 1
    hbm_kv_format: str = "uniform"


@dataclass(frozen=True)
class TransferConfig:
    """Configuration for the transfer and compute timing model.

    This simulator is not doing real CUDA copies here. Instead, it uses a
    timing model that says:

    transfer time = setup overhead + page_bytes / bandwidth

    It also models:
    - a limit on how many transfers may be in flight at once
    - a decode kernel duration
    - overlap between copy time and compute time
    """

    page_bytes: int
    bandwidth_bytes_per_ms: float
    transfer_setup_ms: float = 0.0
    max_inflight_transfers: int = 1
    decode_kernel_ms: float = 0.05


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level simulator configuration."""

    steps: int
    cache: CacheConfig
    transfer: TransferConfig


@dataclass(frozen=True)
class WorkloadStep:
    """All workload information for one decode step.

    ``required_pages``:
    - Pages that must be present before the step's attention kernel can run.

    ``predicted_pages``:
    - Pages that a controller may want to prefetch for the next step(s).

    ``head_weighted_scores``:
    - Required score signal for this project.
    - Each score summarizes how important a page is under a head-weighted view
      of activity.

    ``query_head_weights``:
    - Per-layer head weights for the current step. These explain where the
      head-weighted scores came from and can be used by future controllers.

    ``per_page_head_activity``:
    - Optional per-page activity vector across heads.
    - This is the more primitive signal that the head-weighted score is built
      from.
    - Carrying it in the step object gives us a clean place to store richer
      future traces collected from real model runs.

    ``per_page_features``:
    - Optional extra features for each page.
    - This gives the interface a place for future signals like recency proxy,
      reuse estimates, or externally computed model statistics.

    ``referenced_layers``:
    - Optional explicit summary of which layers appear in this step.
    - Even though every ``KVPageId`` already stores a layer id, carrying the
      layer list separately is convenient for controllers and logging.

    ``request_id``:
    - Logical request/sequence identifier.
    - This makes replay traces closer to real engine logs where multiple
      requests may be interleaved or compared.

    ``decode_position``:
    - Which decode step within the request this record corresponds to.

    ``sequence_length``:
    - Total decoded sequence length visible to attention at this step.

    ``kv_block_size_tokens``:
    - How many token positions are grouped into one KV block/page.
    - This is intentionally vLLM-like because vLLM reasons in terms of
      block-sized KV chunks.

    ``layer_block_tables``:
    - Optional per-layer logical block table.
    - In vLLM-like engines this maps a sequence's logical KV blocks to the
      block ids/pages that attention needs to read.
    """

    step_idx: int
    required_pages: tuple[KVPageId, ...]
    predicted_pages: tuple[KVPageId, ...]
    head_weighted_scores: dict[KVPageId, float]
    query_head_weights: dict[int, tuple[float, ...]]
    per_page_head_activity: dict[KVPageId, tuple[float, ...]] = field(default_factory=dict)
    per_page_features: dict[KVPageId, dict[str, float]] = field(default_factory=dict)
    referenced_layers: tuple[int, ...] = ()
    request_id: str = ""
    decode_position: int = 0
    sequence_length: int = 0
    kv_block_size_tokens: int = 16
    layer_block_tables: dict[int, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ControllerContext:
    """What the simulator tells the controller before each decode step.

    Think of this as the controller's "dashboard":
    - what is needed now
    - what may be needed soon
    - what is already resident
    - what is already being transferred
    - how much churn and miss pressure we have seen recently
    - what the head-weighted page scores are
    - what extra per-page features are available
    - how many pages each layer is currently using
    - how the transfer subsystem looks right now
    """

    step_idx: int
    required_pages: tuple[KVPageId, ...]
    predicted_pages: tuple[KVPageId, ...]
    resident_pages: frozenset[KVPageId]
    inflight_pages: frozenset[KVPageId]
    hbm_capacity_pages: int
    transfer_backlog: int
    inflight_transfer_count: int
    miss_rate: float
    churn: int
    per_layer_occupancy: dict[int, int]
    per_layer_pressure: dict[int, float]
    head_weighted_scores: dict[KVPageId, float]
    per_page_head_activity: dict[KVPageId, tuple[float, ...]]
    per_page_features: dict[KVPageId, dict[str, float]]
    query_head_weights: dict[int, tuple[float, ...]]
    request_id: str
    decode_position: int
    sequence_length: int
    kv_block_size_tokens: int
    layer_block_tables: dict[int, tuple[int, ...]]
    layer_budgets: dict[int, LayerBudget]
    slot_to_page: dict[int, KVPageId]
    page_to_slot: dict[KVPageId, int]
    queued_transfers: tuple[KVPageId, ...]
    transfer_completion_times_ms: dict[KVPageId, float]
    overlap_budget_ms: float


@dataclass(frozen=True)
class PolicyOutput:
    """What a controller chooses to do for the step.

    This is intentionally small and interpretable.
    A controller can:
    - evict pages
    - prefetch pages
    - update per-layer budgets
    - mark CPU pages for compression later

    Even though compression is not implemented yet, keeping the field here
    makes the interface stable as the project grows.
    """

    evict_pages: tuple[KVPageId, ...] = ()
    prefetch_pages: tuple[KVPageId, ...] = ()
    layer_budgets: tuple[LayerBudget, ...] = ()
    compress_cpu_pages: tuple[KVPageId, ...] = ()


# Keep the older name as an alias so early step-1 examples keep working.
PolicyDecision = PolicyOutput


@dataclass(frozen=True)
class TransferRequest:
    """One scheduled page movement into HBM.

    ``submit_time_ms`` is when the request was issued.
    ``ready_time_ms`` is when the page becomes usable by the kernel.

    A request can be demand-driven or speculative prefetch.
    """

    page: KVPageId
    kind: TransferKind
    ready_time_ms: float
    submit_time_ms: float
    page_bytes: int
    compressed: bool = False


@dataclass
class TransferState:
    """State of the copy/transfer subsystem.

    This is separate from ``CacheState`` because it describes transfers rather
    than residency.

    It tracks:
    - queued pages that have been requested but cannot start yet
    - in-flight transfer requests
    - completion times for transfers already submitted
    - a backlog count
    - an overlap budget that says how much copy time may be hidden under compute
    """

    queued_pages: list[KVPageId] = field(default_factory=list)
    inflight_requests: dict[KVPageId, TransferRequest] = field(default_factory=dict)
    completion_times_ms: dict[KVPageId, float] = field(default_factory=dict)
    backlog: int = 0
    overlap_budget_ms: float = 0.0


@dataclass
class StepMetrics:
    """Measurements collected for one simulated step.

    These metrics are what we will eventually write to CSV and compare across
    controllers and baselines.

    A few especially important fields:
    - ``demand_misses``: pages that were not already ready when the step needed them
    - ``stall_ms``: time the decode step had to wait before it could run
    - ``overlap_ms``: how much copy time can be hidden behind compute
    - ``churn``: rough signal that we are evicting pages only to need them again soon
    """

    step_idx: int
    required_pages: int
    predicted_pages: int
    demand_misses: int = 0
    prefetch_submitted: int = 0
    evictions: int = 0
    stall_ms: float = 0.0
    overlap_ms: float = 0.0
    transfer_wait_ms: float = 0.0
    compute_ms: float = 0.0
    copy_busy_ms: float = 0.0
    queue_delay_ms: float = 0.0
    transfer_backlog: int = 0
    inflight_transfers: int = 0
    churn: int = 0
    demand_miss_pages: tuple[KVPageId, ...] = ()
    prefetched_pages: tuple[KVPageId, ...] = ()
    evicted_pages: tuple[KVPageId, ...] = ()
    accessed_pages: tuple[KVPageId, ...] = ()
    resident_pages_end: tuple[KVPageId, ...] = ()
    notes: list[str] = field(default_factory=list)
