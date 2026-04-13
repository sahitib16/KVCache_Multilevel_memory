from __future__ import annotations

from .interfaces import ResidencyController
from .scheduler import OverlapTransferScheduler
from .state import CacheState
from .types import ControllerContext, KVPageId, PolicyOutput, SimulationConfig, StepMetrics, TransferKind, WorkloadStep


class OverlapAwareSimulator:
    """Page-level KV residency simulator with optional overlap timing.

    The most important part of this class is *not* timing fidelity.
    Its primary job is to enforce page-level correctness:
    - which pages are required for the step
    - which pages are resident already
    - which pages must be transferred
    - which pages may be evicted or prefetched

    Timing is modeled too, but it is secondary. The simulator should first be
    read as a page-wise state machine with a small timing layer attached.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = CacheState(
            hbm_capacity_pages=config.cache.hbm_capacity_pages,
            layers=config.cache.layers,
        )
        self.scheduler = OverlapTransferScheduler(config.transfer)

    def seed_cpu_pages(self, trace: list[WorkloadStep]) -> None:
        """Populate the backing CPU tier with every page used by the trace.

        In a real engine, these pages would already exist somewhere in host
        memory or on disk. Here we simply register them so the simulator knows
        they are available to be moved into HBM.
        """
        all_pages: set[KVPageId] = set()
        for step in trace:
            all_pages |= set(step.required_pages)
            all_pages |= set(step.predicted_pages)
        self.state.mark_cpu_resident(all_pages)

    def _make_context(self, step: WorkloadStep) -> ControllerContext:
        """Build the read-only snapshot the controller sees for this step."""
        required_by_layer: dict[int, int] = {}
        for page in step.required_pages:
            required_by_layer[page.layer_id] = required_by_layer.get(page.layer_id, 0) + 1

        per_layer_pressure: dict[int, float] = {}
        visible_layers = set(step.referenced_layers) | set(required_by_layer) | set(self.state.layer_budgets)
        for layer_id in visible_layers:
            budget = self.state.layer_budgets.get(layer_id)
            allowed_pages = budget.max_resident_pages if budget is not None else self.config.cache.hbm_capacity_pages
            per_layer_pressure[layer_id] = required_by_layer.get(layer_id, 0) / max(1, allowed_pages)

        return ControllerContext(
            step_idx=step.step_idx,
            required_pages=step.required_pages,
            predicted_pages=step.predicted_pages,
            resident_pages=frozenset(self.state.resident_pages),
            inflight_pages=frozenset(self.state.inflight_pages.keys()),
            hbm_capacity_pages=self.config.cache.hbm_capacity_pages,
            transfer_backlog=self.state.transfer_state.backlog,
            inflight_transfer_count=self.scheduler.inflight_count(),
            miss_rate=self.state.miss_rate(),
            churn=self.state.churn,
            per_layer_occupancy=dict(self.state.per_layer_occupancy),
            per_layer_pressure=per_layer_pressure,
            head_weighted_scores=step.head_weighted_scores,
            per_page_head_activity=step.per_page_head_activity,
            per_page_features=step.per_page_features,
            query_head_weights=step.query_head_weights,
            request_id=step.request_id,
            decode_position=step.decode_position,
            sequence_length=step.sequence_length,
            kv_block_size_tokens=step.kv_block_size_tokens,
            layer_block_tables=step.layer_block_tables,
            layer_budgets=dict(self.state.layer_budgets),
            slot_to_page=dict(self.state.slot_to_page),
            page_to_slot=dict(self.state.page_to_slot),
            queued_transfers=tuple(self.state.transfer_state.queued_pages),
            transfer_completion_times_ms=dict(self.state.transfer_state.completion_times_ms),
            overlap_budget_ms=self.state.transfer_state.overlap_budget_ms,
        )

    def _check_step_feasible(self, step: WorkloadStep) -> None:
        """Reject traces that require more pages at once than HBM can ever hold."""
        unique_required_pages = len(set(step.required_pages))
        if unique_required_pages > self.config.cache.hbm_capacity_pages:
            raise RuntimeError(
                "Step requires more pages than total HBM capacity. "
                f"required={unique_required_pages}, capacity={self.config.cache.hbm_capacity_pages}"
            )

    def _apply_budget_and_eviction_policy(
        self,
        step: WorkloadStep,
        decision: PolicyOutput,
        metrics: StepMetrics,
        evicted_pages: list[KVPageId],
    ) -> None:
        """Apply controller-chosen budgets and explicit evictions."""
        self.state.apply_layer_budgets(decision.layer_budgets)
        self._enforce_layer_budgets(step.required_pages, metrics, evicted_pages)
        self._evict_if_needed(step.required_pages, decision, metrics, evicted_pages)

    def _prefetch_decision_pages(
        self,
        step: WorkloadStep,
        decision: PolicyOutput,
        metrics: StepMetrics,
        prefetched_pages: list[KVPageId],
        evicted_pages: list[KVPageId],
    ) -> None:
        """Submit controller-chosen prefetches when capacity allows."""
        for page in decision.prefetch_pages:
            if self.state.has_resident(page) or self.state.has_inflight(page):
                continue
            self._evict_if_needed(step.required_pages, decision, metrics, evicted_pages)
            if len(self.state.resident_pages) + len(self.state.inflight_pages) >= self.config.cache.hbm_capacity_pages:
                continue
            self.scheduler.submit(page, TransferKind.PREFETCH, self.state, metrics)
            metrics.prefetch_submitted += 1
            prefetched_pages.append(page)

    def _submit_required_pages(
        self,
        step: WorkloadStep,
        decision: PolicyOutput,
        metrics: StepMetrics,
        demand_miss_pages: list[KVPageId],
        evicted_pages: list[KVPageId],
    ) -> int:
        """Ensure every required page is resident or in flight."""
        demand_misses = 0
        for page in step.required_pages:
            if self.state.has_resident(page) or self.state.has_inflight(page):
                continue
            self._evict_if_needed(step.required_pages, decision, metrics, evicted_pages)
            if len(self.state.resident_pages) + len(self.state.inflight_pages) >= self.config.cache.hbm_capacity_pages:
                self._evict_if_needed(step.required_pages, PolicyOutput(), metrics, evicted_pages)
            self.scheduler.submit(page, TransferKind.DEMAND, self.state, metrics)
            demand_misses += 1
            demand_miss_pages.append(page)
        return demand_misses

    def _verify_required_pages_ready(self, step: WorkloadStep) -> None:
        """Enforce the core simulator invariant: decode only launches on-resident pages."""
        missing_residency = [page for page in step.required_pages if page not in self.state.resident_pages]
        if missing_residency:
            raise RuntimeError(f"Decode launch attempted before residency was ready: {missing_residency}")
        missing_slots = [page for page in step.required_pages if page not in self.state.page_to_slot]
        if missing_slots:
            raise RuntimeError(f"Decode launch attempted without HBM slot assignment: {missing_slots}")

    def _record_page_accesses(self, step: WorkloadStep, accessed_pages: list[KVPageId]) -> None:
        """Mark the step's required pages as accessed once decode can legally run."""
        for page in step.required_pages:
            self.state.mark_access(page, step.step_idx)
            accessed_pages.append(page)

    def _finalize_metrics(
        self,
        metrics: StepMetrics,
        demand_misses: int,
        demand_miss_pages: list[KVPageId],
        prefetched_pages: list[KVPageId],
        evicted_pages: list[KVPageId],
        accessed_pages: list[KVPageId],
    ) -> None:
        """Write final page-wise and transfer-wise summaries into `StepMetrics`."""
        metrics.demand_misses = demand_misses
        metrics.inflight_transfers = self.scheduler.inflight_count()
        metrics.transfer_backlog = self.state.transfer_state.backlog
        metrics.churn = self.state.churn
        metrics.demand_miss_pages = tuple(demand_miss_pages)
        metrics.prefetched_pages = tuple(prefetched_pages)
        metrics.evicted_pages = tuple(evicted_pages)
        metrics.accessed_pages = tuple(accessed_pages)
        metrics.resident_pages_end = tuple(sorted(self.state.resident_pages))

    def _evict_if_needed(
        self,
        required_pages: tuple[KVPageId, ...],
        decision: PolicyOutput,
        metrics: StepMetrics,
        evicted_pages: list[KVPageId] | None = None,
    ) -> None:
        """Apply explicit evictions and, if necessary, fallback capacity evictions.

        ``protected`` pages are pages we should not throw out right now:
        - pages required by the current step
        - pages currently being transferred

        If a controller does not free enough space explicitly, we fall back to
        evicting the least recently used unprotected page.
        """
        protected = set(required_pages) | set(self.state.inflight_pages.keys())
        for page in decision.evict_pages:
            if page in protected:
                continue
            if page not in self.state.resident_pages:
                continue
            self.state.evict(page, metrics.step_idx)
            metrics.evictions += 1
            if evicted_pages is not None:
                evicted_pages.append(page)

        while len(self.state.resident_pages) + len(self.state.inflight_pages) >= self.config.cache.hbm_capacity_pages:
            candidates = [page for page in self.state.resident_pages if page not in protected]
            if not candidates:
                break
            victim = min(candidates, key=lambda page: self.state.last_access_step.get(page, -1))
            self.state.evict(victim, metrics.step_idx)
            metrics.evictions += 1
            if evicted_pages is not None:
                evicted_pages.append(victim)

    def _enforce_layer_budgets(
        self,
        required_pages: tuple[KVPageId, ...],
        metrics: StepMetrics,
        evicted_pages: list[KVPageId] | None = None,
    ) -> None:
        """Evict over-budget layers until their occupancy fits the policy budgets.

        Why this helper exists:
        - controllers can now emit per-layer HBM budgets as part of `PolicyOutput`
        - those budgets only matter if the simulator actively enforces them

        Enforcement rule:
        - never evict a page required by the current decode step
        - never evict a page that is currently in flight
        - within an over-budget layer, fall back to LRU-style victim choice

        This keeps the behavior simple and understandable while making the
        budget-control knob real for both static layer-aware policies and the
        contextual bandit.
        """

        protected = set(required_pages) | set(self.state.inflight_pages.keys())

        for layer_id, budget in self.state.layer_budgets.items():
            max_pages = budget.max_resident_pages
            while self.state.per_layer_occupancy.get(layer_id, 0) > max_pages:
                candidates = [
                    page
                    for page in self.state.resident_pages
                    if page.layer_id == layer_id and page not in protected
                ]
                if not candidates:
                    break
                victim = min(candidates, key=lambda page: self.state.last_access_step.get(page, -1))
                self.state.evict(victim, metrics.step_idx)
                metrics.evictions += 1
                if evicted_pages is not None:
                    evicted_pages.append(victim)

    def run_step(self, step: WorkloadStep, controller: ResidencyController) -> StepMetrics:
        """Run one decode step.

        The intended reading order is:
        1. complete any already-finished transfers
        2. ask the controller for a page-management decision
        3. apply evictions / budgets / prefetches
        4. guarantee required pages are present
        5. launch decode only after the required set is resident
        6. record page-wise statistics
        """

        self._check_step_feasible(step)
        self.scheduler.advance_to(self.scheduler.now_ms, self.state)
        metrics = StepMetrics(
            step_idx=step.step_idx,
            required_pages=len(step.required_pages),
            predicted_pages=len(step.predicted_pages),
        )
        demand_miss_pages: list[KVPageId] = []
        prefetched_pages: list[KVPageId] = []
        evicted_pages: list[KVPageId] = []
        accessed_pages: list[KVPageId] = []

        context = self._make_context(step)
        decision = controller.decide(context)
        self._apply_budget_and_eviction_policy(step, decision, metrics, evicted_pages)
        self._prefetch_decision_pages(step, decision, metrics, prefetched_pages, evicted_pages)
        demand_misses = self._submit_required_pages(step, decision, metrics, demand_miss_pages, evicted_pages)
        self.scheduler.wait_for_pages(step.required_pages, self.state, metrics)
        self._verify_required_pages_ready(step)
        self._record_page_accesses(step, accessed_pages)
        self.scheduler.overlap_compute(self.state, metrics)
        self._finalize_metrics(
            metrics,
            demand_misses,
            demand_miss_pages,
            prefetched_pages,
            evicted_pages,
            accessed_pages,
        )
        self.state.register_miss_batch(demand_misses, len(step.required_pages))
        controller.observe(context, decision, metrics)
        return metrics

    def run(self, trace: list[WorkloadStep], controller: ResidencyController) -> list[StepMetrics]:
        """Run an entire workload trace through one controller."""
        self.seed_cpu_pages(trace)
        return [self.run_step(step, controller) for step in trace]
