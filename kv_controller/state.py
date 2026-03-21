from __future__ import annotations

from collections import defaultdict

from .types import KVPageId, LayerBudget, TransferRequest, TransferState


class CacheState:
    """Mutable residency state for the simulator core.

    This object stores the simulator's current memory picture.

    The main things it tracks are:
    - which pages are already resident in HBM
    - which pages are still only in CPU memory
    - which pages are currently being transferred
    - when each page was last used
    - how full each layer is in HBM
    - simple long-running statistics like miss rate and churn

    Why keep this separate from the simulator loop:
    - The simulator controls *time and events*.
    - The state object controls *who lives where right now*.
    - That separation makes the code easier to reason about and test.
    """

    def __init__(self, hbm_capacity_pages: int, layers: int):
        # Maximum number of pages that can be resident in the GPU hot tier.
        self.hbm_capacity_pages = hbm_capacity_pages

        # Pages currently available in HBM and ready for compute.
        self.resident_pages: set[KVPageId] = set()

        # Explicit HBM slot bookkeeping. This is important because paged-KV
        # engines do not just care whether a page is resident; they also care
        # which HBM slot currently stores it.
        self.slot_to_page: dict[int, KVPageId] = {}
        self.page_to_slot: dict[KVPageId, int] = {}
        self.reserved_slots: dict[int, KVPageId] = {}

        # Pages available somewhere in CPU-side storage.
        # In this early version we mostly treat CPU as the large backing store.
        self.cpu_pages: set[KVPageId] = set()

        # Pages that have been requested for transfer but are not ready yet.
        self.inflight_pages: dict[KVPageId, TransferRequest] = {}

        # Last decode step that touched each page.
        self.last_access_step: dict[KVPageId, int] = {}

        # Last step in which a page was evicted. Useful for future analysis.
        self.last_evicted_step: dict[KVPageId, int] = {}

        # How many pages each layer currently occupies in HBM.
        self.per_layer_occupancy: dict[int, int] = defaultdict(int)

        # Start with equal layer budgets so every layer gets some share.
        default_budget = max(1, hbm_capacity_pages // max(1, layers))
        self.layer_budgets: dict[int, LayerBudget] = {
            layer_id: LayerBudget(layer_id=layer_id, max_resident_pages=default_budget)
            for layer_id in range(layers)
        }

        # Running stats used to build controller context.
        self.completed_steps = 0
        self.total_required_pages = 0
        self.total_demand_misses = 0
        self.churn = 0
        self.transfer_state = TransferState()

    def has_resident(self, page: KVPageId) -> bool:
        """Return True if the page is already ready in HBM."""
        return page in self.resident_pages

    def has_inflight(self, page: KVPageId) -> bool:
        """Return True if a transfer for this page is already underway."""
        return page in self.inflight_pages

    def mark_cpu_resident(self, pages: set[KVPageId]) -> None:
        """Register pages as available in CPU-side backing storage."""
        self.cpu_pages |= pages

    def mark_access(self, page: KVPageId, step_idx: int) -> None:
        """Remember that a page was used at this step."""
        self.last_access_step[page] = step_idx

    def register_miss_batch(self, misses: int, required_pages: int) -> None:
        """Update rolling miss statistics after a step finishes."""
        self.completed_steps += 1
        self.total_demand_misses += misses
        self.total_required_pages += required_pages

    def miss_rate(self) -> float:
        """Return cumulative demand miss rate across the run so far."""
        if self.total_required_pages == 0:
            return 0.0
        return self.total_demand_misses / self.total_required_pages

    def reserve_transfer(self, request: TransferRequest) -> None:
        """Mark a page as in flight as soon as a transfer is submitted.

        We reserve an HBM slot at submission time, not completion time.
        This mirrors what a real engine would need to know: if a copy is headed
        into HBM, we must already know where it will land.
        """
        if request.page not in self.page_to_slot:
            occupied_slots = set(self.slot_to_page) | set(self.reserved_slots)
            free_slots = [slot for slot in range(self.hbm_capacity_pages) if slot not in occupied_slots]
            if not free_slots:
                raise RuntimeError("No free HBM slot available when reserving transfer.")
            slot = min(free_slots)
            self.page_to_slot[request.page] = slot
            self.reserved_slots[slot] = request.page

        self.inflight_pages[request.page] = request
        self.transfer_state.inflight_requests[request.page] = request
        self.transfer_state.completion_times_ms[request.page] = request.ready_time_ms
        self.transfer_state.backlog = len(self.transfer_state.queued_pages)

    def complete_transfer(self, page: KVPageId) -> None:
        """Move a page from the in-flight set into the resident HBM set."""
        request = self.inflight_pages.pop(page, None)
        if request is None:
            return
        self.resident_pages.add(page)
        self.per_layer_occupancy[page.layer_id] += 1
        self.transfer_state.inflight_requests.pop(page, None)
        slot = self.page_to_slot.get(page)
        if slot is None:
            raise RuntimeError("Transferred page has no reserved HBM slot.")
        reserved_page = self.reserved_slots.pop(slot, None)
        if reserved_page != page:
            raise RuntimeError("Transferred page completed into the wrong reserved HBM slot.")
        self.slot_to_page[slot] = page

    def evict(self, page: KVPageId, step_idx: int) -> None:
        """Remove a page from HBM.

        This method also updates:
        - per-layer occupancy
        - churn signal if the page was used very recently
        - eviction history
        """
        if page not in self.resident_pages:
            return
        self.resident_pages.remove(page)
        slot = self.page_to_slot.pop(page, None)
        if slot is not None:
            self.slot_to_page.pop(slot, None)
            self.reserved_slots.pop(slot, None)
        self.per_layer_occupancy[page.layer_id] = max(0, self.per_layer_occupancy[page.layer_id] - 1)
        if page in self.last_access_step:
            last_access = self.last_access_step[page]
            if last_access >= step_idx - 1:
                self.churn += 1
        self.last_evicted_step[page] = step_idx

    def apply_layer_budgets(self, budgets: tuple[LayerBudget, ...]) -> None:
        """Update layer budgets chosen by a controller."""
        for budget in budgets:
            self.layer_budgets[budget.layer_id] = budget

    def enqueue_transfer(self, page: KVPageId) -> None:
        """Track a page that wants transfer service but cannot start yet."""
        self.transfer_state.queued_pages.append(page)
        self.transfer_state.backlog = len(self.transfer_state.queued_pages)

    def dequeue_transfer(self, page: KVPageId) -> None:
        """Remove a page from the queued-transfer list once it starts moving."""
        if page in self.transfer_state.queued_pages:
            self.transfer_state.queued_pages.remove(page)
        self.transfer_state.backlog = len(self.transfer_state.queued_pages)
