from __future__ import annotations

from .interfaces import ResidencyController
from .scheduler import OverlapTransferScheduler
from .state import CacheState
from .types import ControllerContext, KVPageId, PolicyOutput, SimulationConfig, StepMetrics, TransferKind, WorkloadStep


class OverlapAwareSimulator:
    """Reusable engine-level simulator for page residency and transfer overlap.

    This is the top-level object that "runs the world".

    It is responsible for:
    - holding onto cache state
    - holding onto the transfer scheduler
    - building controller context each step
    - applying policy decisions
    - ensuring required pages are resident before decode
    - collecting metrics

    The simulator itself does not contain a smart policy.
    Instead, it asks a controller what to do, then enforces the system rules.
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
            head_weighted_scores=step.head_weighted_scores,
            per_page_head_activity=step.per_page_head_activity,
            per_page_features=step.per_page_features,
            query_head_weights=step.query_head_weights,
            layer_budgets=dict(self.state.layer_budgets),
            slot_to_page=dict(self.state.slot_to_page),
            page_to_slot=dict(self.state.page_to_slot),
            queued_transfers=tuple(self.state.transfer_state.queued_pages),
            transfer_completion_times_ms=dict(self.state.transfer_state.completion_times_ms),
            overlap_budget_ms=self.state.transfer_state.overlap_budget_ms,
        )

    def _evict_if_needed(
        self,
        required_pages: tuple[KVPageId, ...],
        decision: PolicyOutput,
        metrics: StepMetrics,
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

        while len(self.state.resident_pages) + len(self.state.inflight_pages) >= self.config.cache.hbm_capacity_pages:
            candidates = [page for page in self.state.resident_pages if page not in protected]
            if not candidates:
                break
            victim = min(candidates, key=lambda page: self.state.last_access_step.get(page, -1))
            self.state.evict(victim, metrics.step_idx)
            metrics.evictions += 1

    def _enforce_layer_budgets(
        self,
        required_pages: tuple[KVPageId, ...],
        metrics: StepMetrics,
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

    def run_step(self, step: WorkloadStep, controller: ResidencyController) -> StepMetrics:
        """Run one decode step from controller decision to decode completion."""

        # First, let any transfers that should already be complete become
        # visible in the state.
        self.scheduler.advance_to(self.scheduler.now_ms, self.state)

        # Initialize metric storage for this step.
        metrics = StepMetrics(
            step_idx=step.step_idx,
            required_pages=len(step.required_pages),
            predicted_pages=len(step.predicted_pages),
        )

        # Ask the controller what to do, given the current system state.
        context = self._make_context(step)
        decision = controller.decide(context)

        # Apply any layer budget changes first so later policy logic can honor
        # the new budget picture.
        self.state.apply_layer_budgets(decision.layer_budgets)
        self._enforce_layer_budgets(step.required_pages, metrics)
        self._evict_if_needed(step.required_pages, decision, metrics)

        # Submit proactive prefetches chosen by the controller.
        for page in decision.prefetch_pages:
            if self.state.has_resident(page) or self.state.has_inflight(page):
                continue
            self._evict_if_needed(step.required_pages, decision, metrics)
            if len(self.state.resident_pages) + len(self.state.inflight_pages) >= self.config.cache.hbm_capacity_pages:
                continue
            self.scheduler.submit(page, TransferKind.PREFETCH, self.state, metrics)
            metrics.prefetch_submitted += 1

        # Now guarantee the current step's required pages are on the way.
        demand_misses = 0
        for page in step.required_pages:
            if self.state.has_resident(page):
                self.state.mark_access(page, step.step_idx)
                continue
            if not self.state.has_inflight(page):
                self._evict_if_needed(step.required_pages, decision, metrics)
                if len(self.state.resident_pages) + len(self.state.inflight_pages) >= self.config.cache.hbm_capacity_pages:
                    # Final safety valve: if we are still full, fall back to the
                    # simulator's default eviction behavior.
                    self._evict_if_needed(step.required_pages, PolicyOutput(), metrics)
                self.scheduler.submit(page, TransferKind.DEMAND, self.state, metrics)
                demand_misses += 1

        metrics.demand_misses = demand_misses

        # The decode step must wait until every required page is actually ready.
        self.scheduler.wait_for_pages(step.required_pages, self.state, metrics)

        # Engine-level correctness invariant:
        # the decode kernel is only allowed to launch after every required page
        # is resident in HBM and mapped to a slot.
        missing_residency = [page for page in step.required_pages if page not in self.state.resident_pages]
        if missing_residency:
            raise RuntimeError(f"Decode launch attempted before residency was ready: {missing_residency}")
        missing_slots = [page for page in step.required_pages if page not in self.state.page_to_slot]
        if missing_slots:
            raise RuntimeError(f"Decode launch attempted without HBM slot assignment: {missing_slots}")

        # Once the pages are available, the current step touches them.
        for page in step.required_pages:
            self.state.mark_access(page, step.step_idx)

        # Run the simulated decode kernel and allow overlap with background copy.
        self.scheduler.overlap_compute(self.state, metrics)

        # Record some end-of-step state summaries.
        metrics.inflight_transfers = self.scheduler.inflight_count()
        metrics.transfer_backlog = self.state.transfer_state.backlog
        metrics.churn = self.state.churn
        self.state.register_miss_batch(demand_misses, len(step.required_pages))
        controller.observe(context, decision, metrics)
        return metrics

    def run(self, trace: list[WorkloadStep], controller: ResidencyController) -> list[StepMetrics]:
        """Run an entire workload trace through one controller."""
        self.seed_cpu_pages(trace)
        return [self.run_step(step, controller) for step in trace]
