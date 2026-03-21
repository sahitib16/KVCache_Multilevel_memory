from __future__ import annotations

from collections import defaultdict

from .interfaces import ResidencyController
from .scoring import PassthroughHeadWeightedScorer
from .types import ControllerContext, KVPageId, LayerBudget, PolicyOutput, WorkloadStep


def _score_lookup(context: ControllerContext, page: KVPageId) -> float:
    return context.head_weighted_scores.get(page, context.per_page_features.get(page, {}).get("head_weighted_score", 0.0))


def _lru_order(context: ControllerContext, pages: list[KVPageId]) -> list[KVPageId]:
    # Lower score means "older / worse" only because the current simulator core
    # does not yet pass last-access directly into the context. For a pure LRU
    # controller in Step 3, we simply avoid explicit evictions and let the
    # simulator's fallback LRU behavior handle victim choice.
    return pages


class LRUController(ResidencyController):
    """Baseline controller that relies on simulator fallback LRU eviction."""

    def __init__(self, prefetch_k: int = 0):
        self.prefetch_k = prefetch_k

    def decide(self, context: ControllerContext) -> PolicyOutput:
        return PolicyOutput(prefetch_pages=context.predicted_pages[: self.prefetch_k])


class ScoreBasedController(ResidencyController):
    """Baseline controller that uses head-weighted score for prefetch/eviction."""

    def __init__(self, prefetch_k: int = 0):
        self.prefetch_k = prefetch_k

    def decide(self, context: ControllerContext) -> PolicyOutput:
        predicted = sorted(
            context.predicted_pages,
            key=lambda page: _score_lookup(context, page),
            reverse=True,
        )

        protected = set(context.required_pages) | set(context.inflight_pages)
        resident_candidates = [page for page in context.resident_pages if page not in protected]

        incoming_prefetch = [
            page for page in predicted[: self.prefetch_k]
            if page not in context.resident_pages and page not in context.inflight_pages
        ]
        free_slots = context.hbm_capacity_pages - len(context.resident_pages) - len(context.inflight_pages)
        need_slots = max(0, len(incoming_prefetch) - free_slots)

        evict_pages = tuple(
            sorted(resident_candidates, key=lambda page: _score_lookup(context, page))[:need_slots]
        )
        return PolicyOutput(
            evict_pages=evict_pages,
            prefetch_pages=tuple(incoming_prefetch),
        )


class FutureTraceOracle:
    """Helper that exposes future page-use information for oracle policies."""

    def __init__(self, trace: list[WorkloadStep]):
        self.trace = trace
        self._future_steps_by_page: dict[KVPageId, list[int]] = defaultdict(list)
        for step in trace:
            for page in step.required_pages:
                self._future_steps_by_page[page].append(step.step_idx)

    def next_use_after(self, page: KVPageId, step_idx: int) -> int | None:
        for future_step in self._future_steps_by_page.get(page, []):
            if future_step > step_idx:
                return future_step
        return None

    def next_required_pages(self, step_idx: int) -> tuple[KVPageId, ...]:
        next_idx = step_idx + 1
        if next_idx >= len(self.trace):
            return ()
        return self.trace[next_idx].required_pages


class BeladyOracleController(ResidencyController):
    """Future-aware eviction oracle using Belady's farthest-next-use rule."""

    def __init__(self, trace: list[WorkloadStep]):
        self.oracle = FutureTraceOracle(trace)

    def _choose_evictions(self, context: ControllerContext, pages_to_make_room_for: int) -> tuple[KVPageId, ...]:
        protected = set(context.required_pages) | set(context.inflight_pages)
        candidates = [page for page in context.resident_pages if page not in protected]
        candidates.sort(
            key=lambda page: self.oracle.next_use_after(page, context.step_idx)
            if self.oracle.next_use_after(page, context.step_idx) is not None
            else float("inf"),
            reverse=True,
        )
        return tuple(candidates[:pages_to_make_room_for])

    def decide(self, context: ControllerContext) -> PolicyOutput:
        missing_required = [
            page for page in context.required_pages
            if page not in context.resident_pages and page not in context.inflight_pages
        ]
        free_slots = context.hbm_capacity_pages - len(context.resident_pages) - len(context.inflight_pages)
        need_slots = max(0, len(missing_required) - free_slots)
        return PolicyOutput(evict_pages=self._choose_evictions(context, need_slots))


class PerfectPrefetchOracleController(ResidencyController):
    """Future-aware oracle that prefetches exact next-step required pages."""

    def __init__(self, trace: list[WorkloadStep], prefetch_k: int | None = None):
        self.oracle = FutureTraceOracle(trace)
        self.prefetch_k = prefetch_k

    def _choose_evictions(self, context: ControllerContext, pages_to_make_room_for: int) -> tuple[KVPageId, ...]:
        protected = set(context.required_pages) | set(context.inflight_pages)
        candidates = [page for page in context.resident_pages if page not in protected]
        candidates.sort(
            key=lambda page: self.oracle.next_use_after(page, context.step_idx)
            if self.oracle.next_use_after(page, context.step_idx) is not None
            else float("inf"),
            reverse=True,
        )
        return tuple(candidates[:pages_to_make_room_for])

    def decide(self, context: ControllerContext) -> PolicyOutput:
        next_required = list(self.oracle.next_required_pages(context.step_idx))
        if self.prefetch_k is not None:
            next_required = next_required[: self.prefetch_k]

        prefetch_pages = [
            page for page in next_required
            if page not in context.resident_pages and page not in context.inflight_pages
        ]

        free_slots = context.hbm_capacity_pages - len(context.resident_pages) - len(context.inflight_pages)
        need_slots = max(0, len(prefetch_pages) - free_slots)

        return PolicyOutput(
            evict_pages=self._choose_evictions(context, need_slots),
            prefetch_pages=tuple(prefetch_pages),
        )
