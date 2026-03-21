from __future__ import annotations

import heapq

from .state import CacheState
from .types import KVPageId, StepMetrics, TransferConfig, TransferKind, TransferRequest


class OverlapTransferScheduler:
    """Discrete-event transfer scheduler with copy/compute overlap.

    This class is the timing engine for data movement.

    Important idea:
    - We are not using real CUDA streams here.
    - Instead, we simulate time with a small event system.

    The scheduler answers questions like:
    - If I submit a page copy now, when will it be ready?
    - If there are already too many copies in flight, how long must I wait?
    - Can compute run while some copies are still finishing?

    Why use a heap:
    - We always care about the transfer that completes the soonest.
    - A min-heap makes "what finishes next?" cheap to answer.
    """

    def __init__(self, config: TransferConfig):
        self.config = config

        # Global simulated time in milliseconds.
        self.now_ms = 0.0

        # Heap entries look like:
        #   (ready_time_ms, insertion_sequence, TransferRequest)
        #
        # The insertion sequence breaks ties safely when two transfers finish at
        # the same time.
        self._active_heap: list[tuple[float, int, TransferRequest]] = []
        self._seq = 0

    def inflight_count(self) -> int:
        """How many transfers are currently active."""
        return len(self._active_heap)

    def _transfer_duration_ms(self, page_bytes: int) -> float:
        """Convert page size into simulated transfer time."""
        return self.config.transfer_setup_ms + (page_bytes / self.config.bandwidth_bytes_per_ms)

    def advance_to(self, target_ms: float, state: CacheState) -> float:
        """Advance simulated time and complete transfers that finish before target.

        Example:
        - current time is 3.0 ms
        - next transfer completes at 3.4 ms
        - target is 5.0 ms

        Then we:
        1. pop the 3.4 ms transfer
        2. mark its page resident
        3. continue until all transfers due before 5.0 ms are done
        4. move current time to 5.0 ms

        The returned value is a rough "copy engine busy time" accumulated during
        the advance.
        """
        busy_ms = 0.0
        while self._active_heap and self._active_heap[0][0] <= target_ms:
            ready_time, _, request = heapq.heappop(self._active_heap)
            if ready_time > self.now_ms:
                busy_ms += ready_time - self.now_ms
                self.now_ms = ready_time
            state.complete_transfer(request.page)
            state.transfer_state.overlap_budget_ms = max(
                0.0,
                state.transfer_state.overlap_budget_ms - (ready_time - request.submit_time_ms),
            )
        if target_ms > self.now_ms:
            self.now_ms = target_ms
        return busy_ms

    def submit(
        self,
        page: KVPageId,
        kind: TransferKind,
        state: CacheState,
        metrics: StepMetrics,
        compressed: bool = False,
    ) -> TransferRequest:
        """Queue or start a transfer for one page.

        If there is an available transfer slot, the page starts moving now.
        If not, we wait until the earliest in-flight transfer finishes.

        This is how the simulator models a limited number of copy "lanes".
        """
        self.advance_to(self.now_ms, state)
        start_ms = self.now_ms
        if self.inflight_count() >= self.config.max_inflight_transfers:
            state.enqueue_transfer(page)
            earliest_ready, _, _ = self._active_heap[0]
            start_ms = max(start_ms, earliest_ready)
            metrics.queue_delay_ms += max(0.0, start_ms - self.now_ms)
            self.advance_to(start_ms, state)
            state.dequeue_transfer(page)

        ready_ms = start_ms + self._transfer_duration_ms(self.config.page_bytes)
        request = TransferRequest(
            page=page,
            kind=kind,
            ready_time_ms=ready_ms,
            submit_time_ms=self.now_ms,
            page_bytes=self.config.page_bytes,
            compressed=compressed,
        )
        self._seq += 1
        heapq.heappush(self._active_heap, (ready_ms, self._seq, request))
        state.reserve_transfer(request)
        state.transfer_state.overlap_budget_ms += max(0.0, ready_ms - start_ms)
        metrics.copy_busy_ms += ready_ms - start_ms
        return request

    def wait_for_pages(
        self,
        pages: tuple[KVPageId, ...],
        state: CacheState,
        metrics: StepMetrics,
    ) -> None:
        """Block until all requested pages are resident.

        This method represents the "decode cannot launch yet" rule.

        The controller may have scheduled transfers earlier, but if any page
        required by the current step is still missing, decode has to wait.
        The waiting time is recorded as stall.
        """
        while True:
            missing = [page for page in pages if not state.has_resident(page)]
            if not missing:
                return
            next_ready = min(
                state.inflight_pages[page].ready_time_ms
                for page in missing
                if page in state.inflight_pages
            )
            wait_ms = max(0.0, next_ready - self.now_ms)
            metrics.transfer_wait_ms += wait_ms
            metrics.stall_ms += wait_ms
            self.advance_to(next_ready, state)

    def overlap_compute(self, state: CacheState, metrics: StepMetrics) -> None:
        """Advance compute and let copies finish in parallel.

        Conceptually this models:
        - compute stream doing the decode kernel
        - copy stream continuing independent transfers

        Any transfers that finish during this window become resident without
        adding stall to the current step.
        """
        start_ms = self.now_ms
        end_ms = start_ms + self.config.decode_kernel_ms
        self.advance_to(end_ms, state)
        metrics.compute_ms += self.config.decode_kernel_ms
        metrics.overlap_ms += max(0.0, min(metrics.copy_busy_ms, self.config.decode_kernel_ms))
        state.transfer_state.overlap_budget_ms = max(
            0.0,
            state.transfer_state.overlap_budget_ms - self.config.decode_kernel_ms,
        )
