from __future__ import annotations

from abc import ABC, abstractmethod

from .types import ControllerContext, KVPageId, PolicyOutput, WorkloadStep


class HeadWeightedScorer(ABC):
    """Required page scoring interface.

    All controllers in this project should consume a head-weighted score rather
    than treating score-based logic as optional.

    Why an interface instead of one hard-coded function:
    - We know head-weighted scoring is required.
    - But we may still want multiple *implementations* of that idea later:
      offline-estimated weights, proxy weights, trace-derived weights, etc.
    - An interface lets us swap implementations without changing the simulator.
    """

    @abstractmethod
    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        """Return a head-weighted score for pages relevant to this step."""


class ResidencyController(ABC):
    """Engine-level KV residency controller interface.

    A controller is the "brain" that decides:
    - what to evict
    - what to prefetch
    - how to adjust layer budgets

    The simulator itself does not decide policy.
    It only enforces the physical rules:
    - limited HBM capacity
    - transfers take time
    - decode cannot start until required pages are ready
    """

    @abstractmethod
    def decide(self, context: ControllerContext) -> PolicyOutput:
        """Choose evictions, prefetches, and layer budget updates."""

    def observe(self, context: ControllerContext, decision: PolicyOutput, metrics) -> None:
        """Optional feedback hook called after a step completes.

        Static controllers can ignore this.
        Adaptive / learning-based controllers can use it to update internal
        state from the observed outcome of the action they chose.
        """
