from __future__ import annotations

from .interfaces import HeadWeightedScorer
from .types import KVPageId, WorkloadStep


class PassthroughHeadWeightedScorer(HeadWeightedScorer):
    """Return the score already attached to the workload step.

    This is the simplest concrete scorer and is useful as the default in Step 3.
    The synthetic trace generator already computes head-weighted page scores, so
    this class simply exposes them through the scorer interface.
    """

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        return dict(step.head_weighted_scores)


class NormalizedHeadWeightedScorer(HeadWeightedScorer):
    """Return normalized head-weighted scores in the range [0, 1].

    Normalization is helpful when different traces or layers produce scores on
    different scales but we still want a score-based controller to behave
    consistently.
    """

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        raw = dict(step.head_weighted_scores)
        if not raw:
            return {}
        lo = min(raw.values())
        hi = max(raw.values())
        if hi == lo:
            return {page: 1.0 for page in raw}
        return {page: (score - lo) / (hi - lo) for page, score in raw.items()}
