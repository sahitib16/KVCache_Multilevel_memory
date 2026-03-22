from __future__ import annotations

"""Concrete score adapters for the simulator.

The project requires head-weighted page scoring, but there can still be
multiple useful *views* of that score:
- use it exactly as supplied by the workload
- normalize it so different traces are easier to compare

This file keeps those choices separate from the controllers.
That way:
- the workload is responsible for producing score signals
- the scorer is responsible for exposing / transforming them
- the controller is responsible for acting on them
"""

from .interfaces import HeadWeightedScorer
from .types import KVPageId, WorkloadStep


class PassthroughHeadWeightedScorer(HeadWeightedScorer):
    """Return the score already attached to the workload step.

    This is the most literal scorer possible.

    Use this when:
    - the workload already computed the score you want
    - you do not want any extra transformation
    - you want the controller to consume the exact signal produced upstream

    In the current simulator, synthetic traces already include
    `head_weighted_scores`, so this scorer is a natural default.
    """

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        # Return a copy so downstream code can mutate its local view without
        # accidentally changing the original step object.
        return dict(step.head_weighted_scores)


class NormalizedHeadWeightedScorer(HeadWeightedScorer):
    """Return normalized head-weighted scores in the range [0, 1].

    Why normalization can help:
    - Different traces may naturally produce larger or smaller score values.
    - Different layers may have different score scales.
    - Controllers that compare scores across steps often behave more
      consistently when the score range is standardized.

    Normalization rule:
    - if all scores are identical, return 1.0 for every page
    - otherwise linearly map:

        normalized = (score - min_score) / (max_score - min_score)
    """

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        raw = dict(step.head_weighted_scores)
        if not raw:
            return {}

        lo = min(raw.values())
        hi = max(raw.values())

        # If every page has the same score, there is no meaningful ordering.
        # Returning 1.0 everywhere keeps the interface simple and stable.
        if hi == lo:
            return {page: 1.0 for page in raw}

        return {
            page: (score - lo) / (hi - lo)
            for page, score in raw.items()
        }
