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

from dataclasses import replace

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


class HeadActivityRecomputedScorer(HeadWeightedScorer):
    """Recompute the score directly from stored per-head page activity.

    Why this scorer matters:
    - it uses the more primitive replayable signal instead of trusting the
      already-baked aggregate score
    - it is the simplest "realism check" now that traces preserve
      `per_page_head_activity`

    Formula:
        score(page) = sum_h query_head_weight[layer, h] * page_head_activity(page, h)
    """

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        scores: dict[KVPageId, float] = {}
        for page, activity in step.per_page_head_activity.items():
            layer_weights = step.query_head_weights.get(page.layer_id, ())
            scores[page] = sum(weight * value for weight, value in zip(layer_weights, activity))
        return scores


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


class LayerNormalizedHeadActivityScorer(HeadWeightedScorer):
    """Recompute scores from head activity, then normalize within each layer.

    Why layer-local normalization can help:
    - some layers may naturally produce larger raw score magnitudes
    - cross-layer competition can then become more about scale than about true
      relative importance
    - layer-normalized scores reduce that effect while keeping within-layer
      ranking intact
    """

    def __init__(self):
        self._base = HeadActivityRecomputedScorer()

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        raw = self._base.score_step(step)
        by_layer: dict[int, list[tuple[KVPageId, float]]] = {}
        for page, score in raw.items():
            by_layer.setdefault(page.layer_id, []).append((page, score))

        normalized: dict[KVPageId, float] = {}
        for layer_id, rows in by_layer.items():
            values = [score for _, score in rows]
            lo = min(values)
            hi = max(values)
            if hi == lo:
                for page, _ in rows:
                    normalized[page] = 1.0
                continue
            for page, score in rows:
                normalized[page] = (score - lo) / (hi - lo)
        return normalized


class PredictedBoostedHeadActivityScorer(HeadWeightedScorer):
    """Recompute scores from head activity and slightly boost predicted pages.

    This is a simple proxy for "pages likely needed soon should count a bit
    more" without changing the underlying page ordering too aggressively.
    """

    def __init__(self, predicted_boost: float = 1.15):
        self.predicted_boost = predicted_boost
        self._base = HeadActivityRecomputedScorer()

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        base_scores = self._base.score_step(step)
        predicted = set(step.predicted_pages)
        return {
            page: score * (self.predicted_boost if page in predicted else 1.0)
            for page, score in base_scores.items()
        }


def apply_scorer_to_trace(trace: list[WorkloadStep], scorer: HeadWeightedScorer) -> list[WorkloadStep]:
    """Return a new trace whose step scores come from `scorer`.

    This helper is the key replay-side tuning primitive:
    - load one fixed trace
    - apply several scorers to that exact trace
    - compare policy quality without changing the workload itself
    """

    rescored_trace: list[WorkloadStep] = []
    for step in trace:
        rescored = scorer.score_step(step)
        updated_features = {
            page: {
                **features,
                "head_weighted_score": rescored.get(page, features.get("head_weighted_score", 0.0)),
            }
            for page, features in step.per_page_features.items()
        }
        rescored_trace.append(
            replace(
                step,
                head_weighted_scores=rescored,
                per_page_features=updated_features,
            )
        )
    return rescored_trace
