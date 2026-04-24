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

from collections import defaultdict, deque
from dataclasses import replace

from .interfaces import HeadWeightedScorer
from .types import KVPageId, WorkloadStep


def _normalize_scores(raw: dict[KVPageId, float]) -> dict[KVPageId, float]:
    """Min-max normalize one step's score dictionary to `[0, 1]`."""
    if not raw:
        return {}
    lo = min(raw.values())
    hi = max(raw.values())
    if hi == lo:
        return {page: 1.0 for page in raw}
    return {
        page: (score - lo) / (hi - lo)
        for page, score in raw.items()
    }


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
        return _normalize_scores(dict(step.head_weighted_scores))


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


class ReuseDistanceHybridScorer(HeadWeightedScorer):
    """Combine normalized head score with inverse reuse distance.

    This is the first novelty-phase scorer:

        score = alpha * normalized_head_score + beta * inverse_reuse_distance

    where `inverse_reuse_distance` is expected to be attached to
    `per_page_features` by `attach_reuse_distance_features(...)`.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.25):
        self.alpha = alpha
        self.beta = beta

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        head_scores = _normalize_scores(dict(step.head_weighted_scores))
        pages = set(head_scores) | set(step.per_page_features)
        hybrid: dict[KVPageId, float] = {}
        for page in pages:
            head_score = head_scores.get(page, 0.0)
            reuse_inv = float(step.per_page_features.get(page, {}).get("reuse_distance_inverse", 0.0))
            hybrid[page] = self.alpha * head_score + self.beta * reuse_inv
        return hybrid


class PageStatsHybridScorer(HeadWeightedScorer):
    """Combine head score with causal page-history and tile-history features.

    This is a stronger reuse-aware scorer than `ReuseDistanceHybridScorer`.
    Instead of only asking "when was this exact page last accessed globally?",
    it uses:
    - request-aware inverse reuse distance
    - recent per-page frequency
    - recent per-request page frequency
    - recent tile hotness

    All signals are causal: they are computed from previous accesses only.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta_reuse: float = 0.20,
        gamma_page_hotness: float = 0.20,
        delta_tile_hotness: float = 0.10,
    ):
        self.alpha = alpha
        self.beta_reuse = beta_reuse
        self.gamma_page_hotness = gamma_page_hotness
        self.delta_tile_hotness = delta_tile_hotness

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        head_scores = _normalize_scores(dict(step.head_weighted_scores))
        pages = set(head_scores) | set(step.per_page_features)

        page_hotness = _normalize_feature(step, pages, "page_recent_frequency")
        request_hotness = _normalize_feature(step, pages, "request_page_recent_frequency")
        tile_hotness = _normalize_feature(step, pages, "tile_recent_frequency")

        hybrid: dict[KVPageId, float] = {}
        for page in pages:
            features = step.per_page_features.get(page, {})
            reuse_inv = float(features.get("request_reuse_distance_inverse", features.get("reuse_distance_inverse", 0.0)))
            hybrid[page] = (
                self.alpha * head_scores.get(page, 0.0)
                + self.beta_reuse * reuse_inv
                + self.gamma_page_hotness * max(page_hotness.get(page, 0.0), request_hotness.get(page, 0.0))
                + self.delta_tile_hotness * tile_hotness.get(page, 0.0)
            )
        return hybrid


class RegimeAwarePageStatsScorer(HeadWeightedScorer):
    """Choose between normalized and page-stats hybrid using causal regime cues.

    Current design:
    - use the simpler normalized scorer in prefetch-hostile regimes
    - switch to a tuned page-history scorer only when the step shows a large
      predicted-page footprint

    Why predicted footprint is the main trigger:
    - on our current benchmark suite, request-local reuse exists in both the
      main and hard traces
    - what separates them more cleanly is whether the workload offers a large
      predicted set that prefetch can exploit
    - this makes predicted-page footprint a better causal proxy for
      "prefetch-friendly regime" than reuse alone

    The high-pressure scorer uses slightly softer page-history weights than the
    standalone `PageStatsHybridScorer`, because that produced a better
    main-vs-hard compromise in replay experiments.

    This is intentionally lightweight and interpretable. It is meant to test
    the claim that different scorers are practical in different regimes.
    """

    def __init__(
        self,
        predicted_ratio_threshold: float = 1.00,
        normalized_scorer: HeadWeightedScorer | None = None,
        high_reuse_scorer: HeadWeightedScorer | None = None,
    ):
        self.predicted_ratio_threshold = predicted_ratio_threshold
        self.normalized_scorer = normalized_scorer or NormalizedHeadWeightedScorer()
        self.high_reuse_scorer = high_reuse_scorer or PageStatsHybridScorer(
            beta_reuse=0.12,
            gamma_page_hotness=0.18,
            delta_tile_hotness=0.05,
        )

    def score_step(self, step: WorkloadStep) -> dict[KVPageId, float]:
        if not step.per_page_features:
            return self.normalized_scorer.score_step(step)

        predicted_ratio = len(step.predicted_pages) / max(1, len(step.required_pages))

        if predicted_ratio >= self.predicted_ratio_threshold:
            return self.high_reuse_scorer.score_step(step)
        return self.normalized_scorer.score_step(step)


def _normalize_feature(step: WorkloadStep, pages: set[KVPageId], feature_name: str) -> dict[KVPageId, float]:
    raw = {
        page: float(step.per_page_features.get(page, {}).get(feature_name, 0.0))
        for page in pages
    }
    return _normalize_scores(raw)


def attach_reuse_distance_features(
    trace: list[WorkloadStep],
    *,
    short_threshold: float = 2.0,
    ema_decay: float = 0.5,
    recent_window: int = 16,
    tile_size_pages: int = 4,
) -> list[WorkloadStep]:
    """Return a trace with reuse-distance features attached per page per step.

    These features are computed causally from past accesses only, which keeps
    them usable by a practical controller during replay experiments.
    """

    last_access_step: dict[KVPageId, int] = {}
    request_last_access_step: dict[tuple[str, KVPageId], int] = {}
    reuse_ema: dict[KVPageId, float] = {}
    recent_page_accesses: deque[KVPageId] = deque(maxlen=recent_window)
    recent_request_page_accesses: dict[str, deque[KVPageId]] = defaultdict(lambda: deque(maxlen=recent_window))
    recent_tile_accesses: deque[tuple[int, int]] = deque(maxlen=recent_window)
    updated_trace: list[WorkloadStep] = []

    for step in trace:
        candidate_pages = set(step.required_pages) | set(step.predicted_pages) | set(step.head_weighted_scores)
        updated_features = {
            page: dict(features)
            for page, features in step.per_page_features.items()
        }

        for page in candidate_pages:
            last_seen = last_access_step.get(page)
            if last_seen is None:
                reuse_distance = float("inf")
                reuse_inverse = 0.0
                reuse_short = 0.0
                reuse_long = 0.0
                reuse_avg = reuse_ema.get(page, 0.0)
            else:
                reuse_distance = float(step.step_idx - last_seen)
                reuse_inverse = 1.0 / max(1.0, reuse_distance)
                reuse_short = 1.0 if reuse_distance <= short_threshold else 0.0
                reuse_long = 1.0 if reuse_distance > short_threshold else 0.0
                previous_avg = reuse_ema.get(page, reuse_distance)
                reuse_avg = ema_decay * reuse_distance + (1.0 - ema_decay) * previous_avg
                reuse_ema[page] = reuse_avg

            request_key = (step.request_id, page)
            request_last_seen = request_last_access_step.get(request_key)
            if request_last_seen is None:
                request_reuse_distance = float("inf")
                request_reuse_inverse = 0.0
                request_reuse_short = 0.0
                request_reuse_long = 0.0
            else:
                request_reuse_distance = float(step.decode_position - request_last_seen)
                request_reuse_inverse = 1.0 / max(1.0, request_reuse_distance)
                request_reuse_short = 1.0 if request_reuse_distance <= short_threshold else 0.0
                request_reuse_long = 1.0 if request_reuse_distance > short_threshold else 0.0

            tile_id = page.page_id // max(1, tile_size_pages)
            tile_key = (page.layer_id, tile_id)
            page_recent_frequency = sum(1 for recent_page in recent_page_accesses if recent_page == page)
            request_recent_pages = recent_request_page_accesses[step.request_id]
            request_page_recent_frequency = sum(1 for recent_page in request_recent_pages if recent_page == page)
            tile_recent_frequency = sum(1 for recent_tile in recent_tile_accesses if recent_tile == tile_key)

            page_features = updated_features.setdefault(page, {})
            page_features["reuse_distance"] = reuse_distance if reuse_distance != float("inf") else -1.0
            page_features["reuse_distance_inverse"] = reuse_inverse
            page_features["reuse_distance_short"] = reuse_short
            page_features["reuse_distance_long"] = reuse_long
            page_features["reuse_distance_mavg"] = reuse_avg
            page_features["request_reuse_distance"] = (
                request_reuse_distance if request_reuse_distance != float("inf") else -1.0
            )
            page_features["request_reuse_distance_inverse"] = request_reuse_inverse
            page_features["request_reuse_distance_short"] = request_reuse_short
            page_features["request_reuse_distance_long"] = request_reuse_long
            page_features["page_recent_frequency"] = float(page_recent_frequency)
            page_features["request_page_recent_frequency"] = float(request_page_recent_frequency)
            page_features["tile_recent_frequency"] = float(tile_recent_frequency)
            page_features["tile_id"] = float(tile_id)

        updated_trace.append(
            replace(
                step,
                per_page_features=updated_features,
            )
        )

        for page in step.required_pages:
            last_access_step[page] = step.step_idx
            request_last_access_step[(step.request_id, page)] = step.decode_position
            recent_page_accesses.append(page)
            recent_request_page_accesses[step.request_id].append(page)
            recent_tile_accesses.append((page.layer_id, page.page_id // max(1, tile_size_pages)))

    return updated_trace


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
