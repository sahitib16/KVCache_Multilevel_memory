from __future__ import annotations

from dataclasses import dataclass
import random

from .types import KVPageId, WorkloadStep


@dataclass(frozen=True)
class SyntheticTraceConfig:
    """Configuration for the built-in synthetic workload generator.

    This generator is useful during early development because it gives us a
    controlled trace without needing a real model or paged-KV engine.

    The generated workload tries to capture three ideas:
    - local pages that are predictably reused
    - sparse pages that create irregular long-range pressure
    - head-weighted page scores that can guide policy decisions
    """

    steps: int
    layers: int
    pages_per_layer: int
    local_window_pages: int
    sparse_pages_per_step: int
    predicted_prefetch_pages: int
    attention_heads: int
    query_jitter: float = 0.0
    seed: int = 0


class SyntheticTraceGenerator:
    """Synthetic multi-layer trace generator with required head-weighted scores.

    The generator builds a synthetic decode trace in three stages:

    1. Create a per-page "activity signature" for each attention head.
       This is a stand-in for "how much does head h care about this page?"

    2. Create base head weights per layer.
       This is a stand-in for "how important is each head in this layer?"

    3. For each decode step:
       - optionally jitter the head weights
       - assemble local + sparse page candidates
       - compute head-weighted page scores
       - sort/select pages into a workload step

    This is not claiming to be a faithful model of real attention.
    It is a controllable sandbox that already carries the right *kind* of
    information for the controller.
    """

    def __init__(self, config: SyntheticTraceConfig):
        self.config = config

        # We keep our own RNG so the generator is deterministic when seeded.
        self._rng = random.Random(config.seed)

        # Precomputed per-page, per-head activity. This does not change per step.
        self._head_activity = self._build_head_activity()

        # Base head weights per layer. Step-to-step jitter is applied on top.
        self._base_head_weights = self._build_base_head_weights()

    def _build_head_activity(self) -> dict[KVPageId, tuple[float, ...]]:
        """Create a synthetic per-page activity vector for each attention head."""
        activity: dict[KVPageId, tuple[float, ...]] = {}
        for layer_id in range(self.config.layers):
            for page_id in range(self.config.pages_per_layer):
                values = []

                # Small bias so deeper layers can naturally end up looking a bit
                # "heavier" in synthetic traces.
                local_bias = 1.0 + (layer_id / max(1, self.config.layers))
                for _ in range(self.config.attention_heads):
                    values.append(local_bias * self._rng.uniform(0.1, 1.0))
                activity[KVPageId(layer_id=layer_id, page_id=page_id)] = tuple(values)
        return activity

    def _build_base_head_weights(self) -> dict[int, tuple[float, ...]]:
        """Create normalized base head weights for each layer."""
        weights: dict[int, tuple[float, ...]] = {}
        for layer_id in range(self.config.layers):
            layer_weights = [self._rng.uniform(0.5, 1.5) for _ in range(self.config.attention_heads)]
            total = sum(layer_weights) or 1.0
            weights[layer_id] = tuple(value / total for value in layer_weights)
        return weights

    def _head_weights_for_step(self, step_idx: int) -> dict[int, tuple[float, ...]]:
        """Return per-layer head weights for a single step.

        ``query_jitter`` lets weights drift over time.
        This is a simple way to create more or less predictable future demand.
        """
        weights: dict[int, tuple[float, ...]] = {}
        step_rng = random.Random(self.config.seed + step_idx * 9973)
        for layer_id, base_weights in self._base_head_weights.items():
            jittered = []
            for value in base_weights:
                delta = step_rng.uniform(-self.config.query_jitter, self.config.query_jitter)
                jittered.append(max(0.01, value + delta))
            total = sum(jittered) or 1.0
            weights[layer_id] = tuple(value / total for value in jittered)
        return weights

    def _score(self, page: KVPageId, layer_weights: tuple[float, ...]) -> float:
        """Compute the required head-weighted score for one page.

        Formula:
            score(page) = sum_h weight[layer, h] * activity(page, h)
        """
        activity = self._head_activity[page]
        return sum(weight * signal for weight, signal in zip(layer_weights, activity))

    def _required_pages_for_step(self, step_idx: int, head_weights: dict[int, tuple[float, ...]]) -> tuple[KVPageId, ...]:
        """Build the set of pages needed at a step.

        Current recipe:
        - a local sliding window for each layer
        - a few sparse pages outside that window
        - then sort candidates by head-weighted score so the step has a stable
          ordering of "more important first"
        """
        required: list[KVPageId] = []
        for layer_id in range(self.config.layers):
            # The local window shifts one position per step.
            window_start = step_idx % max(1, self.config.pages_per_layer - self.config.local_window_pages + 1)
            local_pages = [
                KVPageId(layer_id=layer_id, page_id=page_id)
                for page_id in range(window_start, window_start + self.config.local_window_pages)
            ]

            # Sparse pages are chosen from outside the local window to create
            # irregular pressure that a naive policy cannot solve perfectly.
            sparse_pool = [
                KVPageId(layer_id=layer_id, page_id=page_id)
                for page_id in range(self.config.pages_per_layer)
                if page_id < window_start or page_id >= window_start + self.config.local_window_pages
            ]
            sparse_pages = self._rng.sample(
                sparse_pool,
                k=min(self.config.sparse_pages_per_step, len(sparse_pool)),
            )
            candidates = local_pages + sparse_pages
            candidates.sort(key=lambda page: self._score(page, head_weights[page.layer_id]), reverse=True)
            required.extend(candidates)
        return tuple(required)

    def _predicted_pages_for_step(self, step_idx: int) -> tuple[KVPageId, ...]:
        """Build a simple prediction set by peeking at the next synthetic step."""
        next_idx = min(step_idx + 1, self.config.steps - 1)
        head_weights = self._head_weights_for_step(next_idx)
        predicted = self._required_pages_for_step(next_idx, head_weights)
        return predicted[: self.config.predicted_prefetch_pages]

    def generate(self) -> list[WorkloadStep]:
        """Generate the full synthetic trace."""
        trace: list[WorkloadStep] = []
        for step_idx in range(self.config.steps):
            head_weights = self._head_weights_for_step(step_idx)
            required_pages = self._required_pages_for_step(step_idx, head_weights)
            predicted_pages = self._predicted_pages_for_step(step_idx)

            # We only need scores for pages that appear in this step's required
            # or predicted sets.
            page_set = set(required_pages) | set(predicted_pages)
            head_weighted_scores = {
                page: self._score(page, head_weights[page.layer_id])
                for page in page_set
            }
            per_page_head_activity = {
                page: self._head_activity[page]
                for page in page_set
            }
            per_page_features = {
                page: {
                    "head_weighted_score": head_weighted_scores[page],
                    "layer_id": float(page.layer_id),
                    "page_id": float(page.page_id),
                    "is_predicted": 1.0 if page in predicted_pages else 0.0,
                    "is_required": 1.0 if page in required_pages else 0.0,
                }
                for page in page_set
            }
            referenced_layers = tuple(sorted({page.layer_id for page in page_set}))
            trace.append(
                WorkloadStep(
                    step_idx=step_idx,
                    required_pages=required_pages,
                    predicted_pages=predicted_pages,
                    head_weighted_scores=head_weighted_scores,
                    query_head_weights=head_weights,
                    per_page_head_activity=per_page_head_activity,
                    per_page_features=per_page_features,
                    referenced_layers=referenced_layers,
                )
            )
        return trace
