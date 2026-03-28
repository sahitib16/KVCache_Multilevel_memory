from __future__ import annotations

"""Controller implementations for the KV residency simulator.

This file is where the simulator stops being "just infrastructure" and starts
acting like a policy testbed.

There are three groups of logic here:

1. Static baseline controllers
   These use a fixed rule every step.
   Examples:
   - LRU-style behavior
   - score-based prefetch / eviction

2. Oracle controllers
   These are not deployable in a real system because they look into the future.
   They exist to answer:
   - "How much headroom is there if we made perfect decisions?"

3. Adaptive / learning-based controllers
   These observe the current state and try to choose between a small action set.
   In Step 4, the first adaptive controller is a lightweight contextual bandit.

Important philosophy:
- The simulator enforces physics and correctness.
- The controller chooses *which* actions to take under those rules.
- The policy layer should stay interpretable.
"""

from dataclasses import dataclass
from collections import defaultdict
import math

import numpy as np

from .interfaces import ResidencyController
from .types import ControllerContext, KVPageId, LayerBudget, PolicyOutput, StepMetrics, WorkloadStep


def _score_lookup(context: ControllerContext, page: KVPageId) -> float:
    """Return the best available score for a page.

    In the current project, head-weighted score is the primary signal.
    We keep this helper so the policy code does not need to know whether the
    score came directly from `head_weighted_scores` or from the richer
    `per_page_features` dictionary.
    """

    return context.head_weighted_scores.get(
        page,
        context.per_page_features.get(page, {}).get("head_weighted_score", 0.0),
    )


def _free_slots(context: ControllerContext) -> int:
    """How many HBM slots are currently unclaimed.

    A slot is considered unavailable if:
    - it already holds a resident page
    - or it is reserved by an in-flight transfer
    """

    return context.hbm_capacity_pages - len(context.resident_pages) - len(context.inflight_pages)


def _protected_pages(context: ControllerContext) -> set[KVPageId]:
    """Pages we should not explicitly evict right now."""

    return set(context.required_pages) | set(context.inflight_pages)


def _incoming_prefetch_pages(
    context: ControllerContext,
    predicted_pages: list[KVPageId],
    prefetch_k: int,
) -> list[KVPageId]:
    """Filter and clip the predicted set down to useful prefetch requests."""

    return [
        page
        for page in predicted_pages[:prefetch_k]
        if page not in context.resident_pages and page not in context.inflight_pages
    ]


def _guarded_prefetch_k(context: ControllerContext, base_prefetch_k: int) -> int:
    """Throttle prefetch depth when the system already looks pressured.

    Why this matters:
    - aggressive prefetch can be good when bandwidth is idle
    - the same aggressiveness can be harmful when HBM is already crowded or
      recent behavior suggests churn / miss pressure

    This helper gives the bandit a simple extra knob:
    "turn transfer-pressure guarding on or off"
    """

    if base_prefetch_k <= 0:
        return 0

    occupancy_ratio = (
        (len(context.resident_pages) + len(context.inflight_pages)) / max(1, context.hbm_capacity_pages)
    )
    if context.transfer_backlog > 0:
        return 0
    if occupancy_ratio >= 0.95:
        return min(base_prefetch_k, 1)
    if context.churn > 0 and context.miss_rate > 0.35:
        return min(base_prefetch_k, 1)
    return base_prefetch_k


def _lowest_score_resident_pages(context: ControllerContext, count: int) -> tuple[KVPageId, ...]:
    """Pick `count` resident pages with the lowest head-weighted score."""

    protected = _protected_pages(context)
    candidates = [page for page in context.resident_pages if page not in protected]
    candidates.sort(key=lambda page: _score_lookup(context, page))
    return tuple(candidates[:count])


def _adaptive_layer_budgets(context: ControllerContext) -> tuple[LayerBudget, ...]:
    """Allocate HBM budget across layers using current score mass.

    Intuition:
    - if a layer currently appears more important under the head-weighted score
      signal, give it more of the scarce HBM budget
    - keep at least one page per layer when possible so no layer is completely
      starved by rounding

    This is intentionally simple, but it gives us a real layer-budget control
    knob for both static policies and the bandit.
    """

    layers = sorted(context.layer_budgets)
    if not layers:
        return ()

    score_mass = {layer_id: 0.0 for layer_id in layers}
    for page in set(context.required_pages) | set(context.predicted_pages):
        score_mass[page.layer_id] += max(0.0, _score_lookup(context, page))

    total_mass = sum(score_mass.values())
    capacity = context.hbm_capacity_pages
    min_per_layer = 1 if capacity >= len(layers) else 0
    budgets = {layer_id: min_per_layer for layer_id in layers}
    remaining = max(0, capacity - min_per_layer * len(layers))

    if total_mass <= 0 or remaining == 0:
        # Fall back to near-equal split if no useful score signal is present.
        for idx in range(remaining):
            budgets[layers[idx % len(layers)]] += 1
    else:
        fractional = []
        assigned = 0
        for layer_id in layers:
            exact = remaining * (score_mass[layer_id] / total_mass)
            whole = int(math.floor(exact))
            budgets[layer_id] += whole
            assigned += whole
            fractional.append((exact - whole, layer_id))
        leftover = remaining - assigned
        fractional.sort(reverse=True)
        for _, layer_id in fractional[:leftover]:
            budgets[layer_id] += 1

    return tuple(
        LayerBudget(layer_id=layer_id, max_resident_pages=budgets[layer_id])
        for layer_id in layers
    )


class LRUController(ResidencyController):
    """Baseline controller that relies on simulator fallback LRU eviction.

    Why this controller is so small:
    - the simulator already has default eviction behavior when capacity is tight
    - that fallback is currently LRU-like
    - so this controller only decides whether to issue any prefetch requests

    This makes it a good "minimal baseline" for comparison.
    """

    def __init__(self, prefetch_k: int = 0, guard_prefetch: bool = False):
        self.prefetch_k = prefetch_k
        self.guard_prefetch = guard_prefetch

    def decide(self, context: ControllerContext) -> PolicyOutput:
        effective_prefetch_k = (
            _guarded_prefetch_k(context, self.prefetch_k) if self.guard_prefetch else self.prefetch_k
        )
        return PolicyOutput(prefetch_pages=context.predicted_pages[: effective_prefetch_k])


class ScoreBasedController(ResidencyController):
    """Baseline controller that uses head-weighted score for decisions.

    High-level behavior:
    - rank predicted pages by score
    - prefetch the top-K predicted pages
    - if space is needed, explicitly evict the currently resident pages with
      the lowest score

    This is the first real "signal-aware" baseline in the new simulator.
    """

    def __init__(self, prefetch_k: int = 0, guard_prefetch: bool = False):
        self.prefetch_k = prefetch_k
        self.guard_prefetch = guard_prefetch

    def decide(self, context: ControllerContext) -> PolicyOutput:
        effective_prefetch_k = (
            _guarded_prefetch_k(context, self.prefetch_k) if self.guard_prefetch else self.prefetch_k
        )
        # Score the predicted set from highest value to lowest value.
        predicted = sorted(
            context.predicted_pages,
            key=lambda page: _score_lookup(context, page),
            reverse=True,
        )

        incoming_prefetch = _incoming_prefetch_pages(context, predicted, effective_prefetch_k)
        need_slots = max(0, len(incoming_prefetch) - _free_slots(context))
        evict_pages = _lowest_score_resident_pages(context, need_slots)

        return PolicyOutput(
            evict_pages=evict_pages,
            prefetch_pages=tuple(incoming_prefetch),
        )


class LayerAwareScoreController(ResidencyController):
    """Score-based controller with adaptive per-layer budget updates.

    This policy uses the same score-based prefetch/eviction logic as
    `ScoreBasedController`, but it also emits a layer budget update each step.
    The simulator then enforces those budgets before processing the step.
    """

    def __init__(self, prefetch_k: int = 0, guard_prefetch: bool = False):
        self.prefetch_k = prefetch_k
        self.guard_prefetch = guard_prefetch

    def decide(self, context: ControllerContext) -> PolicyOutput:
        effective_prefetch_k = (
            _guarded_prefetch_k(context, self.prefetch_k) if self.guard_prefetch else self.prefetch_k
        )
        predicted = sorted(
            context.predicted_pages,
            key=lambda page: _score_lookup(context, page),
            reverse=True,
        )
        incoming_prefetch = _incoming_prefetch_pages(context, predicted, effective_prefetch_k)
        need_slots = max(0, len(incoming_prefetch) - _free_slots(context))
        evict_pages = _lowest_score_resident_pages(context, need_slots)
        return PolicyOutput(
            evict_pages=evict_pages,
            prefetch_pages=tuple(incoming_prefetch),
            layer_budgets=_adaptive_layer_budgets(context),
        )


class FutureTraceOracle:
    """Helper that exposes future page-use information for oracle policies.

    Since oracle policies are allowed to see the future, we precompute a small
    index from page -> future step numbers. That lets policies ask:
    - "When is this page used next?"
    - "What pages are required on the next step?"

    This helper is intentionally separate so multiple oracle policies can reuse
    the same future-knowledge logic.
    """

    def __init__(self, trace: list[WorkloadStep]):
        self.trace = trace
        self._future_steps_by_page: dict[KVPageId, list[int]] = defaultdict(list)
        for step in trace:
            for page in step.required_pages:
                self._future_steps_by_page[page].append(step.step_idx)

    def next_use_after(self, page: KVPageId, step_idx: int) -> int | None:
        """Return the next step after `step_idx` that touches `page`."""

        for future_step in self._future_steps_by_page.get(page, []):
            if future_step > step_idx:
                return future_step
        return None

    def next_required_pages(self, step_idx: int) -> tuple[KVPageId, ...]:
        """Return the exact required-page set for the next step."""

        next_idx = step_idx + 1
        if next_idx >= len(self.trace):
            return ()
        return self.trace[next_idx].required_pages


class BeladyOracleController(ResidencyController):
    """Future-aware eviction oracle using Belady's farthest-next-use rule.

    Belady's idea:
    - if you must evict something, throw out the page that will be needed
      farthest in the future
    - if a page is never used again, it is the best eviction candidate

    This is an upper bound for eviction quality, not a deployable policy.
    """

    def __init__(self, trace: list[WorkloadStep]):
        self.oracle = FutureTraceOracle(trace)

    def _choose_evictions(self, context: ControllerContext, pages_to_make_room_for: int) -> tuple[KVPageId, ...]:
        protected = _protected_pages(context)
        candidates = [page for page in context.resident_pages if page not in protected]
        candidates.sort(
            key=lambda page: self.oracle.next_use_after(page, context.step_idx)
            if self.oracle.next_use_after(page, context.step_idx) is not None
            else float("inf"),
            reverse=True,
        )
        return tuple(candidates[:pages_to_make_room_for])

    def decide(self, context: ControllerContext) -> PolicyOutput:
        # Belady here is focused on eviction quality for required pages.
        missing_required = [
            page for page in context.required_pages
            if page not in context.resident_pages and page not in context.inflight_pages
        ]
        need_slots = max(0, len(missing_required) - _free_slots(context))
        return PolicyOutput(evict_pages=self._choose_evictions(context, need_slots))


class PerfectPrefetchOracleController(ResidencyController):
    """Future-aware oracle that prefetches exact next-step required pages.

    This policy is also not deployable in a real engine because it knows the
    next step exactly. Its value is as an upper bound:
    - "How much better could prefetching be if the predictions were perfect?"
    """

    def __init__(self, trace: list[WorkloadStep], prefetch_k: int | None = None):
        self.oracle = FutureTraceOracle(trace)
        self.prefetch_k = prefetch_k

    def _choose_evictions(self, context: ControllerContext, pages_to_make_room_for: int) -> tuple[KVPageId, ...]:
        protected = _protected_pages(context)
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
        need_slots = max(0, len(prefetch_pages) - _free_slots(context))

        return PolicyOutput(
            evict_pages=self._choose_evictions(context, need_slots),
            prefetch_pages=tuple(prefetch_pages),
        )


@dataclass(frozen=True)
class BanditAction:
    """One action choice for the contextual bandit.

    We keep the action space intentionally small and interpretable:
    - choose an eviction rule
    - choose a prefetch depth

    This matches the project direction of lightweight online control rather
    than a large opaque RL policy.

    `budget_mode`:
    - `False` means use the chosen policy without adaptive layer budgets
    - `True` means turn on layer-aware budgeting for the score-based path

    `guard_mode`:
    - `False` means always use the requested prefetch depth
    - `True` means dynamically clamp prefetch when transfer or occupancy
      pressure suggests the system is already stressed
    """

    eviction_rule: str
    prefetch_k: int
    budget_mode: bool = False
    guard_mode: bool = False


def build_bandit_action_menu(name: str) -> tuple[BanditAction, ...]:
    """Return a named action-menu preset for bandit experiments.

    Why named menus:
    - we want to compare a few interpretable action sets
    - putting them in one helper keeps experiments reproducible
    - replay-based tuning can then ask:
      "does a smaller or more score-heavy menu generalize better?"
    """

    menus = {
        "full": (
            BanditAction("lru", 0, False, False),
            BanditAction("lru", 2, False, False),
            BanditAction("score", 0, False, False),
            BanditAction("score", 2, False, False),
            BanditAction("score", 4, False, False),
            BanditAction("score", 2, True, False),
            BanditAction("score", 2, False, True),
            BanditAction("score", 4, False, True),
            BanditAction("score", 2, True, True),
        ),
        "trimmed": (
            BanditAction("lru", 0, False, False),
            BanditAction("lru", 2, False, False),
            BanditAction("score", 0, False, False),
            BanditAction("score", 2, False, False),
            BanditAction("score", 4, False, False),
            BanditAction("score", 2, True, False),
            BanditAction("score", 2, False, True),
            BanditAction("score", 4, False, True),
        ),
        "score_heavy": (
            BanditAction("lru", 0, False, False),
            BanditAction("score", 0, False, False),
            BanditAction("score", 2, False, False),
            BanditAction("score", 4, False, False),
            BanditAction("score", 2, True, False),
            BanditAction("score", 2, False, True),
        ),
    }
    if name not in menus:
        raise ValueError(f"Unknown bandit action menu: {name}")
    return menus[name]


class ContextualBanditController(ResidencyController):
    """Lightweight contextual bandit over a small controller action space.

    This is the first adaptive / learning-based controller in the project.

    How it works:
    1. Build a small feature vector from the current controller context.
    2. Score each possible action using LinUCB.
    3. Execute the chosen action by delegating to an underlying static policy.
    4. After the step finishes, observe the reward and update the model for
       that chosen action only.

    Why LinUCB:
    - simple
    - data-efficient
    - keeps the learned behavior easy to inspect
    - does not require a heavy RL training loop
    """

    def __init__(
        self,
        actions: tuple[BanditAction, ...] | None = None,
        alpha: float = 0.35,
        prefetch_penalty: float = 0.001,
        eviction_penalty: float = 0.001,
        miss_penalty: float = 0.10,
        useful_prefetch_bonus: float = 0.04,
        wasted_prefetch_penalty: float = 0.015,
        miss_reduction_bonus: float = 0.02,
        miss_increase_penalty: float = 0.02,
        bootstrap_action: BanditAction | None = None,
        bootstrap_steps: int = 2,
    ):
        self.actions = actions or build_bandit_action_menu("full")
        self.alpha = alpha
        self.prefetch_penalty = prefetch_penalty
        self.eviction_penalty = eviction_penalty
        self.miss_penalty = miss_penalty
        self.useful_prefetch_bonus = useful_prefetch_bonus
        self.wasted_prefetch_penalty = wasted_prefetch_penalty
        self.miss_reduction_bonus = miss_reduction_bonus
        self.miss_increase_penalty = miss_increase_penalty
        self.bootstrap_action = bootstrap_action or BanditAction("score", 2, True, False)
        self.bootstrap_steps = bootstrap_steps

        # Feature dimension is fixed by `_features`.
        self._dim = 13

        # LinUCB keeps one linear model per action:
        #   A_a = X^T X + I
        #   b_a = X^T r
        self._A: dict[BanditAction, np.ndarray] = {
            action: np.eye(self._dim, dtype=np.float64) for action in self.actions
        }
        self._b: dict[BanditAction, np.ndarray] = {
            action: np.zeros(self._dim, dtype=np.float64) for action in self.actions
        }

        # Bookkeeping from the most recent step so `observe` can update using
        # the action that was actually taken.
        self.last_action: BanditAction | None = None
        self.last_features: np.ndarray | None = None
        self.last_ucb_scores: dict[BanditAction, float] = {}
        self.total_reward = 0.0
        self.steps_observed = 0

        # Lightweight memory of the most recent observed outcome.
        # This helps the bandit react to short-term regime changes instead of
        # only relying on cumulative signals from the simulator context.
        self.prev_stall_ms = 0.0
        self.prev_demand_misses = 0.0
        self.prev_evictions = 0.0

        # We keep one-step delayed credit assignment because proactive actions
        # like prefetching mainly affect the *next* decode step's stall/misses,
        # not the step on which they were issued.
        self.pending_action: BanditAction | None = None
        self.pending_features: np.ndarray | None = None
        self.pending_prefetched_pages: frozenset[KVPageId] = frozenset()
        self.total_useful_prefetches = 0
        self.total_wasted_prefetches = 0
        self.total_miss_reduction = 0.0
        self.total_miss_increase = 0.0
        self.action_counts: dict[BanditAction, int] = {action: 0 for action in self.actions}

    def _features(self, context: ControllerContext) -> np.ndarray:
        """Turn the current controller context into a small numeric feature vector.

        The features are intentionally simple and interpretable:
        - bias term
        - cumulative miss rate so far
        - HBM occupancy ratio
        - transfer backlog ratio
        - in-flight transfer ratio
        - churn scaled down
        - average required-page score
        - average predicted-page score
        - max per-layer pressure
        - mean per-layer pressure
        - previous step stall
        - previous step demand misses
        - previous step evictions
        """

        occupancy_ratio = (
            (len(context.resident_pages) + len(context.inflight_pages)) / max(1, context.hbm_capacity_pages)
        )
        backlog_ratio = context.transfer_backlog / max(1, context.hbm_capacity_pages)
        inflight_ratio = context.inflight_transfer_count / max(1, context.hbm_capacity_pages)
        avg_required_score = (
            sum(_score_lookup(context, page) for page in context.required_pages) / max(1, len(context.required_pages))
        )
        avg_predicted_score = (
            sum(_score_lookup(context, page) for page in context.predicted_pages) / max(1, len(context.predicted_pages))
        )
        max_layer_pressure = max(context.per_layer_pressure.values(), default=0.0)
        mean_layer_pressure = sum(context.per_layer_pressure.values()) / max(1, len(context.per_layer_pressure))

        return np.array(
            [
                1.0,
                context.miss_rate,
                occupancy_ratio,
                backlog_ratio,
                inflight_ratio,
                context.churn / 100.0,
                avg_required_score,
                avg_predicted_score,
                max_layer_pressure,
                mean_layer_pressure,
                self.prev_stall_ms,
                self.prev_demand_misses / 100.0,
                self.prev_evictions / 100.0,
            ],
            dtype=np.float64,
        )

    def _feature_names(self) -> list[str]:
        """Return human-readable names for the linear bandit feature vector."""

        return [
            "bias",
            "miss_rate",
            "occupancy_ratio",
            "backlog_ratio",
            "inflight_ratio",
            "scaled_churn",
            "avg_required_score",
            "avg_predicted_score",
            "max_layer_pressure",
            "mean_layer_pressure",
            "prev_stall_ms",
            "prev_demand_misses",
            "prev_evictions",
        ]

    def _linucb_score(self, action: BanditAction, features: np.ndarray) -> float:
        """Compute UCB score for one action under the current context."""

        A_inv = np.linalg.inv(self._A[action])
        theta = A_inv @ self._b[action]
        exploitation = float(theta @ features)
        exploration = float(self.alpha * math.sqrt(features @ A_inv @ features))
        return exploitation + exploration

    def _update_model(self, action: BanditAction, features: np.ndarray, reward: float) -> None:
        """Apply one linear-bandit update to a chosen action.

        Keeping this in a helper makes the delayed-credit logic in `observe`
        much easier to read.
        """

        self._A[action] = self._A[action] + np.outer(features, features)
        self._b[action] = self._b[action] + reward * features
        self.total_reward += reward

    def _action_to_policy(self, action: BanditAction) -> ResidencyController:
        """Translate an abstract bandit action into a concrete static controller."""

        if action.budget_mode and action.eviction_rule == "score":
            return LayerAwareScoreController(
                prefetch_k=action.prefetch_k,
                guard_prefetch=action.guard_mode,
            )
        if action.eviction_rule == "score":
            return ScoreBasedController(
                prefetch_k=action.prefetch_k,
                guard_prefetch=action.guard_mode,
            )
        return LRUController(prefetch_k=action.prefetch_k, guard_prefetch=action.guard_mode)

    def decide(self, context: ControllerContext) -> PolicyOutput:
        """Choose the best action under LinUCB, then delegate to its policy."""

        # Warm-start the first few steps with a reasonable score-aware policy.
        # Without this, the bandit starts with almost no information and can
        # easily waste early steps on poor "cold start" actions.
        if self.steps_observed < self.bootstrap_steps:
            chosen = self.bootstrap_action
            features = self._features(context)
            self.last_action = chosen
            self.last_features = features
            self.last_ucb_scores = {chosen: 0.0}
            self.action_counts[chosen] += 1
            return self._action_to_policy(chosen).decide(context)

        features = self._features(context)
        ucb_scores = {
            action: self._linucb_score(action, features)
            for action in self.actions
        }
        chosen = max(self.actions, key=lambda action: ucb_scores[action])

        self.last_action = chosen
        self.last_features = features
        self.last_ucb_scores = ucb_scores
        self.action_counts[chosen] += 1

        decision = self._action_to_policy(chosen).decide(context)
        return decision

    def observe(self, context: ControllerContext, decision: PolicyOutput, metrics: StepMetrics) -> None:
        """Update the chosen action's model using the observed step reward.

        Credit assignment here is intentionally split in two pieces.

        1. Delayed performance reward:
           Prefetch and budget decisions made on step `t` mostly show up as
           stall / miss changes on step `t+1`, so we credit the *previous*
           action with the current step's latency and miss outcome.

           We also score whether those previous prefetches were actually
           useful:
           - bonus if a prefetched page is required on the current step
           - penalty if a prefetched page was not consumed by the current step

        2. Immediate control-cost reward:
           The current action is immediately charged for how much proactive
           work it injected this step (prefetch volume and evictions).

        This is still a simple bandit, but it matches the engine better than
        attributing every outcome only to the same-step action.
        """

        if self.last_action is None or self.last_features is None:
            return

        delayed_reward = -(
            metrics.stall_ms
            + self.miss_penalty * metrics.demand_misses
        )
        immediate_control_cost = -(
            self.prefetch_penalty * metrics.prefetch_submitted
            + self.eviction_penalty * metrics.evictions
        )

        useful_prefetches = len(self.pending_prefetched_pages & set(context.required_pages))
        wasted_prefetches = len(self.pending_prefetched_pages - set(context.required_pages))
        miss_reduction = max(0.0, self.prev_demand_misses - float(metrics.demand_misses))
        miss_increase = max(0.0, float(metrics.demand_misses) - self.prev_demand_misses)
        delayed_reward += (
            self.useful_prefetch_bonus * useful_prefetches
            - self.wasted_prefetch_penalty * wasted_prefetches
            + self.miss_reduction_bonus * miss_reduction
            - self.miss_increase_penalty * miss_increase
        )
        self.total_useful_prefetches += useful_prefetches
        self.total_wasted_prefetches += wasted_prefetches
        self.total_miss_reduction += miss_reduction
        self.total_miss_increase += miss_increase

        # The previous action receives credit for the current step's realized
        # performance, because that is when its proactive decisions paid off.
        if self.pending_action is not None and self.pending_features is not None:
            self._update_model(self.pending_action, self.pending_features, delayed_reward)
        else:
            # On the first observed step there is no previous action yet, so we
            # fall back to crediting the current action with the current result.
            self._update_model(self.last_action, self.last_features, delayed_reward)

        # The current action is immediately charged for how much proactive
        # pressure it placed on the system this step.
        self._update_model(self.last_action, self.last_features, immediate_control_cost)

        self.pending_action = self.last_action
        self.pending_features = self.last_features
        self.pending_prefetched_pages = frozenset(decision.prefetch_pages)
        self.steps_observed += 1
        self.prev_stall_ms = metrics.stall_ms
        self.prev_demand_misses = float(metrics.demand_misses)
        self.prev_evictions = float(metrics.evictions)

    def diagnostics(self) -> dict[str, object]:
        """Return lightweight debugging info for summaries and tests."""

        theta_by_action = {}
        for action in self.actions:
            A_inv = np.linalg.inv(self._A[action])
            theta_by_action[str(action)] = (A_inv @ self._b[action]).tolist()

        return {
            "steps_observed": self.steps_observed,
            "total_reward": self.total_reward,
            "last_action": self.last_action,
            "last_ucb_scores": {str(action): score for action, score in self.last_ucb_scores.items()},
            "feature_names": self._feature_names(),
            "theta_by_action": theta_by_action,
            "action_counts": {str(action): count for action, count in self.action_counts.items()},
            "total_useful_prefetches": self.total_useful_prefetches,
            "total_wasted_prefetches": self.total_wasted_prefetches,
            "prefetch_hit_rate": (
                self.total_useful_prefetches
                / max(1, self.total_useful_prefetches + self.total_wasted_prefetches)
            ),
            "total_miss_reduction": self.total_miss_reduction,
            "total_miss_increase": self.total_miss_increase,
        }
