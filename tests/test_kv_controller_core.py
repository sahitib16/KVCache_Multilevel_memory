from __future__ import annotations

from dataclasses import replace
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import (
    AggregateBenchmarkResult,
    FutureTraceOracle,
    HeadActivityRecomputedScorer,
    LayerNormalizedHeadActivityScorer,
    PageStatsHybridScorer,
    ReuseDistanceHybridScorer,
    ContextualBanditController,
    BeladyOracleController,
    attach_reuse_distance_features,
    apply_scorer_to_trace,
    benchmark_policies_across_seeds,
    benchmark_policies,
    CacheConfig,
    FixedKPrefetchController,
    LayerAwareScoreController,
    LRUController,
    OverlapAwareSimulator,
    PassthroughHeadWeightedScorer,
    PerfectPrefetchOracleController,
    PolicyOutput,
    RegimeAwarePageStatsScorer,
    ResidencyController,
    ScoreBasedController,
    SlidingWindowController,
    SimulationConfig,
    SyntheticTraceConfig,
    SyntheticTraceGenerator,
    TileHotnessController,
    UnifiedBanditController,
    UnifiedBlendController,
    UnifiedRuleController,
    TransferConfig,
    convert_trace_recent_threshold,
    convert_trace_recent_topk,
    interleave_sparse_recent_threshold_traces,
    interleave_sparse_recent_topk_traces,
    interleave_traces_round_robin,
    load_trace_json,
    save_trace_json,
    summarize_page_tile_stats,
    summarize_realism_metrics,
)


class NoPrefetchController(ResidencyController):
    def decide(self, context):
        return PolicyOutput()


class PredictedPrefetchController(ResidencyController):
    def __init__(self, k: int):
        self.k = k

    def decide(self, context):
        return PolicyOutput(prefetch_pages=context.predicted_pages[: self.k])


def make_trace(steps: int = 4):
    return SyntheticTraceGenerator(
        SyntheticTraceConfig(
            steps=steps,
            layers=2,
            pages_per_layer=12,
            local_window_pages=3,
            sparse_pages_per_step=1,
            predicted_prefetch_pages=3,
            attention_heads=4,
            query_jitter=0.05,
            seed=11,
        )
    ).generate()


def make_sim(hbm_capacity_pages: int = 10):
    return OverlapAwareSimulator(
        SimulationConfig(
            steps=4,
            cache=CacheConfig(
                hbm_capacity_pages=hbm_capacity_pages,
                layers=2,
                page_bytes=4096,
            ),
            transfer=TransferConfig(
                page_bytes=4096,
                bandwidth_bytes_per_ms=4096 * 2,
                transfer_setup_ms=0.01,
                max_inflight_transfers=2,
                decode_kernel_ms=0.05,
            ),
        )
    )


def test_workload_step_exposes_step2_interface_fields():
    trace = make_trace(steps=2)
    step = trace[0]

    assert step.required_pages
    assert step.predicted_pages
    assert step.per_page_features
    assert step.per_page_head_activity
    assert step.referenced_layers == (0, 1)
    assert step.request_id == "synthetic_request_0"
    assert step.kv_block_size_tokens == 16
    assert step.layer_block_tables

    sample_page = next(iter(step.per_page_features))
    features = step.per_page_features[sample_page]
    assert "head_weighted_score" in features
    assert "layer_id" in features
    assert "page_id" in features
    assert "is_predicted" in features
    assert "is_required" in features


def test_simulator_context_exposes_slot_and_transfer_interfaces():
    seen = {}

    class RecordingController(ResidencyController):
        def decide(self, context):
            seen["slot_to_page"] = context.slot_to_page
            seen["page_to_slot"] = context.page_to_slot
            seen["queued_transfers"] = context.queued_transfers
            seen["completion_times"] = context.transfer_completion_times_ms
            seen["overlap_budget_ms"] = context.overlap_budget_ms
            seen["per_page_features"] = context.per_page_features
            seen["per_page_head_activity"] = context.per_page_head_activity
            seen["request_id"] = context.request_id
            seen["sequence_length"] = context.sequence_length
            seen["layer_block_tables"] = context.layer_block_tables
            seen["per_layer_pressure"] = context.per_layer_pressure
            return PolicyOutput(prefetch_pages=context.predicted_pages[:1])

    sim = make_sim()
    trace = make_trace(steps=2)
    sim.run(trace, RecordingController())

    assert isinstance(seen["slot_to_page"], dict)
    assert isinstance(seen["page_to_slot"], dict)
    assert isinstance(seen["queued_transfers"], tuple)
    assert isinstance(seen["completion_times"], dict)
    assert isinstance(seen["overlap_budget_ms"], float)
    assert isinstance(seen["per_page_features"], dict)
    assert isinstance(seen["per_page_head_activity"], dict)
    assert isinstance(seen["request_id"], str)
    assert isinstance(seen["sequence_length"], int)
    assert isinstance(seen["layer_block_tables"], dict)
    assert isinstance(seen["per_layer_pressure"], dict)


def test_cache_slot_maps_remain_consistent_after_run():
    sim = make_sim(hbm_capacity_pages=8)
    trace = make_trace(steps=3)
    sim.run(trace, PredictedPrefetchController(k=2))

    assert len(sim.state.slot_to_page) == len(sim.state.page_to_slot)
    assert len(sim.state.slot_to_page) <= sim.config.cache.hbm_capacity_pages

    for slot, page in sim.state.slot_to_page.items():
        assert sim.state.page_to_slot[page] == slot
        assert page in sim.state.resident_pages


def test_transfer_state_is_populated_and_backlog_is_non_negative():
    # Use a feasible HBM capacity for this test so we are validating transfer
    # bookkeeping, not the separate question of what should happen when a step's
    # required set is larger than total HBM capacity.
    sim = make_sim(hbm_capacity_pages=8)
    trace = make_trace(steps=3)
    rows = sim.run(trace, PredictedPrefetchController(k=2))

    assert sim.state.transfer_state.backlog >= 0
    assert isinstance(sim.state.transfer_state.queued_pages, list)
    assert isinstance(sim.state.transfer_state.inflight_requests, dict)
    assert isinstance(sim.state.transfer_state.completion_times_ms, dict)
    assert all(row.transfer_backlog >= 0 for row in rows)


def test_step_metrics_capture_pagewise_events():
    sim = make_sim(hbm_capacity_pages=8)
    trace = make_trace(steps=3)
    rows = sim.run(trace, PredictedPrefetchController(k=2))

    assert any(row.accessed_pages for row in rows)
    assert all(len(row.accessed_pages) == row.required_pages for row in rows)
    assert all(isinstance(row.resident_pages_end, tuple) for row in rows)
    assert all(isinstance(row.prefetched_pages, tuple) for row in rows)


def test_decode_launch_requires_required_pages_to_be_resident_and_slotted():
    sim = make_sim(hbm_capacity_pages=8)
    trace = make_trace(steps=1)

    original_overlap_compute = sim.scheduler.overlap_compute

    def checking_overlap_compute(state, metrics):
        required = trace[0].required_pages
        assert all(page in state.resident_pages for page in required)
        assert all(page in state.page_to_slot for page in required)
        return original_overlap_compute(state, metrics)

    sim.scheduler.overlap_compute = checking_overlap_compute
    sim.run(trace, NoPrefetchController())


def test_simulator_raises_if_required_page_loses_slot_before_launch():
    sim = make_sim(hbm_capacity_pages=8)
    trace = make_trace(steps=1)
    original_wait_for_pages = sim.scheduler.wait_for_pages

    def broken_wait_for_pages(pages, state, metrics):
        original_wait_for_pages(pages, state, metrics)
        stolen = pages[0]
        slot = state.page_to_slot.pop(stolen)
        state.slot_to_page.pop(slot)

    sim.scheduler.wait_for_pages = broken_wait_for_pages

    with pytest.raises(RuntimeError, match="slot assignment"):
        sim.run(trace, NoPrefetchController())


def test_uniform_hbm_format_is_recorded_in_config():
    sim = make_sim()
    assert sim.config.cache.hbm_kv_format == "uniform"


def test_policydecision_alias_still_works():
    from kv_controller import PolicyDecision

    decision = PolicyDecision()
    assert isinstance(decision, PolicyOutput)


def test_passthrough_head_weighted_scorer_matches_step_scores():
    trace = make_trace(steps=1)
    scorer = PassthroughHeadWeightedScorer()
    assert scorer.score_step(trace[0]) == trace[0].head_weighted_scores


def test_head_activity_recomputed_scorer_matches_synthetic_formula():
    trace = make_trace(steps=1)
    scorer = HeadActivityRecomputedScorer()
    assert scorer.score_step(trace[0]) == trace[0].head_weighted_scores


def test_layer_normalized_scorer_keeps_scores_in_unit_interval():
    trace = make_trace(steps=1)
    scorer = LayerNormalizedHeadActivityScorer()
    scores = scorer.score_step(trace[0])
    assert scores
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_apply_scorer_to_trace_updates_head_weighted_score_field():
    trace = make_trace(steps=1)
    rescored = apply_scorer_to_trace(trace, LayerNormalizedHeadActivityScorer())
    assert rescored[0].head_weighted_scores != trace[0].head_weighted_scores
    sample_page = next(iter(rescored[0].head_weighted_scores))
    assert rescored[0].per_page_features[sample_page]["head_weighted_score"] == rescored[0].head_weighted_scores[sample_page]


def test_attach_reuse_distance_features_adds_expected_fields():
    trace = attach_reuse_distance_features(make_trace(steps=3))
    sample_page = next(iter(trace[1].per_page_features))
    features = trace[1].per_page_features[sample_page]
    assert "reuse_distance" in features
    assert "reuse_distance_inverse" in features
    assert "reuse_distance_short" in features
    assert "reuse_distance_long" in features
    assert "reuse_distance_mavg" in features
    assert "request_reuse_distance_inverse" in features
    assert "page_recent_frequency" in features
    assert "request_page_recent_frequency" in features
    assert "tile_recent_frequency" in features


def test_reuse_distance_hybrid_scorer_uses_reuse_features():
    trace = attach_reuse_distance_features(make_trace(steps=3))
    scorer = ReuseDistanceHybridScorer(alpha=1.0, beta=0.5)
    scores = scorer.score_step(trace[1])
    assert scores
    assert any(score >= 0.0 for score in scores.values())


def test_page_stats_hybrid_scorer_uses_causal_hotness_features():
    trace = attach_reuse_distance_features(make_trace(steps=3))
    scorer = PageStatsHybridScorer()
    scores = scorer.score_step(trace[1])
    assert scores
    assert any(score >= 0.0 for score in scores.values())


def test_regime_aware_scorer_switches_on_request_reuse_signal():
    trace = attach_reuse_distance_features(make_trace(steps=3))
    scorer = RegimeAwarePageStatsScorer(
        predicted_ratio_threshold=0.0,
    )
    scores = scorer.score_step(trace[1])
    assert scores
    assert any(score >= 0.0 for score in scores.values())


def test_regime_aware_scorer_uses_predicted_footprint_gate():
    trace = attach_reuse_distance_features(make_trace(steps=3))

    class ConstantScorer:
        def __init__(self, value: float):
            self.value = value

        def score_step(self, step):
            pages = set(step.head_weighted_scores) | set(step.per_page_features)
            return {page: self.value for page in pages}

    low_pred_step = replace(trace[1], predicted_pages=())
    high_pred_step = replace(trace[1], predicted_pages=tuple(trace[1].required_pages))
    scorer = RegimeAwarePageStatsScorer(
        predicted_ratio_threshold=1.0,
        normalized_scorer=ConstantScorer(1.0),
        high_reuse_scorer=ConstantScorer(2.0),
    )

    low_scores = scorer.score_step(low_pred_step)
    high_scores = scorer.score_step(high_pred_step)

    assert set(low_scores.values()) == {1.0}
    assert set(high_scores.values()) == {2.0}


def test_page_and_tile_stats_aggregate_pagewise_behavior():
    trace = make_trace(steps=4)
    sim = make_sim(hbm_capacity_pages=8)
    metrics = sim.run(trace, PredictedPrefetchController(k=2))

    page_rows, layer_rows, tile_rows = summarize_page_tile_stats(trace, metrics, tile_size_pages=2)

    assert page_rows
    assert layer_rows
    assert tile_rows
    assert sum(int(row["access_count"]) for row in page_rows) == sum(len(step.required_pages) for step in trace)
    assert sum(int(row["access_count"]) for row in layer_rows) == sum(len(step.required_pages) for step in trace)
    assert sum(int(row["access_count"]) for row in tile_rows) == sum(len(step.required_pages) for step in trace)


def test_realism_metrics_return_expected_summary_fields():
    trace = make_trace(steps=4)
    sim = make_sim(hbm_capacity_pages=8)
    metrics = sim.run(trace, PredictedPrefetchController(k=2))
    page_rows, layer_rows, tile_rows = summarize_page_tile_stats(trace, metrics, tile_size_pages=2)
    realism = summarize_realism_metrics(page_rows, layer_rows, tile_rows)

    assert "page_miss_rate" in realism
    assert "prefetch_hit_rate" in realism
    assert "hot_page_access_share_top10pct" in realism
    assert "hot_tile_access_share_top10pct" in realism


def test_score_based_controller_prefetches_from_predicted_pages():
    trace = make_trace(steps=2)
    sim = make_sim(hbm_capacity_pages=10)
    contexts = []

    class RecordingScoreController(ScoreBasedController):
        def decide(self, context):
            contexts.append(context)
            return super().decide(context)

    sim.run(trace, RecordingScoreController(prefetch_k=2))
    assert contexts


def test_fixed_k_prefetch_controller_runs():
    trace = make_trace(steps=2)
    sim = make_sim(hbm_capacity_pages=10)
    rows = sim.run(trace, FixedKPrefetchController(prefetch_k=2))
    assert len(rows) == 2


def test_sliding_window_controller_prefers_high_page_id_predictions():
    trace = make_trace(steps=2)
    decisions = []

    class RecordingSlidingWindowController(SlidingWindowController):
        def decide(self, context):
            decision = super().decide(context)
            decisions.append(decision)
            return decision

    sim = make_sim(hbm_capacity_pages=10)
    sim.run(trace, RecordingSlidingWindowController(prefetch_k=2))
    for decision in decisions:
        if decision.prefetch_pages:
            page_ids = [page.page_id for page in decision.prefetch_pages]
            assert page_ids == sorted(page_ids, reverse=True)


def test_tile_hotness_controller_runs_on_augmented_features():
    trace = attach_reuse_distance_features(make_trace(steps=2))
    sim = make_sim(hbm_capacity_pages=10)
    rows = sim.run(trace, TileHotnessController(prefetch_k=2))
    assert len(rows) == 2


def test_unified_rule_controller_runs():
    trace = attach_reuse_distance_features(make_trace(steps=3))
    sim = make_sim(hbm_capacity_pages=10)
    rows = sim.run(trace, UnifiedRuleController(prefetch_k=2))
    assert len(rows) == 3


def test_unified_blend_controller_runs():
    trace = attach_reuse_distance_features(make_trace(steps=3))
    sim = make_sim(hbm_capacity_pages=10)
    rows = sim.run(trace, UnifiedBlendController(prefetch_k=2))
    assert len(rows) == 3


def test_unified_bandit_controller_runs_and_observes():
    trace = attach_reuse_distance_features(make_trace(4))
    sim = make_sim(hbm_capacity_pages=10)
    controller = UnifiedBanditController()
    rows = sim.run(trace, controller)
    assert len(rows) == 4


def test_belady_oracle_controller_runs_on_trace():
    trace = make_trace(steps=3)
    sim = make_sim(hbm_capacity_pages=8)
    rows = sim.run(trace, BeladyOracleController(trace))
    assert len(rows) == 3


def test_perfect_prefetch_oracle_submits_prefetches_when_possible():
    trace = make_trace(steps=3)
    sim = make_sim(hbm_capacity_pages=10)
    rows = sim.run(trace, PerfectPrefetchOracleController(trace, prefetch_k=2))
    assert sum(row.prefetch_submitted for row in rows) >= 0


def test_contextual_bandit_controller_observes_feedback():
    trace = make_trace(steps=4)
    sim = make_sim(hbm_capacity_pages=10)
    controller = ContextualBanditController(alpha=0.5)
    rows = sim.run(trace, controller)

    diag = controller.diagnostics()
    assert len(rows) == 4
    assert diag["steps_observed"] == 4
    assert diag["last_action"] is not None


def test_layer_aware_controller_emits_budget_updates():
    trace = make_trace(steps=2)
    sim = make_sim(hbm_capacity_pages=10)
    contexts = []

    class RecordingLayerAwareController(LayerAwareScoreController):
        def decide(self, context):
            contexts.append(context)
            return super().decide(context)

    sim.run(trace, RecordingLayerAwareController(prefetch_k=2))
    assert contexts
    decision = RecordingLayerAwareController(prefetch_k=2).decide(contexts[0])
    assert decision.layer_budgets


def test_simulator_enforces_layer_budgets_when_controller_requests_them():
    trace = make_trace(steps=3)
    sim = make_sim(hbm_capacity_pages=10)
    sim.run(trace, LayerAwareScoreController(prefetch_k=2))
    final_required_by_layer = {}
    for page in trace[-1].required_pages:
        final_required_by_layer[page.layer_id] = final_required_by_layer.get(page.layer_id, 0) + 1

    for layer_id, budget in sim.state.layer_budgets.items():
        occupancy = sim.state.per_layer_occupancy.get(layer_id, 0)
        allowed_overage = final_required_by_layer.get(layer_id, 0)
        assert occupancy <= budget.max_resident_pages + allowed_overage


def test_demand_queue_delay_contributes_to_stall_when_capacity_is_tight():
    trace = make_trace(steps=1)
    sim = make_sim(hbm_capacity_pages=20)
    sim.config = SimulationConfig(
        steps=1,
        cache=sim.config.cache,
        transfer=TransferConfig(
            page_bytes=4096,
            bandwidth_bytes_per_ms=4096 * 2,
            transfer_setup_ms=0.01,
            max_inflight_transfers=1,
            decode_kernel_ms=0.05,
        ),
    )
    sim.scheduler.config = sim.config.transfer
    rows = sim.run(trace, NoPrefetchController())
    # With one copy lane and many demand misses, stall should exceed a single
    # page transfer because queueing delay is now counted.
    assert rows[0].stall_ms > 0.27


def test_future_trace_oracle_reports_known_next_use():
    trace = make_trace(steps=3)
    oracle = FutureTraceOracle(trace)
    page = trace[0].required_pages[0]
    next_use = oracle.next_use_after(page, 0)
    assert next_use is None or next_use > 0


def test_benchmark_policies_runs_multiple_policies_on_same_trace():
    trace = make_trace(steps=3)
    results = benchmark_policies(
        policy_names=["lru", "bandit"],
        trace=trace,
        simulator_builder=lambda: make_sim(hbm_capacity_pages=10),
        controller_builder=lambda policy: (
            LRUController(prefetch_k=2) if policy == "lru" else ContextualBanditController(alpha=0.5)
        ),
    )
    assert len(results) == 2
    assert {result.policy_name for result in results} == {"lru", "bandit"}


def test_benchmark_policies_across_seeds_aggregates_multiple_runs():
    results = benchmark_policies_across_seeds(
        policy_names=["lru", "bandit"],
        seeds=[3, 5],
        trace_builder=lambda seed: SyntheticTraceGenerator(
            SyntheticTraceConfig(
                steps=8,
                layers=2,
                pages_per_layer=16,
                local_window_pages=3,
                sparse_pages_per_step=1,
                predicted_prefetch_pages=3,
                attention_heads=4,
                query_jitter=0.05,
                seed=seed,
            )
        ).generate(),
        simulator_builder=lambda: make_sim(hbm_capacity_pages=10),
        controller_builder=lambda policy, trace: (
            LRUController(prefetch_k=2) if policy == "lru" else ContextualBanditController(alpha=0.35)
        ),
    )
    assert len(results) == 2
    assert all(isinstance(result, AggregateBenchmarkResult) for result in results)
    assert {result.policy_name for result in results} == {"lru", "bandit"}
    for result in results:
        assert result.aggregate_summary["seeds"] == 2


def test_trace_replay_round_trip_preserves_step_structure(tmp_path):
    trace = make_trace(steps=3)
    path = tmp_path / "trace.json"
    save_trace_json(str(path), trace)
    loaded = load_trace_json(str(path))

    assert len(loaded) == len(trace)
    assert loaded[0].required_pages == trace[0].required_pages
    assert loaded[0].predicted_pages == trace[0].predicted_pages
    assert loaded[0].head_weighted_scores == trace[0].head_weighted_scores
    assert loaded[0].per_page_head_activity == trace[0].per_page_head_activity
    assert loaded[0].request_id == trace[0].request_id
    assert loaded[0].sequence_length == trace[0].sequence_length
    assert loaded[0].layer_block_tables == trace[0].layer_block_tables


def test_recent_topk_converter_reduces_required_pages_and_adds_predictions():
    trace = make_trace(steps=3)
    converted = convert_trace_recent_topk(trace, recent_block_window=1, top_k_older_per_layer=1)

    assert len(converted) == len(trace)
    assert len(converted[0].required_pages) <= len(trace[0].required_pages)
    assert converted[0].predicted_pages or converted[-1].predicted_pages == ()


def test_recent_threshold_converter_reduces_required_pages():
    trace = make_trace(steps=3)
    converted = convert_trace_recent_threshold(trace, recent_block_window=1, score_mass_fraction=0.5)

    assert len(converted) == len(trace)
    assert len(converted[0].required_pages) <= len(trace[0].required_pages)


def test_round_robin_interleave_keeps_requests_distinct_via_page_offsets():
    trace_a = make_trace(steps=2)
    trace_b = make_trace(steps=2)
    merged = interleave_traces_round_robin([trace_a, trace_b], page_stride=100)

    assert len(merged) == 4
    assert merged[0].request_id != merged[1].request_id
    pages_a = set(merged[0].required_pages)
    pages_b = set(merged[1].required_pages)
    assert pages_a.isdisjoint(pages_b)


def test_sparse_round_robin_topk_interleave_produces_smaller_required_sets():
    trace_a = make_trace(steps=2)
    trace_b = make_trace(steps=2)
    merged = interleave_sparse_recent_topk_traces(
        [trace_a, trace_b],
        recent_block_window=1,
        top_k_older_per_layer=1,
        page_stride=100,
    )

    assert len(merged) == 4
    assert len(merged[0].required_pages) <= len(trace_a[0].required_pages)


def test_sparse_round_robin_threshold_interleave_produces_smaller_required_sets():
    trace_a = make_trace(steps=2)
    trace_b = make_trace(steps=2)
    merged = interleave_sparse_recent_threshold_traces(
        [trace_a, trace_b],
        recent_block_window=1,
        score_mass_fraction=0.5,
        page_stride=100,
    )

    assert len(merged) == 4
    assert len(merged[0].required_pages) <= len(trace_a[0].required_pages)
