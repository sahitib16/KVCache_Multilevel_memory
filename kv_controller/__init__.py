"""Core simulator package for overlap-aware multi-tier KV residency control.

This file is intentionally small, but it matters because it is the public
"front door" of the package.

Why this file exists:
- It lets other files import the package with short, readable imports like
  ``from kv_controller import OverlapAwareSimulator``.
- It defines which classes and functions count as the package's public API.
- It gives us one central place to document the package at a high level.

What this package currently contains:
- Type definitions that describe pages, steps, transfers, and metrics.
- Interfaces for scorers and controllers.
- Mutable cache state used by the simulator.
- A discrete-event overlap scheduler.
- A synthetic workload generator.
- The main simulator loop.
"""

from .benchmark import (
    AggregateBenchmarkResult,
    BenchmarkResult,
    aggregate_policy_summaries,
    benchmark_policies,
    benchmark_policies_across_seeds,
    collect_step_rows,
    qtile,
    summarize_run,
    write_aggregate_summary_csv,
    write_step_csv,
    write_summary_csv,
)
from .interfaces import HeadWeightedScorer, ResidencyController
from .policies import (
    BanditAction,
    BeladyOracleController,
    build_bandit_action_menu,
    ContextualBanditController,
    FutureTraceOracle,
    LayerAwareScoreController,
    LRUController,
    PerfectPrefetchOracleController,
    ScoreBasedController,
)
from .replay import load_trace_json, save_trace_json, trace_to_json_rows
from .scoring import (
    attach_reuse_distance_features,
    HeadActivityRecomputedScorer,
    LayerNormalizedHeadActivityScorer,
    NormalizedHeadWeightedScorer,
    PageStatsHybridScorer,
    PassthroughHeadWeightedScorer,
    PredictedBoostedHeadActivityScorer,
    ReuseDistanceHybridScorer,
    apply_scorer_to_trace,
)
from .score_diagnostics import diagnose_trace_scores
from .scheduler import OverlapTransferScheduler
from .simulator import OverlapAwareSimulator
from .stats import summarize_page_tile_stats, summarize_realism_metrics, write_rows_csv
from .state import CacheState
from .trace_transforms import (
    convert_trace_recent_threshold,
    convert_trace_recent_topk,
    interleave_sparse_recent_threshold_traces,
    interleave_sparse_recent_topk_traces,
    interleave_traces_round_robin,
)
from .types import (
    CacheConfig,
    ControllerContext,
    KVPageId,
    LayerBudget,
    PolicyDecision,
    PolicyOutput,
    SimulationConfig,
    StepMetrics,
    TransferConfig,
    TransferKind,
    TransferRequest,
    TransferState,
    WorkloadStep,
)
from .workload import SyntheticTraceConfig, SyntheticTraceGenerator

__all__ = [
    "BanditAction",
    "BeladyOracleController",
    "AggregateBenchmarkResult",
    "BenchmarkResult",
    "aggregate_policy_summaries",
    "attach_reuse_distance_features",
    "benchmark_policies",
    "benchmark_policies_across_seeds",
    "build_bandit_action_menu",
    "CacheConfig",
    "CacheState",
    "collect_step_rows",
    "ControllerContext",
    "ContextualBanditController",
    "convert_trace_recent_threshold",
    "convert_trace_recent_topk",
    "diagnose_trace_scores",
    "FutureTraceOracle",
    "HeadWeightedScorer",
    "HeadActivityRecomputedScorer",
    "KVPageId",
    "LayerNormalizedHeadActivityScorer",
    "LayerAwareScoreController",
    "LayerBudget",
    "load_trace_json",
    "LRUController",
    "NormalizedHeadWeightedScorer",
    "PageStatsHybridScorer",
    "OverlapTransferScheduler",
    "OverlapAwareSimulator",
    "PassthroughHeadWeightedScorer",
    "PerfectPrefetchOracleController",
    "PredictedBoostedHeadActivityScorer",
    "ReuseDistanceHybridScorer",
    "PolicyDecision",
    "PolicyOutput",
    "qtile",
    "ResidencyController",
    "ScoreBasedController",
    "apply_scorer_to_trace",
    "interleave_traces_round_robin",
    "interleave_sparse_recent_threshold_traces",
    "interleave_sparse_recent_topk_traces",
    "save_trace_json",
    "SimulationConfig",
    "summarize_page_tile_stats",
    "summarize_realism_metrics",
    "StepMetrics",
    "SyntheticTraceConfig",
    "SyntheticTraceGenerator",
    "TransferConfig",
    "TransferKind",
    "TransferRequest",
    "TransferState",
    "trace_to_json_rows",
    "WorkloadStep",
    "write_rows_csv",
    "summarize_run",
    "write_step_csv",
    "write_summary_csv",
    "write_aggregate_summary_csv",
]
