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
    ContextualBanditController,
    FutureTraceOracle,
    LayerAwareScoreController,
    LRUController,
    PerfectPrefetchOracleController,
    ScoreBasedController,
)
from .scoring import NormalizedHeadWeightedScorer, PassthroughHeadWeightedScorer
from .scheduler import OverlapTransferScheduler
from .simulator import OverlapAwareSimulator
from .state import CacheState
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
    "benchmark_policies",
    "benchmark_policies_across_seeds",
    "CacheConfig",
    "CacheState",
    "collect_step_rows",
    "ControllerContext",
    "ContextualBanditController",
    "FutureTraceOracle",
    "HeadWeightedScorer",
    "KVPageId",
    "LayerAwareScoreController",
    "LayerBudget",
    "LRUController",
    "NormalizedHeadWeightedScorer",
    "OverlapTransferScheduler",
    "OverlapAwareSimulator",
    "PassthroughHeadWeightedScorer",
    "PerfectPrefetchOracleController",
    "PolicyDecision",
    "PolicyOutput",
    "qtile",
    "ResidencyController",
    "ScoreBasedController",
    "SimulationConfig",
    "StepMetrics",
    "SyntheticTraceConfig",
    "SyntheticTraceGenerator",
    "TransferConfig",
    "TransferKind",
    "TransferRequest",
    "TransferState",
    "WorkloadStep",
    "summarize_run",
    "write_step_csv",
    "write_summary_csv",
    "write_aggregate_summary_csv",
]
