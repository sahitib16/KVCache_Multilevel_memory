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

from .interfaces import HeadWeightedScorer, ResidencyController
from .policies import (
    BeladyOracleController,
    FutureTraceOracle,
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
    "CacheConfig",
    "CacheState",
    "ControllerContext",
    "FutureTraceOracle",
    "HeadWeightedScorer",
    "KVPageId",
    "LayerBudget",
    "LRUController",
    "NormalizedHeadWeightedScorer",
    "OverlapTransferScheduler",
    "OverlapAwareSimulator",
    "PassthroughHeadWeightedScorer",
    "PerfectPrefetchOracleController",
    "PolicyDecision",
    "PolicyOutput",
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
    "BeladyOracleController",
]
