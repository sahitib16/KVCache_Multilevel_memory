from __future__ import annotations

"""Benchmark helpers for repeatable policy comparison.

Why this file exists:
- The simulator knows how to run one controller on one trace.
- The driver script knows how to parse command-line arguments.
- But we also need reusable logic for:
  - running several policies on the *same* trace
  - collecting step-level rows
  - building summary rows
  - writing CSV outputs consistently

Putting that logic here keeps the benchmarking path reusable from:
- the CLI driver
- future notebooks
- future integration harnesses
"""

from dataclasses import dataclass
import csv
import os
import statistics
from typing import Callable

from .policies import ContextualBanditController
from .simulator import OverlapAwareSimulator


@dataclass(frozen=True)
class BenchmarkResult:
    """One policy run over one fixed trace."""

    policy_name: str
    metrics: list
    summary: dict[str, object]


@dataclass(frozen=True)
class AggregateBenchmarkResult:
    """Summary of one policy evaluated over multiple random seeds.

    This is especially important for adaptive controllers because one short run
    on one seed can be very misleading. By aggregating across seeds we can ask:
    - does the policy usually help?
    - how stable is it?
    - what is the tradeoff between misses and stall on average?
    """

    policy_name: str
    per_seed_summaries: list[dict[str, object]]
    aggregate_summary: dict[str, object]


def qtile(values: list[float], q: float) -> float:
    """Small quantile helper used for summary metrics."""

    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize_run(policy_name: str, metrics, simulator, controller) -> dict[str, object]:
    """Convert one run into a compact summary row.

    The intent is to keep all policies comparable on the same set of key fields:
    - demand misses
    - stall behavior
    - prefetch / eviction volume
    - final residency / transfer state
    - optional adaptive-controller diagnostics
    """

    stall_values = [row.stall_ms for row in metrics]
    demand_miss_values = [float(row.demand_misses) for row in metrics]

    summary = {
        "policy": policy_name,
        "steps": len(metrics),
        "total_demand_misses": sum(row.demand_misses for row in metrics),
        "mean_demand_misses": statistics.mean(demand_miss_values) if demand_miss_values else 0.0,
        "total_prefetches_submitted": sum(row.prefetch_submitted for row in metrics),
        "total_stall_ms": sum(row.stall_ms for row in metrics),
        "mean_stall_ms": statistics.mean(stall_values) if stall_values else 0.0,
        "p95_stall_ms": qtile(stall_values, 0.95),
        "p99_stall_ms": qtile(stall_values, 0.99),
        "total_compute_ms": sum(row.compute_ms for row in metrics),
        "total_evictions": sum(row.evictions for row in metrics),
        "final_resident_pages": len(simulator.state.resident_pages),
        "final_slot_map_size": len(simulator.state.slot_to_page),
        "final_inflight_transfers": len(simulator.state.inflight_pages),
        "final_queued_transfers": len(simulator.state.transfer_state.queued_pages),
        "final_backlog": simulator.state.transfer_state.backlog,
        "hbm_format": simulator.config.cache.hbm_kv_format,
    }

    if isinstance(controller, ContextualBanditController):
        diag = controller.diagnostics()
        summary["bandit_steps_observed"] = diag["steps_observed"]
        summary["bandit_total_reward"] = diag["total_reward"]
        summary["bandit_last_action"] = str(diag["last_action"])

    return summary


def collect_step_rows(policy_name: str, metrics) -> list[dict[str, object]]:
    """Flatten step metrics into CSV-friendly dictionaries."""

    rows = []
    for row in metrics:
        rows.append(
            {
                "policy": policy_name,
                "step_idx": row.step_idx,
                "required_pages": row.required_pages,
                "predicted_pages": row.predicted_pages,
                "demand_misses": row.demand_misses,
                "prefetch_submitted": row.prefetch_submitted,
                "evictions": row.evictions,
                "stall_ms": row.stall_ms,
                "overlap_ms": row.overlap_ms,
                "transfer_wait_ms": row.transfer_wait_ms,
                "compute_ms": row.compute_ms,
                "copy_busy_ms": row.copy_busy_ms,
                "queue_delay_ms": row.queue_delay_ms,
                "transfer_backlog": row.transfer_backlog,
                "inflight_transfers": row.inflight_transfers,
                "churn": row.churn,
            }
        )
    return rows


def benchmark_policies(
    policy_names: list[str],
    trace,
    simulator_builder: Callable[[], OverlapAwareSimulator],
    controller_builder: Callable[[str], object],
) -> list[BenchmarkResult]:
    """Run a suite of policies on the same fixed trace.

    Important design rule:
    - one shared trace for fairness
    - one fresh simulator per policy so state does not leak between runs
    """

    results: list[BenchmarkResult] = []
    for policy_name in policy_names:
        simulator = simulator_builder()
        controller = controller_builder(policy_name)
        metrics = simulator.run(trace, controller)
        summary = summarize_run(policy_name, metrics, simulator, controller)
        results.append(BenchmarkResult(policy_name=policy_name, metrics=metrics, summary=summary))
    return results


def aggregate_policy_summaries(
    policy_name: str,
    per_seed_summaries: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate a list of one-run summaries into one cross-seed summary row.

    We report both central tendency and spread for the most important metrics.
    This makes it easier to judge whether an adaptive controller is:
    - consistently helpful
    - only winning on one lucky seed
    - trading misses for stall in a stable way
    """

    if not per_seed_summaries:
        return {"policy": policy_name, "seeds": 0}

    def values(key: str) -> list[float]:
        return [float(summary[key]) for summary in per_seed_summaries]

    mean_stall_values = values("mean_stall_ms")
    p95_stall_values = values("p95_stall_ms")
    p99_stall_values = values("p99_stall_ms")
    miss_values = values("total_demand_misses")
    prefetch_values = values("total_prefetches_submitted")
    eviction_values = values("total_evictions")

    return {
        "policy": policy_name,
        "seeds": len(per_seed_summaries),
        "avg_total_demand_misses": statistics.mean(miss_values),
        "avg_mean_stall_ms": statistics.mean(mean_stall_values),
        "avg_p95_stall_ms": statistics.mean(p95_stall_values),
        "avg_p99_stall_ms": statistics.mean(p99_stall_values),
        "avg_total_prefetches_submitted": statistics.mean(prefetch_values),
        "avg_total_evictions": statistics.mean(eviction_values),
        "std_total_demand_misses": statistics.pstdev(miss_values) if len(miss_values) > 1 else 0.0,
        "std_mean_stall_ms": statistics.pstdev(mean_stall_values) if len(mean_stall_values) > 1 else 0.0,
        "std_p95_stall_ms": statistics.pstdev(p95_stall_values) if len(p95_stall_values) > 1 else 0.0,
        "std_p99_stall_ms": statistics.pstdev(p99_stall_values) if len(p99_stall_values) > 1 else 0.0,
        "std_total_prefetches_submitted": statistics.pstdev(prefetch_values) if len(prefetch_values) > 1 else 0.0,
        "std_total_evictions": statistics.pstdev(eviction_values) if len(eviction_values) > 1 else 0.0,
    }


def benchmark_policies_across_seeds(
    policy_names: list[str],
    seeds: list[int],
    trace_builder: Callable[[int], object],
    simulator_builder: Callable[[], OverlapAwareSimulator],
    controller_builder: Callable[[str, object], object],
) -> list[AggregateBenchmarkResult]:
    """Run each policy across multiple seeds and aggregate the summaries.

    Design rule:
    - every policy sees the *same seed list*
    - within one seed, every policy sees the *same trace*
    - each policy/seed pair gets a fresh simulator and fresh controller
    """

    results: list[AggregateBenchmarkResult] = []
    for policy_name in policy_names:
        per_seed_summaries: list[dict[str, object]] = []
        for seed in seeds:
            trace = trace_builder(seed)
            simulator = simulator_builder()
            controller = controller_builder(policy_name, trace)
            metrics = simulator.run(trace, controller)
            summary = summarize_run(policy_name, metrics, simulator, controller)
            summary["seed"] = seed
            per_seed_summaries.append(summary)
        aggregate_summary = aggregate_policy_summaries(policy_name, per_seed_summaries)
        results.append(
            AggregateBenchmarkResult(
                policy_name=policy_name,
                per_seed_summaries=per_seed_summaries,
                aggregate_summary=aggregate_summary,
            )
        )
    return results


def write_step_csv(path: str, results: list[BenchmarkResult]) -> None:
    """Write step-level metrics for one or many policies."""

    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for result in results:
        rows.extend(collect_step_rows(result.policy_name, result.metrics))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: str, results: list[BenchmarkResult]) -> None:
    """Write one summary row per policy."""

    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [result.summary for result in results]
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_summary_csv(path: str, results: list[AggregateBenchmarkResult]) -> None:
    """Write one aggregated summary row per policy for multi-seed studies."""

    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [result.aggregate_summary for result in results]
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
