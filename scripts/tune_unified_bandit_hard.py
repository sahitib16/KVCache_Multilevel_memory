#!/usr/bin/env python3
"""Sweep reward parameters for UnifiedBanditController on the hard benchmark only.

Why this script exists:
- The unified bandit is tuned by default across all three regimes.
- The hard regime (round_robin_interleave) has different reward structure:
    prefetch_hit_rate ≈ 1, so wasted prefetch penalty should be near zero.
    Each miss is costlier, so miss_penalty should be higher.
- This script sweeps those hard-regime-specific coefficients in isolation
  so we can empirically confirm that the regime scaling direction is correct.

Usage:
    python scripts/tune_unified_bandit_hard.py
    python scripts/tune_unified_bandit_hard.py --benchmark-dir results/broader_real_benchmark/benchmarks
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import (
    attach_reuse_distance_features,
    CacheConfig,
    OverlapAwareSimulator,
    ScoreBasedController,
    SimulationConfig,
    TransferConfig,
    UnifiedBanditController,
    UnifiedThompsonController,
    load_trace_json,
    summarize_page_tile_stats,
    summarize_realism_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep reward parameters for the unified bandit on the hard benchmark."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="results/broader_real_benchmark/benchmarks",
    )
    parser.add_argument("--capacity", type=int, default=84)
    parser.add_argument("--top-k", type=int, default=10, help="Print top-K configurations.")
    return parser


def build_simulator(trace, capacity: int) -> OverlapAwareSimulator:
    layers = max((p.layer_id for step in trace for p in step.required_pages), default=-1) + 1
    return OverlapAwareSimulator(
        SimulationConfig(
            steps=len(trace),
            cache=CacheConfig(hbm_capacity_pages=capacity, layers=layers, page_bytes=4096),
            transfer=TransferConfig(
                page_bytes=4096,
                bandwidth_bytes_per_ms=16384.0,
                transfer_setup_ms=0.02,
                max_inflight_transfers=2,
                decode_kernel_ms=0.05,
            ),
        )
    )


def compute_regime_prior(trace, capacity: int) -> dict[str, float]:
    sim = build_simulator(trace, capacity)
    metrics = sim.run(trace, ScoreBasedController(prefetch_k=2))
    page_rows, layer_rows, tile_rows = summarize_page_tile_stats(trace, metrics, tile_size_pages=4)
    realism = summarize_realism_metrics(page_rows, layer_rows, tile_rows)
    return {
        "page_miss_rate": float(realism["page_miss_rate"]),
        "prefetch_hit_rate": float(realism["prefetch_hit_rate"]),
        "evictions_per_access": float(realism["evictions_per_access"]),
    }


def run_bandit(
    trace,
    capacity: int,
    regime_prior: dict[str, float],
    miss_penalty: float,
    budget_wasted_penalty: float,
    budget_useful_bonus: float,
    controller_class,
) -> dict:
    sim = build_simulator(trace, capacity)
    controller = controller_class(
        regime_prior_metrics=regime_prior,
        miss_penalty=miss_penalty,
        budget_wasted_prefetch_penalty=budget_wasted_penalty,
        budget_useful_prefetch_bonus=budget_useful_bonus,
    )
    step_metrics = sim.run(trace, controller)
    total_miss = sum(m.demand_misses for m in step_metrics)
    mean_stall = sum(m.stall_ms for m in step_metrics) / max(1, len(step_metrics))
    total_prefetch = sum(m.prefetch_submitted for m in step_metrics)
    p95_idx = max(0, int(len(step_metrics) * 0.95) - 1)
    stalls_sorted = sorted(m.stall_ms for m in step_metrics)
    p95_stall = stalls_sorted[p95_idx] if stalls_sorted else 0.0
    return {
        "total_miss": total_miss,
        "mean_stall": mean_stall,
        "p95_stall": p95_stall,
        "total_prefetch": total_prefetch,
        "miss_penalty": miss_penalty,
        "budget_wasted_penalty": budget_wasted_penalty,
        "budget_useful_bonus": budget_useful_bonus,
    }


def main() -> None:
    args = build_parser().parse_args()

    hard_path = os.path.join(args.benchmark_dir, "round_robin_interleave.json")
    print(f"Loading hard benchmark: {hard_path}")
    trace = attach_reuse_distance_features(load_trace_json(hard_path))
    regime_prior = compute_regime_prior(trace, args.capacity)

    print(f"\nRegime prior for hard benchmark (capacity={args.capacity}):")
    for k, v in regime_prior.items():
        print(f"  {k}: {v:.4f}")

    # Sweep parameters that matter most for the hard regime.
    miss_penalties = [0.08, 0.10, 0.12, 0.15, 0.20]
    wasted_penalties = [0.001, 0.005, 0.01, 0.025, 0.05]
    useful_bonuses = [0.02, 0.04, 0.06, 0.08]

    print(f"\nSweeping {len(miss_penalties) * len(wasted_penalties) * len(useful_bonuses)} "
          f"parameter combinations on hard benchmark...")

    results_bandit = []
    results_thompson = []

    for mp, wp, ub in itertools.product(miss_penalties, wasted_penalties, useful_bonuses):
        r_b = run_bandit(trace, args.capacity, regime_prior, mp, wp, ub, UnifiedBanditController)
        r_t = run_bandit(trace, args.capacity, regime_prior, mp, wp, ub, UnifiedThompsonController)
        r_b["controller"] = "unified_bandit"
        r_t["controller"] = "unified_thompson"
        results_bandit.append(r_b)
        results_thompson.append(r_t)

    # Rank by miss count first, then stall.
    for label, results in [("UnifiedBanditController", results_bandit), ("UnifiedThompsonController", results_thompson)]:
        results.sort(key=lambda r: (r["total_miss"], r["mean_stall"]))
        print(f"\n--- Top {args.top_k} configurations for {label} ---")
        print(f"{'miss_pen':>8} {'wasted_pen':>10} {'useful_b':>9} | {'total_miss':>10} {'mean_stall':>10} {'p95_stall':>9} {'prefetch':>8}")
        print("-" * 75)
        for r in results[: args.top_k]:
            print(
                f"{r['miss_penalty']:>8.3f} {r['budget_wasted_penalty']:>10.4f} {r['budget_useful_bonus']:>9.3f} | "
                f"{r['total_miss']:>10} {r['mean_stall']:>10.4f} {r['p95_stall']:>9.4f} {r['total_prefetch']:>8}"
            )

    # Also show the default (no override) for comparison.
    print("\n--- Defaults (regime-scaled, no explicit override) ---")
    for cls, label in [(UnifiedBanditController, "unified_bandit"), (UnifiedThompsonController, "unified_thompson")]:
        sim = build_simulator(trace, args.capacity)
        controller = cls(regime_prior_metrics=regime_prior)
        step_metrics = sim.run(trace, controller)
        total_miss = sum(m.demand_misses for m in step_metrics)
        mean_stall = sum(m.stall_ms for m in step_metrics) / max(1, len(step_metrics))
        total_prefetch = sum(m.prefetch_submitted for m in step_metrics)
        print(f"  {label}: total_miss={total_miss}  mean_stall={mean_stall:.4f}  prefetch={total_prefetch}")


if __name__ == "__main__":
    main()
