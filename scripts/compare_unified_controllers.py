#!/usr/bin/env python3
"""Compare single-controller designs across the three-regime benchmark suite."""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import (
    attach_reuse_distance_features,
    benchmark_policies,
    CacheConfig,
    ContextualBanditController,
    OverlapAwareSimulator,
    ScoreBasedController,
    SimulationConfig,
    TransferConfig,
    UnifiedBanditController,
    UnifiedBlendController,
    UnifiedRuleController,
    UnifiedThompsonController,
    load_trace_json,
    summarize_page_tile_stats,
    summarize_realism_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare unified controller options on the three-regime suite.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="results/broader_real_benchmark/benchmarks",
    )
    parser.add_argument("--main-capacity", type=int, default=32)
    parser.add_argument("--middle-capacity", type=int, default=24)
    parser.add_argument("--hard-capacity", type=int, default=84)
    parser.add_argument("--prefetch-k", type=int, default=2)
    return parser


def build_simulator(trace, capacity: int) -> OverlapAwareSimulator:
    layers = max((page.layer_id for step in trace for page in step.required_pages), default=-1) + 1
    return OverlapAwareSimulator(
        SimulationConfig(
            steps=len(trace),
            cache=CacheConfig(
                hbm_capacity_pages=capacity,
                layers=layers,
                page_bytes=4096,
            ),
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


def build_controller(name: str, prefetch_k: int, regime_prior: dict[str, float] | None = None):
    if name == "score":
        return ScoreBasedController(prefetch_k=prefetch_k)
    if name == "bandit":
        return ContextualBanditController()
    if name == "unified_rule":
        return UnifiedRuleController(prefetch_k=prefetch_k)
    if name == "unified_blend":
        return UnifiedBlendController(prefetch_k=prefetch_k)
    if name == "unified_bandit":
        return UnifiedBanditController(regime_prior_metrics=regime_prior)
    if name == "unified_thompson":
        return UnifiedThompsonController(regime_prior_metrics=regime_prior)
    raise ValueError(f"Unknown policy: {name}")


def print_realism_gate(label: str, capacity: int, regime_prior: dict[str, float]) -> None:
    """Print realism gate metrics so every benchmark in the report is validated."""
    pmr = regime_prior.get("page_miss_rate", float("nan"))
    phr = regime_prior.get("prefetch_hit_rate", float("nan"))
    epa = regime_prior.get("evictions_per_access", float("nan"))
    hard = pmr < 0.98 and phr >= 0.50
    hostile = pmr >= 0.99 or phr <= 0.05 or epa >= 1.0
    regime_tag = "hard-reuse" if hard else ("hostile" if hostile else "middle")
    print(
        f"  realism: page_miss_rate={pmr:.4f}  prefetch_hit_rate={phr:.4f}"
        f"  evictions/access={epa:.4f}  [{regime_tag}]"
    )


def print_results(label: str, results) -> None:
    print(f"\n{label}")
    print("policy | total_miss | mean_stall | p95_stall | prefetch | evictions")
    for result in results:
        row = result.summary
        print(
            f"{str(row['policy']):>14} | "
            f"{int(row['total_demand_misses']):>10} | "
            f"{float(row['mean_stall_ms']):>10.4f} | "
            f"{float(row['p95_stall_ms']):>9.4f} | "
            f"{int(row['total_prefetches_submitted']):>8} | "
            f"{int(row['total_evictions']):>9}"
        )


def main() -> None:
    args = build_parser().parse_args()
    suite = [
        ("MAIN", os.path.join(args.benchmark_dir, "recent_threshold_round_robin_interleave.json"), args.main_capacity),
        ("MIDDLE", os.path.join(args.benchmark_dir, "recent_topk_round_robin_interleave.json"), args.middle_capacity),
        ("HARD", os.path.join(args.benchmark_dir, "round_robin_interleave.json"), args.hard_capacity),
    ]
    policy_names = ["score", "bandit", "unified_rule", "unified_blend", "unified_bandit", "unified_thompson"]

    for label, trace_path, capacity in suite:
        trace = attach_reuse_distance_features(load_trace_json(trace_path))
        regime_prior = compute_regime_prior(trace, capacity)
        print_realism_gate(f"{label} | capacity={capacity}", capacity, regime_prior)
        results = benchmark_policies(
            policy_names=policy_names,
            trace=trace,
            simulator_builder=lambda trace=trace, capacity=capacity: build_simulator(trace, capacity),
            controller_builder=lambda name, regime_prior=regime_prior: build_controller(
                name, args.prefetch_k, regime_prior
            ),
        )
        print_results(f"{label} | capacity={capacity}", results)


if __name__ == "__main__":
    main()
