#!/usr/bin/env python3
"""Small bandit-tuning harness for Step 4 experiments.

Why this script exists:
- The contextual bandit now has several interpretable reward coefficients.
- We want to tune them systematically instead of changing one constant by hand
  and guessing from one run.
- We also want to see some light "feature importance" style diagnostics from
  the linear bandit after a run.

What this script does:
- sweeps a compact grid of bandit reward settings
- evaluates each setting across multiple seeds
- reports the best settings under a simple ranking objective
- prints the bandit's action counts and learned linear coefficients for the
  top configuration

This is not meant to be the final research harness. It is a practical Step 4
tool for understanding the miss/stall tradeoff better.
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
    CacheConfig,
    ContextualBanditController,
    OverlapAwareSimulator,
    SimulationConfig,
    SyntheticTraceConfig,
    SyntheticTraceGenerator,
    TransferConfig,
    benchmark_policies_across_seeds,
)


def build_parser() -> argparse.ArgumentParser:
    """Collect tuning arguments in one place."""

    parser = argparse.ArgumentParser(description="Sweep contextual-bandit reward settings.")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pages-per-layer", type=int, default=24)
    parser.add_argument("--local-window-pages", type=int, default=4)
    parser.add_argument("--sparse-pages-per-step", type=int, default=2)
    parser.add_argument("--predicted-prefetch-pages", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--query-jitter", type=float, default=0.05)
    parser.add_argument("--hbm-capacity-pages", type=int, default=20)
    parser.add_argument("--page-bytes", type=int, default=4096)
    parser.add_argument("--bandwidth-bytes-per-ms", type=float, default=16384.0)
    parser.add_argument("--transfer-setup-ms", type=float, default=0.02)
    parser.add_argument("--max-inflight-transfers", type=int, default=2)
    parser.add_argument("--decode-kernel-ms", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def build_trace(args, seed: int):
    """Build one synthetic trace for one seed."""

    return SyntheticTraceGenerator(
        SyntheticTraceConfig(
            steps=args.steps,
            layers=args.layers,
            pages_per_layer=args.pages_per_layer,
            local_window_pages=args.local_window_pages,
            sparse_pages_per_step=args.sparse_pages_per_step,
            predicted_prefetch_pages=args.predicted_prefetch_pages,
            attention_heads=args.attention_heads,
            query_jitter=args.query_jitter,
            seed=seed,
        )
    ).generate()


def build_simulator(args) -> OverlapAwareSimulator:
    """Build a fresh simulator per run."""

    return OverlapAwareSimulator(
        SimulationConfig(
            steps=args.steps,
            cache=CacheConfig(
                hbm_capacity_pages=args.hbm_capacity_pages,
                layers=args.layers,
                page_bytes=args.page_bytes,
            ),
            transfer=TransferConfig(
                page_bytes=args.page_bytes,
                bandwidth_bytes_per_ms=args.bandwidth_bytes_per_ms,
                transfer_setup_ms=args.transfer_setup_ms,
                max_inflight_transfers=args.max_inflight_transfers,
                decode_kernel_ms=args.decode_kernel_ms,
            ),
        )
    )


def ranking_score(summary: dict[str, float]) -> tuple[float, float, float]:
    """Return a sortable objective tuple for "good" configurations.

    Ordering rule:
    - first minimize average mean stall
    - then minimize average total misses
    - then minimize average p99 stall

    This mirrors the current project priority:
    stall is primary, but miss explosion is not acceptable.
    """

    return (
        float(summary["avg_mean_stall_ms"]),
        float(summary["avg_total_demand_misses"]),
        float(summary["avg_p99_stall_ms"]),
    )


def print_top_results(results: list[dict[str, object]], top_k: int) -> None:
    """Print the best tuning results in a compact table."""

    print("rank | miss_pen | useful | wasted | miss_red | miss_inc | avg_stall | avg_miss | avg_p99")
    for idx, row in enumerate(results[:top_k], start=1):
        print(
            f"{idx:>4} | "
            f"{float(row['miss_penalty']):>8.3f} | "
            f"{float(row['useful_prefetch_bonus']):>6.3f} | "
            f"{float(row['wasted_prefetch_penalty']):>6.3f} | "
            f"{float(row['miss_reduction_bonus']):>8.3f} | "
            f"{float(row['miss_increase_penalty']):>8.3f} | "
            f"{float(row['avg_mean_stall_ms']):>9.4f} | "
            f"{float(row['avg_total_demand_misses']):>8.2f} | "
            f"{float(row['avg_p99_stall_ms']):>7.4f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    # Compact grid so the sweep stays practical in a normal development loop.
    miss_penalties = [0.08, 0.10, 0.12]
    useful_bonuses = [0.04, 0.06]
    wasted_penalties = [0.015, 0.02, 0.03]
    miss_reduction_bonuses = [0.02, 0.04]
    miss_increase_penalties = [0.02, 0.04]

    ranked_rows: list[dict[str, object]] = []
    best_config = None

    for miss_penalty, useful_bonus, wasted_penalty, miss_reduction_bonus, miss_increase_penalty in itertools.product(
        miss_penalties,
        useful_bonuses,
        wasted_penalties,
        miss_reduction_bonuses,
        miss_increase_penalties,
    ):
        aggregate_results = benchmark_policies_across_seeds(
            policy_names=["bandit"],
            seeds=seeds,
            trace_builder=lambda seed: build_trace(args, seed),
            simulator_builder=lambda: build_simulator(args),
            controller_builder=lambda _policy_name, _trace: ContextualBanditController(
                alpha=0.35,
                miss_penalty=miss_penalty,
                useful_prefetch_bonus=useful_bonus,
                wasted_prefetch_penalty=wasted_penalty,
                miss_reduction_bonus=miss_reduction_bonus,
                miss_increase_penalty=miss_increase_penalty,
            ),
        )
        summary = aggregate_results[0].aggregate_summary
        row = {
            "miss_penalty": miss_penalty,
            "useful_prefetch_bonus": useful_bonus,
            "wasted_prefetch_penalty": wasted_penalty,
            "miss_reduction_bonus": miss_reduction_bonus,
            "miss_increase_penalty": miss_increase_penalty,
            **summary,
        }
        ranked_rows.append(row)

        if best_config is None or ranking_score(row) < ranking_score(best_config):
            best_config = row

    ranked_rows.sort(key=ranking_score)
    print_top_results(ranked_rows, args.top_k)

    if best_config is None:
        return

    print("\nBEST CONFIG")
    for key in [
        "miss_penalty",
        "useful_prefetch_bonus",
        "wasted_prefetch_penalty",
        "miss_reduction_bonus",
        "miss_increase_penalty",
        "avg_mean_stall_ms",
        "avg_total_demand_misses",
        "avg_p95_stall_ms",
        "avg_p99_stall_ms",
    ]:
        print(f"{key}: {best_config[key]}")

    # Run one representative seed with the best config so we can inspect the
    # learned linear diagnostics.
    representative_seed = seeds[0]
    trace = build_trace(args, representative_seed)
    simulator = build_simulator(args)
    controller = ContextualBanditController(
        alpha=0.35,
        miss_penalty=float(best_config["miss_penalty"]),
        useful_prefetch_bonus=float(best_config["useful_prefetch_bonus"]),
        wasted_prefetch_penalty=float(best_config["wasted_prefetch_penalty"]),
        miss_reduction_bonus=float(best_config["miss_reduction_bonus"]),
        miss_increase_penalty=float(best_config["miss_increase_penalty"]),
    )
    simulator.run(trace, controller)
    diag = controller.diagnostics()

    print("\nACTION COUNTS")
    for action, count in diag["action_counts"].items():
        print(f"{action}: {count}")

    print("\nFEATURE WEIGHTS")
    feature_names = diag["feature_names"]
    theta_by_action = diag["theta_by_action"]
    for action, coeffs in theta_by_action.items():
        print(f"\n{action}")
        for name, coeff in zip(feature_names, coeffs):
            print(f"  {name}: {coeff:.4f}")


if __name__ == "__main__":
    main()
