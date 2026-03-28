#!/usr/bin/env python3
"""Tune bandit reward coefficients on replay traces instead of synthetic traces.

This script is the replay-trace analogue of `tune_bandit_reward.py`.
It is meant for the next stage where the main benchmark is:
- `recent_threshold_round_robin_interleave`

We sweep a compact reward grid and evaluate the bandit directly on one or more
fixed replay traces.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate_replay_scores import infer_trace_shape

from kv_controller import (
    CacheConfig,
    ContextualBanditController,
    OverlapAwareSimulator,
    SimulationConfig,
    TransferConfig,
    benchmark_policies,
    build_bandit_action_menu,
    load_trace_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune the contextual bandit on replay traces.")
    parser.add_argument(
        "--trace-jsons",
        type=str,
        required=True,
        help="Comma-separated replay traces to evaluate.",
    )
    parser.add_argument(
        "--capacity",
        type=str,
        default="min",
        help="Replay capacity setting: min, min+N, min*N.F, or an integer page count.",
    )
    parser.add_argument("--page-bytes", type=int, default=4096)
    parser.add_argument("--bandwidth-bytes-per-ms", type=float, default=16384.0)
    parser.add_argument("--transfer-setup-ms", type=float, default=0.02)
    parser.add_argument("--max-inflight-transfers", type=int, default=2)
    parser.add_argument("--decode-kernel-ms", type=float, default=0.05)
    parser.add_argument("--bandit-menu", type=str, default="full")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def _capacity_choice(spec: str, min_required: int) -> int:
    import math

    if spec == "min":
        return min_required
    if spec.startswith("min+"):
        return min_required + int(spec[4:])
    if spec.startswith("min*"):
        return int(math.ceil(min_required * float(spec[4:])))
    return int(spec)


def build_simulator(trace, args) -> OverlapAwareSimulator:
    steps, layers, min_required = infer_trace_shape(trace)
    capacity = _capacity_choice(args.capacity, min_required)
    return OverlapAwareSimulator(
        SimulationConfig(
            steps=steps,
            cache=CacheConfig(
                hbm_capacity_pages=capacity,
                layers=layers,
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


def ranking_key(row: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(row["avg_mean_stall_ms"]),
        float(row["avg_total_demand_misses"]),
        float(row["avg_total_evictions"]),
    )


def main() -> None:
    args = build_parser().parse_args()
    traces = [load_trace_json(path.strip()) for path in args.trace_jsons.split(",") if path.strip()]

    miss_penalties = [0.08, 0.10, 0.12]
    useful_bonuses = [0.04, 0.06]
    wasted_penalties = [0.015, 0.02, 0.03]
    miss_reduction_bonuses = [0.02, 0.04]
    miss_increase_penalties = [0.02, 0.04]

    ranked_rows: list[dict[str, float]] = []

    for miss_penalty, useful_bonus, wasted_penalty, miss_reduction_bonus, miss_increase_penalty in itertools.product(
        miss_penalties,
        useful_bonuses,
        wasted_penalties,
        miss_reduction_bonuses,
        miss_increase_penalties,
    ):
        total_mean_stall = 0.0
        total_misses = 0.0
        total_evictions = 0.0

        for trace in traces:
            simulator = build_simulator(trace, args)
            controller = ContextualBanditController(
                alpha=0.35,
                actions=build_bandit_action_menu(args.bandit_menu),
                miss_penalty=miss_penalty,
                useful_prefetch_bonus=useful_bonus,
                wasted_prefetch_penalty=wasted_penalty,
                miss_reduction_bonus=miss_reduction_bonus,
                miss_increase_penalty=miss_increase_penalty,
            )
            result = benchmark_policies(
                policy_names=["bandit"],
                trace=trace,
                simulator_builder=lambda simulator=simulator: simulator,
                controller_builder=lambda _policy: controller,
            )[0].summary
            total_mean_stall += float(result["mean_stall_ms"])
            total_misses += float(result["total_demand_misses"])
            total_evictions += float(result["total_evictions"])

        row = {
            "miss_penalty": miss_penalty,
            "useful_prefetch_bonus": useful_bonus,
            "wasted_prefetch_penalty": wasted_penalty,
            "miss_reduction_bonus": miss_reduction_bonus,
            "miss_increase_penalty": miss_increase_penalty,
            "avg_mean_stall_ms": total_mean_stall / len(traces),
            "avg_total_demand_misses": total_misses / len(traces),
            "avg_total_evictions": total_evictions / len(traces),
        }
        ranked_rows.append(row)

    ranked_rows.sort(key=ranking_key)

    print("rank | miss_pen | useful | wasted | miss_red | miss_inc | avg_stall | avg_miss | avg_evict")
    for idx, row in enumerate(ranked_rows[: args.top_k], start=1):
        print(
            f"{idx:>4} | "
            f"{row['miss_penalty']:>8.3f} | "
            f"{row['useful_prefetch_bonus']:>6.3f} | "
            f"{row['wasted_prefetch_penalty']:>6.3f} | "
            f"{row['miss_reduction_bonus']:>8.3f} | "
            f"{row['miss_increase_penalty']:>8.3f} | "
            f"{row['avg_mean_stall_ms']:>9.4f} | "
            f"{row['avg_total_demand_misses']:>8.2f} | "
            f"{row['avg_total_evictions']:>9.2f}"
        )

    if ranked_rows:
        print("\nBEST CONFIG")
        for key, value in ranked_rows[0].items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
