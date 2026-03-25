#!/usr/bin/env python3
"""Evaluate score variants on a fixed replayed trace.

This script exists for the next phase of the project:
- keep the workload fixed via replay
- change only the score function
- compare how score-aware policies respond

That lets us ask questions like:
- does layer-normalized scoring help score-based eviction?
- does a predicted-page boost help or hurt?
- does the bandit benefit from a different score signal on the same trace?
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import (
    build_bandit_action_menu,
    CacheConfig,
    ContextualBanditController,
    HeadActivityRecomputedScorer,
    LayerNormalizedHeadActivityScorer,
    LRUController,
    NormalizedHeadWeightedScorer,
    OverlapAwareSimulator,
    PassthroughHeadWeightedScorer,
    PredictedBoostedHeadActivityScorer,
    ScoreBasedController,
    SimulationConfig,
    SyntheticTraceConfig,
    SyntheticTraceGenerator,
    TransferConfig,
    apply_scorer_to_trace,
    benchmark_policies_across_seeds,
    benchmark_policies,
    load_trace_json,
    save_trace_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare score variants on one replayed trace.")
    parser.add_argument("--trace-json", type=str, default="")
    parser.add_argument("--save-trace-json", type=str, default="")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-list",
        type=str,
        default="",
        help="Comma-separated seed list for multi-seed replay scoring evaluation.",
    )
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
    parser.add_argument("--prefetch-k", type=int, default=2)
    parser.add_argument(
        "--policies",
        type=str,
        default="lru,score,bandit",
        help="Comma-separated policy list to run on each rescored trace.",
    )
    parser.add_argument(
        "--bandit-menus",
        type=str,
        default="trimmed",
        help="Comma-separated bandit action-menu presets to compare when bandit is enabled.",
    )
    return parser


def build_or_load_trace(args):
    if args.trace_json:
        return load_trace_json(args.trace_json)

    trace = SyntheticTraceGenerator(
        SyntheticTraceConfig(
            steps=args.steps,
            layers=args.layers,
            pages_per_layer=args.pages_per_layer,
            local_window_pages=args.local_window_pages,
            sparse_pages_per_step=args.sparse_pages_per_step,
            predicted_prefetch_pages=args.predicted_prefetch_pages,
            attention_heads=args.attention_heads,
            query_jitter=args.query_jitter,
            seed=args.seed,
        )
    ).generate()
    if args.save_trace_json:
        save_trace_json(args.save_trace_json, trace)
    return trace


def build_simulator(args) -> OverlapAwareSimulator:
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


def build_controller(policy_name: str, prefetch_k: int, bandit_menu: str = "trimmed"):
    if policy_name == "lru":
        return LRUController(prefetch_k=prefetch_k)
    if policy_name == "score":
        return ScoreBasedController(prefetch_k=prefetch_k)
    if policy_name == "bandit":
        return ContextualBanditController(actions=build_bandit_action_menu(bandit_menu))
    raise ValueError(f"Unknown policy: {policy_name}")


def print_results_for_scorer(scorer_name: str, summaries: list[dict[str, object]]) -> None:
    print(f"\nSCORER: {scorer_name}")
    print("policy | total_miss | mean_stall | p95_stall | prefetch | evictions")
    for row in summaries:
        print(
            f"{str(row['policy']):>16} | "
            f"{int(row['total_demand_misses']):>10} | "
            f"{float(row['mean_stall_ms']):>10.4f} | "
            f"{float(row['p95_stall_ms']):>9.4f} | "
            f"{int(row['total_prefetches_submitted']):>8} | "
            f"{int(row['total_evictions']):>9}"
        )


def print_aggregate_results(label: str, summaries: list[dict[str, object]]) -> None:
    print(f"\n{label}")
    print("policy | avg_miss | avg_mean_stall | avg_p95 | avg_p99 | avg_prefetch | avg_evictions")
    for row in summaries:
        print(
            f"{str(row['policy']):>16} | "
            f"{float(row['avg_total_demand_misses']):>8.2f} | "
            f"{float(row['avg_mean_stall_ms']):>14.4f} | "
            f"{float(row['avg_p95_stall_ms']):>7.4f} | "
            f"{float(row['avg_p99_stall_ms']):>7.4f} | "
            f"{float(row['avg_total_prefetches_submitted']):>12.2f} | "
            f"{float(row['avg_total_evictions']):>13.2f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    base_trace = build_or_load_trace(args)
    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    bandit_menus = [name.strip() for name in args.bandit_menus.split(",") if name.strip()]

    scorers = {
        "passthrough": PassthroughHeadWeightedScorer(),
        "normalized": NormalizedHeadWeightedScorer(),
        "recomputed": HeadActivityRecomputedScorer(),
        "layer_normalized": LayerNormalizedHeadActivityScorer(),
        "predicted_boosted": PredictedBoostedHeadActivityScorer(),
    }

    if args.seed_list:
        seeds = [int(seed.strip()) for seed in args.seed_list.split(",") if seed.strip()]
        for scorer_name, scorer in scorers.items():
            for bandit_menu in bandit_menus:
                aggregate_results = benchmark_policies_across_seeds(
                    policy_names=policy_names,
                    seeds=seeds,
                    trace_builder=lambda seed: apply_scorer_to_trace(
                        build_or_load_trace(
                            argparse.Namespace(**{**vars(args), "seed": seed, "trace_json": args.trace_json})
                        ),
                        scorer,
                    ),
                    simulator_builder=lambda: build_simulator(args),
                    controller_builder=lambda policy_name, _trace: build_controller(
                        policy_name,
                        args.prefetch_k,
                        bandit_menu=bandit_menu,
                    ),
                )
                print_aggregate_results(
                    f"SCORER: {scorer_name} | BANDIT MENU: {bandit_menu}",
                    [result.aggregate_summary for result in aggregate_results],
                )
        return

    for scorer_name, scorer in scorers.items():
        for bandit_menu in bandit_menus:
            rescored_trace = apply_scorer_to_trace(base_trace, scorer)
            results = benchmark_policies(
                policy_names=policy_names,
                trace=rescored_trace,
                simulator_builder=lambda: build_simulator(args),
                controller_builder=lambda policy_name: build_controller(
                    policy_name,
                    args.prefetch_k,
                    bandit_menu=bandit_menu,
                ),
            )
            print_results_for_scorer(f"{scorer_name} | bandit_menu={bandit_menu}", [result.summary for result in results])


if __name__ == "__main__":
    main()
