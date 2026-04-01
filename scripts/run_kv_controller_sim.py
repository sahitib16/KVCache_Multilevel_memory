#!/usr/bin/env python3
"""CLI driver for the `kv_controller` simulator.

This script is the easiest way to answer practical questions like:
- "Does the simulator run end-to-end?"
- "How does LRU compare to score-based control on the same trace?"
- "What happens if I try the new contextual bandit?"
- "Can I dump step-level and summary metrics to CSV?"

This is not yet the final full benchmark harness for the project, but it is
now much more than a smoke test:
- it can run one policy or a suite of policies
- it can print step-by-step metrics
- it can write reproducible CSV outputs
- it can exercise both static and adaptive controllers
"""

from __future__ import annotations

import argparse
import os
import sys

# When this script is executed directly, Python puts `scripts/` on sys.path.
# The local package lives one directory above that, so we add the repo root
# explicitly to make direct execution reliable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import (
    aggregate_policy_summaries,
    BeladyOracleController,
    CacheConfig,
    ContextualBanditController,
    benchmark_policies,
    benchmark_policies_across_seeds,
    load_trace_json,
    LRUController,
    OverlapAwareSimulator,
    PerfectPrefetchOracleController,
    PolicyOutput,
    ResidencyController,
    ScoreBasedController,
    SimulationConfig,
    SyntheticTraceConfig,
    SyntheticTraceGenerator,
    TransferConfig,
    save_trace_json,
    summarize_page_tile_stats,
    write_aggregate_summary_csv,
    write_rows_csv,
    write_step_csv,
    write_summary_csv,
)


class GreedyPrefetchController(ResidencyController):
    """Very small baseline controller for simple simulator validation.

    Behavior:
    - no explicit evictions
    - prefetch the first K predicted pages

    This policy is intentionally unsophisticated. It exists mostly so the
    driver always has a very simple baseline available.
    """

    def __init__(self, prefetch_k: int):
        self.prefetch_k = prefetch_k

    def decide(self, context):
        return PolicyOutput(prefetch_pages=context.predicted_pages[: self.prefetch_k])


def build_parser() -> argparse.ArgumentParser:
    """Define all CLI arguments in one place."""

    parser = argparse.ArgumentParser(description="Run the kv_controller simulator on a synthetic trace.")

    # Workload-shape arguments.
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pages-per-layer", type=int, default=24)
    parser.add_argument("--local-window-pages", type=int, default=4)
    parser.add_argument("--sparse-pages-per-step", type=int, default=2)
    parser.add_argument("--predicted-prefetch-pages", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--query-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-list",
        type=str,
        default="",
        help="Comma-separated seed list for multi-seed aggregate benchmarking.",
    )
    parser.add_argument(
        "--trace-json",
        type=str,
        default="",
        help="Replay an offline trace from JSON instead of generating a synthetic one.",
    )
    parser.add_argument(
        "--save-trace-json",
        type=str,
        default="",
        help="Save the generated synthetic trace to JSON for later replay.",
    )

    # Memory/transfer model arguments.
    parser.add_argument("--hbm-capacity-pages", type=int, default=20)
    parser.add_argument("--page-bytes", type=int, default=4096)
    parser.add_argument("--bandwidth-bytes-per-ms", type=float, default=16384.0)
    parser.add_argument("--transfer-setup-ms", type=float, default=0.02)
    parser.add_argument("--max-inflight-transfers", type=int, default=2)
    parser.add_argument("--decode-kernel-ms", type=float, default=0.05)

    # Policy arguments.
    parser.add_argument("--prefetch-k", type=int, default=2)
    parser.add_argument(
        "--policy",
        choices=["greedy_prefetch", "lru", "score", "belady", "perfect_prefetch", "bandit"],
        default="greedy_prefetch",
        help="Run a single policy.",
    )
    parser.add_argument(
        "--policy-suite",
        type=str,
        default="",
        help="Comma-separated list of policies to compare on the same trace.",
    )
    # Match the tuned controller default unless the user explicitly overrides it.
    parser.add_argument("--bandit-alpha", type=float, default=0.35)

    # Output arguments.
    parser.add_argument("--print-steps", action="store_true")
    parser.add_argument("--step-csv", type=str, default="")
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--aggregate-summary-csv", type=str, default="")
    parser.add_argument("--page-stats-csv", type=str, default="")
    parser.add_argument("--layer-stats-csv", type=str, default="")
    parser.add_argument("--tile-stats-csv", type=str, default="")
    parser.add_argument("--tile-size-pages", type=int, default=4)
    return parser


def build_trace(args, seed_override: int | None = None) -> list:
    """Generate one synthetic trace.

    When `seed_override` is provided, it replaces the CLI seed. This keeps the
    same trace-shape arguments while allowing fair multi-seed evaluation.
    """

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
            seed=args.seed if seed_override is None else seed_override,
        )
    ).generate()
    if args.save_trace_json and seed_override is None:
        save_trace_json(args.save_trace_json, trace)
    return trace


def build_simulator(args) -> OverlapAwareSimulator:
    """Create a fresh simulator instance.

    We build a fresh simulator per policy so policies do not share any state.
    The trace stays fixed, but residency/transfer history should reset.
    """

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


def build_controller(policy_name: str, args, trace) -> ResidencyController:
    """Map a policy name to a concrete controller instance."""

    if policy_name == "greedy_prefetch":
        return GreedyPrefetchController(prefetch_k=args.prefetch_k)
    if policy_name == "lru":
        return LRUController(prefetch_k=args.prefetch_k)
    if policy_name == "score":
        return ScoreBasedController(prefetch_k=args.prefetch_k)
    if policy_name == "belady":
        return BeladyOracleController(trace)
    if policy_name == "perfect_prefetch":
        return PerfectPrefetchOracleController(trace, prefetch_k=args.prefetch_k)
    if policy_name == "bandit":
        return ContextualBanditController(alpha=args.bandit_alpha)
    raise ValueError(f"Unknown policy: {policy_name}")


def print_step_rows(metrics) -> None:
    """Pretty-print per-step metrics for one run."""

    print("step | req | pred | demand_miss | prefetch | stall_ms | compute_ms | evictions | backlog")
    for row in metrics:
        print(
            f"{row.step_idx:>4} | "
            f"{row.required_pages:>3} | "
            f"{row.predicted_pages:>4} | "
            f"{row.demand_misses:>11} | "
            f"{row.prefetch_submitted:>8} | "
            f"{row.stall_ms:>8.4f} | "
            f"{row.compute_ms:>10.4f} | "
            f"{row.evictions:>9} | "
            f"{row.transfer_backlog:>7}"
        )


def print_summary_table(summaries: list[dict[str, object]]) -> None:
    """Print a compact comparison table across policies."""

    print("\nSUMMARY")
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


def print_aggregate_summary_table(summaries: list[dict[str, object]]) -> None:
    """Print a compact cross-seed comparison table.

    This table focuses on the metrics that matter most for Step 4 tuning:
    - average misses
    - average mean stall
    - average tail stall
    - average prefetch volume
    - average evictions
    """

    print("\nAGGREGATE SUMMARY")
    print("policy | seeds | avg_miss | avg_mean_stall | avg_p95 | avg_p99 | avg_prefetch | avg_evictions")
    for row in summaries:
        print(
            f"{str(row['policy']):>16} | "
            f"{int(row['seeds']):>5} | "
            f"{float(row['avg_total_demand_misses']):>8.2f} | "
            f"{float(row['avg_mean_stall_ms']):>14.4f} | "
            f"{float(row['avg_p95_stall_ms']):>7.4f} | "
            f"{float(row['avg_p99_stall_ms']):>7.4f} | "
            f"{float(row['avg_total_prefetches_submitted']):>12.2f} | "
            f"{float(row['avg_total_evictions']):>13.2f}"
        )


def main() -> None:
    args = build_parser().parse_args()

    # Support either one policy or a comma-separated suite.
    policies = (
        [p.strip() for p in args.policy_suite.split(",") if p.strip()]
        if args.policy_suite
        else [args.policy]
    )

    default_suite_seed_list = "0,1,2,3,4"
    effective_seed_list = args.seed_list
    if args.policy_suite and not effective_seed_list and not args.trace_json:
        # Policy-suite comparisons are now aggregate-by-default because adaptive
        # controller quality is too noisy to judge from one seed alone.
        effective_seed_list = default_suite_seed_list

    if effective_seed_list:
        seeds = [int(seed.strip()) for seed in effective_seed_list.split(",") if seed.strip()]
        aggregate_results = benchmark_policies_across_seeds(
            policy_names=policies,
            seeds=seeds,
            trace_builder=lambda seed: build_trace(args, seed_override=seed),
            simulator_builder=lambda: build_simulator(args),
            controller_builder=lambda policy_name, trace: build_controller(policy_name, args, trace),
        )
        aggregate_summaries = [result.aggregate_summary for result in aggregate_results]
        print_aggregate_summary_table(aggregate_summaries)
        write_aggregate_summary_csv(args.aggregate_summary_csv or args.summary_csv, aggregate_results)
        return

    trace = build_trace(args)
    results = benchmark_policies(
        policy_names=policies,
        trace=trace,
        simulator_builder=lambda: build_simulator(args),
        controller_builder=lambda policy_name: build_controller(policy_name, args, trace),
    )
    summaries = [result.summary for result in results]

    if args.print_steps and len(results) == 1:
        print_step_rows(results[0].metrics)

    print_summary_table(summaries)

    # In single-policy mode, also print the more verbose key/value summary for
    # convenience.
    if len(summaries) == 1:
        row = summaries[0]
        print(f"\nsteps: {row['steps']}")
        print(f"total demand misses: {row['total_demand_misses']}")
        print(f"mean demand misses: {float(row['mean_demand_misses']):.4f}")
        print(f"total prefetches submitted: {row['total_prefetches_submitted']}")
        print(f"total stall ms: {float(row['total_stall_ms']):.4f}")
        print(f"mean stall ms: {float(row['mean_stall_ms']):.4f}")
        print(f"p95 stall ms: {float(row['p95_stall_ms']):.4f}")
        print(f"p99 stall ms: {float(row['p99_stall_ms']):.4f}")
        print(f"total compute ms: {float(row['total_compute_ms']):.4f}")
        print(f"total evictions: {row['total_evictions']}")
        print(f"final resident pages: {row['final_resident_pages']}")
        print(f"final slot map size: {row['final_slot_map_size']}")
        print(f"final inflight transfers: {row['final_inflight_transfers']}")
        print(f"final queued transfers: {row['final_queued_transfers']}")
        print(f"final backlog: {row['final_backlog']}")
        print(f"hbm format: {row['hbm_format']}")
        if "bandit_steps_observed" in row:
            print(f"bandit steps observed: {row['bandit_steps_observed']}")
            print(f"bandit total reward: {float(row['bandit_total_reward']):.4f}")
            print(f"bandit last action: {row['bandit_last_action']}")

    write_step_csv(args.step_csv, results)
    write_summary_csv(args.summary_csv, results)

    if len(results) == 1 and (args.page_stats_csv or args.layer_stats_csv or args.tile_stats_csv):
        page_rows, layer_rows, tile_rows = summarize_page_tile_stats(
            trace=trace,
            metrics=results[0].metrics,
            tile_size_pages=args.tile_size_pages,
        )
        write_rows_csv(args.page_stats_csv, page_rows)
        write_rows_csv(args.layer_stats_csv, layer_rows)
        write_rows_csv(args.tile_stats_csv, tile_rows)


if __name__ == "__main__":
    main()
