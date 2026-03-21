#!/usr/bin/env python3
"""Small CLI driver for the new kv_controller simulator core.

This script is intentionally simple.
Its job is not to be the final benchmark harness, but to make it easy to:
- generate a synthetic trace
- run it through a controller
- print readable summaries
- confirm the Step 1 / Step 2 simulator core is behaving sensibly
"""

from __future__ import annotations

import argparse
import os
import sys

# When this script is executed directly from the repository root, Python puts
# `scripts/` on sys.path, not the repo root itself. Add the repo root so the
# local `kv_controller` package can be imported reliably.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import (
    BeladyOracleController,
    CacheConfig,
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
)


class GreedyPrefetchController(ResidencyController):
    """Very small baseline controller for smoke-testing the simulator.

    Behavior:
    - do not make explicit evictions
    - prefetch the first K predicted pages for the next step

    This is deliberately simple because this script is for validating the
    simulator core, not for claiming a strong policy.
    """

    def __init__(self, prefetch_k: int):
        self.prefetch_k = prefetch_k

    def decide(self, context):
        return PolicyOutput(prefetch_pages=context.predicted_pages[: self.prefetch_k])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the kv_controller simulator core on a synthetic trace.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pages-per-layer", type=int, default=24)
    parser.add_argument("--local-window-pages", type=int, default=4)
    parser.add_argument("--sparse-pages-per-step", type=int, default=2)
    parser.add_argument("--predicted-prefetch-pages", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--query-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hbm-capacity-pages", type=int, default=20)
    parser.add_argument("--page-bytes", type=int, default=4096)
    parser.add_argument("--bandwidth-bytes-per-ms", type=float, default=16384.0)
    parser.add_argument("--transfer-setup-ms", type=float, default=0.02)
    parser.add_argument("--max-inflight-transfers", type=int, default=2)
    parser.add_argument("--decode-kernel-ms", type=float, default=0.05)

    parser.add_argument("--prefetch-k", type=int, default=2)
    parser.add_argument(
        "--policy",
        choices=["greedy_prefetch", "lru", "score", "belady", "perfect_prefetch"],
        default="greedy_prefetch",
    )
    parser.add_argument("--print-steps", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

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

    simulator = OverlapAwareSimulator(
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

    if args.policy == "greedy_prefetch":
        controller: ResidencyController = GreedyPrefetchController(prefetch_k=args.prefetch_k)
    elif args.policy == "lru":
        controller = LRUController(prefetch_k=args.prefetch_k)
    elif args.policy == "score":
        controller = ScoreBasedController(prefetch_k=args.prefetch_k)
    elif args.policy == "belady":
        controller = BeladyOracleController(trace)
    else:
        controller = PerfectPrefetchOracleController(trace, prefetch_k=args.prefetch_k)

    metrics = simulator.run(trace, controller)

    if args.print_steps:
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

    total_demand_misses = sum(row.demand_misses for row in metrics)
    total_prefetch = sum(row.prefetch_submitted for row in metrics)
    total_stall_ms = sum(row.stall_ms for row in metrics)
    total_compute_ms = sum(row.compute_ms for row in metrics)
    total_evictions = sum(row.evictions for row in metrics)

    print("\nSUMMARY")
    print(f"steps: {len(metrics)}")
    print(f"total demand misses: {total_demand_misses}")
    print(f"total prefetches submitted: {total_prefetch}")
    print(f"total stall ms: {total_stall_ms:.4f}")
    print(f"total compute ms: {total_compute_ms:.4f}")
    print(f"total evictions: {total_evictions}")
    print(f"final resident pages: {len(simulator.state.resident_pages)}")
    print(f"final slot map size: {len(simulator.state.slot_to_page)}")
    print(f"final inflight transfers: {len(simulator.state.inflight_pages)}")
    print(f"final queued transfers: {len(simulator.state.transfer_state.queued_pages)}")
    print(f"final backlog: {simulator.state.transfer_state.backlog}")
    print(f"hbm format: {simulator.config.cache.hbm_kv_format}")


if __name__ == "__main__":
    main()
