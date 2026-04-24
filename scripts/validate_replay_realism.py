#!/usr/bin/env python3
"""Validate replay workloads using page-wise and tile-wise statistics.

This script exists for research defensibility rather than controller tuning.

It answers questions like:
- is this replay workload creating meaningful page pressure?
- do we see hot/cold concentration at the page or tile level?
- are prefetches useful or mostly wasted?
- is the scorer comparison happening on a workload with real structure?
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import (
    attach_reuse_distance_features,
    CacheConfig,
    HeadActivityRecomputedScorer,
    LayerNormalizedHeadActivityScorer,
    LRUController,
    NormalizedHeadWeightedScorer,
    OverlapAwareSimulator,
    PageStatsHybridScorer,
    PassthroughHeadWeightedScorer,
    PredictedBoostedHeadActivityScorer,
    RegimeAwarePageStatsScorer,
    ReuseDistanceHybridScorer,
    ScoreBasedController,
    SimulationConfig,
    TransferConfig,
    apply_scorer_to_trace,
    load_trace_json,
    summarize_page_tile_stats,
    summarize_realism_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate replay workload realism with page/tile stats.")
    parser.add_argument("--trace-json", type=str, required=True)
    parser.add_argument("--policy", choices=["lru", "score"], default="score")
    parser.add_argument(
        "--scorers",
        type=str,
        default="normalized,layer_normalized",
        help="Comma-separated scorer list.",
    )
    parser.add_argument("--prefetch-k", type=int, default=2)
    parser.add_argument("--tile-size-pages", type=int, default=4)
    parser.add_argument("--hbm-capacity-pages", type=int, default=0)
    parser.add_argument("--page-bytes", type=int, default=4096)
    parser.add_argument("--bandwidth-bytes-per-ms", type=float, default=16384.0)
    parser.add_argument("--transfer-setup-ms", type=float, default=0.02)
    parser.add_argument("--max-inflight-transfers", type=int, default=2)
    parser.add_argument("--decode-kernel-ms", type=float, default=0.05)
    return parser


def infer_trace_shape(trace) -> tuple[int, int, int]:
    steps = len(trace)
    layers = max((page.layer_id for step in trace for page in step.required_pages), default=-1) + 1
    min_required_capacity = max((len(step.required_pages) for step in trace), default=0)
    return steps, layers, min_required_capacity


def build_simulator(args, trace):
    steps, layers, min_required_capacity = infer_trace_shape(trace)
    hbm_capacity_pages = args.hbm_capacity_pages or min_required_capacity
    if hbm_capacity_pages < min_required_capacity:
        raise ValueError(
            f"HBM capacity {hbm_capacity_pages} is too small for this trace. "
            f"Minimum required pages: {min_required_capacity}."
        )
    return OverlapAwareSimulator(
        SimulationConfig(
            steps=steps,
            cache=CacheConfig(
                hbm_capacity_pages=hbm_capacity_pages,
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


def build_controller(policy_name: str, prefetch_k: int):
    if policy_name == "lru":
        return LRUController(prefetch_k=prefetch_k)
    if policy_name == "score":
        return ScoreBasedController(prefetch_k=prefetch_k)
    raise ValueError(f"Unknown policy: {policy_name}")


def main() -> None:
    args = build_parser().parse_args()
    trace = load_trace_json(args.trace_json)
    trace = attach_reuse_distance_features(trace)

    scorers = {
        "passthrough": PassthroughHeadWeightedScorer(),
        "normalized": NormalizedHeadWeightedScorer(),
        "recomputed": HeadActivityRecomputedScorer(),
        "layer_normalized": LayerNormalizedHeadActivityScorer(),
        "predicted_boosted": PredictedBoostedHeadActivityScorer(),
        "reuse_hybrid": ReuseDistanceHybridScorer(),
        "page_stats_hybrid": PageStatsHybridScorer(),
        "regime_aware": RegimeAwarePageStatsScorer(),
    }

    scorer_names = [name.strip() for name in args.scorers.split(",") if name.strip()]
    for scorer_name in scorer_names:
        scorer = scorers[scorer_name]
        rescored_trace = apply_scorer_to_trace(trace, scorer)
        sim = build_simulator(args, rescored_trace)
        controller = build_controller(args.policy, args.prefetch_k)
        metrics = sim.run(rescored_trace, controller)
        page_rows, layer_rows, tile_rows = summarize_page_tile_stats(
            rescored_trace,
            metrics,
            tile_size_pages=args.tile_size_pages,
        )
        realism = summarize_realism_metrics(page_rows, layer_rows, tile_rows)
        print(f"\nSCORER: {scorer_name} | POLICY: {args.policy}")
        for key, value in realism.items():
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
