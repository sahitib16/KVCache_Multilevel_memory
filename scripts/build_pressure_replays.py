#!/usr/bin/env python3
"""Build pressure-inducing replay traces from real dense traces.

This script implements three concrete ways to make the controller face harder
choices while still using real per-head signals from the collected traces:

1. `recent_topk`
   Keep the most recent block(s) mandatory and add top-k older attended blocks.

2. `recent_threshold`
   Keep the most recent block(s) mandatory and add older blocks until a chosen
   fraction of attention-score mass is covered.

3. `round_robin_interleave`
   Interleave multiple request traces so they compete for the same HBM.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate_replay_scores import infer_trace_shape

from kv_controller import (
    convert_trace_recent_threshold,
    convert_trace_recent_topk,
    interleave_traces_round_robin,
    load_trace_json,
    save_trace_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pressure-inducing replay traces from real dense traces.")
    parser.add_argument(
        "--input-traces",
        type=str,
        required=True,
        help="Comma-separated replay JSON files collected from the real-trace pipeline.",
    )
    parser.add_argument("--output-dir", type=str, default="results/pressure_replays")
    parser.add_argument("--recent-block-window", type=int, default=1)
    parser.add_argument("--top-k-older-per-layer", type=int, default=1)
    parser.add_argument("--score-mass-fraction", type=float, default=0.5)
    parser.add_argument("--page-stride", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_paths = [path.strip() for path in args.input_traces.split(",") if path.strip()]
    traces = [load_trace_json(path) for path in input_paths]

    # Method 1: sparse old-context replay using top-k older pages.
    topk_trace = convert_trace_recent_topk(
        traces[0],
        recent_block_window=args.recent_block_window,
        top_k_older_per_layer=args.top_k_older_per_layer,
    )
    topk_path = os.path.join(args.output_dir, "recent_topk.json")
    save_trace_json(topk_path, topk_trace)

    # Method 2: sparse old-context replay using cumulative score mass.
    threshold_trace = convert_trace_recent_threshold(
        traces[0],
        recent_block_window=args.recent_block_window,
        score_mass_fraction=args.score_mass_fraction,
    )
    threshold_path = os.path.join(args.output_dir, "recent_threshold.json")
    save_trace_json(threshold_path, threshold_trace)

    # Method 3: interleave multiple requests so they compete for residency.
    interleaved_trace = interleave_traces_round_robin(traces, page_stride=args.page_stride)
    interleaved_path = os.path.join(args.output_dir, "round_robin_interleave.json")
    save_trace_json(interleaved_path, interleaved_trace)

    for label, trace, path in [
        ("recent_topk", topk_trace, topk_path),
        ("recent_threshold", threshold_trace, threshold_path),
        ("round_robin_interleave", interleaved_trace, interleaved_path),
    ]:
        steps, layers, min_required_capacity = infer_trace_shape(trace)
        print(
            f"{label}: path={path} steps={steps} layers={layers} "
            f"min_required_capacity={min_required_capacity}"
        )


if __name__ == "__main__":
    main()
