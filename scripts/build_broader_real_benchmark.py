#!/usr/bin/env python3
"""Collect a broader real-trace set and build a three-regime benchmark suite.

This script packages the next project phase into one reproducible workflow:

1. collect a wider set of real traces from a small model
2. save the raw replay JSON files
3. build `recent_threshold_round_robin_interleave` as the main benchmark
4. build `recent_topk_round_robin_interleave` as a middle-ground benchmark
5. also build the dense `round_robin_interleave` stress trace for comparison

The goal is to create a better main benchmark before retuning the bandit.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from batch_real_head_replay import PROMPT_PRESETS
from collect_real_head_traces import collect_trace
from evaluate_replay_scores import infer_trace_shape

from kv_controller import (
    interleave_sparse_recent_topk_traces,
    interleave_sparse_recent_threshold_traces,
    load_trace_json,
    interleave_traces_round_robin,
    save_trace_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a broader real-trace benchmark suite.")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument(
        "--prompt-presets",
        type=str,
        default="fox,hierarchy,memory,controller,scientific,dialogue,procedural,code",
        help="Comma-separated prompt preset names to collect.",
    )
    parser.add_argument(
        "--decode-steps",
        type=str,
        default="8,12,12,20,12,12,12,12",
        help="Comma-separated decode lengths paired with prompt presets.",
    )
    parser.add_argument(
        "--kv-block-sizes",
        type=str,
        default="16,16,8,16,16,16,16,8",
        help="Comma-separated KV block sizes paired with prompt presets.",
    )
    parser.add_argument("--recent-block-window", type=int, default=1)
    parser.add_argument("--top-k-older-per-layer", type=int, default=1)
    parser.add_argument("--score-mass-fraction", type=float, default=0.5)
    parser.add_argument("--page-stride", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/broader_real_benchmark")
    parser.add_argument(
        "--reuse-existing-raw",
        action="store_true",
        help="Reuse raw traces already present in <output-dir>/raw_traces instead of recollecting them.",
    )
    return parser


def _split_strings(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def _split_ints(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _expand_by_index(values: list[int], index: int) -> int:
    return values[index] if index < len(values) else values[-1]


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = os.path.join(args.output_dir, "raw_traces")
    benchmark_dir = os.path.join(args.output_dir, "benchmarks")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(benchmark_dir, exist_ok=True)

    prompt_names = _split_strings(args.prompt_presets)
    decode_steps = _split_ints(args.decode_steps)
    kv_block_sizes = _split_ints(args.kv_block_sizes)

    traces = []
    for index, prompt_name in enumerate(prompt_names):
        if prompt_name not in PROMPT_PRESETS:
            raise ValueError(f"Unknown prompt preset: {prompt_name}")

        raw_path = os.path.join(raw_dir, f"{prompt_name}.json")
        if args.reuse_existing_raw and os.path.exists(raw_path):
            trace = load_trace_json(raw_path)
        else:
            collector_args = SimpleNamespace(
                model=args.model,
                prompt=PROMPT_PRESETS[prompt_name],
                max_new_tokens=_expand_by_index(decode_steps, index),
                kv_block_size_tokens=_expand_by_index(kv_block_sizes, index),
                request_id=f"{prompt_name}_request",
                output_json="",
                device=args.device,
            )
            trace = collect_trace(collector_args)
            save_trace_json(raw_path, trace)

        traces.append(trace)
        steps, layers, min_required = infer_trace_shape(trace)
        print(
            f"raw trace {prompt_name}: path={raw_path} steps={steps} "
            f"layers={layers} min_required_capacity={min_required}"
        )

    threshold_interleaved = interleave_sparse_recent_threshold_traces(
        traces,
        recent_block_window=args.recent_block_window,
        score_mass_fraction=args.score_mass_fraction,
        page_stride=args.page_stride,
    )
    threshold_path = os.path.join(benchmark_dir, "recent_threshold_round_robin_interleave.json")
    save_trace_json(threshold_path, threshold_interleaved)

    middle_interleaved = interleave_sparse_recent_topk_traces(
        traces,
        recent_block_window=args.recent_block_window,
        top_k_older_per_layer=args.top_k_older_per_layer,
        page_stride=args.page_stride,
    )
    middle_path = os.path.join(benchmark_dir, "recent_topk_round_robin_interleave.json")
    save_trace_json(middle_path, middle_interleaved)

    dense_interleaved = interleave_traces_round_robin(traces, page_stride=args.page_stride)
    dense_path = os.path.join(benchmark_dir, "round_robin_interleave.json")
    save_trace_json(dense_path, dense_interleaved)

    for label, trace, path in [
        ("recent_threshold_round_robin_interleave", threshold_interleaved, threshold_path),
        ("recent_topk_round_robin_interleave", middle_interleaved, middle_path),
        ("round_robin_interleave", dense_interleaved, dense_path),
    ]:
        steps, layers, min_required = infer_trace_shape(trace)
        print(
            f"{label}: path={path} steps={steps} layers={layers} "
            f"min_required_capacity={min_required}"
        )


if __name__ == "__main__":
    main()
