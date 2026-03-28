#!/usr/bin/env python3
"""Collect and replay a batch of real head-summary traces.

This script automates the workflow that was previously done by hand:

1. run a small real model on several prompt settings
2. collect replayable head-summary traces
3. infer each trace's minimum feasible HBM capacity
4. replay each trace under several capacity settings
5. compare scorer/policy combinations in one summary table

The goal is not to be the final experiment harness for the whole project.
The goal is to make the "real trace + replay" loop easy enough that we can
iterate quickly on head-weighted scoring without manually typing many commands.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# These helpers already exist in neighboring scripts. Importing them keeps the
# batch harness thin and ensures we use the same trace-generation and replay
# logic as the manual commands.
from collect_real_head_traces import collect_trace
from evaluate_replay_scores import infer_trace_shape

from kv_controller import (
    CacheConfig,
    ContextualBanditController,
    HeadActivityRecomputedScorer,
    LayerNormalizedHeadActivityScorer,
    NormalizedHeadWeightedScorer,
    OverlapAwareSimulator,
    PassthroughHeadWeightedScorer,
    PredictedBoostedHeadActivityScorer,
    ScoreBasedController,
    SimulationConfig,
    TransferConfig,
    apply_scorer_to_trace,
    benchmark_policies,
    build_bandit_action_menu,
    save_trace_json,
)


# A few prompt presets are enough to start. They are intentionally small and
# easy to understand. We can expand this later when we want broader coverage.
PROMPT_PRESETS = {
    "fox": "The quick brown fox jumps over the lazy dog.",
    "hierarchy": (
        "In a distant future, researchers built a memory hierarchy for transformer "
        "inference where key-value blocks could move between CPU and GPU, and every "
        "decode step depended on whether those blocks were already resident when "
        "attention began."
    ),
    "memory": (
        "Memory movement dominates decode latency when spilled KV blocks are not "
        "already present in GPU memory."
    ),
    "controller": (
        "The controller should learn which pages to keep hot, which to evict, and "
        "which to prefetch before the next decode step begins."
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch collection and replay for real head-summary traces.")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument(
        "--prompt-presets",
        type=str,
        default="fox,hierarchy,memory,controller",
        help="Comma-separated prompt preset names to collect.",
    )
    parser.add_argument(
        "--decode-steps",
        type=str,
        default="8,12,20",
        help="Comma-separated decode lengths. These are paired with prompts in order; if only one is provided it is reused.",
    )
    parser.add_argument(
        "--kv-block-sizes",
        type=str,
        default="16,16,8,16",
        help="Comma-separated KV block sizes. These are paired with prompts in order; if only one is provided it is reused.",
    )
    parser.add_argument(
        "--capacities",
        type=str,
        default="min,min+2,min+4",
        help="Comma-separated replay capacities. Supported forms: min, min+N, min*N.F",
    )
    parser.add_argument(
        "--scorers",
        type=str,
        default="recomputed,layer_normalized",
        help="Comma-separated scorer variants to evaluate.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="score,bandit",
        help="Comma-separated policies to evaluate.",
    )
    parser.add_argument("--bandit-menu", type=str, default="full")
    parser.add_argument("--prefetch-k", type=int, default=2)
    parser.add_argument("--page-bytes", type=int, default=4096)
    parser.add_argument("--bandwidth-bytes-per-ms", type=float, default=16384.0)
    parser.add_argument("--transfer-setup-ms", type=float, default=0.02)
    parser.add_argument("--max-inflight-transfers", type=int, default=2)
    parser.add_argument("--decode-kernel-ms", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/real_head_batch")
    parser.add_argument("--summary-csv", type=str, default="")
    return parser


def _split_ints(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _split_strings(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def _expand_by_index(values: list[int], index: int) -> int:
    """Use the matching list item when available, otherwise reuse the last one."""

    if not values:
        raise ValueError("Expected at least one value.")
    return values[index] if index < len(values) else values[-1]


def _capacity_choices(spec: list[str], min_required: int) -> list[int]:
    """Translate symbolic capacity settings into concrete HBM page counts.

    Examples:
    - `min` -> exactly the minimum feasible capacity
    - `min+2` -> minimum + 2 pages
    - `min*1.25` -> 25% more than the minimum, rounded up
    """

    import math

    capacities: list[int] = []
    for token in spec:
        if token == "min":
            capacities.append(min_required)
        elif token.startswith("min+"):
            capacities.append(min_required + int(token[4:]))
        elif token.startswith("min*"):
            capacities.append(int(math.ceil(min_required * float(token[4:]))))
        else:
            capacities.append(int(token))
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(capacities))


def _build_scorer(name: str):
    scorers = {
        "passthrough": PassthroughHeadWeightedScorer(),
        "normalized": NormalizedHeadWeightedScorer(),
        "recomputed": HeadActivityRecomputedScorer(),
        "layer_normalized": LayerNormalizedHeadActivityScorer(),
        "predicted_boosted": PredictedBoostedHeadActivityScorer(),
    }
    if name not in scorers:
        raise ValueError(f"Unknown scorer: {name}")
    return scorers[name]


def _build_controller(name: str, prefetch_k: int, bandit_menu: str):
    if name == "score":
        return ScoreBasedController(prefetch_k=prefetch_k)
    if name == "bandit":
        return ContextualBanditController(actions=build_bandit_action_menu(bandit_menu))
    raise ValueError(f"Unknown policy: {name}")


def _build_simulator(
    steps: int,
    layers: int,
    capacity: int,
    args: argparse.Namespace,
) -> OverlapAwareSimulator:
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


def collect_one_trace(
    *,
    model: str,
    prompt: str,
    decode_steps: int,
    kv_block_size_tokens: int,
    request_id: str,
    device: str,
):
    """Collect one real trace using the same collector as the manual script."""

    collector_args = SimpleNamespace(
        model=model,
        prompt=prompt,
        max_new_tokens=decode_steps,
        kv_block_size_tokens=kv_block_size_tokens,
        request_id=request_id,
        output_json="",
        device=device,
    )
    return collect_trace(collector_args)


def write_summary_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompt_names = _split_strings(args.prompt_presets)
    decode_steps = _split_ints(args.decode_steps)
    kv_block_sizes = _split_ints(args.kv_block_sizes)
    capacity_spec = _split_strings(args.capacities)
    scorer_names = _split_strings(args.scorers)
    policy_names = _split_strings(args.policies)

    summary_rows: list[dict[str, object]] = []

    for index, prompt_name in enumerate(prompt_names):
        if prompt_name not in PROMPT_PRESETS:
            raise ValueError(f"Unknown prompt preset: {prompt_name}")

        prompt = PROMPT_PRESETS[prompt_name]
        trace = collect_one_trace(
            model=args.model,
            prompt=prompt,
            decode_steps=_expand_by_index(decode_steps, index),
            kv_block_size_tokens=_expand_by_index(kv_block_sizes, index),
            request_id=f"{prompt_name}_request",
            device=args.device,
        )

        trace_path = os.path.join(args.output_dir, f"{prompt_name}.json")
        save_trace_json(trace_path, trace)

        steps, layers, min_required_capacity = infer_trace_shape(trace)
        capacities = _capacity_choices(capacity_spec, min_required_capacity)

        print(
            f"\nTRACE {prompt_name}: steps={steps} layers={layers} "
            f"min_required_capacity={min_required_capacity} capacities={capacities}"
        )

        for scorer_name in scorer_names:
            rescored_trace = apply_scorer_to_trace(trace, _build_scorer(scorer_name))
            for capacity in capacities:
                results = benchmark_policies(
                    policy_names=policy_names,
                    trace=rescored_trace,
                    simulator_builder=lambda capacity=capacity: _build_simulator(
                        steps, layers, capacity, args
                    ),
                    controller_builder=lambda policy_name: _build_controller(
                        policy_name, args.prefetch_k, args.bandit_menu
                    ),
                )
                for result in results:
                    row = {
                        "trace_name": prompt_name,
                        "model": args.model,
                        "steps": steps,
                        "layers": layers,
                        "kv_block_size_tokens": rescored_trace[0].kv_block_size_tokens if rescored_trace else 0,
                        "min_required_capacity": min_required_capacity,
                        "capacity": capacity,
                        "scorer": scorer_name,
                        "policy": result.summary["policy"],
                        "total_demand_misses": result.summary["total_demand_misses"],
                        "mean_stall_ms": result.summary["mean_stall_ms"],
                        "p95_stall_ms": result.summary["p95_stall_ms"],
                        "p99_stall_ms": result.summary["p99_stall_ms"],
                        "total_prefetches_submitted": result.summary["total_prefetches_submitted"],
                        "total_evictions": result.summary["total_evictions"],
                    }
                    summary_rows.append(row)
                    print(
                        f"{prompt_name:>12} | scorer={scorer_name:<16} | capacity={capacity:>3} | "
                        f"policy={row['policy']:<6} | miss={int(row['total_demand_misses']):>4} | "
                        f"mean_stall={float(row['mean_stall_ms']):.4f} | "
                        f"evictions={int(row['total_evictions']):>3}"
                    )

    summary_csv = args.summary_csv or os.path.join(args.output_dir, "summary.csv")
    write_summary_csv(summary_csv, summary_rows)
    print(f"\nWrote batch summary to {summary_csv}")


if __name__ == "__main__":
    main()
