#!/usr/bin/env python3
"""Collect real head-summary traces from public prompt datasets.

Why this script exists:
- hand-written prompts got the pipeline working
- now we want broader validation on public prompt sources
- this gives us a reproducible path to collect traces from datasets rather than
  inventing prompts by hand

Current dataset adapters:
- `HuggingFaceH4/mt_bench_prompts`
- `OpenAssistant/oasst1`
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collect_real_head_traces import collect_trace
from evaluate_replay_scores import infer_trace_shape

from kv_controller import save_trace_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect real head traces from public prompt datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mt_bench", "oasst1"],
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--kv-block-size-tokens", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/dataset_head_traces")
    return parser


def _load_dataset_rows(dataset_name: str, split: str):
    from datasets import load_dataset

    if dataset_name == "mt_bench":
        return load_dataset("HuggingFaceH4/mt_bench_prompts", split=split)
    if dataset_name == "oasst1":
        return load_dataset("OpenAssistant/oasst1", split=split)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _extract_prompts(dataset_name: str, rows, count: int, offset: int) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []

    if dataset_name == "mt_bench":
        for row in rows:
            if len(prompts) >= count:
                break
            prompt = row.get("prompt") or row.get("question") or row.get("turns", [""])[0]
            category = row.get("category", "mt_bench")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            prompt = str(prompt).strip()
            if prompt:
                prompts.append((f"mt_bench_{category}_{len(prompts)+offset}", prompt))
        return prompts[offset:offset + count] if offset else prompts

    if dataset_name == "oasst1":
        # OASST1 is a conversation tree. We want root English user prompts.
        for row in rows:
            if len(prompts) >= count + offset:
                break
            role = str(row.get("role", "")).lower()
            lang = str(row.get("lang", "")).lower()
            parent_id = row.get("parent_id")
            text = str(row.get("text", "")).strip()
            if role == "prompter" and lang == "en" and parent_id in (None, "", "None") and text:
                prompts.append((f"oasst1_{len(prompts)}", text))
        return prompts[offset:offset + count]

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = _load_dataset_rows(args.dataset, args.split)
    prompts = _extract_prompts(args.dataset, rows, args.count, args.offset)

    if not prompts:
        raise RuntimeError("No prompts were extracted from the chosen dataset.")

    for request_id, prompt in prompts:
        collector_args = SimpleNamespace(
            model=args.model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            kv_block_size_tokens=args.kv_block_size_tokens,
            request_id=request_id,
            output_json="",
            device=args.device,
        )
        trace = collect_trace(collector_args)
        output_path = os.path.join(args.output_dir, f"{request_id}.json")
        save_trace_json(output_path, trace)
        steps, layers, min_required = infer_trace_shape(trace)
        print(
            f"{request_id}: path={output_path} steps={steps} layers={layers} "
            f"min_required_capacity={min_required}"
        )


if __name__ == "__main__":
    main()
