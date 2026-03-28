#!/usr/bin/env python3
"""Collect replayable decode-time head traces from a small real model.

This script is the first real-data bridge for the head-importance plan.

It intentionally chooses a simple collection path:
- eager-mode inference through `transformers`
- one request at a time
- greedy decode
- attention summaries from the newest token at each step

That makes it slower than a production engine, but much easier to debug.
The goal here is data correctness and schema validation, not serving speed.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import KVPageId, WorkloadStep, save_trace_json


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for the real-trace collector."""

    parser = argparse.ArgumentParser(description="Collect replayable head-summary traces from a small real model.")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="The quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--kv-block-size-tokens", type=int, default=16)
    parser.add_argument("--request-id", type=str, default="real_request_0")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _load_model_and_tokenizer(model_name: str, device: str):
    """Load a causal LM and tokenizer from `transformers`.

    We import lazily so the rest of the repo does not require `transformers`
    unless the user actually wants to collect real traces.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "This script requires `transformers` and `torch` to be installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    chosen_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    model.to(chosen_device)
    model.eval()
    return model, tokenizer, torch, chosen_device


def _block_table(sequence_length: int, block_size_tokens: int) -> tuple[int, ...]:
    """Return logical block ids covering the current sequence length."""

    num_blocks = (sequence_length + block_size_tokens - 1) // block_size_tokens
    return tuple(range(num_blocks))


def _predicted_block_table(sequence_length: int, block_size_tokens: int) -> tuple[int, ...]:
    """Return the likely next-step logical block ids.

    For dense decode, the next step usually needs the same old blocks and
    sometimes a new block if the sequence crosses a block boundary.
    """

    current_blocks = set(_block_table(sequence_length, block_size_tokens))
    next_blocks = set(_block_table(sequence_length + 1, block_size_tokens))
    return tuple(sorted(next_blocks))


def _head_concentration_weights(last_token_attention, torch) -> tuple[float, ...]:
    """Turn per-head attention distributions into normalized head weights.

    We use a simple concentration proxy:
    - sharper heads receive slightly higher weight
    - then we normalize within the layer
    """

    concentrations = torch.sum(last_token_attention * last_token_attention, dim=-1)
    total = torch.sum(concentrations)
    if float(total) <= 0.0:
        return tuple(float(1.0 / last_token_attention.shape[0]) for _ in range(last_token_attention.shape[0]))
    return tuple(float(value / total) for value in concentrations)


def _block_activity(last_token_attention, block_size_tokens: int, torch) -> dict[int, tuple[float, ...]]:
    """Aggregate last-token attention mass into block/page activity per head."""

    seq_len = int(last_token_attention.shape[-1])
    num_heads = int(last_token_attention.shape[0])
    block_ids = _block_table(seq_len, block_size_tokens)
    activity: dict[int, tuple[float, ...]] = {}
    for block_id in block_ids:
        start = block_id * block_size_tokens
        end = min(seq_len, start + block_size_tokens)
        block_mass = torch.sum(last_token_attention[:, start:end], dim=-1)
        activity[block_id] = tuple(float(value) for value in block_mass[:num_heads])
    return activity


def collect_trace(args) -> list[WorkloadStep]:
    """Run a small real model and convert decode-time attentions into replay steps."""

    model, tokenizer, torch, device = _load_model_and_tokenizer(args.model, args.device)
    tokenized = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)

    trace: list[WorkloadStep] = []
    with torch.no_grad():
        for decode_position in range(args.max_new_tokens):
            outputs = model(input_ids=input_ids, output_attentions=True, use_cache=False)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            attentions = outputs.attentions

            sequence_length = int(input_ids.shape[1])
            required_pages: list[KVPageId] = []
            predicted_pages: list[KVPageId] = []
            head_weighted_scores: dict[KVPageId, float] = {}
            query_head_weights: dict[int, tuple[float, ...]] = {}
            per_page_head_activity: dict[KVPageId, tuple[float, ...]] = {}
            per_page_features: dict[KVPageId, dict[str, float]] = {}
            layer_block_tables: dict[int, tuple[int, ...]] = {}

            current_blocks = _block_table(sequence_length, args.kv_block_size_tokens)
            predicted_blocks = _predicted_block_table(sequence_length, args.kv_block_size_tokens)

            for layer_id, layer_attention in enumerate(attentions):
                # Shape: [batch, heads, query_len, key_len]
                last_token_attention = layer_attention[0, :, -1, :]
                layer_weights = _head_concentration_weights(last_token_attention, torch)
                query_head_weights[layer_id] = layer_weights
                layer_block_tables[layer_id] = current_blocks

                block_activity = _block_activity(last_token_attention, args.kv_block_size_tokens, torch)
                for block_id in current_blocks:
                    page = KVPageId(layer_id=layer_id, page_id=block_id)
                    required_pages.append(page)
                    per_page_head_activity[page] = block_activity[block_id]
                    score = sum(weight * value for weight, value in zip(layer_weights, block_activity[block_id]))
                    head_weighted_scores[page] = score
                    per_page_features[page] = {
                        "head_weighted_score": score,
                        "layer_id": float(layer_id),
                        "page_id": float(block_id),
                        "is_required": 1.0,
                        "is_predicted": 1.0 if block_id in predicted_blocks else 0.0,
                        "sequence_length": float(sequence_length),
                    }
                for block_id in predicted_blocks:
                    predicted_pages.append(KVPageId(layer_id=layer_id, page_id=block_id))

            trace.append(
                WorkloadStep(
                    step_idx=decode_position,
                    required_pages=tuple(required_pages),
                    predicted_pages=tuple(predicted_pages),
                    head_weighted_scores=head_weighted_scores,
                    query_head_weights=query_head_weights,
                    per_page_head_activity=per_page_head_activity,
                    per_page_features=per_page_features,
                    referenced_layers=tuple(range(len(attentions))),
                    request_id=args.request_id,
                    decode_position=decode_position,
                    sequence_length=sequence_length,
                    kv_block_size_tokens=args.kv_block_size_tokens,
                    layer_block_tables=layer_block_tables,
                )
            )

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return trace


def main() -> None:
    args = build_parser().parse_args()
    trace = collect_trace(args)
    save_trace_json(args.output_json, trace)
    print(f"Wrote {len(trace)} replay steps to {args.output_json}")


if __name__ == "__main__":
    main()
