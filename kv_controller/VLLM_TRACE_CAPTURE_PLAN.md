# vLLM Trace Capture Plan

This document defines the next bridge from our page-level simulator to real
vLLM-shaped traces.

The goal is **not** live control yet. The goal is shadow/logging mode:
capture real block-table and KV-cache-manager state, convert that into our
replay format, and then evaluate policies offline.

## Current vLLM Concepts To Match

Recent vLLM documentation describes KV cache management around blocks rather
than arbitrary Python page objects.

Important concepts:

- `KVCacheBlock`
  - immutable `block_id`
  - optional `block_hash`
  - `ref_cnt`
  - free-queue links
- `KVCacheManager`
  - owns request-to-block mappings
  - exposes block-oriented helpers such as `get_blocks(...)` and
    `get_block_ids(...)`
  - supports freeing and evicting blocks
- `BlockTable` / `MultiGroupBlockTable`
  - maps request rows to block IDs used by attention kernels
  - can represent multiple KV cache groups
  - may map allocation block sizes to different kernel block sizes

Implication for our simulator:
- our `KVPageId(layer_id, page_id)` should map to vLLM-style block IDs
- replay traces should preserve request id, decode position, sequence length,
  block size, and per-layer/group block tables
- we should avoid pretending that our page id is exactly vLLM's physical block
  id until trace capture confirms the mapping

## Minimal Fields To Capture

For each decode/scheduling step:

- `request_id`
- `decode_step_idx`
- `sequence_length`
- `num_computed_tokens` if available
- `kv_block_size_tokens`
- `kernel_block_size_tokens` if different
- `kv_cache_group_id`
- per-request block IDs:
  - logical block index
  - physical block ID
  - block table row position
- optional block metadata:
  - `block_hash`
  - `ref_cnt`
  - whether the block is full / computed
- scheduling metadata:
  - running request IDs
  - newly allocated block IDs
  - freed block IDs
  - evicted block IDs if available
- prefix-cache metadata if available:
  - number of computed/prefix blocks
  - common-prefix block counts

## What We Should Not Capture Yet

- CUDA timing as a primary signal
- kernel-level memory transactions
- live controller decisions
- anything requiring modification of the attention kernel

The first useful artifact is a JSONL stream of block-table and KV-cache state.

## Proposed Capture Stages

### Stage 1: Local API Probe

Run a script against the installed vLLM package to identify:

- vLLM version
- whether v1 modules are importable
- whether `KVCacheManager` exposes:
  - `get_block_ids`
  - `get_blocks`
  - `get_computed_blocks`
  - `free`
  - `evict_blocks`
- whether `BlockTable` / `MultiGroupBlockTable` expose:
  - `block_size`
  - `blocks_per_kv_block`
  - `block_table`
  - `slot_mapping`

This is implemented by:

```bash
python scripts/probe_vllm_trace_points.py
```

### Stage 2: Shadow Logging Patch

Create a tiny wrapper around scheduler / KV cache manager methods that logs
block mappings without changing behavior.

Target hook points:

- after block allocation
- after scheduling a decode batch
- after block table commit
- after freeing / evicting blocks

Output:

```json
{
  "event": "decode_step",
  "request_id": "...",
  "decode_position": 17,
  "sequence_length": 128,
  "kv_cache_group_id": 0,
  "block_ids": [1, 2, 9, 10],
  "block_size": 16,
  "kernel_block_size": 16
}
```

### Stage 3: Convert vLLM Logs To Replay

Convert captured block logs into `WorkloadStep` records:

- `request_id` comes from vLLM request id
- `decode_position` comes from generated-token position
- `sequence_length` comes from request state
- `required_pages` comes from block IDs needed by attention
- `layer_block_tables` stores the block IDs by layer/group
- scores can initially be placeholder normalized scores until head signals are
  joined later

### Stage 4: Offline Evaluation

Replay those vLLM-shaped traces through:

- `validate_replay_realism.py`
- `evaluate_replay_scores.py`
- page/tile CSV export

Only after this should we consider shadow controller decisions.

## Open Questions

- Which vLLM version will we target first?
- Is vLLM v1 enabled in the target environment?
- Are we tracing decode-only, prefill+decode, or both?
- Do we need per-layer block identity, or is per-KV-cache-group block identity
  sufficient for the first replay?
- Can offload/swap events be captured cleanly from public APIs, or do we need a
  small internal patch?

## Current Decision

Start with Stage 1 and Stage 2 only:

- probe the installed vLLM API surface
- add shadow logging of block IDs and request/decode metadata
- do not integrate controller decisions into vLLM yet
