# Real Trace Logging Plan

This document defines the first real-model logging path for head-importance
collection.

The goal is not to jump straight into full engine integration.
The goal is to collect replayable decode-time traces from a real model that
look close enough to a future vLLM-style integration to be useful now.

## Why This Exists

We already have:
- a simulator
- a replay format
- replay-time score experiments

What we do not yet have is:
- real decode-time head signals

This plan defines the exact fields to log so we can move from synthetic head
activity toward real head summaries.

## What We Are Logging

For each decode step we want:

- `request_id`
  - logical request / sequence id

- `step_idx`
  - index in the replay file

- `decode_position`
  - decode step within the request

- `sequence_length`
  - number of tokens currently visible to attention

- `kv_block_size_tokens`
  - block/page size in tokens

- `layer_block_tables`
  - per-layer logical block table
  - this is the main piece that makes the trace more vLLM-like

- `required_pages`
  - pages needed by the current decode step

- `predicted_pages`
  - pages that might be useful to prefetch for the next step

- `query_head_weights`
  - per-layer head-importance weights for this decode step

- `per_page_head_activity`
  - the most important primitive signal
  - for each page, how much each head attends to token positions inside that
    page/block

- `head_weighted_scores`
  - aggregate score derived from the two signals above

## How We Will Approximate These Fields From A Small Real Model

For a first real-model collector:

1. Run a small decoder-only transformer in eager mode.
2. At each decode step, request attention weights.
3. For the newest query token:
   - take the attention distribution from each head
   - bucket token positions into KV blocks of size `kv_block_size_tokens`
   - sum attention mass within each block
4. That block-summed attention becomes `per_page_head_activity`.
5. Compute a per-head concentration measure within each layer to derive
   `query_head_weights`.
6. Combine them into:

   `score(page) = sum_h query_head_weight[layer, h] * page_head_activity(page, h)`

## Why We Start With A Small Real Model

- easier to run
- easier to instrument
- easier to debug
- enough to validate the logging and replay pipeline

This is not the final model choice for the project.
It is the first data-collection step.

## What Counts As "More vLLM-Like"

We are not yet integrating into vLLM internals.
But we are intentionally aligning the trace schema with vLLM-like concepts:

- block/page sized KV chunks
- per-layer block tables
- request-level decode steps
- required vs predicted block/page sets

That makes later trace replay or shadow-mode integration easier.

## Important Non-Goals Right Now

- not modifying attention kernels
- not depending on a special vLLM dataset
- not training a separate importance model first

## First Collector Implementation

The first collector should:

- use `transformers`
- run greedy decode for a small number of new tokens
- collect attentions for each step
- emit a replay JSON file using the current replay schema

## Later Extensions

After the first collector works:

- compare collected real traces against synthetic traces
- add multiple prompts / requests
- add adapters for future engine-exported logs
- only later consider direct vLLM trace ingestion
