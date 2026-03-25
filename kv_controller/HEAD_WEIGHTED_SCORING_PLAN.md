# Head-Weighted Scoring Plan

This document describes the step-by-step approach for moving the project from
synthetic head-weighted scores toward replayable, more realistic score signals.

The key idea is:

1. define the data we need
2. carry that data through the simulator and replay path
3. experiment with score functions offline
4. only later move toward real engine instrumentation

## Goal

We want a head-weighted score that is useful for control decisions:
- eviction
- prefetch
- layer budgeting
- eventually adaptive action selection

The score is valuable only if it improves control outcomes such as:
- lower stall
- lower misses
- lower churn
- better tail latency

## Step 1: Define And Preserve The Raw Head Signal

What we need:
- page identity
- layer identity
- per-layer query head weights
- per-page per-head activity

Implementation target:
- every `WorkloadStep` should be able to carry:
  - aggregated `head_weighted_scores`
  - raw `per_page_head_activity`
  - `query_head_weights`

Why this is first:
- if we only store the final score, we cannot later try alternate score
  functions without recollecting the trace
- storing the raw head activity makes offline experimentation much easier

## Step 2: Use Replay As The Main Experiment Interface

What we want:
- save traces to JSON
- reload them later
- compare multiple score functions on the exact same trace

Why:
- this avoids re-running generation or future instrumentation for every test
- it gives us a stable offline evaluation path

## Step 3: Add Score Adapters

Examples:
- raw head-weighted score
- normalized score
- layer-normalized score
- score with recency prior
- score with reuse-distance prior

Why:
- we want to compare score *functions* cleanly without changing the simulator

## Step 4: Build Better Diagnostics

We should log:
- score distributions
- which pages were scored highly but still evicted
- which prefetched pages were useful vs wasted
- how score rank correlates with next-step usage

Why:
- this tells us whether a score is informative or just numerically noisy

## Step 5: Move From Synthetic To Replayed Traces

At first:
- use synthetic traces with preserved per-head activity

Then:
- use externally saved traces in the replay format

Eventually:
- replay traces exported from a real inference stack

Why this progression:
- it lets us validate tooling before depending on live model instrumentation

## Step 6: Collect Real Head Signals Later

When we are ready for more realism:
- run a manageable model in eager or instrumented mode
- log attention or head-summary signals during decode
- convert those signals into page-level per-head activity
- save them into the replay format

Important note:
- we are not depending on a pre-existing public "KV page importance dataset"
- this is more likely to come from our own instrumentation pipeline

## Step 7: Evaluate Score Quality By Control Value

A better score is not just one that "looks reasonable."
It should improve:
- score-based eviction vs LRU
- score-based prefetch vs simple prefetch
- bandit performance when score-based actions are available

## Immediate Next Action

The first implementation step is:

- extend the trace/replay path to preserve per-page per-head activity

That gives us the raw signal needed for the next score-function experiments.
