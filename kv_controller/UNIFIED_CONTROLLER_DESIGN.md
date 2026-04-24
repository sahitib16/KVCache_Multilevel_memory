# Unified Controller Design

## Goal

The end goal is one controller that can eventually sit in a `vLLM`-like KV
manager path and make online page decisions:
- what to evict
- what to prefetch
- how aggressive to be under different page-pressure regimes

This should *not* remain a permanent collection of separate systems. The
current comparisons exist only to decide what the single integrated controller
should learn.

## Proposed Single-Controller Interface

Inputs:
- current resident pages
- in-flight / queued transfer pressure
- required pages
- predicted pages
- per-page head score
- reuse-aware page features
- tile hotness
- per-layer pressure

Outputs:
- effective page priority scores
- prefetch set
- eviction set

Design principle:
- one controller object
- one online `decide(context) -> PolicyOutput` call
- one `observe(...)` feedback path for adaptive versions

## Current Controller Options

### Option A: `UnifiedRuleController`

Behavior:
- choose one scorer view from:
  - normalized
  - layer-normalized
  - page-stats hybrid
- use simple interpretable rules over causal features

Why include it:
- strongest interpretability baseline
- easiest to reason about for a paper and for future engine integration

### Option B: `UnifiedBlendController`

Behavior:
- softly blend normalized, layer-normalized, and page-stats scores
- use causal regime features as blend weights

Why include it:
- avoids brittle hard switching
- tests whether a continuous controller is better than discrete regime logic

### Option C: `UnifiedBanditController`

Behavior:
- one lightweight online model chooses among scorer modes and prefetch depths
- still emits one ordinary `PolicyOutput`

Why include it:
- closest current approximation to the desired final system:
  one adaptive controller inside the online decision loop

## Current Research Hypothesis

The evidence so far suggests:
- prefetch-hostile regime:
  - normalized scoring works best
- middle regime:
  - layer-normalized scoring can work best
- hard high-pressure regime:
  - reuse/page-stats-aware scoring works best

So the unified-controller problem is:

> can one online controller infer which regime it is in and choose the right
> behavior without hand-picking a separate global scorer?

## Recommended Near-Term Path

1. Compare the three unified-controller options on:
   - main benchmark
   - middle benchmark
   - hard benchmark
2. Use page/tile realism gates to interpret the workload regime
3. Keep only the best one or two unified designs
4. Improve the winner instead of adding more one-off baselines

## Why This Fits The Final vLLM Goal

Even though we are still in simulator/replay mode, these designs already map
to a future engine integration:
- feature extraction from live page/block state
- one controller call per decode step
- one page-priority / prefetch / eviction decision path

That is the intended bridge from simulator research to an integrated
`vLLM`-facing controller.
