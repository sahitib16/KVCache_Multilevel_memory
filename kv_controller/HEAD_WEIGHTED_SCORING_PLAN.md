# Head-Weighted Scoring Plan

This document describes the step-by-step approach for moving the project from
synthetic head-weighted scores toward replayable, more realistic score signals.

The key idea is:

1. define the data we need
2. carry that data through the simulator and replay path
3. experiment with score functions offline
4. collect real model signals
5. only later consider a learned predictor if needed

## Goal

We want a head-weighted score that is useful for control decisions:
- eviction
- prefetch
- layer budgeting
- eventually adaptive action selection

The score is valuable only if it improves control outcomes such as:
- lower misses
- lower churn
- better prefetch usefulness
- lower harmful eviction / miss pressure

Timing can still be observed, but for this project phase the main validation
target is page-level behavior rather than simulated milliseconds.

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
- page-wise / tile-wise realism summaries:
  - hot-page concentration
  - hot-tile concentration
  - page miss rate
  - prefetch hit / waste rate
  - eviction-per-access rate
  - reuse-distance summaries

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

## Step 6: Collect Real Head Signals From A Real Model

This is the next major transition.

Important clarification:
- we do **not** need to train a separate model first
- the first goal is to collect real decode-time head signals and convert them
  into page-level importance traces

Recommended first path:
- use a manageable real model
- instrument decode-time execution
- log head-level summaries
- map those summaries onto pages
- replay those traces through the current controller

Why this comes before training:
- simpler
- easier to debug
- gives us a direct signal path from real execution to controller evaluation
- avoids prematurely adding another learned system before we know the signal is
  useful

When we are ready for this step:
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

## Step 8: Only If Needed, Train A Predictor Later

A learned predictor is optional, not the default next step.

We should consider it only if:
- real logged head signals are useful
- direct score construction from those logs is still too weak
- a learned mapping would likely improve generalization

In that later phase, we could:
- define page-importance labels or utility labels
- train a small predictor from logged features
- compare it against the direct analytic score

## Current Default Position

Right now the project should assume:
- first collect real head signals
- first try direct score construction from those signals
- only later consider training a separate predictor

## Immediate Next Action

The next implementation steps are:

1. keep using replay traces as the experiment interface
2. make the replay schema and page concepts more compatible with a future
   vLLM-style integration
3. design the exact logging fields needed from a small real model run
4. collect real head-summary traces
5. replay them through the controller

## Current Status

Completed so far:
- Step 1 is done:
  - replay traces now preserve `per_page_head_activity`
  - `WorkloadStep` carries both raw head activity and aggregate scores
- Step 2 is effectively done and working:
  - replay traces can be saved and reloaded
  - alternate replay-time scorers can now be applied to the same trace
- Step 3 is effectively done for the first scorer family:
  - multiple score adapters exist
  - replay-time scorer comparisons are now possible
- Step 4 is now effectively done for the current analytic scorer family:
  - diagnostics script exists for score distributions and next-step hit rates
  - realism validation script now exists for page-wise / tile-wise workload checks
- Step 6 is in progress:
  - real head-summary traces can now be collected from:
    - manually supplied prompts
    - public prompt datasets through the dataset collector
 - Step 3 has now entered the first novelty extension:
   - reuse-distance features can be attached to replay traces
   - a reuse-distance hybrid scorer now exists for controlled comparison

Current replay-time default:
- scorer view: `NormalizedHeadWeightedScorer`
  - larger public-dataset validation now favors `normalized` as the strongest
    current replay-time default on control outcomes
- bandit action menu: `full`
  - broader replay studies favored the full menu over the trimmed menu

Next:
- use the page/tile realism validator as a gate before trusting new replay
  workloads
- keep the analytic scorer path unless broader evidence clearly says it is too weak
- compare the new reuse-distance hybrid against the current normalized default
- only consider a learned predictor later if the analytic scorer family stops
  improving controller outcomes
