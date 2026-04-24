# KV Controller Logbook

This logbook records what has been implemented so far in the new
`kv_controller/` system, why it was added, and what the current results mean.

It is meant to be a narrative companion to:
- [`PLAN.md`](/home/sbulusu31/kv_multilevel/kv_controller/PLAN.md)
- [`README.md`](/home/sbulusu31/kv_multilevel/kv_controller/README.md)

## Project Goal

Build an overlap-aware multi-tier KV page residency controller that:
- operates at the engine / KV-manager layer
- does not modify fused attention kernels
- minimizes decode stall and tail latency when KV spills between CPU and GPU

## Latest Update: Fixed Policies, Regime-Aware Scoring, and Realism Gates

### What We Added
- Broadened the fixed-policy family in `kv_controller/policies.py`:
  - `FixedKPrefetchController`
  - `SlidingWindowController`
  - `TileHotnessController`
- Added richer reuse/page-history features in `kv_controller/scoring.py`:
  - request-local reuse distance
  - recent per-page frequency
  - recent per-request page frequency
  - recent tile frequency
- Added:
  - `PageStatsHybridScorer`
  - `RegimeAwarePageStatsScorer`
- Integrated realism gates through `scripts/validate_replay_realism.py`.

### Realism-Gate Results

Main benchmark:
- Trace: `results/dataset_head_traces/pressure_16/recent_threshold_round_robin_interleave.json`
- Capacity: `32`
- Under `score`:
  - `page_miss_rate = 1.0000`
  - `prefetch_hit_rate = 0.0000`
  - `prefetch_waste_rate = 1.0000`
  - `evictions_per_access = 1.0101`

Interpretation:
- this benchmark is strongly prefetch-hostile
- it is still useful, but should not be the only workload used for claims

Hard benchmark:
- Trace: `results/dataset_head_traces/pressure_16/round_robin_interleave.json`
- Capacity: `60`
- Under `score`:
  - `page_miss_rate = 0.9332`
  - `prefetch_hit_rate = 1.0000`
  - `prefetch_waste_rate = 0.0000`
  - `evictions_per_access = 0.9896`

Interpretation:
- this benchmark has real repeated-access structure
- aggressive reuse/page-history-aware prefetch can pay off here

### Fixed-Policy Results

Main benchmark:

| policy | total_miss | mean_stall | prefetch | evictions |
| --- | ---: | ---: | ---: | ---: |
| `fixed_k_prefetch` | 4155 | 2.9995 | 74 | 4197 |
| `sliding_window` | 4155 | 2.9995 | 74 | 4197 |
| `tile_hotness` | 4155 | 2.9995 | 74 | 4197 |
| `score` | 4155 | 2.9995 | 74 | 4197 |

Interpretation:
- the main benchmark does not separate the richer fixed-policy family

Hard benchmark:

| policy | total_miss | mean_stall | prefetch | evictions |
| --- | ---: | ---: | ---: | ---: |
| `fixed_k_prefetch` | 5368 | 4.0472 | 384 | 5692 |
| `sliding_window` | 5376 | 4.0584 | 384 | 5700 |
| `tile_hotness` | 5364 | 4.0416 | 384 | 5688 |
| `score` | 5364 | 4.0416 | 384 | 5688 |

Interpretation:
- `tile_hotness` ties `score` on the hard benchmark
- `fixed_k_prefetch` and `sliding_window` are slightly worse

### Scorer Results

Main benchmark (`bandit`):

| scorer | total_miss | mean_stall | prefetch | evictions |
| --- | ---: | ---: | ---: | ---: |
| `normalized` | 4155 | 2.9517 | 24 | 4147 |
| `page_stats_hybrid` | 4155 | 2.9770 | 46 | 4169 |
| `regime_aware` | 4155 | 2.9784 | 49 | 4172 |

Interpretation:
- `normalized` is still best on the main prefetch-hostile benchmark
- `regime_aware` is better than pure `page_stats_hybrid` on the main benchmark,
  but it still gives back some of the `normalized` advantage

Hard benchmark (`bandit`):

| scorer | total_miss | mean_stall | prefetch | evictions |
| --- | ---: | ---: | ---: | ---: |
| `normalized` | 5405 | 4.0359 | 345 | 5690 |
| `page_stats_hybrid` | 5180 | 3.8827 | 568 | 5688 |
| `regime_aware` | 5174 | 3.8784 | 574 | 5688 |

Interpretation:
- `page_stats_hybrid` clearly wins on the hard benchmark
- the improved `regime_aware` scorer now slightly beats `page_stats_hybrid`
  on the hard benchmark
- the remaining open problem is preserving the main-benchmark gains of
  `normalized` at the same time

### Thought Process And Failed Selector Ideas

#### 1. Reuse-only trigger

What we tried:
- switch to the reuse/page-stats hybrid whenever average request-local inverse
  reuse was high

What failed:
- both the main and hard benchmarks show strong request-local reuse
- the selector therefore fired in both regimes
- that made it too aggressive on the main benchmark

What we learned:
- reuse alone is not the right discriminator

#### 2. Reuse plus predicted-page footprint

What we tried:
- switch only when both:
  - request-local reuse was high
  - predicted-page footprint was large

What improved:
- it matched the hard benchmark much better

What still failed:
- it was still too eager on the main benchmark
- a few main-benchmark steps with large predicted-page sets were enough to
  give back part of the `normalized` win

What we learned:
- predicted-page footprint matters more than reuse
- but a single hard switch is still brittle

#### 3. Soft blend between `normalized` and `page_stats_hybrid`

What we tried:
- blend the two scorers based on predicted footprint and working-set size

What failed:
- this preserved the main-benchmark result
- but it threw away too much of the hard-benchmark gain

What we learned:
- the issue is not just "how much page-history signal to add"
- the issue is "which steps should receive that signal at all"

#### 4. Predicted footprint plus working-set threshold

What we tried:
- only switch when:
  - predicted ratio was high
  - required-page count was above a threshold

What improved:
- it protected the main benchmark better

What failed:
- it also removed too much of the hard-benchmark benefit

What we learned:
- absolute working-set size is useful context, but too coarse as a hard gate

#### 5. Current best design

What we now use:
- `RegimeAwarePageStatsScorer`
- trigger on large predicted-page footprint
- when triggered, use a tuned high-pressure scorer with slightly softer
  page-history weights than the standalone `PageStatsHybridScorer`

Why this is better:
- it stays interpretable
- it uses causal step-local signals only
- it now slightly improves the hard benchmark beyond the plain page-stats
  hybrid

Current outcome:
- main benchmark:
  - still worse than `normalized`
  - but better than earlier regime-aware attempts
- hard benchmark:
  - now slightly better than `page_stats_hybrid`

Honest conclusion:
- the regime-aware direction is real
- it is not fully solved yet
- but it is now a meaningful novelty path rather than a placeholder

## Three-Regime Benchmark Update

To avoid overfitting the selector to only two extremes, we added a middle
benchmark to the broader real-trace suite:

- main:
  `recent_threshold_round_robin_interleave`
- middle:
  `recent_topk_round_robin_interleave`
- hard:
  `round_robin_interleave`

### Middle-Benchmark Realism

Under `score`:
- `page_miss_rate = 0.9283`
- `prefetch_hit_rate = 0.0000`
- `prefetch_waste_rate = 1.0000`
- `evictions_per_access = 0.9383`
- `mean_page_reuse_distance = 7.5651`
- `short_reuse_fraction = 0.0815`

Interpretation:
- this is less hostile than the main benchmark
- but still not as reuse-friendly as the dense hard stress benchmark
- it gives us a bridge regime for scorer selection

### Middle-Benchmark Scorer Results (`bandit`)

| scorer | total_miss | mean_stall | prefetch | evictions |
| --- | ---: | ---: | ---: | ---: |
| `normalized` | 2150 | 2.9646 | 28 | 2154 |
| `layer_normalized` | 2150 | 2.9484 | 24 | 2150 |
| `page_stats_hybrid` | 2150 | 2.9673 | 32 | 2158 |
| `regime_aware` | 2150 | 2.9646 | 28 | 2154 |

Interpretation:
- the middle benchmark does not favor the current regime-aware scorer
- it behaves like `normalized` there
- the best scorer in this bridge regime is `layer_normalized`

### What This Changed

The selector problem is now clearly a three-regime problem:
- main benchmark best:
  - `normalized`
- middle benchmark best:
  - `layer_normalized`
- hard benchmark best:
  - `regime_aware` / tuned page-history scorer

This is actually a stronger research story than a single global scorer:
- different scoring strategies are practical in different page-pressure regimes
- the remaining challenge is making the regime selector itself robust and
  interpretable

## Step 1: Reusable Simulator Core

What we built:
- [`types.py`](/home/sbulusu31/kv_multilevel/kv_controller/types.py)
  Shared dataclasses for pages, workload steps, transfer requests, metrics, and configs.
- [`interfaces.py`](/home/sbulusu31/kv_multilevel/kv_controller/interfaces.py)
  Abstract interfaces for scorers and controllers.
- [`state.py`](/home/sbulusu31/kv_multilevel/kv_controller/state.py)
  Mutable residency state.
- [`scheduler.py`](/home/sbulusu31/kv_multilevel/kv_controller/scheduler.py)
  Discrete-event transfer scheduler.
- [`workload.py`](/home/sbulusu31/kv_multilevel/kv_controller/workload.py)
  Synthetic multi-layer workload generator.
- [`simulator.py`](/home/sbulusu31/kv_multilevel/kv_controller/simulator.py)
  End-to-end simulator loop.
- [`__init__.py`](/home/sbulusu31/kv_multilevel/kv_controller/__init__.py)
  Public package entrypoint.

Why Step 1 mattered:
- it replaced one-off experimental scripts with reusable infrastructure
- it gave the project a stable vocabulary for pages, steps, transfers, and policies
- it made later controller work possible without duplicating logic

## Step 2: Explicit Engine-Level Interfaces

What we added:
- `WorkloadStep` now carries:
  - required pages
  - predicted pages
  - per-page features
  - referenced layers
- `CacheState` now tracks:
  - HBM residency
  - CPU residency
  - page-to-slot map
  - slot-to-page map
  - access and churn data
- `TransferState` now tracks:
  - queued transfers
  - in-flight transfers
  - completion times
  - backlog
  - overlap budget
- `PolicyOutput` is now the formal controller return type
- the simulator enforces:
  - decode may not launch until all required pages are resident
  - every resident page must have an HBM slot assignment

Why Step 2 mattered:
- it turned the simulator into something structurally compatible with a real paged-KV engine
- it made the controller operate on page-level engine state instead of script-local variables

## Step 3: Static Baselines, Oracles, and Benchmarking

What we added:
- [`scoring.py`](/home/sbulusu31/kv_multilevel/kv_controller/scoring.py)
  Concrete scorer implementations:
  - `PassthroughHeadWeightedScorer`
  - `NormalizedHeadWeightedScorer`
- [`policies.py`](/home/sbulusu31/kv_multilevel/kv_controller/policies.py)
  Static and oracle controllers:
  - `LRUController`
  - `ScoreBasedController`
  - `BeladyOracleController`
  - `PerfectPrefetchOracleController`
  - `FutureTraceOracle`
- [`benchmark.py`](/home/sbulusu31/kv_multilevel/kv_controller/benchmark.py)
  Reusable multi-policy benchmarking helpers.
- [`scripts/run_kv_controller_sim.py`](/home/sbulusu31/kv_multilevel/scripts/run_kv_controller_sim.py)
  Driver script with:
  - single-policy mode
  - policy-suite comparison
  - step CSV output
  - summary CSV output

Why Step 3 mattered:
- it created actual baselines and upper bounds
- it made policy comparison repeatable on the same trace
- it showed that oracle eviction quality can materially improve miss behavior

What remained partial after Step 3:
- the stall model was still too flat
- the score-based baseline was only marginally different from LRU
- the new benchmark path existed, but older script parity was still incomplete

## Step 4: First Adaptive / Learning-Based Controller

What we added:
- `ResidencyController.observe(...)` feedback hook in
  [`interfaces.py`](/home/sbulusu31/kv_multilevel/kv_controller/interfaces.py)
- simulator callback to `controller.observe(...)` in
  [`simulator.py`](/home/sbulusu31/kv_multilevel/kv_controller/simulator.py)
- `ContextualBanditController` in
  [`policies.py`](/home/sbulusu31/kv_multilevel/kv_controller/policies.py)

How the adaptive controller works:
- it builds a context feature vector from current simulator state
- it chooses among a small action set:
  - eviction rule `{lru, score}`
  - prefetch depth `{0, 2, 4}`
- it uses a lightweight LinUCB-style contextual bandit
- it delegates the chosen action to an existing static controller
- it updates its model after each step from observed reward

Why Step 4 mattered:
- it introduced adaptive behavior without jumping to heavy RL
- it kept the action space interpretable
- it let us test whether adaptive control can beat fixed baselines

## Important Fixes Made After Initial Step 4

### 1. Stall accounting fix

Problem:
- `stall_ms` was coming out almost perfectly flat across policies.

Root cause:
- the simulator only counted the final wait in `wait_for_pages(...)`
- it did **not** count queueing delay while demand transfers were waiting for a
  free copy lane

Fix:
- in [`scheduler.py`](/home/sbulusu31/kv_multilevel/kv_controller/scheduler.py),
  demand-side queue delay is now added to `stall_ms` and `transfer_wait_ms`

Why this matters:
- more misses should create more demand-transfer contention
- more contention should increase stall
- without this fix, policy differences were hidden in miss counts but not
  reflected in latency

### 2. Bandit cold-start / reward fix

Problem:
- the bandit was performing worse than the static baselines

Likely reasons:
- weak reward signal because stall was too flat
- poor early exploration on very short traces
- penalties weighted in a way that over-punished prefetch/eviction relative to miss reduction

Fixes applied:
- lower exploration `alpha`
- stronger miss penalty
- weaker prefetch/eviction penalties
- bootstrap warm-start action of `score + prefetch_k=2`
- feature vector now also includes previous-step stall, misses, and evictions

Why this matters:
- the bandit now starts from a more sensible action
- it can react to short-term regime changes more directly
- the reward better emphasizes what we care about

### 3. Layer-budget and transfer-pressure adaptive knobs

Problem:
- the first bandit action space only changed eviction rule and prefetch depth
- that was not enough to express two system behaviors we actually care about:
  - per-layer HBM partitioning
  - backing off prefetch when the system is already stressed

Fixes applied:
- added `budget_mode` to `BanditAction`
  - when enabled, the bandit can choose a score-based controller that also
    emits adaptive per-layer HBM budgets
- implemented layer-budget enforcement in
  [`simulator.py`](/home/sbulusu31/kv_multilevel/kv_controller/simulator.py)
  so those budget updates are now real, not just logged
- added `guard_mode` to `BanditAction`
  - when enabled, prefetch depth is dynamically clamped under transfer backlog,
    very high occupancy, or miss+churn pressure

Why this matters:
- it gives the adaptive controller more meaningful levers than just "LRU vs
  score" and "prefetch depth"
- it makes the project closer to the intended engine-level controller design

### 4. Bandit credit-assignment fix

Problem:
- the bandit was crediting most reward to the same step that issued the action
- but prefetch decisions mainly pay off on the following step

Fix applied:
- the bandit now uses one-step delayed credit assignment for stall/miss reward
- immediate control cost for the current step's prefetch/eviction work is still
  charged on the same step

Why this matters:
- it matches the systems behavior better
- it gives prefetch-heavy actions a fairer learning signal

### 5. CLI consistency fix

Problem:
- the CLI driver was still defaulting to a higher bandit exploration value than
  the tuned controller default

Fix applied:
- [`run_kv_controller_sim.py`](/home/sbulusu31/kv_multilevel/scripts/run_kv_controller_sim.py)
  now defaults `--bandit-alpha` to `0.35`

Why this matters:
- command-line runs now match the tuned controller more closely
- it avoids accidentally judging the bandit from a noisier configuration

## Current Results Snapshot

These results came from:

```bash
python scripts/run_kv_controller_sim.py --policy-suite lru,score,belady,perfect_prefetch,bandit
```

Observed summary before the latest stall/bandit fixes:

```text
policy             total_miss   mean_stall   p95_stall   prefetch   evictions
lru                       56       0.2700       0.2700         11          47
score                     57       0.2700       0.2700          9          46
belady                    48       0.2700       0.2700          0          28
perfect_prefetch          52       0.2700       0.2700         11          43
bandit                    59       0.2700       0.2700          9          48
```

Interpretation:
- `Belady` was best on misses and evictions.
  This shows genuine headroom in eviction quality.
- `PerfectPrefetch` helped, but less than perfect eviction.
  This suggests eviction quality matters a lot in the current regime.
- `LRU` and `score` were nearly tied.
  The current score signal or score-based policy was not yet strong enough to beat LRU.
- `Bandit` was worst.
  That meant the adaptive controller existed, but the environment/reward setup
  was not yet helping it.
- identical stall across all policies was a simulator issue, not a policy truth.

Observed summary after the stall-accounting fix and initial bandit tuning:

```text
policy             total_miss   mean_stall   p95_stall   prefetch   evictions
lru                       56       0.8325       1.0800         11          47
score                     57       0.7875       1.0800          9          46
belady                    51       0.6525       0.8100          0          32
perfect_prefetch          53       0.7650       0.8100         11          44
bandit                    55       0.7650       0.8100         10          45
```

Interpretation:
- the flat-stall bug was real and the fix worked
- after the fix, better policies do translate into lower stall
- the bandit stopped being obviously worst and became competitive on stall

Observed summary after adding layer-budget mode, prefetch guard mode, and
delayed credit assignment on the default 12-step trace:

```text
policy             total_miss   mean_stall   p95_stall   prefetch   evictions
lru                       56       0.8325       1.0800         11          47
score                     57       0.7875       1.0800          9          46
belady                    51       0.6525       0.8100          0          32
perfect_prefetch          53       0.7650       0.8100         11          44
bandit                    57       0.7650       0.8100          7          44
```

Interpretation:
- on the short default trace, the bandit now matches `perfect_prefetch` on
  mean stall and beats `lru` / `score` on stall
- it became more conservative, which reduced prefetch traffic and evictions
- that conservatism helped stall but did not yet improve misses

Observed summary on a longer 64-step trace:

```text
policy             total_miss   mean_stall   p95_stall   prefetch   evictions
lru                      258       0.7594       1.0800         67         305
score                    262       0.7467       1.0800         61         303
belady                   212       0.5062       0.8100          0         192
perfect_prefetch         248       0.7130       1.0800         56         284
bandit                   271       0.6961       1.0800         30         281
```

Interpretation:
- once the learner gets more than 12 steps, the adaptive controller becomes
  the best deployable policy on mean stall in this setup
- it does that by becoming more selective about prefetching
- the tradeoff is clear: lower stall, but currently more misses than the other
  non-oracle policies
- this tells us the next bandit improvement should probably focus on
  better reward balancing between stall and miss minimization

## Current Commands Used For Testing

Syntax and test suite:

```bash
python -m py_compile kv_controller/*.py scripts/run_kv_controller_sim.py tests/test_kv_controller_core.py
pytest tests/test_kv_controller_core.py
```

Policy comparisons:

```bash
python scripts/run_kv_controller_sim.py --policy lru --print-steps
python scripts/run_kv_controller_sim.py --policy score --prefetch-k 2 --print-steps
python scripts/run_kv_controller_sim.py --policy belady
python scripts/run_kv_controller_sim.py --policy perfect_prefetch --prefetch-k 2
python scripts/run_kv_controller_sim.py --policy bandit --print-steps
python scripts/run_kv_controller_sim.py --policy-suite lru,score,belady,perfect_prefetch,bandit
```

CSV output:

```bash
python scripts/run_kv_controller_sim.py --policy score --step-csv results/kv_steps.csv
python scripts/run_kv_controller_sim.py --policy-suite lru,score,belady,perfect_prefetch,bandit --summary-csv results/kv_summary.csv
```

## Current State Of The Project

Implemented:
- simulator core
- page-level interfaces
- overlap-aware scheduling
- baseline controllers
- oracle controllers
- first adaptive controller
- adaptive layer-budget mode
- adaptive transfer-pressure prefetch guard
- benchmark helpers
- automated tests

Still pending:
- real-model-derived head scoring
- compression-aware control
- better vLLM integration path
- richer latency model and more realistic overlap behavior
- stronger bandit tradeoff handling between stall and misses

## Next Focus

The next high-value work items are:
- confirm the new stall accounting produces policy-sensitive latency
- reevaluate the bandit after the stall fix
- decide whether the next adaptive step should be:
  - richer action space
  - adaptive layer budgeting
  - better score signals
  - or a more realistic vLLM-shaped trace/replay setup

## New Entry: Step 4 Reward / Replay Round

This round focused on three goals:
- improve the bandit's miss/stall tradeoff
- evaluate adaptive behavior across multiple seeds by default
- add an offline trace-replay layer before any live-engine work

### Changes made

1. Richer bandit reward in
   [`policies.py`](/home/sbulusu31/kv_multilevel/kv_controller/policies.py)

The bandit now tracks more than just stall and raw miss count.
It also reasons about:
- useful prefetches
- wasted prefetches
- miss reduction relative to the previous observed step
- miss increase relative to the previous observed step

This makes the reward closer to the actual system question:
"Did proactive movement help the next step, or did it just add pressure?"

2. Better bandit diagnostics in
   [`benchmark.py`](/home/sbulusu31/kv_multilevel/kv_controller/benchmark.py)

The benchmark summary now records:
- total useful prefetches
- total wasted prefetches
- prefetch hit rate
- total miss reduction
- total miss increase

This makes it much easier to understand *why* the bandit is behaving a certain
way instead of only looking at misses/stall at the end.

3. Multi-seed comparison by default for policy suites in
   [`run_kv_controller_sim.py`](/home/sbulusu31/kv_multilevel/scripts/run_kv_controller_sim.py)

Policy-suite comparisons now default to aggregate evaluation over seeds
`0,1,2,3,4` unless a different `--seed-list` is provided.

Why this matters:
- one-seed conclusions are too noisy for adaptive controllers
- aggregate results are a much safer basis for tuning

4. Offline trace replay layer in
   [`replay.py`](/home/sbulusu31/kv_multilevel/kv_controller/replay.py)

Added helpers to:
- save a trace to JSON
- load a trace from JSON
- replay it through the existing simulator

This creates the bridge between:
- synthetic trace development
- future engine-collected trace analysis

It is intentionally offline and non-invasive.

### Results after this round

5-seed aggregate, 64-step traces:

```text
policy             avg_miss   avg_mean_stall   avg_p95   avg_p99   avg_prefetch   avg_evictions
lru                 267.40         0.7636      1.0800    1.3500         61.80          309.20
score               267.00         0.7577      1.1340    1.3500         60.20          307.20
belady              213.40         0.5164      0.8640    1.1880          0.00          193.80
perfect_prefetch    257.00         0.7282      1.0800    1.3500         56.80          293.80
bandit              272.60         0.7332      1.0800    1.3500         46.60          299.20
```

8-seed aggregate, 64-step traces:

```text
policy             avg_miss   avg_mean_stall   avg_p95   avg_p99   avg_prefetch   avg_evictions
lru                 267.25         0.7694      1.0800    1.3163         63.12          310.38
score               266.12         0.7594      1.1138    1.3163         61.25          307.38
perfect_prefetch    256.88         0.7314      1.0800    1.3163         59.12          296.00
bandit              272.62         0.7314      1.0800    1.3163         45.00          297.62
```

### Interpretation

- The richer reward did not fully fix the miss/stall tradeoff.
- It *did* keep the bandit competitive on mean stall across seeds.
- The bandit now gets similar mean stall to `perfect_prefetch` while using
  substantially fewer prefetches.
- The miss cost is still too high, so the current adaptive controller remains
  too conservative.
- This means the next tuning step should focus on better valuing miss
  avoidance, not on adding many more knobs yet.

### Replay validation

Trace replay round-trip was validated by:
- saving a generated trace to JSON
- reloading it
- confirming the replayed run matched the original run

That means the project now has a clean offline path for future real traces
without jumping directly into live-engine integration.

## New Entry: Reward Sweep And Raw Head-Activity Preservation

This round had two goals:
- tune the bandit's miss/stall tradeoff more systematically
- start the head-weighted scoring roadmap by preserving raw head activity

### Changes made

1. Added a reward sweep harness in
   [`tune_bandit_reward.py`](/home/sbulusu31/kv_multilevel/scripts/tune_bandit_reward.py)

This script:
- sweeps bandit reward coefficients
- evaluates them across multiple seeds
- prints top configurations
- prints action counts and linear feature weights for inspection

Why:
- we wanted a more principled tuning workflow than one manual constant change
  at a time

2. Exposed more bandit diagnostics in
   [`policies.py`](/home/sbulusu31/kv_multilevel/kv_controller/policies.py)

The bandit now reports:
- action counts
- feature names
- linear coefficients per action

Why:
- the model is linear, so this is the simplest form of "feature importance"
- it helps explain which actions the bandit is favoring and why

3. Added the head-weighted scoring approach document at
   [`HEAD_WEIGHTED_SCORING_PLAN.md`](/home/sbulusu31/kv_multilevel/kv_controller/HEAD_WEIGHTED_SCORING_PLAN.md)

4. Implemented the first step of that plan:
   preserve raw per-page per-head activity through the workload and replay path

Files updated:
- [`types.py`](/home/sbulusu31/kv_multilevel/kv_controller/types.py)
- [`workload.py`](/home/sbulusu31/kv_multilevel/kv_controller/workload.py)
- [`simulator.py`](/home/sbulusu31/kv_multilevel/kv_controller/simulator.py)
- [`replay.py`](/home/sbulusu31/kv_multilevel/kv_controller/replay.py)

Why:
- we no longer want replay traces to store only the final aggregated score
- preserving the underlying head activity lets us experiment with alternate
  score functions later without recollecting traces

### Results after the reward sweep update

5-seed aggregate, 64-step traces:

```text
policy             avg_miss   avg_mean_stall   avg_p95   avg_p99   avg_prefetch   avg_evictions
lru                 267.40         0.7636      1.0800    1.3500         61.80          309.20
score               267.00         0.7577      1.1340    1.3500         60.20          307.20
belady              213.40         0.5164      0.8640    1.1880          0.00          193.80
perfect_prefetch    257.00         0.7282      1.0800    1.3500         56.80          293.80
bandit              268.60         0.7282      1.0800    1.4040         53.80          302.40
```

8-seed aggregate, 64-step traces:

```text
policy             avg_miss   avg_mean_stall   avg_p95   avg_p99   avg_prefetch   avg_evictions
lru                 267.25         0.7694      1.0800    1.3163         63.12          310.38
score               266.12         0.7594      1.1138    1.3163         61.25          307.38
perfect_prefetch    256.88         0.7314      1.0800    1.3163         59.12          296.00
bandit              267.62         0.7298      1.0800    1.3163         53.62          301.25
```

### Interpretation

- This is a better balance than the previous bandit defaults.
- On 8 seeds, the bandit now slightly beats `perfect_prefetch` on mean stall.
- The miss gap is still present, but much smaller than before.
- The bandit remains more conservative on prefetch volume than the best static
  prefetch baseline.
- That means the next phase should move from generic simulator tuning toward
  replay-based score and action experiments.

### Detailed tuning-harness interpretation

The reward sweep in
[`tune_bandit_reward.py`](/home/sbulusu31/kv_multilevel/scripts/tune_bandit_reward.py)
produced the following top configurations on seeds `0,1,2,3,4`:

```text
rank | miss_pen | useful | wasted | miss_red | miss_inc | avg_stall | avg_miss | avg_p99
   1 |    0.100 |  0.040 |  0.015 |    0.020 |    0.040 |    0.7256 |   272.80 |  1.4040
   2 |    0.120 |  0.060 |  0.030 |    0.020 |    0.040 |    0.7282 |   268.60 |  1.4040
   3 |    0.120 |  0.040 |  0.030 |    0.040 |    0.020 |    0.7290 |   272.20 |  1.3500
   4 |    0.120 |  0.040 |  0.030 |    0.040 |    0.040 |    0.7298 |   273.00 |  1.4040
   5 |    0.120 |  0.060 |  0.020 |    0.020 |    0.040 |    0.7305 |   268.60 |  1.4040
```

What that means:
- the single best configuration under the sweep's ranking objective minimizes
  mean stall most aggressively, but still gives up too many misses
- the settings near rank `2` are better balanced
  - they keep stall very close to the best sweep result
  - they improve misses noticeably relative to the most stall-aggressive option
- that is why the controller defaults were updated toward the better-balanced
  configuration rather than the absolute lowest-stall one

The best pure-stall configuration from the sweep was:

```text
miss_penalty: 0.1
useful_prefetch_bonus: 0.04
wasted_prefetch_penalty: 0.015
miss_reduction_bonus: 0.02
miss_increase_penalty: 0.04
avg_mean_stall_ms: 0.7256250000000025
avg_total_demand_misses: 272.8
avg_p95_stall_ms: 1.0800000000000125
avg_p99_stall_ms: 1.4040000000000012
```

Why this was *not* adopted as the final default:
- its stall is excellent
- but its miss count is still too high to be the best overall tradeoff

### Action-count interpretation

The representative-seed diagnostics showed:

```text
BanditAction(eviction_rule='lru', prefetch_k=0, budget_mode=False, guard_mode=False): 17
BanditAction(eviction_rule='lru', prefetch_k=2, budget_mode=False, guard_mode=False): 5
BanditAction(eviction_rule='score', prefetch_k=0, budget_mode=False, guard_mode=False): 4
BanditAction(eviction_rule='score', prefetch_k=2, budget_mode=False, guard_mode=False): 7
BanditAction(eviction_rule='score', prefetch_k=4, budget_mode=False, guard_mode=False): 13
BanditAction(eviction_rule='score', prefetch_k=2, budget_mode=True, guard_mode=False): 6
BanditAction(eviction_rule='score', prefetch_k=2, budget_mode=False, guard_mode=True): 8
BanditAction(eviction_rule='score', prefetch_k=4, budget_mode=False, guard_mode=True): 2
BanditAction(eviction_rule='score', prefetch_k=2, budget_mode=True, guard_mode=True): 2
```

Interpretation:
- the bandit is not collapsing to one single action
- it uses a mix of:
  - conservative `lru, prefetch=0`
  - aggressive `score, prefetch=4`
  - medium score-based actions with and without guard/budget modes
- that is a good sign because it means the adaptive controller is genuinely
  switching behavior by context rather than behaving like a disguised static
  policy
- at the same time, the relatively high count for `lru, prefetch=0` confirms
  the "conservative" tendency we have seen in the aggregate metrics

### Feature-weight interpretation

The printed linear coefficients are not causal proofs, but they are useful
diagnostics because this is a linear bandit.

High-level takeaways from the representative run:
- `prev_stall_ms` is often one of the stronger terms for guarded score-based
  actions
  - that suggests the bandit is using recent latency pain to decide when to
    become more defensive
- `avg_required_score` and `avg_predicted_score` matter across many actions
  - that means the head-weighted score signal is already influencing decisions,
    even though it is still synthetic
- the guarded score-based actions tend to get more negative weights on recent
  stall/churn terms
  - consistent with "use guard mode when recent conditions look bad"
- the most restrictive action
  `BanditAction(eviction_rule='score', prefetch_k=2, budget_mode=True, guard_mode=True)`
  has unusual coefficients and was chosen rarely
  - that suggests it may not yet be worth keeping in the action menu long-term

### Replay validation interpretation

The two identical `lru` summaries on the `8`-step trace were the replay check.

They show:
- one run from a freshly generated synthetic trace
- one run from the saved JSON replay file

Because the summaries match exactly, the replay layer is behaving correctly.
That matters because future scoring experiments can now run against a fixed
replayed trace rather than regenerated synthetic data.

## New Entry: Replay Score Tuning And Action-Menu Comparison

This round focused on:
- using replayed traces to compare score variants directly
- comparing a small set of bandit action menus against those score variants
- deciding whether the trimmed menu should remain the default experimental menu

### Changes made

1. Added new score variants in
   [`scoring.py`](/home/sbulusu31/kv_multilevel/kv_controller/scoring.py)

New scorers:
- `HeadActivityRecomputedScorer`
- `LayerNormalizedHeadActivityScorer`
- `PredictedBoostedHeadActivityScorer`
- `apply_scorer_to_trace(...)`

Why:
- we want replay traces to support alternate score functions without changing
  the trace itself

2. Added replay score evaluation script in
   [`evaluate_replay_scores.py`](/home/sbulusu31/kv_multilevel/scripts/evaluate_replay_scores.py)

This script can now:
- compare scorers on one replayed trace
- compare scorers across multiple seeds
- compare multiple bandit action menus against those scorers

3. Added named bandit action-menu presets in
   [`policies.py`](/home/sbulusu31/kv_multilevel/kv_controller/policies.py)

Current presets:
- `full`
- `trimmed`
- `score_heavy`

### Multi-seed replay results

Using seeds `0,1,2,3,4`, policies `score,bandit`, and menus
`full, trimmed, score_heavy`, the replay scorer comparison showed:

#### Passthrough scorer

```text
full        bandit avg_miss 271.60   avg_mean_stall 0.7180
trimmed     bandit avg_miss 269.40   avg_mean_stall 0.7282
score_heavy bandit avg_miss 272.00   avg_mean_stall 0.7197
```

Interpretation:
- `full` and `score_heavy` are more aggressive on stall
- `trimmed` gives up some stall but improves misses

#### Normalized scorer

```text
full        bandit avg_miss 266.00   avg_mean_stall 0.7214
trimmed     bandit avg_miss 267.00   avg_mean_stall 0.7256
score_heavy bandit avg_miss 268.80   avg_mean_stall 0.7229
```

Interpretation:
- this was the best overall replay result in the comparison
- `full + normalized` beats the score baseline on both misses and stall
- this is the strongest sign so far that score realism matters materially

#### Layer-normalized scorer

```text
full        bandit avg_miss 273.80   avg_mean_stall 0.7189
trimmed     bandit avg_miss 273.20   avg_mean_stall 0.7383
score_heavy bandit avg_miss 274.80   avg_mean_stall 0.7223
```

Interpretation:
- layer normalization did not help in this replay study
- it pushed the bandit toward lower stall but significantly worse misses

#### Predicted-boosted scorer

```text
full        bandit avg_miss 274.80   avg_mean_stall 0.7256
trimmed     bandit avg_miss 268.80   avg_mean_stall 0.7223
score_heavy bandit avg_miss 271.60   avg_mean_stall 0.7256
```

Interpretation:
- naive prediction boosting is not clearly beneficial yet
- it likely needs more careful shaping before it becomes a good default score

### Menu decision

Conclusion from the replay evidence:
- the trimmed menu does **not** clearly dominate
- the `full` menu paired with the `normalized` scorer gave the best overall
  miss/stall balance in this study
- the `trimmed` menu is still useful as a simpler comparison point, but it
  should not be treated as the final answer yet

So the current decision is:
- keep named menus for experimentation
- do not lock the project to `trimmed` as the only preferred menu
- prioritize replay-driven scorer/menu combinations rather than pruning the
  action space too aggressively

### What this means for head-weighted importance progress

Done:
- raw per-page head activity is preserved in traces and replay files
- replay-based alternate scorers now exist
- replay-based scorer comparison now exists

Next:
- promote the strongest replay scorer candidates into broader aggregate studies
- decide whether `normalized` should become the main replay-time score view
- only after that move toward real collected head summaries from model runs

## New Entry: Broader Replay Aggregate And Default Lock-In

This round answered two questions:
- does the best replay scorer/menu combination hold up beyond 5 seeds?
- should the trimmed menu remain the preferred replay-time menu?

### Broader replay results

8-seed replay aggregate comparing `full` vs `trimmed` menus:

```text
passthrough + full        bandit avg_miss 270.50   avg_mean_stall 0.7235
passthrough + trimmed     bandit avg_miss 268.25   avg_mean_stall 0.7330

normalized + full         bandit avg_miss 267.88   avg_mean_stall 0.7283
normalized + trimmed      bandit avg_miss 267.88   avg_mean_stall 0.7320

recomputed + full         bandit avg_miss 270.50   avg_mean_stall 0.7235
recomputed + trimmed      bandit avg_miss 268.25   avg_mean_stall 0.7330
```

12-seed replay aggregate using the strongest menu candidate (`full`):

```text
passthrough + full        bandit avg_miss 267.25   avg_mean_stall 0.7225
normalized + full         bandit avg_miss 267.92   avg_mean_stall 0.7249
recomputed + full         bandit avg_miss 267.25   avg_mean_stall 0.7225
layer_normalized + full   bandit avg_miss 270.00   avg_mean_stall 0.7214
predicted_boosted + full  bandit avg_miss 269.92   avg_mean_stall 0.7246
```

### Interpretation

- The trimmed menu does not win the broader replay study.
- `full` consistently gives better mean stall than `trimmed`.
- `normalized + full` looked strongest in the smaller 5-seed comparison, but
  the broader 12-seed replay study favored `passthrough/recomputed + full`.
- On synthetic traces, `passthrough` and `recomputed` are identical because the
  original synthetic score is already built from the stored head activity.
- Between those two, `recomputed` is the better conceptual default because it
  depends on the primitive replay-preserved signal rather than trusting a
  pre-baked aggregate score.

### Decision

Replay-time defaults are now treated as:
- scorer: `HeadActivityRecomputedScorer`
- bandit menu: `full`

This does not mean the search is over, but it gives the project one clear
default replay configuration before moving further down the head-importance
pipeline.

## New Entry: First Real Head-Trace Collection Attempt

This round was the first end-to-end attempt to:
- collect a replay trace from a real model
- feed it back through the simulator and controller

### What was done

Real trace collection command:

```bash
python scripts/collect_real_head_traces.py \
  --model gpt2 \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --max-new-tokens 8 \
  --kv-block-size-tokens 16 \
  --output-json /tmp/real_head_trace.json
```

This succeeded and wrote a replay file.

Then replay evaluation was attempted with:

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json /tmp/real_head_trace.json \
  --policies score,bandit
```

This failed with:

```text
RuntimeError: No free HBM slot available when reserving transfer.
```

Model-library APIs used in this first collector:
- `transformers.AutoTokenizer.from_pretrained(...)`
- `transformers.AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`
- model forward pass with:
  - `output_attentions=True`
  - `use_cache=False`
- next-token selection from `outputs.logits[:, -1, :]` via greedy `argmax`

Attention extraction path:
- read `outputs.attentions`
- take the newest query token's attention distribution from each layer/head
- bucket token positions into KV blocks
- sum attention mass per block to form `per_page_head_activity`
- compute per-layer head weights from a head-concentration proxy
- combine them into `head_weighted_scores`

### What this error means

This is not a parser bug and not a scorer bug.
It means the current simulator configuration did not have enough HBM capacity
to hold the real trace's required KV working set.

In this case:
- GPT-2 has 12 layers
- with `kv_block_size_tokens = 16`, once the sequence grows beyond 16 tokens,
  each layer needs 2 KV blocks/pages instead of 1
- that means the required set can reach roughly:
  - `12 layers * 2 blocks = 24 required pages`
- the simulator default HBM capacity used by the replay evaluator was only
  `20 pages`

So the simulator was being asked to launch a decode step whose required set was
larger than total available HBM slots.

That is why it failed during transfer reservation.

### Why this is still useful

This was actually a good validation point:
- the real trace path is working
- the replay file is valid enough to reach the simulator
- the simulator is enforcing a real capacity invariant instead of silently
  producing nonsense

### Immediate implication

Real-trace replay should be run with a capacity that can accommodate the
required working set for the chosen model / block size / sequence length.

For the GPT-2 example, a safer next run is something like:

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json /tmp/real_head_trace.json \
  --hbm-capacity-pages 32 \
  --policies score,bandit
```

### Follow-up fix and successful replay

After increasing HBM capacity, a second issue appeared:

```text
KeyError: 4
```

What that meant:
- the replay evaluator was still building the simulator using the CLI default
  layer count (`2`)
- but the real GPT-2 trace contains `12` layers
- that caused the controller's layer-budget logic to see pages from layers not
  present in the simulator's configured budget table

Fix applied:
- the replay evaluator now infers layer count and step count directly from the
  loaded trace
- it also reports a clearer error if HBM capacity is below the trace's minimum
  required working-set size

Internal capacity check for the GPT-2 trace showed:

```text
steps: 8
layers: 12
max_required_pages_per_step: 24
required_pages_per_step: [12, 12, 12, 12, 12, 12, 12, 24]
```

So a capacity of at least `24` pages is required just to make residency
possible. Using `32` pages is a safe first replay setting.

Successful replay command:

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json /tmp/real_head_trace.json \
  --hbm-capacity-pages 32 \
  --policies score,bandit
```

Observed result:

```text
SCORER: passthrough | bandit_menu=full
score   total_miss 22   mean_stall 0.4050   p95 1.6200   prefetch 2   evictions 0
bandit  total_miss 22   mean_stall 0.4050   p95 1.6200   prefetch 2   evictions 0
```

Other scorer variants were almost identical, with only a tiny change for
`layer_normalized`.

### Interpretation of the first successful real replay

- This first real trace is valid, but it is not yet stressful enough to
  separate controller quality.
- There are no evictions and almost no prefetch activity.
- That means the controller is not yet operating in the high-pressure regime
  where scorer or policy differences can show up strongly.

So this is a data-pipeline success, not yet a meaningful controller benchmark.

The next real-trace collection runs should increase pressure by changing one or
more of:
- longer prompts
- more decode steps
- smaller KV block size
- tighter HBM capacity during replay

### Framework-level vs kernel-level note

The current collector uses framework-level signals:
- model forward pass in `transformers`
- `output_attentions=True`
- attention tensors captured in Python / framework space

This is different from kernel-level collection:
- framework-level means we read logical attention outputs from the model API
- kernel-level would mean instrumenting or reading behavior much closer to the
  fused attention implementation itself

Why we started at framework level:
- much easier to implement and debug
- good enough to validate the head-importance data path
- avoids modifying or depending on fused kernels early

Why it differs from kernel level:
- framework-level is higher-level and slower, but simpler
- kernel-level is closer to production execution, but much harder to instrument
  and less flexible for early experimentation

## New Entry: Real Trace Sweep And Interpretation

Commands run:

```bash
python scripts/collect_real_head_traces.py \
  --model gpt2 \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --max-new-tokens 8 \
  --kv-block-size-tokens 16 \
  --output-json /tmp/real_head_trace_01.json

python scripts/evaluate_replay_scores.py \
  --trace-json /tmp/real_head_trace_01.json \
  --hbm-capacity-pages 32 \
  --policies score,bandit

python scripts/collect_real_head_traces.py \
  --model gpt2 \
  --prompt "In a distant future, researchers built a memory hierarchy for transformer inference where key-value blocks could move between CPU and GPU, and every decode step depended on whether those blocks were already resident when attention began." \
  --max-new-tokens 12 \
  --kv-block-size-tokens 16 \
  --output-json /tmp/real_head_trace_02.json

python scripts/evaluate_replay_scores.py \
  --trace-json /tmp/real_head_trace_02.json \
  --hbm-capacity-pages 48 \
  --policies score,bandit

python scripts/collect_real_head_traces.py \
  --model gpt2 \
  --prompt "Memory movement dominates decode latency when spilled KV blocks are not already present in GPU memory." \
  --max-new-tokens 12 \
  --kv-block-size-tokens 8 \
  --output-json /tmp/real_head_trace_03.json

python scripts/evaluate_replay_scores.py \
  --trace-json /tmp/real_head_trace_03.json \
  --hbm-capacity-pages 64 \
  --policies score,bandit

python scripts/collect_real_head_traces.py \
  --model gpt2 \
  --prompt "The controller should learn which pages to keep hot, which to evict, and which to prefetch before the next decode step begins." \
  --max-new-tokens 20 \
  --kv-block-size-tokens 16 \
  --output-json /tmp/real_head_trace_04.json

python scripts/evaluate_replay_scores.py \
  --trace-json /tmp/real_head_trace_04.json \
  --hbm-capacity-pages 64 \
  --policies score,bandit
```

Important note:
- the earlier `FileNotFoundError` was caused by a filename mismatch
- the collector wrote `/tmp/real_head_trace_01.json`
- the evaluator was first pointed at `/tmp/real_head_trace.json`

Observed results:

```text
Trace 01:
score   total_miss 22   mean_stall 0.4050   p95 1.6200   prefetch 2   evictions 0
bandit  total_miss 22   mean_stall 0.4050   p95 1.6200   prefetch 2   evictions 0

Trace 02:
score   total_miss 46   mean_stall 0.5400   p95 1.6200   prefetch 2   evictions 0
bandit  total_miss 46   mean_stall 0.5400   p95 1.6200   prefetch 2   evictions 0

Trace 03:
score   total_miss 46   mean_stall 0.5400   p95 1.6200   prefetch 2   evictions 0
bandit  total_miss 46   mean_stall 0.5400   p95 1.6200   prefetch 2   evictions 0

Trace 04:
score   total_miss 34   mean_stall 0.2430   p95 1.6200   prefetch 2   evictions 0
bandit  total_miss 34   mean_stall 0.2430   p95 1.6200   prefetch 2   evictions 0
```

Across scorer variants:
- `passthrough`, `normalized`, `recomputed`, and `predicted_boosted` were effectively identical on these traces
- `layer_normalized` sometimes improved misses by `1-2` and increased prefetch from `2` to `3` or `4`, but still did not change stall

### Interpretation

1. The real-data pipeline works.
- Real GPT-2 attentions were collected.
- They were converted into replay traces with page/head metadata.
- Those traces ran through the simulator successfully.

2. These traces are not yet stressful enough.
- Every run ended with `evictions = 0`.
- Prefetch counts remained very small.
- The bandit and score-based controller therefore made the same decisions.

3. More data alone is not enough.
- We need more real traces.
- But we also need more pressure.
- If HBM capacity is much larger than the active working set, controller choice will not matter much.

4. These runs validate the data path, not policy superiority.
- They show that the project can now do:
  `real model -> real attentions -> replay trace -> controller evaluation`
- They do not yet show that one controller beats another on real traces.

5. Trace 04's lower stall is a trace property, not a controller win.
- It means this particular replay configuration created fewer misses per step.
- Because there was still no eviction pressure, policy quality did not separate.

### What this informs next

- Automate batch collection and replay instead of running one trace at a time.
- Sweep:
  - prompt length
  - decode length
  - KV block size
  - replay HBM capacity
  - later, model choice
- Choose some replay capacities that are only slightly above the trace's minimum feasible capacity so that the controller has to make meaningful residency decisions.

## New Entry: Batch Real-Trace Replay Study

Commands run:

```bash
pytest tests/test_kv_controller_core.py
python -m py_compile scripts/batch_real_head_replay.py

python scripts/batch_real_head_replay.py

python scripts/batch_real_head_replay.py \
  --output-dir results/real_head_batch \
  --summary-csv results/real_head_batch/summary.csv

python scripts/batch_real_head_replay.py \
  --prompt-presets fox,hierarchy,memory,controller \
  --decode-steps 8,12,12,20 \
  --kv-block-sizes 16,16,8,16 \
  --capacities min,min+2,min+4 \
  --scorers recomputed,layer_normalized \
  --policies score,bandit \
  --bandit-menu full \
  --output-dir results/real_head_pressure \
  --summary-csv results/real_head_pressure/summary.csv

python scripts/batch_real_head_replay.py \
  --capacities min,min+2,min+4,min*1.25 \
  --output-dir results/real_head_capacity_sweep \
  --summary-csv results/real_head_capacity_sweep/summary.csv
```

### What the batch study showed

1. The batch harness works.
- It collected multiple real traces automatically.
- It inferred minimum feasible capacity from each trace.
- It replayed those traces under several capacities.
- It wrote structured CSV summaries.

2. The results were extremely stable across capacities.
- `fox` stayed at:
  - recomputed: miss `22`, mean stall `0.4050`, evictions `0`
  - layer_normalized: miss `21`, mean stall `0.4050`, evictions `0`
- `hierarchy` stayed at:
  - recomputed: miss `46`, mean stall `0.5400`, evictions `0`
  - layer_normalized: miss `44`, mean stall `0.5400`, evictions `0`
- `memory` stayed at:
  - recomputed: miss `58` or `46` depending on decode configuration, but always identical across tested capacities
  - layer_normalized: `1` fewer miss than recomputed, same stall, evictions `0`
- `controller` stayed at:
  - recomputed: miss `34`, mean stall `0.2430`, evictions `0`
  - layer_normalized: miss `33`, mean stall `0.2430`, evictions `0`

3. Score-based and bandit policies remained identical.
- same misses
- same stall
- same prefetch count
- same zero-eviction behavior

### What this means

The big lesson is more specific than "we need more data."
The current real-trace replay setup is structurally too easy for a different reason:

- in the real collector, every decode step marks all current causal prefix blocks as `required_pages`
- for dense causal attention, that required set grows monotonically as decode continues
- the final step's required set is effectively the full set of blocks ever used
- the batch harness then sets replay capacity to `min_required_capacity` or slightly above it

So in practice:
- capacity at `min` is already enough to hold the entire lifetime working set
- once pages arrive, they never need to be evicted
- therefore controller choice barely matters

This is why there are:
- no evictions
- almost no prefetch differences
- identical bandit and score behavior
- no sensitivity to capacity changes above `min`

Important interpretation:
- this is not a failure of the bandit
- and not a failure of the scorer
- it is a limitation of the current real-trace semantics

### Are we collecting per-head information yet?

Yes.

From real model runs we already store:
- `per_page_head_activity`
- `query_head_weights`
- derived `head_weighted_scores`

So the project *has* crossed the threshold of collecting real per-head signals.
What it does *not* yet have is a real-trace formulation that creates realistic controller pressure.

### What the scorer comparison still tells us

- `layer_normalized` is consistently slightly better than `recomputed` on miss count, usually by `1-2` misses
- but because there is still no eviction pressure, that improvement does not translate into stall changes
- so scorer ranking on these traces is still weak evidence

### Recommended next steps

1. Change the real-trace formulation, not just the amount of data.
- Collecting more traces in the current dense-prefix format will mostly produce the same pattern.

2. Introduce pressure in one of these ways:
- multi-request replay with interleaved decode steps from several sequences
- trace extraction that distinguishes:
  - mandatory pages
  - highly attended pages
  - lower-priority pages
- top-k attended block replay, where not every historical block is treated as equally required
- a local-window-plus-sparse-attended-block trace representation

3. Keep the current collector output because it is still valuable.
- It gives us real per-head signals.
- Those signals can be transformed into richer replay formats offline.

4. The next concrete implementation should probably be:
- an offline converter from dense real traces into a pressure-inducing replay trace, for example:
  - always require the most recent block(s)
  - treat the top-k older attended blocks as the scored sparse set
  - let the controller decide residency under a tighter capacity

### Conclusion

- The batch automation succeeded.
- Real per-head data collection succeeded.
- The current bottleneck is replay semantics, not data collection.

## New Entry: First Pressure-Replay Results

Commands run:

```bash
python scripts/build_pressure_replays.py \
  --input-traces /tmp/real_head_trace_01.json,/tmp/real_head_trace_02.json,/tmp/real_head_trace_03.json \
  --output-dir results/pressure_replays

python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/recent_topk.json \
  --hbm-capacity-pages 24 \
  --policies score,bandit

python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/recent_threshold.json \
  --hbm-capacity-pages 24 \
  --policies score,bandit

python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/round_robin_interleave.json \
  --hbm-capacity-pages 48 \
  --policies score,bandit

python scripts/build_pressure_replays.py \
  --input-traces /tmp/real_head_trace_01.json,/tmp/real_head_trace_02.json,/tmp/real_head_trace_03.json \
  --recent-block-window 1 \
  --top-k-older-per-layer 1 \
  --score-mass-fraction 0.35 \
  --output-dir results/pressure_replays_harsh
```

Observed results:

1. `recent_topk`
- score: total miss `22`, mean stall `0.3988`, prefetch `2`, evictions `0`
- bandit: total miss `20`, mean stall `0.3650`, prefetch `4`, evictions `0`

2. `recent_threshold`
- score: total miss `22`, mean stall `0.3988`, prefetch `2`, evictions `0`
- bandit: total miss `20`, mean stall `0.3650`, prefetch `4`, evictions `0`

3. `round_robin_interleave`
- score: total miss `992`, mean stall `4.4550`, prefetch `64`, evictions `1008`
- bandit: total miss `1021-1028` depending on scorer variant, mean stall `4.5056-4.5141`, prefetch `39-43`, evictions `1016-1019`

### What these results tell us

1. The pressure-inducing transforms work.
- They finally changed controller behavior.
- They produced non-identical outcomes between score-based and bandit policies.

2. `recent_topk` and `recent_threshold` are useful, but still low pressure.
- There are still no evictions.
- However, the bandit now does better than score-based:
  - fewer misses
  - lower mean stall
  - more aggressive prefetch

Interpretation:
- These transforms are sparse enough that extra prefetching helps.
- But they still do not create true residency competition.

3. `round_robin_interleave` is the first genuinely hard replay benchmark.
- It creates very large miss counts.
- It creates thousands of evictions.
- It creates multi-request competition for HBM.

This is the first transformed real trace where the controller is clearly in a high-pressure regime.

4. On the hard interleaved trace, score-based beats the current bandit.
- Score-based has fewer misses.
- Lower mean stall.
- More prefetches.

Interpretation:
- The bandit is currently too conservative under strong real-trace pressure.
- Its reward/action tradeoff was tuned mostly on earlier synthetic settings.
- It is under-prefetching in the interleaved real-pressure regime.

5. `recent_topk` and `recent_threshold` were identical in this run.
- For these particular traces and settings, the threshold selection ended up choosing essentially the same pages as top-k.
- That means the current threshold parameter is not yet differentiating the workload.

### What this informs next

1. Keep `round_robin_interleave` as the main real-pressure benchmark.
- It is the only one so far that clearly forces real controller tradeoffs.

2. Retune the bandit on this benchmark.
- The current bandit was competitive on earlier synthetic studies.
- But on the first hard real benchmark it loses to score-based.
- That is now the clearest tuning target.

3. Explore a combined transform next.
- Interleave traces that have already been sparsified with `recent_topk` or `recent_threshold`.
- That may give a more realistic middle ground between:
  - too easy (`recent_topk` alone)
  - very harsh (`round_robin_interleave` on dense traces)

4. Differentiate the threshold converter more strongly.
- The current `score_mass_fraction` did not separate it from top-k.
- Future runs should try more aggressive values.

### Conclusion

- We now have real per-head data and a real high-pressure replay benchmark.
- The next step is not more collection first.
- It is tuning and comparing controllers on the interleaved real-pressure benchmark, plus building a combined sparse+interleaved replay formulation.

## New Entry: Combined Sparse+Interleaved Replay Formulations

What was implemented:
- `interleave_sparse_recent_topk_traces(...)`
- `interleave_sparse_recent_threshold_traces(...)`

These are new middle-ground replay formulations:
- first sparsify each request using real head-based scores
- then interleave the requests so they compete for HBM

Why this was needed:
- sparse single-request replay was still too easy
- dense round-robin interleave was very harsh
- the combined traces should give a more realistic benchmark between those extremes

Implementation files:
- `kv_controller/trace_transforms.py`
- `scripts/build_pressure_replays.py`
- `kv_controller/HEAD_WEIGHTED_SCORING_WORKFLOW.md`

Validation:
- `pytest tests/test_kv_controller_core.py` -> `28 passed`
- `python -m py_compile kv_controller/*.py scripts/build_pressure_replays.py` passed

Status:
- round-robin interleave remains the main hard benchmark
- combined sparse+interleaved traces are now ready to evaluate before any bandit retuning

## New Entry: Middle-Ground Pressure Benchmark Results

Commands run:

```bash
python scripts/build_pressure_replays.py \
  --input-traces /tmp/real_head_trace_01.json,/tmp/real_head_trace_02.json,/tmp/real_head_trace_03.json \
  --output-dir results/pressure_replays

python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/recent_topk_round_robin_interleave.json \
  --hbm-capacity-pages 24 \
  --policies score,bandit

python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/recent_threshold_round_robin_interleave.json \
  --hbm-capacity-pages 32 \
  --policies score,bandit

python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/round_robin_interleave.json \
  --hbm-capacity-pages 48 \
  --policies score,bandit
```

Observed transformed-trace capacities:
- `recent_topk_round_robin_interleave`: `min_required_capacity = 24`
- `recent_threshold_round_robin_interleave`: `min_required_capacity = 26`
- `round_robin_interleave`: `min_required_capacity = 48`

Observed results:

1. `recent_topk_round_robin_interleave`
- score: total miss `684`, mean stall `2.9616`, prefetch `14`, evictions `674`
- bandit: total miss `684`, mean stall `2.9194` to `2.9362` depending on scorer, prefetch `4` to `8`, evictions `664` to `668`

2. `recent_threshold_round_robin_interleave`
- score: total miss `643`, mean stall `2.8266`, prefetch `18`, evictions `629`
- bandit: total miss `641` to `644` depending on scorer, mean stall `2.7844` to `2.7928`, prefetch `9` to `10`, evictions `618` to `622`

3. `round_robin_interleave`
- score: total miss `992`, mean stall `4.4550`, prefetch `64`, evictions `1008`
- bandit: total miss `1021` to `1028`, mean stall `4.5056` to `4.5141`, prefetch `39` to `43`, evictions `1016` to `1019`

### Interpretation

1. The new combined sparse+interleaved traces made meaningful progress.
- Unlike the earlier single-request sparse traces, these now create hundreds of evictions.
- Unlike the dense interleaved trace, they are not overwhelmingly harsh.
- This means they successfully created the intended middle-ground benchmark.

2. `recent_threshold_round_robin_interleave` is currently the best benchmark of the three.
- It is clearly harder than single-request sparse replay.
- But easier and more discriminating than the dense interleaved benchmark.
- Bandit is slightly better than score-based on mean stall.
- Bandit is roughly tied or slightly better on misses, depending on scorer.

3. `recent_topk_round_robin_interleave` also works, but is slightly weaker.
- Score and bandit tie on total misses.
- Bandit still improves mean stall a bit.
- So it is useful, but less informative than the threshold-interleaved version.

4. Dense `round_robin_interleave` is still valuable, but maybe too harsh as the main tuning target.
- It creates the strongest pressure.
- But in that regime the current bandit loses clearly to score-based control.
- This suggests it is a good stress test, but not necessarily the best first retuning benchmark.

5. The threshold-interleaved trace is the first replay regime that looks like a good "mainline" tuning benchmark.
- controller choices matter
- pressure is real
- the bandit is not collapsing
- the score-based baseline does not dominate completely

### Conclusion

- Yes, this is meaningful progress.
- The project now has:
  - easy sparse replay
  - middle-ground sparse+interleaved replay
  - hard dense-interleaved replay
- The best next benchmark for controller work is `recent_threshold_round_robin_interleave`.

## New Entry: Broader-Tuned Bandit Defaults And Head-Plan Status

What was changed:
- bandit default reward coefficients updated to the broader-tuned values:
  - `miss_penalty = 0.10`
  - `useful_prefetch_bonus = 0.04`
  - `wasted_prefetch_penalty = 0.015`
  - `miss_reduction_bonus = 0.02`
  - `miss_increase_penalty = 0.02`
- explicit `per_layer_pressure` was added to `ControllerContext`
- the bandit's feature vector now includes:
  - `max_layer_pressure`
  - `mean_layer_pressure`

Validation:
- `pytest tests/test_kv_controller_core.py` -> `28 passed`
- `python -m py_compile kv_controller/*.py` passed

### Post-update comparison results

1. Main benchmark: `recent_threshold_round_robin_interleave`

score baseline:
- total miss `2181`
- mean stall `3.0505`
- p95 stall `3.7800`
- prefetch `56`
- evictions `2213`

bandit after default update:
- passthrough/recomputed:
  - total miss `2182`
  - mean stall `3.0024`
  - p95 stall `3.5100`
  - prefetch `23`
  - evictions `2177`
- layer_normalized:
  - total miss `2181`
  - mean stall `3.0073`
  - p95 stall `3.5100`
  - prefetch `25`
  - evictions `2177`

Interpretation:
- the bandit now clearly beats or ties `score` on the main benchmark
- it achieves lower mean stall
- it achieves lower p95 stall
- it does so with far fewer prefetches and fewer evictions
- the best current main-benchmark scorer view is now effectively `layer_normalized`, because it ties misses and still lowers stall

2. Hard stress test: `round_robin_interleave`

score baseline:
- total miss `3258`
- mean stall `4.6494`
- p95 stall `11.3400`
- prefetch `186`
- evictions `3366`

bandit after default update:
- passthrough/recomputed:
  - total miss `3295`
  - mean stall `4.6035`
  - p95 stall `11.0700`
  - prefetch `149`
- layer_normalized:
  - total miss `3280`
  - mean stall `4.5900`
  - p95 stall `11.0700`
  - prefetch `164`

Interpretation:
- the bandit still loses on misses in the hardest regime
- but it improves mean stall and p95 stall
- and the miss gap is smaller than before under the better scorer views
- `layer_normalized` is again the strongest scorer view for the bandit

### Overall interpretation

- the broader-tuned defaults were a good update
- the bandit is now in a healthier state:
  - strong on the main benchmark
  - competitive but not yet dominant on the hard stress test
- reward tuning helped, but did not fully solve the hardest regime
- that suggests the next bottleneck is no longer just reward coefficients
- the next likely bottlenecks are:
  - action menu
  - controller features
  - static policy family

### Head-weighted scoring plan status

Completed:
- Step 1: define and preserve raw head signal
- Step 2: replay as the main experiment interface
- Step 3: score adapters
- Step 5: move from synthetic to replayed traces
- Step 6 (first half): collect real head signals from a real model

Partially completed:
- Step 4: diagnostics
  - we have useful policy/benchmark diagnostics
  - we can still add deeper score-rank correlation and score-distribution views
- Step 7: evaluate score quality by control value
  - this is actively happening now through sparse, interleaved, and broader replay benchmarks

Not started yet:
- Step 8: train a predictor later if needed

Practical status summary:
- the head-based scoring plan is mostly through its core data-pipeline and replay-evaluation stages
- the main missing work is:
  - broader validation
  - better diagnostics
  - deciding whether a learned predictor is necessary at all

### Conclusion

- yes, it was worth updating the bandit defaults before moving on
- yes, adding churn/per-layer pressure context was easy and is now done
- the head-weighted scoring plan should be finished first before moving deeply into reuse-distance and other novelty features
- the remaining head-plan work is now mostly about validation and diagnostics, not basic infrastructure

## New Entry: Public-Dataset Validation And Diagnostics

What was implemented:
- `scripts/collect_dataset_head_traces.py`
  - collects replay traces from public prompt datasets
  - current adapters:
    - `HuggingFaceH4/mt_bench_prompts`
    - `OpenAssistant/oasst1`
- `kv_controller/score_diagnostics.py`
- `scripts/diagnose_replay_scores.py`
- diagnostics bug fix:
  - next-step reuse is now measured against the next step of the *same request* instead of the next global step, which matters for interleaved traces

Validation:
- `pytest tests/test_kv_controller_core.py` -> `28 passed`
- `python -m py_compile kv_controller/*.py scripts/diagnose_replay_scores.py scripts/collect_dataset_head_traces.py` passed

### Public-dataset trace collection

MT-Bench slice:
- `4` prompts collected successfully
- example min feasible capacities: `36`, `48`, `60`, `48`

OpenAssistant slice:
- `4` prompts collected successfully
- example min feasible capacities: `24`, `48`, `24`, `24`

### Dataset-backed main benchmark

`recent_threshold_round_robin_interleave`

score:
- total miss `2158`
- mean stall `3.1163`
- p95 stall `3.5100`
- prefetch `39`
- evictions `2165`

bandit:
- normalized scorer:
  - total miss `2158`
  - mean stall `3.0938`
  - p95 stall `3.5100`
  - prefetch `20`
  - evictions `2146`
- layer_normalized scorer:
  - total miss `2158`
  - mean stall `3.1078`
  - p95 stall `3.5100`
  - prefetch `35`
  - evictions `2161`

### Interpretation of dataset-backed benchmark

- the public-dataset results confirm the main pattern:
  - bandit can match score-based misses while reducing mean stall
  - and it does so with less prefetch/eviction pressure
- however, unlike the earlier broader prompt benchmark, `normalized` slightly beats `layer_normalized` on this dataset slice
- therefore we should *not* lock `layer_normalized` as the universal default scorer yet

### Diagnostics on the dataset-backed traces

- all scorers show extremely high top-k next-step hit rates
- reused-page mean rank is very similar across scorer variants
- `layer_normalized` is slightly better on reused-page rank, but only by a small margin

### Interpretation of diagnostics

- the current pressure replay formulation still makes next-step relevance very easy for all scorers
- diagnostics are now correct, but they are somewhat saturated
- this means:
  - we do have enough data to validate the current analytic scorer family
  - but we do *not* yet have evidence that a learned predictor is needed
  - scorer differences are still relatively small and mostly show up in control outcomes rather than raw reuse diagnostics

### Decision

- collect more real data only in a targeted way
- not because the pipeline is missing data entirely, but because scorer ranking is still close and benchmark conclusions may depend on prompt source
- do not jump to a learned predictor yet
- first finish the head-weighted scoring plan with:
  - broader validation
  - stronger diagnostics
  - scorer-default decision

## 2026-03-30 - Larger Public-Dataset Validation For Scorer Decision

### Goal

- run a larger public-dataset validation so we can settle `normalized` vs `layer_normalized` more cleanly
- use a larger slice of real prompts before deciding whether analytic scoring is enough or whether a learned predictor is needed

### Data collected

- `8` prompts from `HuggingFaceH4/mt_bench_prompts`
- `8` prompts from `OpenAssistant/oasst1`
- total: `16` real prompt traces

Collected trace examples:
- `results/dataset_head_traces/mt_bench_8/mt_bench_writing_0.json`
- `results/dataset_head_traces/oasst1_8/oasst1_0.json`

### Pressure traces built from the 16 collected traces

- `recent_threshold_round_robin_interleave`
  - steps: `192`
  - layers: `12`
  - min required capacity: `27`
- `round_robin_interleave`
  - steps: `192`
  - layers: `12`
  - min required capacity: `60`

Main benchmark used for scorer decision:
- `results/dataset_head_traces/pressure_16/recent_threshold_round_robin_interleave.json`

### Main benchmark results

score:
- total miss `4155`
- mean stall `2.9995`
- p95 stall `3.5100`
- prefetch `74`
- evictions `4197`

bandit + normalized:
- total miss `4155`
- mean stall `2.9517`
- p95 stall `3.5100`
- prefetch `24`
- evictions `4147`

bandit + layer_normalized:
- total miss `4155`
- mean stall `2.9883`
- p95 stall `3.5100`
- prefetch `67`
- evictions `4190`

bandit + recomputed:
- total miss `4155`
- mean stall `2.9700`
- p95 stall `3.5100`
- prefetch `43`
- evictions `4166`

### Diagnostics on the larger dataset-backed main benchmark

normalized:
- top1/top2/top4 next-step hit rate: `1.0000`
- top8 next-step hit rate: `0.9986`
- mean reused-page rank: `0.4132`

layer_normalized:
- top1/top2/top4 next-step hit rate: `1.0000`
- top8 next-step hit rate: `0.9950`
- mean reused-page rank: `0.4095`

### Interpretation

- the larger public-dataset validation makes the scorer decision clearer than before
- on control performance, `normalized` is the strongest replay-time scorer on the main benchmark:
  - same misses as `score`
  - lower mean stall than `layer_normalized`
  - far fewer prefetches and evictions than `layer_normalized`
- `layer_normalized` still looks slightly better on reused-page-rank diagnostics, but that advantage is small and does not translate into better controller outcomes on this benchmark
- diagnostics remain somewhat saturated, so they are useful as supporting evidence but not strong enough to override the control results

### Decision

- keep the analytic scorer path
- do not introduce a learned predictor yet
- adopt `normalized` as the current replay-time default scorer view
- treat `layer_normalized` as a strong alternative for future comparison, not the default

### Why no learned predictor yet

- real per-head data collection is working
- replay-time evaluation on public datasets is working
- the analytic scorer family is still competitive and interpretable
- scorer gaps are present but not large enough yet to justify the complexity of a learned predictor

### Next head-weighted-scoring-plan work

- one more targeted validation round only if needed for confidence
- otherwise close out scorer-selection and move on to the next feature phase

## 2026-03-31 - Converting The Repo From "Benchmark Harness" To Page-Wise Simulator

### Why this change was needed

- feedback from Dr. Shi was fair in an important way:
  - the repo already had a real simulator core
  - but most outputs still looked like benchmark summaries
  - it was missing the page-wise / tile-wise statistics layer that makes a simulator convincing and useful for algorithm study

### What was already true before this change

- the simulator already operated on explicit page identities
- `WorkloadStep` already carried page-level required and predicted sets
- cache state already tracked page residency and HBM slots
- the scheduler already modeled page transfers and overlap

### What was missing

- explicit exported page-wise statistics after a run
- explicit exported tile-wise aggregations
- direct visibility into:
  - which pages missed
  - which pages were prefetched
  - which prefetched pages were later useful vs wasted
  - which pages were evicted
  - how long pages stayed resident
  - per-page reuse-distance statistics

### Implemented in this update

- `StepMetrics` now records:
  - `demand_miss_pages`
  - `prefetched_pages`
  - `evicted_pages`
  - `accessed_pages`
  - `resident_pages_end`
- `OverlapAwareSimulator` now fills those fields every step
- new `stats.py` module now builds:
  - per-page summaries
  - per-layer summaries
  - per-tile summaries
- `run_kv_controller_sim.py` now supports:
  - `--page-stats-csv`
  - `--layer-stats-csv`
  - `--tile-stats-csv`
  - `--tile-size-pages`

Example smoke-test command:

`python scripts/run_kv_controller_sim.py --policy score --steps 8 --page-stats-csv /tmp/page_stats.csv --layer-stats-csv /tmp/layer_stats.csv --tile-stats-csv /tmp/tile_stats.csv`

### Example exported page-level columns

- `access_count`
- `demand_miss_count`
- `prefetch_submit_count`
- `prefetch_hit_count`
- `prefetch_wasted_count`
- `eviction_count`
- `resident_steps`
- `first_access_step`
- `last_access_step`
- `mean_reuse_distance`
- `short_reuse_count`
- `long_reuse_count`

### Example exported tile-level columns

- `layer_id`
- `tile_id`
- `tile_size_pages`
- aggregated access / miss / prefetch / eviction / reuse counts

### Validation

- `pytest tests/test_kv_controller_core.py`
  - result: `30 passed`
- `python -m py_compile kv_controller/*.py scripts/run_kv_controller_sim.py`
  - passed

### Interpretation

- the repo was not "just a benchmark system", but the feedback identified a real gap in observability
- after this update, the simulator now exposes the page-wise and tile-wise statistics needed to study page- or tile-level algorithms directly
- this makes it much easier to validate whether a policy is practical before changing prediction strategies

### Next recommended simulator work

- add tile-aware replay transforms or tile-aware policy baselines if needed
- use the new page/tile CSVs to validate whether the current replay workloads produce realistic hot/cold and reuse patterns

## 2026-04-13 - Closing The Head-Scoring Track And Starting Reuse-Aware Scoring

### Goal of this round

- finish the head-weighted scoring track cleanly enough to stop debating scorer basics
- add a page/tile realism gate so replay workloads can be judged as useful or misleading
- start the next novelty phase with reuse-distance-aware scoring

### What was implemented

- new reuse-distance feature attachment in `scoring.py`
  - `attach_reuse_distance_features(trace)`
  - adds:
    - `reuse_distance`
    - `reuse_distance_inverse`
    - `reuse_distance_short`
    - `reuse_distance_long`
    - `reuse_distance_mavg`
- new scorer:
  - `ReuseDistanceHybridScorer`
  - formula:
    - `score = alpha * normalized_head_score + beta * inverse_reuse_distance`
- new realism summary in `stats.py`
  - `summarize_realism_metrics(...)`
  - reports:
    - page miss rate
    - prefetch hit / waste rate
    - evictions per access
    - mean resident steps per page
    - mean page reuse distance
    - short-reuse fraction
    - hot-page access share
    - hot-tile access share
    - layer access share max
- new script:
  - `scripts/validate_replay_realism.py`
  - purpose:
    - validate whether a replay workload is actually producing meaningful page-level structure
    - compare `normalized`, `layer_normalized`, and `reuse_hybrid` under the same replay trace
- updated:
  - `scripts/evaluate_replay_scores.py`
  - `scripts/diagnose_replay_scores.py`
  so reuse-distance features are attached before scorer comparison

### Validation

- `pytest tests/test_kv_controller_core.py`
  - result: `33 passed`
- `python -m py_compile kv_controller/*.py scripts/evaluate_replay_scores.py scripts/diagnose_replay_scores.py scripts/validate_replay_realism.py`
  - passed

### Smoke-test command

`python scripts/validate_replay_realism.py --trace-json results/dataset_head_traces/pressure_16/recent_threshold_round_robin_interleave.json --hbm-capacity-pages 32 --policy score --scorers normalized,layer_normalized,reuse_hybrid`

### Smoke-test realism output on the current main benchmark

- `unique_pages_touched`: `480`
- `unique_tiles_touched`: `204`
- `page_miss_rate`: `1.0000`
- `prefetch_hit_rate`: `0.0000`
- `prefetch_waste_rate`: `1.0000`
- `evictions_per_access`: `1.0101`
- `mean_page_reuse_distance`: `16.9806`
- `hot_page_access_share_top10pct`: `0.1386`
- `hot_tile_access_share_top10pct`: `0.1215`

### Interpretation

- the current main replay benchmark is clearly not trivial
- it is actually extremely harsh for the plain score-based policy:
  - every touched page misses
  - all prefetches are wasted under this setting
  - evictions roughly match accesses
- this is exactly the kind of evidence we needed to know whether a workload is too easy or too hard
- the realism gate is now a useful research check between infrastructure work and novelty claims

### Research-focus implication

- before claiming a new scorer or controller is better, we can now ask:
  - was it tested on a replay workload with meaningful page pressure?
  - was there real hot/cold structure at the page or tile level?
  - was the workload neither trivial nor completely degenerate?

### Current decision

- keep `normalized` as the replay-time default scorer
- treat head-scoring infrastructure as sufficiently complete for the next phase
- start reuse-distance-aware scoring as the first novelty extension

## 2026-04-18 - Reuse-Hybrid vs Normalized On Main And Hard Replay Benchmarks

### Goal

- evaluate the first reuse-distance-aware scorer against the current normalized head-score default
- test on both:
  - main benchmark: `recent_threshold_round_robin_interleave`
  - hard stress benchmark: dense `round_robin_interleave`

### Main benchmark

- trace: `results/dataset_head_traces/pressure_16/recent_threshold_round_robin_interleave.json`
- HBM capacity: `32`

Score policy:
- all scorer variants had:
  - total miss `4155`
  - mean stall `2.9995`
  - p95 stall `3.5100`
  - prefetch `74`
  - evictions `4197`

Bandit + normalized:
- total miss `4155`
- mean stall `2.9517`
- p95 stall `3.5100`
- prefetch `24`
- evictions `4147`

Bandit + reuse_hybrid:
- total miss `4155`
- mean stall `2.9728`
- p95 stall `3.5100`
- prefetch `37`
- evictions `4160`

### Main benchmark interpretation

- reuse_hybrid does not beat normalized
- both match misses, but normalized gives lower mean stall and less movement
- normalized remains the better default on the main benchmark

### Hard stress benchmark

- trace: `results/dataset_head_traces/pressure_16/round_robin_interleave.json`
- HBM capacity: `60`

Score policy:
- all scorer variants had:
  - total miss `5364`
  - mean stall `4.0416`
  - p95 stall `6.4800`
  - prefetch `384`
  - evictions `5688`

Bandit + normalized:
- total miss `5405`
- mean stall `4.0359`
- p95 stall `6.4800`
- prefetch `345`
- evictions `5690`

Bandit + reuse_hybrid:
- total miss `5481`
- mean stall `4.0359`
- p95 stall `6.4800`
- prefetch `269`
- evictions `5690`

### Hard benchmark interpretation

- reuse_hybrid is worse than normalized on misses
- it reaches the same mean stall only by being more conservative and prefetching less
- this confirms that the current reuse-distance formula is not yet an improvement over normalized head score

### Score diagnostics

- normalized:
  - mean reused-page rank `0.4132`
- reuse_hybrid:
  - mean reused-page rank `0.4214`
- lower is better, so diagnostics also favor normalized

### Decision

- keep `normalized` as the default scorer
- keep reuse_hybrid as an experimental baseline, not the default
- do not claim reuse-distance-aware scoring is beneficial yet

### Next reuse-distance work

- the current reuse feature is too weak for these traces because the benchmark has long reuse distances and no short-reuse signal
- next version should use:
  - reuse-distance buckets
  - page hotness / frequency
  - request-aware reuse rather than only global last-access distance

## 2026-04-18 - Richer Page-Stats Hybrid Scorer And vLLM Trace-Capture Start

### Goal

- improve the reuse-aware scorer beyond raw inverse reuse distance
- test whether richer causal page statistics help on the main and hard replay benchmarks
- begin vLLM trace-capture work without claiming live integration

### New scorer

- `PageStatsHybridScorer`
- formula combines:
  - normalized head score
  - request-aware inverse reuse distance
  - recent global page frequency
  - recent per-request page frequency
  - recent tile hotness
- all features are causal and computed from previous accesses only

### New features attached by `attach_reuse_distance_features(...)`

- `request_reuse_distance`
- `request_reuse_distance_inverse`
- `request_reuse_distance_short`
- `request_reuse_distance_long`
- `page_recent_frequency`
- `request_page_recent_frequency`
- `tile_recent_frequency`
- `tile_id`

### Main benchmark

- trace: `results/dataset_head_traces/pressure_16/recent_threshold_round_robin_interleave.json`
- HBM capacity: `32`

Bandit + normalized:
- total miss `4155`
- mean stall `2.9517`
- p95 stall `3.5100`
- prefetch `24`
- evictions `4147`

Bandit + page_stats_hybrid:
- total miss `4155`
- mean stall `2.9770`
- p95 stall `3.5100`
- prefetch `46`
- evictions `4169`

### Main benchmark interpretation

- page_stats_hybrid does not beat normalized on the main benchmark
- it preserves misses but increases movement and mean stall
- normalized remains the best default for this benchmark

### Hard stress benchmark

- trace: `results/dataset_head_traces/pressure_16/round_robin_interleave.json`
- HBM capacity: `60`

Bandit + normalized:
- total miss `5405`
- mean stall `4.0359`
- p95 stall `6.4800`
- prefetch `345`
- evictions `5690`

Bandit + page_stats_hybrid:
- total miss `5180`
- mean stall `3.8827`
- p95 stall `6.4800`
- prefetch `568`
- evictions `5688`

### Hard benchmark interpretation

- page_stats_hybrid is the first reuse-aware scorer that clearly helps in a hard regime
- it reduces misses by `225` vs normalized and improves mean stall by about `0.1532`
- the tradeoff is much more aggressive prefetching
- this is a promising novelty signal, but not yet a universal default

### Page/tile realism checks

- main benchmark under score policy:
  - page miss rate `1.0000`
  - prefetch hit rate `0.0000`
  - prefetch waste rate `1.0000`
  - evictions per access `1.0101`
- hard benchmark under score policy:
  - page miss rate `0.9332`
  - prefetch hit rate `1.0000`
  - prefetch waste rate `0.0000`
  - evictions per access `0.9896`

### Interpretation of realism checks

- the main benchmark is extremely harsh and prefetch-hostile for the score policy
- the hard benchmark has more useful prefetch structure and is where page_stats_hybrid helps
- this supports the research point that different strategies work best under different page-pressure regimes

### vLLM trace-capture start

- added `VLLM_TRACE_CAPTURE_PLAN.md`
- added `scripts/probe_vllm_trace_points.py`
- current local environment result:
  - `vllm` is not installed
  - probe reports missing modules cleanly

### vLLM research notes

- current vLLM v1 docs describe KV cache around `KVCacheBlock`, `KVCacheManager`, `BlockTable`, and `MultiGroupBlockTable`
- relevant public concepts include:
  - `block_id`
  - `block_hash`
  - `ref_cnt`
  - request-to-block mappings
  - block tables and slot mappings
  - possible difference between allocation block size and kernel block size

### Next vLLM step

- install or use an environment with vLLM
- run `python scripts/probe_vllm_trace_points.py`
- then add shadow logging around block allocation, block-table commit, and block free/evict events

## Unified Controller Design And Comparison

Why this was necessary:
- the actual end goal is one integrated controller inside a `vLLM`-like pipeline
- not three separate permanent systems

Design doc:
- `kv_controller/UNIFIED_CONTROLLER_DESIGN.md`

Implemented controller options:
- `UnifiedRuleController`
- `UnifiedBlendController`
- `UnifiedBanditController`

All three use the same controller interface:
- one `decide(context) -> PolicyOutput`
- one page-priority path
- one eviction/prefetch decision point

### Three-Regime Unified-Controller Comparison

Main benchmark (`recent_threshold_round_robin_interleave`, capacity `32`)

| policy | total_miss | mean_stall | prefetch | evictions |
| --- | ---: | ---: | ---: | ---: |
| `score` | 2181 | 3.0505 | 56 | 2213 |
| `bandit` | 2181 | 3.0073 | 25 | 2181 |
| `unified_rule` | 2181 | 3.0505 | 58 | 2215 |
| `unified_blend` | 2181 | 3.0505 | 58 | 2215 |
| `unified_bandit` | 2181 | 3.0424 | 39 | 2196 |

Middle benchmark (`recent_topk_round_robin_interleave`, capacity `24`)

| policy | total_miss | mean_stall | prefetch | evictions |
| --- | ---: | ---: | ---: | ---: |
| `score` | 2150 | 2.9889 | 47 | 2173 |
| `bandit` | 2150 | 2.9619 | 30 | 2156 |
| `unified_rule` | 2150 | 2.9889 | 49 | 2175 |
| `unified_blend` | 2150 | 2.9889 | 49 | 2175 |
| `unified_bandit` | 2150 | 2.9889 | 31 | 2157 |

Hard benchmark (`round_robin_interleave`, capacity `84`)

| policy | total_miss | mean_stall | prefetch | evictions |
| --- | ---: | ---: | ---: | ---: |
| `score` | 3258 | 4.6494 | 186 | 3366 |
| `bandit` | 3295 | 4.6035 | 149 | 3366 |
| `unified_rule` | 3072 | 4.3983 | 372 | 3366 |
| `unified_blend` | 3072 | 4.3983 | 372 | 3366 |
| `unified_bandit` | 3090 | 4.4145 | 354 | 3365 |

### Interpretation

- `UnifiedRuleController` and `UnifiedBlendController` are now the strongest
  candidates for the final one-controller direction
- they beat both `score` and the old contextual `bandit` by a large margin on
  the hard benchmark
- they do not yet improve the main or middle regimes
- the improved `UnifiedBanditController` is now much stronger in the hard
  regime and nearly matches the rule/blend controllers there

### Current Bottom Line

- the head-scoring idea is working
- normalized head score is still strong in the hostile and middle regimes
- we do not need a trained predictor yet for head scoring
- reuse-aware scoring is working in the hard regime
- we do not yet need training data for reuse either
- the next bottleneck is controller/selector design, not lack of labels

## Adaptive Unified-Controller Update

### What Was Wrong Before

The first `UnifiedBanditController` was too static in practice:
- weak feature set
- no delayed credit assignment
- no meaningful warm start
- too little action diversity

Result:
- it was too conservative and much worse than the rule/blend unified
  controllers

### What Changed

- reused delayed-credit assignment from the older contextual bandit
- added previous-step feedback features:
  - previous stall
  - previous misses
  - previous evictions
- expanded the action space across:
  - scorer mode
  - prefetch depth
- added a heuristic prior / warm-start action from the current regime cues

### New Unified-Bandit Results

Main benchmark:
- old unified-bandit:
  - mean stall `3.0424`
- improved unified-bandit:
  - mean stall `3.0505`

Middle benchmark:
- old unified-bandit:
  - mean stall `2.9889`
- improved unified-bandit:
  - mean stall `2.9889`

Hard benchmark:
- old unified-bandit:
  - total miss `3349`
  - mean stall `4.6494`
  - prefetch `95`
- improved unified-bandit:
  - total miss `3090`
  - mean stall `4.4145`
  - prefetch `354`

### Interpretation

- the adaptive unified controller is now genuinely competitive in the hard
  regime
- it still has a generalization problem on the main and middle regimes
- but it is now much closer to the final one-controller story than before
