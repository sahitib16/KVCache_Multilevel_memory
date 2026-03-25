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
