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
