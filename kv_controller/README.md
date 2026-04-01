# `kv_controller` Package Guide

This folder contains the new reusable simulator core for the project:

> Overlap-Aware Multi-Tier KV Cache Controller

The goal of this package is to move us away from one-off experiment scripts and
toward a clean engine-level simulation framework where we can plug in:
- different residency controllers
- different head-weighted scorers
- different workload traces
- different overlap / transfer models
- oracle and later bandit policies

This README is meant to be updated as the package grows. It now documents the
Step 1 core plus the Step 2 interface formalization.

For the practical real-trace workflow, use
[HEAD_WEIGHTED_SCORING_WORKFLOW.md](/home/sbulusu31/kv_multilevel/kv_controller/HEAD_WEIGHTED_SCORING_WORKFLOW.md).
That guide contains the current commands for:
- collecting real head-summary traces
- replaying them through the simulator
- interpreting common errors like missing files or undersized HBM capacity

## Big Picture

The simulator models the following idea:

1. A decode step needs a set of KV pages.
2. Some pages are already in GPU HBM.
3. Some pages are only in CPU memory.
4. The controller decides what to evict and what to prefetch.
5. Transfers take time and only a limited number can be in flight.
6. Decode is not allowed to start until the required pages are resident.
7. We record stall, overlap, misses, evictions, and churn.

This is exactly the engine-level problem your project wants to solve.

## File Guide

### [`__init__.py`](/home/sbulusu31/kv_multilevel/kv_controller/__init__.py)

Purpose:
- exposes the main public classes of the package
- lets other files import from `kv_controller` directly

What to use it for:
- importing simulator objects in future scripts or notebooks

Example:

```python
from kv_controller import OverlapAwareSimulator, SyntheticTraceGenerator
```

### [`types.py`](/home/sbulusu31/kv_multilevel/kv_controller/types.py)

Purpose:
- defines the shared data structures used everywhere else

Important classes:
- `KVPageId`: unique identifier for a page as `(layer_id, page_id)`
- `WorkloadStep`: everything needed to describe one decode step
- `ControllerContext`: what the controller sees before deciding
- `PolicyOutput`: what the controller returns
- `PolicyDecision`: compatibility alias for the older Step 1 name
- `TransferRequest`: one scheduled CPU→GPU page movement
- `TransferState`: explicit transfer subsystem state
- `StepMetrics`: what gets measured for one step
- `CacheConfig`, `TransferConfig`, `SimulationConfig`: configuration objects

Why this file matters:
- it gives the project a stable vocabulary
- every other file can rely on the same structures
- it now explicitly separates residency state from transfer state

### [`interfaces.py`](/home/sbulusu31/kv_multilevel/kv_controller/interfaces.py)

Purpose:
- defines abstract interfaces for pluggable logic

Important classes:
- `HeadWeightedScorer`
- `ResidencyController`

Why this file matters:
- the simulator should not hard-code a single controller
- the project requires head-weighted scoring, so that concept now has a formal
  interface instead of being hidden inside a script

### [`scoring.py`](/home/sbulusu31/kv_multilevel/kv_controller/scoring.py)

Purpose:
- provides concrete implementations of the required head-weighted scorer interface

Current implementations:
- `PassthroughHeadWeightedScorer`: uses the score already attached to the step
- `NormalizedHeadWeightedScorer`: rescales scores to `[0, 1]`

Why this file matters:
- it turns the scorer interface into something controllers can actually use
- it keeps “how scores are produced” separate from “how policies consume them”

### [`policies.py`](/home/sbulusu31/kv_multilevel/kv_controller/policies.py)

Purpose:
- defines actual controller behavior for Step 3

Current controllers:
- `LRUController`: baseline that leans on simulator fallback LRU eviction
- `ScoreBasedController`: uses head-weighted scores to guide prefetch and eviction
- `BeladyOracleController`: future-aware eviction oracle
- `PerfectPrefetchOracleController`: future-aware prefetch oracle
- `ContextualBanditController`: adaptive controller that learns over a small action set

Helper:
- `FutureTraceOracle`: exposes next-use information from a known trace
- `BanditAction`: interpretable action object for the contextual bandit

Why this file matters:
- this is where the simulator becomes useful for comparing controller quality
- this is also where the first adaptive / learning-based controller lives

### [`benchmark.py`](/home/sbulusu31/kv_multilevel/kv_controller/benchmark.py)

Purpose:
- provides reusable benchmarking helpers for policy comparison

What it does:
- runs multiple policies on the same fixed trace
- collects step-level rows
- builds summary rows
- writes step and summary CSV files

Why this file matters:
- it finishes an important remaining part of Step 3 by making policy
  comparison reusable outside the CLI script

### [`state.py`](/home/sbulusu31/kv_multilevel/kv_controller/state.py)

Purpose:
- stores mutable simulator state

What it tracks:
- which pages are resident in HBM
- which HBM slot each resident page occupies
- which pages are currently in flight
- which pages exist in CPU backing storage
- last access timestamps
- last eviction timestamps
- per-layer occupancy
- rolling miss rate
- churn

Why this file matters:
- it is the source of truth for “where does each page live right now?”

### [`scheduler.py`](/home/sbulusu31/kv_multilevel/kv_controller/scheduler.py)

Purpose:
- simulates copy timing and overlap

What it does:
- submits transfers
- tracks queued copies as well as active ones
- enforces max in-flight limit
- advances simulated time
- marks pages resident when transfers finish
- blocks decode until required pages are ready
- overlaps compute with background transfer completion

Why this file matters:
- this is where the project’s “overlap-aware” part begins
- it replaces the older script pattern of forcing synchronization around every
  copy when we want an engine-level model

Important note:
- this is a discrete-event timing model, not real CUDA stream code
- that is intentional for Step 1 because it keeps the controller architecture
  clean and easy to test

### [`workload.py`](/home/sbulusu31/kv_multilevel/kv_controller/workload.py)

Purpose:
- generates synthetic traces that already include head-weighted page scores

What it currently models:
- multiple layers
- a local sliding window of likely-reused pages
- sparse irregular pages outside the window
- per-head activity values
- per-layer head weights
- optional per-page feature dictionaries
- explicit referenced-layer lists per step
- optional query jitter across steps

Why this file matters:
- it gives us a controllable environment for testing policies before we wire the
  simulator into a real paged-KV engine
- it makes head-weighted scoring a first-class part of the workload, not an
  optional afterthought

### [`simulator.py`](/home/sbulusu31/kv_multilevel/kv_controller/simulator.py)

Purpose:
- runs the full step loop

Per step, it does:
1. build controller context
2. ask the controller for a decision
3. apply budget updates and evictions
4. submit prefetches
5. submit demand transfers for missing required pages
6. wait until required pages are ready
7. run simulated compute
8. emit metrics

Important invariant:
- decode launch is rejected unless every required page is resident in HBM and
  assigned to an HBM slot

Why this file matters:
- this is the main “engine shell” of the new architecture

## How To Use The Current Step 1 Core

Right now, Steps 1-4 give you infrastructure, explicit engine-level interfaces,
baseline/oracle policies, and a first adaptive controller.

Typical usage looks like this:

```python
from kv_controller import (
    CacheConfig,
    LRUController,
    OverlapAwareSimulator,
    SimulationConfig,
    SyntheticTraceConfig,
    SyntheticTraceGenerator,
    TransferConfig,
)
trace = SyntheticTraceGenerator(
    SyntheticTraceConfig(
        steps=10,
        layers=2,
        pages_per_layer=32,
        local_window_pages=4,
        sparse_pages_per_step=2,
        predicted_prefetch_pages=4,
        attention_heads=8,
        query_jitter=0.05,
        seed=0,
    )
).generate()

sim = OverlapAwareSimulator(
    SimulationConfig(
        steps=10,
        cache=CacheConfig(
            hbm_capacity_pages=24,
            layers=2,
            page_bytes=4096,
        ),
        transfer=TransferConfig(
            page_bytes=4096,
            bandwidth_bytes_per_ms=4096 * 4,
            transfer_setup_ms=0.02,
            max_inflight_transfers=2,
            decode_kernel_ms=0.05,
        ),
    )
)

metrics = sim.run(trace, LRUController(prefetch_k=2))
```

You can also use the new driver script:

```bash
python scripts/run_kv_controller_sim.py --policy lru --print-steps
python scripts/run_kv_controller_sim.py --policy score --prefetch-k 2 --print-steps
python scripts/run_kv_controller_sim.py --policy belady
python scripts/run_kv_controller_sim.py --policy perfect_prefetch --prefetch-k 2
python scripts/run_kv_controller_sim.py --policy bandit --print-steps
python scripts/run_kv_controller_sim.py --policy-suite lru,score,belady,perfect_prefetch,bandit
```

## How To Test That The Simulator Core Is Working

There are now two good ways to validate the Step 1 / Step 2 core.

### 1. Run the small driver script

File:
- [`scripts/run_kv_controller_sim.py`](/home/sbulusu31/kv_multilevel/scripts/run_kv_controller_sim.py)

Example:

```bash
python scripts/run_kv_controller_sim.py --print-steps
```

What this gives you:
- a synthetic trace
- one policy or a policy suite
- per-step metrics if you ask for them
- a final summary showing demand misses, stall, evictions, slot-map size, and
  transfer-state status
- optional step-level / summary CSV output

This is the easiest “does the simulator run end to end?” check.

### 2. Run the automated pytest suite

File:
- [`tests/test_kv_controller_core.py`](/home/sbulusu31/kv_multilevel/tests/test_kv_controller_core.py)

Example:

```bash
pytest tests/test_kv_controller_core.py
```

What these tests check:
- `WorkloadStep` exposes the Step 2 page-level interface fields
- controller context exposes slot maps and transfer-state details
- cache slot maps remain internally consistent
- transfer-state bookkeeping exists and stays sane
- required pages are resident and slotted before decode launch
- the simulator raises if that invariant is violated
- HBM format is recorded as uniform
- the old `PolicyDecision` name still works as an alias
- the contextual bandit observes reward feedback after each step

## Current Test Commands

These are the most useful commands for validating and comparing the current
implementation.

Syntax and test suite:

```bash
python -m py_compile kv_controller/*.py scripts/run_kv_controller_sim.py tests/test_kv_controller_core.py
pytest tests/test_kv_controller_core.py
```

Single-policy debug runs:

```bash
python scripts/run_kv_controller_sim.py --policy lru --print-steps
python scripts/run_kv_controller_sim.py --policy score --prefetch-k 2 --print-steps
python scripts/run_kv_controller_sim.py --policy belady
python scripts/run_kv_controller_sim.py --policy perfect_prefetch --prefetch-k 2
python scripts/run_kv_controller_sim.py --policy bandit --print-steps
```

Multi-policy comparison on one fixed trace:

```bash
python scripts/run_kv_controller_sim.py --policy-suite lru,score,belady,perfect_prefetch,bandit
```

Offline trace replay:

```bash
python scripts/run_kv_controller_sim.py --steps 8 --save-trace-json /tmp/kv_trace.json --policy lru
python scripts/run_kv_controller_sim.py --trace-json /tmp/kv_trace.json --policy lru
```

Commands that specifically show the stall-accounting fix and the updated
bandit behavior:

```bash
python scripts/run_kv_controller_sim.py --policy bandit --print-steps
python scripts/run_kv_controller_sim.py --policy-suite lru,score,belady,perfect_prefetch,bandit
python scripts/run_kv_controller_sim.py --steps 64 --policy-suite lru,score,belady,perfect_prefetch,bandit
python scripts/run_kv_controller_sim.py --steps 64 --seed-list 0,1,2,3,4,5,6,7 --policy-suite lru,score,perfect_prefetch,bandit
```

How to read those:
- the first command should now show varying `stall_ms` by step rather than one flat value everywhere
- the second command shows short-run policy comparison on the default 12-step trace
- the third command gives the bandit enough runway to learn, which is usually a fairer adaptive-controller check
- the fourth command is the stronger aggregate evaluation across multiple seeds

CSV output:

```bash
python scripts/run_kv_controller_sim.py --policy score --step-csv results/kv_steps.csv
python scripts/run_kv_controller_sim.py --policy-suite lru,score,belady,perfect_prefetch,bandit --summary-csv results/kv_summary.csv
```

## What “Head-Weighted” Means In The Current Core

The project requirement is that head-weighted scoring is not optional.

In the current synthetic implementation:

1. Every page gets a synthetic per-head activity vector.
2. Every layer gets a per-step head-weight vector.
3. The page score is:

```text
score(page) = sum over heads of weight[layer, head] * activity(page, head)
```

That score is attached to every `WorkloadStep` in `head_weighted_scores`.

This is not the final research version yet, but it creates the right
architecture:
- controllers receive page importance as a required signal
- future scoring implementations can replace the synthetic one without changing
  the simulator interface

## What Is Done vs Not Done Yet

Done in Step 1:
- reusable types
- controller interfaces
- cache state tracking
- overlap-aware scheduler
- synthetic multi-layer workload generation
- simulator step loop
- required head-weighted score plumbing

Done in Step 2:
- `WorkloadStep` now carries optional per-page feature dictionaries
- `WorkloadStep` now carries explicit referenced-layer lists
- `CacheState` now tracks page-to-slot and slot-to-page mappings
- `TransferState` now exists as an explicit transfer subsystem object
- `ControllerContext` now exposes transfer and slot-map details
- `PolicyOutput` is now the formal controller return type
- simulator enforces “all required pages must be resident before launch”

Done in Step 3:
- concrete scorer implementations now exist
- baseline controllers now exist
- future-aware oracle controllers now exist
- the driver script can now switch between baseline and oracle policies
- the driver can compare multiple policies on the same trace and write CSV outputs

Done in Step 4:
- a first adaptive / learning-based controller now exists
- the adaptive controller is a lightweight contextual bandit over a small action set
- the simulator now calls controller feedback hooks after each step
- the driver can run bandit mode and compare it against static/oracle policies
- reusable benchmark helpers now exist outside the driver script
- the bandit now has adaptive layer-budget mode
- the bandit now has adaptive transfer-pressure prefetch guarding
- the bandit uses one-step delayed reward credit so prefetch decisions are credited closer to when they pay off
- stall accounting now includes demand-transfer queue delay, so latency differences between policies are visible

Done in simulator instrumentation:
- each simulated step now records page-level events:
  - demand-miss pages
  - prefetched pages
  - evicted pages
  - accessed pages
  - resident pages at end of step
- the CLI can now export:
  - per-page CSV statistics
  - per-layer CSV statistics
  - per-tile CSV statistics
- this makes the project usable as a page-wise simulator rather than only a
  benchmark harness

## Page-Wise And Tile-Wise Statistics

To export page-wise, layer-wise, and tile-wise statistics from one run:

```bash
python scripts/run_kv_controller_sim.py \
  --policy score \
  --steps 8 \
  --page-stats-csv results/page_stats.csv \
  --layer-stats-csv results/layer_stats.csv \
  --tile-stats-csv results/tile_stats.csv \
  --tile-size-pages 4
```

The exported CSVs include signals such as:
- access count
- demand miss count
- prefetch submit / hit / wasted count
- eviction count
- resident-step count
- first / last access step
- mean reuse distance
- short vs long reuse counts

Not done yet:
- integration with your older experiment scripts
- real model-derived head-weight estimation from eager attention logs
- real vLLM trace replay / shadow-mode adapter
- CPU-tier compression control
- richer adaptive knobs beyond the current bandit action set

## How This README Should Be Updated Later

When you add a new file to `kv_controller`, update this README with:
- the file name
- its purpose
- its main classes/functions
- how it interacts with the rest of the package
- any usage example if it becomes user-facing

When you change an existing file, update the relevant section here so this
README remains the beginner-friendly map of the package.
