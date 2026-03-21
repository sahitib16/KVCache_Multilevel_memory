# Overlap-Aware Multi-Tier KV Controller Plan

## Summary
The repo already proves the core motivation and has a usable experimental base:
- `baseline_decode_hbm.py` establishes the HBM-only lower bound.
- `two_tier_decode_*` proves CPU→GPU transfer dominates once KV spills.
- `irregular_lru_*` adds LRU, fixed prefetch sweeps, step logging, and some query-aware sparse-page experiments.
- `kv_smoketests.py` is the best current “benchmark-style” harness, with fixed traces and tail metrics.

The main gap is architectural: the project is still a set of duplicated scripts with forced `torch.cuda.synchronize()` behavior, not a reusable engine-level controller with explicit overlap, pluggable policies, oracle bounds, or layer-aware state.

Current project status:
- Steps 1-3 are now implemented in the new `kv_controller/` package.
- The simulator core exists, has explicit page-level interfaces, and is covered by automated tests.
- Static baseline controllers and future-aware oracle controllers now exist.
- The first adaptive / learning-based controller has **not** been implemented yet. That begins in Step 4.

## Implementation Changes
1. Consolidate the simulator into a reusable core package.
- Create one controller-oriented module for cache state, workload traces, transfer scheduling, metrics, and policy APIs.
- Keep existing scripts as thin entrypoints or retire them once parity is reached.
- Treat `external/Quest` as vendored dependency only; do not build new logic inside it.

Status:
- Done. The reusable core now lives under `kv_controller/`.

2. Define the engine-level simulator interfaces around pages, not one-off scripts.
- `WorkloadStep`: required pages for step `t`, optional predicted pages for `t+1`, optional per-page score features, optional layer IDs.
- `CacheState`: HBM residency, CPU residency, slot map, per-layer occupancy, last access, churn counters.
- `TransferState`: queued copies, in-flight copies, copy completion times/events, backlog, overlap budget.
- `PolicyOutput`: eviction set, prefetch set, optional per-layer budget updates, optional CPU compression choice.
- Keep HBM format uniform and require all pages needed for a decode step to be resident before launch.

Status:
- Done. These interfaces are now explicit in the new types/state/simulator layers.

3. Replace synchronous transfer accounting with an explicit overlap model.
- Stop using `torch.cuda.synchronize()` inside every page load as the primary timing path.
- Model copies on a copy stream and decode on a compute stream with per-step readiness checks.
- Support configurable limits for in-flight transfers, transfer bandwidth, launch overhead, and compute overlap.
- For the first implementation, a discrete-event simulator is enough; real CUDA streams/events can be added later behind the same interface.
- Report `stall_ms`, `overlapped_ms`, `copy_busy_ms`, `compute_busy_ms`, and queue delay separately.

Status:
- Partially done. A discrete-event overlap-aware scheduler exists and reports the main timing fields.
- Future refinement is still needed so `stall_ms` becomes more discriminative across stronger policies.

4. Rebuild baselines in the new harness.
- Baseline A: HBM-only lower bound.
- Baseline B: demand-fetch only with LRU eviction.
- Baseline C: LRU + fixed next-window prefetch.
- Baseline D: query-aware sparse-page prefetch using current QUEST-style selector path.
- Use one fixed-trace runner so all policies see identical workloads.

Status:
- Partially done. The new harness now has baseline controllers including LRU and score-based behavior.
- The old script baselines still exist and remain useful as historical reference points.
- HBM-only lower-bound parity and QUEST-style replay should still be unified into the new runner later.

5. Add oracle upper bounds before more controller complexity.
- Implement Belady-optimal eviction for the known future trace.
- Implement perfect prefetch with future knowledge under the same HBM capacity and in-flight transfer limits.
- Compare each baseline to both oracles on mean stall and p95/p99 stall.
- Gate the rest of the project on this result: continue only if the oracle gap over best static baseline is material.

Status:
- Mostly done at the policy level. Belady-style and perfect-prefetch oracle controllers now exist.
- Still pending: richer benchmark output so oracle-vs-baseline comparisons are reported systematically, not just through ad hoc smoke summaries.

6. Formalize page scoring as a pluggable signal, not a hard-coded selector.
- Generalize current QUEST/min-max/repo-eval code into a `PageScorer` interface.
- Support:
  - `recency_score`
  - `reuse_distance_proxy`
  - `quest_minmax_score`
  - `quest_repo_eval_score`
  - required head-weighted score `sum_h w[layer,h] * activity(layer,h,page)`
- Use scores only to rank eviction/prefetch priority; do not change page format or kernel behavior.

Status:
- Partially done. Head-weighted scoring is now required, not optional.
- A pluggable scorer layer exists, with passthrough and normalized head-weighted scorers.
- QUEST/min-max/repo-eval integration into the new scorer layer is still pending.

7. Add layer-aware budgets in the simulator.
- Extend page identity to `(layer_id, page_id)` rather than a single flat page index.
- Start with synthetic multi-layer traces if real per-layer traces are not yet available.
- Implement a budget manager that assigns HBM capacity by layer or layer group and can adjust shares every `N` steps.
- Add at least two policies:
  - static equal budget
  - adaptive budget based on recent miss rate / churn / score mass

Status:
- Foundation done. Page identity is already `(layer_id, page_id)`, synthetic multi-layer traces exist, and layer budgets are part of the interfaces.
- Policy logic for actively adjusting budgets is still pending.

8. Add the lightweight contextual bandit only after the static policy space is stable.
- Context:
  - recent miss rate
  - recent stall time
  - churn
  - HBM occupancy
  - per-layer pressure
  - transfer backlog
  - in-flight transfers
- Action space:
  - eviction rule `{LRU, score-based}`
  - prefetch depth `{0, 2, 4, 8}`
  - layer budgeting `{off, on}`
  - compression `{off, on}` if compression exists
- Reward:
  - `-stall_ms - alpha*wasted_prefetch_bytes - beta*churn`
- Start with epsilon-greedy LinUCB or a simple contextual Thompson variant; do not use heavy RL.

Status:
- Not started yet. This is the first step where adaptive / learning-based behavior will be implemented.

Implementation note for Step 4:
- The bandit should sit above the existing static policy building blocks rather than replacing the simulator.
- The first version should choose among a small action set like:
  - eviction rule `{LRU, score-based}`
  - prefetch depth `{0, 2, 4, 8}`
  - layer budgeting `{off, on}`
  - compression `{off, on}` once compression exists

9. Treat CPU-tier compression as optional and isolated.
- Add it only after the non-compressed controller is working.
- Compression policy applies only to CPU-resident cold pages.
- Model:
  - reduced transfer bytes
  - extra dequant/decompress latency before HBM residency
- Keep decompressed HBM pages in the same format as all other pages.

10. Unify experiment drivers and outputs.
- One CLI for:
  - workload type
  - capacity
  - overlap settings
  - policy choice
  - scorer choice
  - budget mode
  - oracle mode
  - seed
- Write one step-level CSV schema and one summary CSV schema for all runs.
- Include `mean`, `p95`, `p99` for stall, miss rate, backlog, churn, wasted prefetch, overlap efficiency, and per-layer occupancy skew.

Status:
- In progress. There is now a simple simulator driver script and automated tests.
- A full benchmark/CSV runner for systematic policy comparison is still pending.

## Test Plan
- Unit tests for cache invariants: slot map consistency, no duplicate residency, eviction correctness, and protected needed pages not evicted within a step.
- Unit tests for transfer scheduler: copy completion ordering, in-flight cap enforcement, readiness gating before decode launch, and correct stall accounting.
- Unit tests for Belady oracle on small hand-built traces where the optimal answer is known.
- Unit tests for scorer adapters: deterministic top-k page ranking and stable exclusion of already-local-window pages.
- Integration tests that reproduce current qualitative findings:
  - predictable sliding window: fixed prefetch nearly removes on-demand misses
  - irregular workload: aggressive static prefetch can increase churn and wasted transfer
  - query-jitter workload: prediction-sensitive prefetch loses value
- Regression tests that compare new harness outputs against current script-level results within a tolerance, especially for HBM-only kernel time and demand-fetch transfer dominance.
- Step 4 adaptive-controller validation should happen in four layers:
  - simulator-only evaluation on fixed traces for reward, regret, mean stall, and tail stall
  - vLLM trace replay, where real decode traces are captured from vLLM and replayed offline through this controller interface
  - vLLM shadow mode, where the adaptive controller runs alongside the existing engine policy and logs what it would have done without changing live behavior
  - gated live mode, where controller decisions are enabled only for a limited slice of traffic with fallback to the default policy
- Acceptance criterion for the project:
  - best dynamic controller beats best static baseline on mean stall and p95/p99 stall on at least one irregular spill regime
  - oracle gap is reported so the improvement is contextualized
  - no kernel changes are required

## Assumptions And Defaults
- `kv_smoketests.py` should be treated as the seed of the new benchmark harness because it already supports fixed traces and summary CSVs.
- The many `*_commented.py` files are documentation artifacts, not primary implementation targets.
- Current scripts in `scripts/` are the active project code; `external/Quest` is third-party and should stay vendored.
- Near-term novelty should come from controller design, overlap-aware scheduling, oracle grounding, and layer-aware budgeting, not from new sparse-attention math.
- If real per-layer/model traces are unavailable initially, synthetic multi-layer traces are acceptable for building and validating the controller architecture before integrating with a paged-KV engine.
- For vLLM integration, the safest path is:
  - first validate adaptive behavior in the simulator
  - then validate interface compatibility with vLLM using trace replay
  - then use shadow mode before allowing live control decisions
