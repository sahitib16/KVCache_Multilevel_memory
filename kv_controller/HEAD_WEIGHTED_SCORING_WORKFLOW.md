# Head-Weighted Scoring Workflow

This guide is the practical companion to
[HEAD_WEIGHTED_SCORING_PLAN.md](/home/sbulusu31/kv_multilevel/kv_controller/HEAD_WEIGHTED_SCORING_PLAN.md).

Use it when you want to:
- collect real head-summary traces from a small model
- replay them through the simulator
- compare scorer variants and controller behavior

## What This Workflow Does

The current real-data path is:

1. run a small real model through `transformers`
2. extract decode-time attention summaries
3. convert those summaries into replayable page/block traces
4. run the simulator on those traces with different scorer views

This is a framework-level collection path:
- real model
- real attention tensors
- replayed through our controller

It is not yet kernel-level or vLLM-integrated.

## Core Commands

### 1. Run the test suite

```bash
pytest tests/test_kv_controller_core.py
```

### 2. Collect a small real trace

```bash
python scripts/collect_real_head_traces.py \
  --model gpt2 \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --max-new-tokens 8 \
  --kv-block-size-tokens 16 \
  --output-json /tmp/real_head_trace_01.json
```

### 3. Replay that real trace

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json /tmp/real_head_trace_01.json \
  --hbm-capacity-pages 32 \
  --policies score,bandit
```

## More Stressful Trace Variants

### Longer prompt

```bash
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
```

### Smaller KV block size

```bash
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
```

### More decode steps

```bash
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

## Important Notes

### File names must match

If the collector writes:

```bash
--output-json /tmp/real_head_trace_01.json
```

then the evaluator must use:

```bash
--trace-json /tmp/real_head_trace_01.json
```

Using `/tmp/real_head_trace.json` instead will fail with `FileNotFoundError`.

### HBM capacity matters

Real traces can require more pages than the synthetic defaults.

The evaluator now infers the minimum required working set from the trace and
will raise a clear error if capacity is too small.

Rule of thumb for early GPT-2 runs:
- `32` pages for short traces
- `48` pages for longer prompts
- `64` pages for smaller blocks or longer decode runs

## How To Read The Output

If you see:
- `evictions = 0`
- very small `prefetch`
- identical `score` and `bandit` results

that usually means the replay trace is too easy under the chosen HBM capacity.

In that case, the run is still useful because it validates:
- the real-data collector
- the replay format
- the head-summary scoring pipeline

But it is not yet a strong controller benchmark.

## Current Interpretation Of The First Real Runs

The first GPT-2 traces show:
- real head-summary collection works
- replay works
- scorer variants can be applied to real traces

But they also show:
- too little eviction pressure
- too little prefetch pressure
- too little separation between controller policies

So the next step is not just "collect more traces." It is:
- collect more traces
- replay them under tighter, pressure-inducing capacities
- automate the experiment matrix so we stop doing one-off runs manually

## Recommended Next Automation

The next useful tool to build is a batch harness that:
- collects multiple real traces automatically
- replays each trace under several HBM capacities
- writes one summary table across:
  - model
  - prompt type
  - decode length
  - block size
  - capacity
  - scorer
  - policy

That will give us a much better basis for the next head-importance phase.

## Automated Batch Harness

The repo now includes:

[`batch_real_head_replay.py`](/home/sbulusu31/kv_multilevel/scripts/batch_real_head_replay.py)

This script:
- collects multiple real traces automatically
- saves the replay JSON files into one output directory
- infers the minimum feasible HBM capacity for each trace
- replays each trace under several capacities
- compares scorer and policy combinations in one CSV summary

### Default batch run

```bash
python scripts/batch_real_head_replay.py
```

### Recommended first run

```bash
python scripts/batch_real_head_replay.py \
  --output-dir results/real_head_batch \
  --summary-csv results/real_head_batch/summary.csv
```

### Run with tighter pressure settings

```bash
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
```

### Run with a broader capacity sweep

```bash
python scripts/batch_real_head_replay.py \
  --capacities min,min+2,min+4,min*1.25 \
  --output-dir results/real_head_capacity_sweep \
  --summary-csv results/real_head_capacity_sweep/summary.csv
```

## Pressure-Inducing Replay Formulations

The repo now also includes:

[`build_pressure_replays.py`](/home/sbulusu31/kv_multilevel/scripts/build_pressure_replays.py)

This script builds three harder replay formulations from the collected real traces:

1. `recent_topk`
- always require the most recent block(s)
- add top-k older attended blocks per layer

2. `recent_threshold`
- always require the most recent block(s)
- add older blocks until a chosen fraction of score mass is covered

3. `round_robin_interleave`
- interleave multiple requests so they compete for the same HBM space

It also writes two combined middle-ground traces:

4. `recent_topk_round_robin_interleave`
- sparsify each request with recent-topk first
- then interleave requests so they compete for HBM

5. `recent_threshold_round_robin_interleave`
- sparsify each request with recent-threshold first
- then interleave requests so they compete for HBM

### Build the three pressure traces

```bash
python scripts/build_pressure_replays.py \
  --input-traces /tmp/real_head_trace_01.json,/tmp/real_head_trace_02.json,/tmp/real_head_trace_03.json \
  --output-dir results/pressure_replays
```

The script will print the `min_required_capacity` for each transformed trace.

### Evaluate `recent_topk`

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/recent_topk.json \
  --hbm-capacity-pages 24 \
  --policies score,bandit
```

### Evaluate `recent_threshold`

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/recent_threshold.json \
  --hbm-capacity-pages 24 \
  --policies score,bandit
```

### Evaluate `round_robin_interleave`

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/round_robin_interleave.json \
  --hbm-capacity-pages 48 \
  --policies score,bandit
```

### Evaluate `recent_topk_round_robin_interleave`

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/recent_topk_round_robin_interleave.json \
  --hbm-capacity-pages 24 \
  --policies score,bandit
```

### Evaluate `recent_threshold_round_robin_interleave`

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json results/pressure_replays/recent_threshold_round_robin_interleave.json \
  --hbm-capacity-pages 24 \
  --policies score,bandit
```

### Build a harsher sparse version

```bash
python scripts/build_pressure_replays.py \
  --input-traces /tmp/real_head_trace_01.json,/tmp/real_head_trace_02.json,/tmp/real_head_trace_03.json \
  --recent-block-window 1 \
  --top-k-older-per-layer 1 \
  --score-mass-fraction 0.35 \
  --output-dir results/pressure_replays_harsh
```

## Broader Main Benchmark

The repo now also includes:

[`build_broader_real_benchmark.py`](/home/sbulusu31/kv_multilevel/scripts/build_broader_real_benchmark.py)

This script:
- collects a broader real-trace set from more prompt styles
- saves the raw traces
- builds `recent_threshold_round_robin_interleave` as the main benchmark
- also builds dense `round_robin_interleave` as the hard stress test

### Build the broader benchmark set

```bash
python scripts/build_broader_real_benchmark.py \
  --output-dir results/broader_real_benchmark
```

### Evaluate the broader main benchmark

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json results/broader_real_benchmark/benchmarks/recent_threshold_round_robin_interleave.json \
  --hbm-capacity-pages 32 \
  --policies score,bandit
```

### Evaluate the broader dense stress trace

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json results/broader_real_benchmark/benchmarks/round_robin_interleave.json \
  --hbm-capacity-pages 48 \
  --policies score,bandit
```

## Replay-Side Bandit Tuning

The repo now also includes:

[`tune_bandit_replay.py`](/home/sbulusu31/kv_multilevel/scripts/tune_bandit_replay.py)

This script sweeps bandit reward coefficients directly on replay traces instead
of synthetic traces.

### Tune on the broader main benchmark

```bash
python scripts/tune_bandit_replay.py \
  --trace-jsons results/broader_real_benchmark/benchmarks/recent_threshold_round_robin_interleave.json \
  --capacity 32 \
  --bandit-menu full
```

### Tune on both the main benchmark and the hard stress test

```bash
python scripts/tune_bandit_replay.py \
  --trace-jsons results/broader_real_benchmark/benchmarks/recent_threshold_round_robin_interleave.json,results/broader_real_benchmark/benchmarks/round_robin_interleave.json \
  --capacity min \
  --bandit-menu full
```

## Public Dataset Validation

The repo now also includes:

[`collect_dataset_head_traces.py`](/home/sbulusu31/kv_multilevel/scripts/collect_dataset_head_traces.py)

This script collects real head-summary traces from public prompt datasets.

Current dataset adapters:
- `HuggingFaceH4/mt_bench_prompts`
- `OpenAssistant/oasst1`

### Collect a small MT-Bench slice

```bash
python scripts/collect_dataset_head_traces.py \
  --dataset mt_bench \
  --split train \
  --count 4 \
  --model gpt2 \
  --max-new-tokens 12 \
  --kv-block-size-tokens 16 \
  --output-dir results/dataset_head_traces/mt_bench
```

### Collect a small OpenAssistant slice

```bash
python scripts/collect_dataset_head_traces.py \
  --dataset oasst1 \
  --split validation \
  --count 4 \
  --model gpt2 \
  --max-new-tokens 12 \
  --kv-block-size-tokens 16 \
  --output-dir results/dataset_head_traces/oasst1
```

### Build a pressure benchmark from the collected dataset traces

```bash
python scripts/build_pressure_replays.py \
  --input-traces results/dataset_head_traces/mt_bench/mt_bench_writing_0.json,results/dataset_head_traces/mt_bench/mt_bench_writing_1.json,results/dataset_head_traces/mt_bench/mt_bench_writing_2.json,results/dataset_head_traces/mt_bench/mt_bench_writing_3.json,results/dataset_head_traces/oasst1/oasst1_0.json,results/dataset_head_traces/oasst1/oasst1_1.json,results/dataset_head_traces/oasst1/oasst1_2.json,results/dataset_head_traces/oasst1/oasst1_3.json \
  --output-dir results/dataset_head_traces/pressure
```

### Evaluate the dataset-backed main benchmark

```bash
python scripts/evaluate_replay_scores.py \
  --trace-json results/dataset_head_traces/pressure/recent_threshold_round_robin_interleave.json \
  --hbm-capacity-pages 32 \
  --policies score,bandit
```

## Score Diagnostics

The repo now also includes:

[`diagnose_replay_scores.py`](/home/sbulusu31/kv_multilevel/scripts/diagnose_replay_scores.py)

This script reports:
- score distribution statistics
- top-k next-step hit rates
- reused-page rank summary

### Run diagnostics on the dataset-backed benchmarks

```bash
python scripts/diagnose_replay_scores.py \
  --trace-jsons results/dataset_head_traces/pressure/recent_threshold_round_robin_interleave.json,results/dataset_head_traces/pressure/round_robin_interleave.json
```
