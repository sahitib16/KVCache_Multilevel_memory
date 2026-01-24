# Multi-Level KV Cache Experiments (FlashInfer-based)

This repository contains a set of controlled experiments exploring **KV cache management across multi-level memory** (GPU HBM + CPU memory) for decode-time attention. The work is motivated by the observation that, while modern attention kernels are highly optimized once KV is in GPU memory, **KV movement and placement decisions dominate latency when KV exceeds HBM capacity**.

The experiments build progressively from simple baselines to harder, more realistic workloads, and are designed to support future work on **adaptive KV placement and prefetch policies**.

---

## Project Structure
kv_multilevel/
├── scripts/
│   ├── baseline_decode_hbm.py
│   ├── two_tier_decode_step.py
│   ├── two_tier_decode_loop.py
│   ├── two_tier_decode_irregular.py
│   ├── irregular_lru_sweep.py
│   ├── irregular_lru_log.py
│   └── summarize_step_logs.py
├── results/              # generated logs, CSVs, summaries (gitignored)
├── README.md
├── .gitignore
└── requirements.txt

---

## Scripts Overview

### `baseline_decode_hbm.py`
Establishes a **gold reference** for decode-time attention when all KV pages reside in GPU HBM.

- Uses FlashInfer’s `BatchDecodeWithPagedKVCacheWrapper`
- KV is fully resident in GPU memory
- Measures stable kernel latency (~tens of microseconds)
- Serves as the lower bound for performance

---

### `two_tier_decode_step.py`
Introduces a **two-tier KV setup** with:
- CPU pinned memory holding all KV pages
- A limited-size GPU HBM cache holding a subset of pages

Demonstrates that once KV is not resident in HBM, **CPU→GPU transfer dominates runtime**, while kernel time remains negligible.

---

### `two_tier_decode_loop.py`
Extends the two-tier setup to a **multi-step decode loop** with a sliding window access pattern.

- Each step requires a predictable window of KV pages
- Shows that naive prefetch can trivially eliminate on-demand misses in easy workloads
- Serves as a sanity check before introducing harder access patterns

---

### `two_tier_decode_irregular.py`
Defines a **harder access pattern**:
- Local sliding window + sparse random KV page requests
- Small HBM cache to force eviction and cache pressure

Used to demonstrate that naive prefetch can **fail badly** under irregular access.

---

### `irregular_lru_sweep.py`
Core experiment script for the current project phase.

- Replaces round-robin eviction with **LRU eviction**
- Separates **on-demand** vs **prefetch** transfer accounting
- Sweeps prefetch sizes `k ∈ {0, 4, 8, 16}`
- Outputs a clean summary table suitable for a systems paper

Key result: small prefetch (k=4) reduces on-demand misses and slightly improves total transfer time, while larger prefetch shows diminishing returns.

---

### `irregular_lru_log.py`
Logs **step-level features** for interpretability:
- on-demand misses
- prefetch misses
- evictions
- transfer times

Produces CSV files used to explain *why* certain prefetch values help or hurt.

---

### `summarize_step_logs.py`
Analyzes step-level CSV logs and reports:
- mean and p95 on-demand misses
- eviction behavior
- fraction of “bad” steps
- correlation between prefetch activity and evictions

Used to identify regimes where static heuristics are unstable and motivate adaptive policies.

---

## Current Findings (Summary)

- KV layout changes inside GPU HBM provide minimal headroom once KV is resident.
- In multi-level memory setups, **KV movement dominates latency**.
- Prefetch has a **non-trivial tradeoff**:
  - small prefetch reduces blocking misses
  - aggressive prefetch causes cache churn and wasted transfer
- There exists a narrow sweet spot that depends on cache capacity and access pattern.

This motivates adaptive policies rather than fixed heuristics.

---

## Planned Next Steps

- Implement a **rule-based adaptive prefetch policy** using step-level signals (misses, evictions).
- Replace the rule with a **lightweight contextual bandit** over a small action set.
- Integrate **QUEST-style query-aware KV page selection** to replace random sparse access with a realistic long-context mechanism (https://arxiv.org/abs/2406.10774).

---

## Environment Notes

- Python environment: custom venv with PyTorch + FlashInfer
- GPU: NVIDIA L40S (HBM experiments assume constrained cache)
- CPU memory uses pinned tensors for realistic transfer behavior

