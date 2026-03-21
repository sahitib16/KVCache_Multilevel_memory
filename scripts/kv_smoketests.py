#!/usr/bin/env python3
"""
kv_smoketests.py

One script that runs 3 smoke tests for KV cache stall significance:

1) HBM-only vs CPU->GPU (pinned) transfer + decode kernel
2) Capacity sweep under irregular page accesses (misses, transfer time, kernel time)
3) Back-of-the-envelope bandwidth estimate (bytes/page, theoretical transfer)

It will try to use FlashInfer's BatchDecodeWithPagedKVCacheWrapper if available.
If FlashInfer is not installed, it falls back to a tiny CUDA "kernel proxy"
(a matmul) so you can still measure copy vs compute.

Run (example):
  python kv_smoketests.py --device cuda --total-pages 2048 --page-size 16 --head-dim 128 --kv-heads 8

Notes:
- This is a SMOKE TEST, not a benchmark harness.
- It intentionally uses synchronizations to expose worst-case stall.
"""

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch


# ----------------------------
# Optional FlashInfer support
# ----------------------------
HAS_FLASHINFER = False
flashinfer = None
FlashInferDecodeWrapper = None
FLASHINFER_IMPORT_ERROR = None
try:
    import flashinfer  # type: ignore
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper as FlashInferDecodeWrapper  # type: ignore

    HAS_FLASHINFER = True
except Exception as e:
    HAS_FLASHINFER = False
    FLASHINFER_IMPORT_ERROR = e


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_cuda(fn, warmup: int = 10, iters: int = 50) -> float:
    """Return average ms of fn() using CUDA events."""
    if not torch.cuda.is_available():
        t0 = time.time()
        for _ in range(warmup):
            fn()
        t1 = time.time()
        for _ in range(iters):
            fn()
        t2 = time.time()
        return (t2 - t1) * 1000.0 / iters

    # warmup
    for _ in range(warmup):
        fn()
    cuda_sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    cuda_sync()
    return start.elapsed_time(end) / iters


def bytes_per_page(kv_heads: int, page_size: int, head_dim: int, dtype_bytes: int) -> int:
    # 2 tensors: K and V
    return 2 * kv_heads * page_size * head_dim * dtype_bytes


def fmt_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


# ----------------------------
# Workload generators (pages touched per step)
# ----------------------------
def sliding_window_pages(step: int, window_pages: int) -> List[int]:
    # simple: pages [step, step+1, ...]
    return list(range(step, step + window_pages))


def irregular_pages(
    step: int,
    window_pages: int,
    num_sparse: int,
    total_pages: int,
    sparse_min_page: int = 0,
) -> List[int]:
    pages = set(sliding_window_pages(step, window_pages))
    # add sparse random long-range pages (can overlap)
    for _ in range(num_sparse):
        pages.add(random.randrange(sparse_min_page, total_pages))
    return sorted(pages)


def irregular_pages_rng(
    step: int,
    window_pages: int,
    num_sparse: int,
    total_pages: int,
    rng: random.Random,
    sparse_min_page: int = 0,
) -> List[int]:
    pages = set(sliding_window_pages(step, window_pages))
    for _ in range(num_sparse):
        pages.add(rng.randrange(sparse_min_page, total_pages))
    return sorted(pages)


# ----------------------------
# FlashInfer decode wrapper (batch=1)
# ----------------------------
@dataclass
class DecodeKernel:
    """Abstraction so we can swap FlashInfer vs proxy kernel."""
    name: str
    # function: (q, k_pages, v_pages, indptr, indices, last_page_len) -> out
    run: callable


def make_flashinfer_decode_kernel(
    kv_heads: int, head_dim: int, page_size: int, dtype: torch.dtype, device: torch.device
) -> DecodeKernel:
    # Keep API aligned with the rest of this repo (flashinfer.decode wrapper).
    workspace = torch.empty((256 * 1024 * 1024 // 4,), device=device, dtype=torch.float32)
    wrapper = FlashInferDecodeWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        use_cuda_graph=False,
    )

    # In FlashInfer, you call wrapper.begin_forward(...) once per configuration.
    # We'll configure each run with given page table sizes.
    # KV layout in this script is NHD: [P, S, H, D], matching wrapper expectation.
    def _run(q, k_pages, v_pages, indptr, indices, last_page_len):
        wrapper.begin_forward(
            indptr,
            indices,
            last_page_len,
            num_qo_heads=q.shape[1],
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode="NONE",
            non_blocking=True,
        )
        out = wrapper.forward(q, (k_pages, v_pages))
        wrapper.end_forward()
        return out

    return DecodeKernel(name="flashinfer.decode", run=_run)


def make_proxy_decode_kernel(device: torch.device) -> DecodeKernel:
    # A tiny compute proxy: y = (q @ k_mean) to emulate "some kernel work".
    # This is NOT attention; it just gives you a compute term to compare with copy.
    def _run(q, k_pages, v_pages, indptr, indices, last_page_len):
        # Reduce k_pages over pages/tokens to get a [B, H, D] tensor, then matmul-ish.
        # q: [B, H, D], k_pages: [P, S, H, D] (NHD)
        k_mean = k_pages.mean(dim=(0, 1))  # [H, D]
        # compute: [B, H, D] * [H, D] elementwise then sum over D
        return (q * k_mean.unsqueeze(0)).sum(dim=-1)  # [B, H]

    return DecodeKernel(name="proxy.matmul", run=_run)


# ----------------------------
# Smoke Test 1: HBM vs CPU->GPU
# ----------------------------
def smoke_test_1(
    kernel: DecodeKernel,
    total_pages: int,
    hbm_cap_pages: int,
    needed_pages: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    print("\n" + "=" * 80)
    print("SMOKE TEST #1 — HBM-only vs CPU→GPU transfer + kernel")
    print("=" * 80)

    # Allocate KV fully on GPU (HBM baseline)
    k_hbm_all = torch.randn((total_pages, page_size, kv_heads, head_dim), device=device, dtype=dtype)
    v_hbm_all = torch.randn((total_pages, page_size, kv_heads, head_dim), device=device, dtype=dtype)

    # Paged table for batch=1: we choose first `needed_pages` pages
    indices_hbm = torch.arange(0, needed_pages, device=device, dtype=torch.int32)
    indptr = torch.tensor([0, needed_pages], device=device, dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device=device, dtype=torch.int32)
    q = torch.randn((1, kv_heads, head_dim), device=device, dtype=dtype)

    def run_hbm_only():
        return kernel.run(q, k_hbm_all, v_hbm_all, indptr, indices_hbm, last_page_len)

    hbm_kernel_ms = time_cuda(run_hbm_only, warmup=20, iters=100)
    print(f"HBM-only: avg kernel+overhead = {hbm_kernel_ms:.4f} ms/step  (kernel={kernel.name})")

    # Two-tier: keep KV pool on CPU pinned, only hbm_cap_pages slots on GPU
    cpu = torch.device("cpu")
    k_cpu = torch.randn((total_pages, page_size, kv_heads, head_dim), device=cpu, dtype=dtype).pin_memory()
    v_cpu = torch.randn((total_pages, page_size, kv_heads, head_dim), device=cpu, dtype=dtype).pin_memory()
    k_hbm_cache = torch.empty((hbm_cap_pages, page_size, kv_heads, head_dim), device=device, dtype=dtype)
    v_hbm_cache = torch.empty((hbm_cap_pages, page_size, kv_heads, head_dim), device=device, dtype=dtype)

    # We'll load the first `needed_pages` pages into the first slots (0..needed_pages-1).
    # If needed_pages > hbm_cap_pages, you will see forced eviction effect (but this is test #1).
    load_pages = list(range(needed_pages))

    def do_transfer_sync():
        # copy pages into slots
        for slot, pid in enumerate(load_pages[:hbm_cap_pages]):
            k_hbm_cache[slot].copy_(k_cpu[pid], non_blocking=True)
            v_hbm_cache[slot].copy_(v_cpu[pid], non_blocking=True)
        cuda_sync()

    # Measure transfer only
    transfer_ms = time_cuda(do_transfer_sync, warmup=10, iters=50)
    print(f"CPU→GPU transfer only: avg = {transfer_ms:.4f} ms  (copied {min(needed_pages, hbm_cap_pages)} pages)")

    # Now run decode using indices that reference the HBM slots
    used = min(needed_pages, hbm_cap_pages)
    indices_slots = torch.arange(0, used, device=device, dtype=torch.int32)
    indptr_slots = torch.tensor([0, used], device=device, dtype=torch.int32)

    def run_transfer_plus_kernel():
        do_transfer_sync()
        return kernel.run(q, k_hbm_cache, v_hbm_cache, indptr_slots, indices_slots, last_page_len)

    total_ms = time_cuda(run_transfer_plus_kernel, warmup=10, iters=50)
    # Kernel component inside total is roughly total - transfer (not perfect due to sync, overhead)
    print(f"Transfer + kernel: avg total = {total_ms:.4f} ms/step")
    if total_ms > 0:
        frac = min(1.0, transfer_ms / total_ms)
        print(f"Approx stall share (transfer/total) ≈ {frac*100:.1f}%")
    print("Interpretation:")
    print("- If transfer dominates here, your multi-tier policy has real upside.")
    print("- If transfer is tiny vs kernel, policy upside is smaller on this setup.")


# ----------------------------
# Smoke Test 2: Capacity sweep under irregular access
# ----------------------------
@dataclass
class LRUCache:
    cap: int
    page_to_slot: Dict[int, int]
    slot_to_page: List[Optional[int]]
    last_used: Dict[int, int]
    rr_ptr: int = 0  # fallback

    @classmethod
    def create(cls, cap: int):
        return cls(cap=cap, page_to_slot={}, slot_to_page=[None] * cap, last_used={})

    def touch(self, page: int, step: int):
        self.last_used[page] = step

    def has(self, page: int) -> bool:
        return page in self.page_to_slot

    def get_slot(self, page: int) -> int:
        return self.page_to_slot[page]

    def choose_victim_slot(self) -> int:
        # empty slot first
        for s, p in enumerate(self.slot_to_page):
            if p is None:
                return s
        # LRU among resident pages
        victim_page = min(self.page_to_slot.keys(), key=lambda p: self.last_used.get(p, -1))
        return self.page_to_slot[victim_page]

    def evict_slot(self, slot: int) -> Optional[int]:
        old = self.slot_to_page[slot]
        if old is not None:
            self.page_to_slot.pop(old, None)
            self.last_used.pop(old, None)
        self.slot_to_page[slot] = None
        return old

    def insert(self, page: int, slot: int, step: int):
        self.page_to_slot[page] = slot
        self.slot_to_page[slot] = page
        self.last_used[page] = step

    def choose_victim_slot_avoiding(self, avoid_pages: set[int]) -> int:
        # Prefer empty slots.
        for s, p in enumerate(self.slot_to_page):
            if p is None:
                return s
        # Then prefer evicting pages not needed by current step.
        candidates = [p for p in self.page_to_slot.keys() if p not in avoid_pages]
        if candidates:
            victim_page = min(candidates, key=lambda p: self.last_used.get(p, -1))
            return self.page_to_slot[victim_page]
        # Fallback (should be rare if len(needed) <= cap).
        return self.choose_victim_slot()


def load_needed_pages(
    needed: List[int],
    step: int,
    cache: LRUCache,
    k_hbm: torch.Tensor,
    v_hbm: torch.Tensor,
    k_cpu: torch.Tensor,
    v_cpu: torch.Tensor,
) -> int:
    """Load all pages required by this step without evicting other needed pages."""
    needed_set = set(needed)
    misses = 0

    # Mark hits as recently used first.
    for p in needed:
        if cache.has(p):
            cache.touch(p, step)

    # Fill misses while protecting the current step's needed set.
    for p in needed:
        if cache.has(p):
            continue
        misses += 1
        slot = cache.choose_victim_slot_avoiding(needed_set)
        cache.evict_slot(slot)
        k_hbm[slot].copy_(k_cpu[p], non_blocking=True)
        v_hbm[slot].copy_(v_cpu[p], non_blocking=True)
        cache.insert(p, slot, step)
        cache.touch(p, step)

    return misses


def smoke_test_2(
    kernel: DecodeKernel,
    total_pages: int,
    cap_list: List[int],
    steps: int,
    window_pages: int,
    num_sparse: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    print("\n" + "=" * 80)
    print("SMOKE TEST #2 — Capacity sweep under irregular access (LRU, demand fetch)")
    print("=" * 80)
    cpu = torch.device("cpu")

    # KV pool on CPU pinned
    k_cpu = torch.randn((total_pages, page_size, kv_heads, head_dim), device=cpu, dtype=dtype).pin_memory()
    v_cpu = torch.randn((total_pages, page_size, kv_heads, head_dim), device=cpu, dtype=dtype).pin_memory()
    q = torch.randn((1, kv_heads, head_dim), device=device, dtype=dtype)
    last_page_len = torch.tensor([page_size], device=device, dtype=torch.int32)

    for cap in cap_list:
        k_hbm = torch.empty((cap, page_size, kv_heads, head_dim), device=device, dtype=dtype)
        v_hbm = torch.empty((cap, page_size, kv_heads, head_dim), device=device, dtype=dtype)
        cache = LRUCache.create(cap)
        truncated_steps = 0

        total_miss = 0
        total_need = 0
        total_xfer_bytes = 0
        total_xfer_ms = 0.0
        total_kernel_ms = 0.0

        # Warm a bit
        for t in range(5):
            needed = irregular_pages(t, window_pages, num_sparse, total_pages)
            if len(needed) > cap:
                needed = needed[-cap:]
                truncated_steps += 1
            _ = load_needed_pages(needed, t, cache, k_hbm, v_hbm, k_cpu, v_cpu)
            cuda_sync()
            # build indices (slots) and run kernel once
            slots = [cache.get_slot(p) for p in needed]
            indices = torch.tensor(slots, device=device, dtype=torch.int32)
            indptr = torch.tensor([0, len(slots)], device=device, dtype=torch.int32)
            kernel.run(q, k_hbm, v_hbm, indptr, indices, last_page_len)
        cuda_sync()

        # Measured loop
        for t in range(steps):
            needed = irregular_pages(t, window_pages, num_sparse, total_pages)
            if len(needed) > cap:
                needed = needed[-cap:]
                truncated_steps += 1
            total_need += len(needed)

            # Demand fetch: measure transfer time (sync) separately from kernel
            t0 = time.time()
            miss = load_needed_pages(needed, t, cache, k_hbm, v_hbm, k_cpu, v_cpu)
            cuda_sync()
            t1 = time.time()

            # kernel time
            slots = [cache.get_slot(p) for p in needed]
            indices = torch.tensor(slots, device=device, dtype=torch.int32)
            indptr = torch.tensor([0, len(slots)], device=device, dtype=torch.int32)

            def run_kernel_once():
                kernel.run(q, k_hbm, v_hbm, indptr, indices, last_page_len)

            k_ms = time_cuda(run_kernel_once, warmup=0, iters=1)

            xfer_ms = (t1 - t0) * 1000.0
            total_xfer_ms += xfer_ms
            total_kernel_ms += k_ms
            total_miss += miss

            # bytes moved (approx): each miss copies 1 page of K and 1 page of V
            dtype_bytes = torch.tensor([], dtype=dtype).element_size()
            total_xfer_bytes += miss * bytes_per_page(kv_heads, page_size, head_dim, dtype_bytes)

        miss_rate = total_miss / max(1, total_need)
        avg_xfer_ms = total_xfer_ms / steps
        avg_k_ms = total_kernel_ms / steps
        avg_total_ms = avg_xfer_ms + avg_k_ms

        print(f"\nHBM cap pages = {cap}")
        print(f"  avg needed pages/step = {total_need/steps:.1f}")
        print(f"  miss rate             = {miss_rate*100:.2f}%")
        print(f"  avg transfer (sync)   = {avg_xfer_ms:.4f} ms/step")
        print(f"  avg kernel            = {avg_k_ms:.4f} ms/step   (kernel={kernel.name})")
        print(f"  avg total (approx)    = {avg_total_ms:.4f} ms/step")
        print(f"  bytes moved (approx)  = {fmt_bytes(total_xfer_bytes)} over {steps} steps")
        if truncated_steps > 0:
            print(f"  note                  = truncated request set to cap on {truncated_steps} steps")

    print("\nInterpretation:")
    print("- If transfer grows sharply as cap shrinks, you’re in the regime where policy matters.")
    print("- If miss rate barely changes or transfer stays tiny, your setup/workload isn’t stressing KV enough.")


# ----------------------------
# Benchmark-grade capacity sweep
# ----------------------------
def qtile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(t, torch.tensor(q, dtype=torch.float64)).item())


def benchmark_capacity_sweep(
    kernel: DecodeKernel,
    total_pages: int,
    cap_list: List[int],
    steps: int,
    warmup_steps: int,
    window_pages: int,
    num_sparse: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    allow_truncate: bool,
    step_csv_path: str,
    summary_csv_path: str,
):
    print("\n" + "=" * 80)
    print("BENCHMARK MODE — Capacity sweep with fixed trace + tail metrics")
    print("=" * 80)

    cpu = torch.device("cpu")
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    q = torch.randn((1, kv_heads, head_dim), device=device, dtype=dtype)
    last_page_len = torch.tensor([page_size], device=device, dtype=torch.int32)

    # Fixed trace shared by all capacities for comparability.
    rng = random.Random(seed)
    trace = [
        irregular_pages_rng(t, window_pages, num_sparse, total_pages, rng)
        for t in range(warmup_steps + steps)
    ]
    max_needed = max(len(x) for x in trace)
    min_cap = min(cap_list)
    if (not allow_truncate) and max_needed > min_cap:
        raise ValueError(
            f"Non-comparable setup: max needed pages per step is {max_needed}, "
            f"but smallest cap is {min_cap}. Reduce workload or cap list, or pass --bench-allow-truncate."
        )

    # Shared logical KV pool.
    k_cpu = torch.randn((total_pages, page_size, kv_heads, head_dim), device=cpu, dtype=dtype).pin_memory()
    v_cpu = torch.randn((total_pages, page_size, kv_heads, head_dim), device=cpu, dtype=dtype).pin_memory()

    step_rows = []
    summary_rows = []
    any_truncation = False

    for cap in cap_list:
        k_hbm = torch.empty((cap, page_size, kv_heads, head_dim), device=device, dtype=dtype)
        v_hbm = torch.empty((cap, page_size, kv_heads, head_dim), device=device, dtype=dtype)
        cache = LRUCache.create(cap)
        truncated_steps = 0

        xfer_list = []
        kernel_list = []
        total_list = []
        miss_list = []
        miss_rate_list = []

        for t, needed_orig in enumerate(trace):
            needed = needed_orig
            if len(needed) > cap:
                if allow_truncate:
                    needed = needed[-cap:]
                    truncated_steps += 1
                else:
                    raise RuntimeError(f"Internal error: needed pages ({len(needed)}) > cap ({cap}) with truncation disabled.")

            t0 = time.time()
            miss = load_needed_pages(needed, t, cache, k_hbm, v_hbm, k_cpu, v_cpu)
            cuda_sync()
            t1 = time.time()

            slots = [cache.get_slot(p) for p in needed]
            indices = torch.tensor(slots, device=device, dtype=torch.int32)
            indptr = torch.tensor([0, len(slots)], device=device, dtype=torch.int32)

            def run_kernel_once():
                kernel.run(q, k_hbm, v_hbm, indptr, indices, last_page_len)

            k_ms = time_cuda(run_kernel_once, warmup=0, iters=1) 
            xfer_ms = (t1 - t0) * 1000.0
            total_ms = xfer_ms + k_ms

            if t >= warmup_steps:
                step_idx = t - warmup_steps
                needed_len = len(needed)
                miss_rate = miss / max(1, needed_len)
                moved_bytes = miss * bytes_per_page(kv_heads, page_size, head_dim, dtype_bytes)

                xfer_list.append(xfer_ms)
                kernel_list.append(k_ms)
                total_list.append(total_ms)
                miss_list.append(float(miss))
                miss_rate_list.append(miss_rate)

                step_rows.append({
                    "cap_pages": cap,
                    "step": step_idx,
                    "needed_pages": needed_len,
                    "misses": miss,
                    "miss_rate": miss_rate,
                    "transfer_ms": xfer_ms,
                    "kernel_ms": k_ms,
                    "total_ms": total_ms,
                    "stall_share": (xfer_ms / total_ms) if total_ms > 0 else 0.0,
                    "bytes_moved": moved_bytes,
                    "kernel_name": kernel.name,
                })

        summary_rows.append({
            "cap_pages": cap,
            "kernel_name": kernel.name,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "window_pages": window_pages,
            "num_sparse": num_sparse,
            "truncated_steps": truncated_steps,
            "mean_misses": sum(miss_list) / max(1, len(miss_list)),
            "mean_miss_rate": sum(miss_rate_list) / max(1, len(miss_rate_list)),
            "mean_transfer_ms": sum(xfer_list) / max(1, len(xfer_list)),
            "mean_kernel_ms": sum(kernel_list) / max(1, len(kernel_list)),
            "mean_total_ms": sum(total_list) / max(1, len(total_list)),
            "p50_total_ms": qtile(total_list, 0.50),
            "p95_total_ms": qtile(total_list, 0.95),
            "p99_total_ms": qtile(total_list, 0.99),
            "p95_transfer_ms": qtile(xfer_list, 0.95),
            "p95_kernel_ms": qtile(kernel_list, 0.95),
            "mean_stall_share": (sum(xfer_list) / max(1, len(xfer_list))) / (sum(total_list) / max(1, len(total_list))),
            "p95_stall_share": qtile(
                [(x / t) if t > 0 else 0.0 for x, t in zip(xfer_list, total_list)],
                0.95,
            ),
            "regime_label": "memory_bound" if (sum(xfer_list) > sum(kernel_list)) else "compute_bound",
        })
        if truncated_steps > 0:
            any_truncation = True

    os.makedirs(os.path.dirname(step_csv_path) or ".", exist_ok=True)
    with open(step_csv_path, "w", newline="") as f:
        fieldnames = [
            "cap_pages", "step", "needed_pages", "misses", "miss_rate",
            "transfer_ms", "kernel_ms", "total_ms", "stall_share", "bytes_moved", "kernel_name",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(step_rows)

    os.makedirs(os.path.dirname(summary_csv_path) or ".", exist_ok=True)
    with open(summary_csv_path, "w", newline="") as f:
        fieldnames = [
            "cap_pages", "kernel_name", "steps", "warmup_steps", "window_pages", "num_sparse",
            "truncated_steps", "mean_misses", "mean_miss_rate", "mean_transfer_ms", "mean_kernel_ms",
            "mean_total_ms", "p50_total_ms", "p95_total_ms", "p99_total_ms", "p95_transfer_ms", "p95_kernel_ms",
            "mean_stall_share", "p95_stall_share", "regime_label", "cross_cap_comparable",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            r["cross_cap_comparable"] = 0 if any_truncation else 1
            w.writerow(r)

    print("cap | mean_total | p95_total | p99_total | stall_share | regime | mean_transfer | mean_kernel | mean_miss_rate")
    for r in summary_rows:
        print(
            f"{r['cap_pages']:>3} | "
            f"{r['mean_total_ms']:>10.4f} | {r['p95_total_ms']:>9.4f} | {r['p99_total_ms']:>9.4f} | "
            f"{100.0*r['mean_stall_share']:>10.2f}% | {r['regime_label']:^12} | "
            f"{r['mean_transfer_ms']:>13.4f} | {r['mean_kernel_ms']:>11.4f} | "
            f"{100.0*r['mean_miss_rate']:>13.2f}%"
        )
        if r["truncated_steps"] > 0:
            print(f"      note: truncated request set on {r['truncated_steps']} steps (consider disabling for strict comparability).")

    print(f"\n[write] step metrics -> {step_csv_path}")
    print(f"[write] summary      -> {summary_csv_path}")
    if any_truncation:
        print("[warn] cross-cap comparability is reduced because at least one cap used truncation.")


# ----------------------------
# Smoke Test 3: bandwidth estimate
# ----------------------------
def smoke_test_3(
    kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    assumed_GBps: float,
    misses_per_step: int,
    layers: int,
):
    print("\n" + "=" * 80)
    print("SMOKE TEST #3 — Back-of-the-envelope bandwidth estimate")
    print("=" * 80)

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    bpp = bytes_per_page(kv_heads, page_size, head_dim, dtype_bytes)

    # per-step bytes if misses_per_step pages miss for each layer
    per_step = bpp * misses_per_step * layers

    # time = bytes / bandwidth
    bw_Bps = assumed_GBps * (1024**3)
    t_s = per_step / bw_Bps
    t_ms = t_s * 1000.0

    print(f"dtype bytes/elem: {dtype_bytes}")
    print(f"bytes/page (K+V): {fmt_bytes(bpp)}  [kv_heads={kv_heads}, page_size={page_size}, head_dim={head_dim}]")
    print(f"Assume misses/step: {misses_per_step}, layers: {layers}")
    print(f"Bytes moved/step (approx): {fmt_bytes(per_step)}")
    print(f"Assumed CPU→GPU bandwidth: {assumed_GBps:.1f} GB/s")
    print(f"Theoretical transfer time/step: {t_ms:.4f} ms (lower bound, ignores overhead/sync)")
    print("\nInterpretation:")
    print("- Real measured stall can be much higher due to per-transfer overhead, synchronization, limited inflight, and contention.")
    print("- If this lower bound is already comparable to your kernel time, movement is definitely important.")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--total-pages", type=int, default=2048)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--kv-heads", type=int, default=8)

    # Smoke test 1
    ap.add_argument("--smoke1-needed-pages", type=int, default=128)
    ap.add_argument("--smoke1-hbm-cap-pages", type=int, default=128)

    # Smoke test 2
    ap.add_argument("--smoke2-steps", type=int, default=200)
    ap.add_argument("--smoke2-window-pages", type=int, default=64)
    ap.add_argument("--smoke2-num-sparse", type=int, default=16)
    ap.add_argument("--smoke2-cap-list", type=str, default="256,128,64,32,16")

    # Smoke test 3
    ap.add_argument("--assumed-bandwidth-GBps", type=float, default=25.0)
    ap.add_argument("--assumed-misses-per-step", type=int, default=32)
    ap.add_argument("--assumed-layers", type=int, default=32)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-flashinfer", action="store_true")
    ap.add_argument("--benchmark-grade", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=400)
    ap.add_argument("--bench-warmup-steps", type=int, default=50)
    ap.add_argument("--bench-window-pages", type=int, default=16)
    ap.add_argument("--bench-num-sparse", type=int, default=8)
    ap.add_argument("--bench-cap-list", type=str, default="256,128,64,32,16")
    ap.add_argument("--bench-seed", type=int, default=123)
    ap.add_argument("--bench-allow-truncate", action="store_true")
    ap.add_argument("--bench-step-csv", type=str, default="results/benchmark_steps.csv")
    ap.add_argument("--bench-summary-csv", type=str, default="results/benchmark_summary.csv")

    args = ap.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print("Config:")
    print(f"  device={device}, dtype={dtype}")
    print(f"  total_pages={args.total_pages}, page_size={args.page_size}, kv_heads={args.kv_heads}, head_dim={args.head_dim}")
    print(f"  flashinfer_available={HAS_FLASHINFER} (forced off? {args.no_flashinfer})")
    if not HAS_FLASHINFER and FLASHINFER_IMPORT_ERROR is not None:
        print(f"  flashinfer_import_error={type(FLASHINFER_IMPORT_ERROR).__name__}: {FLASHINFER_IMPORT_ERROR}")

    if HAS_FLASHINFER and not args.no_flashinfer and device.type == "cuda":
        try:
            kernel = make_flashinfer_decode_kernel(args.kv_heads, args.head_dim, args.page_size, dtype, device)
        except Exception as e:
            print(f"[WARN] FlashInfer present but wrapper init failed: {e}")
            print("[WARN] Falling back to proxy kernel.")
            kernel = make_proxy_decode_kernel(device)
    else:
        kernel = make_proxy_decode_kernel(device)

    if args.benchmark_grade:
        bench_caps = [int(x.strip()) for x in args.bench_cap_list.split(",") if x.strip()]
        benchmark_capacity_sweep(
            kernel=kernel,
            total_pages=args.total_pages,
            cap_list=bench_caps,
            steps=args.bench_steps,
            warmup_steps=args.bench_warmup_steps,
            window_pages=args.bench_window_pages,
            num_sparse=args.bench_num_sparse,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            dtype=dtype,
            device=device,
            seed=args.bench_seed,
            allow_truncate=args.bench_allow_truncate,
            step_csv_path=args.bench_step_csv,
            summary_csv_path=args.bench_summary_csv,
        )
        print("\nDone benchmark run.")
        return

    # Smoke tests
    smoke_test_1(
        kernel=kernel,
        total_pages=args.total_pages,
        hbm_cap_pages=args.smoke1_hbm_cap_pages,
        needed_pages=args.smoke1_needed_pages,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        dtype=dtype,
        device=device,
    )

    cap_list = [int(x.strip()) for x in args.smoke2_cap_list.split(",") if x.strip()]
    smoke_test_2(
        kernel=kernel,
        total_pages=args.total_pages,
        cap_list=cap_list,
        steps=args.smoke2_steps,
        window_pages=args.smoke2_window_pages,
        num_sparse=args.smoke2_num_sparse,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        dtype=dtype,
        device=device,
    )

    smoke_test_3(
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        dtype=dtype,
        assumed_GBps=args.assumed_bandwidth_GBps,
        misses_per_step=args.assumed_misses_per_step,
        layers=args.assumed_layers,
    )

    print("\nDone.")
    print("If Smoke #1 shows transfer dominates total time, your project’s policy layer is worth pursuing.")


if __name__ == "__main__":
    main()
