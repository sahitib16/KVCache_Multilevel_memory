"""
two_tier_decode_step_commented.py

Goal:
- Demonstrate a 2-tier KV cache:
  * CPU tier (pinned memory): holds ALL logical KV pages
  * GPU HBM tier: holds ONLY a limited number of pages (hbm_capacity)
- Force misses by requesting more pages than fit in HBM.
- Show: CPU->GPU transfer time dominates; kernel time stays tiny.

This is the "multi-level memory matters" proof.
"""

import os, time
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
torch.set_grad_enabled(False)

device = "cuda"
dtype = torch.float16

# Basic dimensions
B = 1
Hq = 16
Hkv = 4
D = 64
page_size = 16

# Logical KV and HBM cache capacity
num_pages_total = 2048   # total logical pages (stored on CPU)
active_pages = 512       # pages needed for this request
hbm_capacity = 256       # pages that can fit in GPU hot cache (force misses)

# Decode query [B, Hq, D]
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# CPU tier: pinned KV store (page-locked CPU memory)
# pin_memory=True makes DMA transfers faster
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)

# GPU tier: only hbm_capacity pages exist here
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)

# Mapping between logical pages and GPU slots
slot_to_page = torch.full((hbm_capacity,), -1, device="cpu", dtype=torch.int32)  # slot -> page_id
page_to_slot = {}  # page_id -> slot

# Simple eviction policy: round robin
evict_ptr = 0

def prefetch_page(page_id: int):
    """
    Ensure logical page_id is resident in GPU HBM. Return (slot, transfer_ms).
    - If already in cache: transfer_ms = 0.
    - If missing: evict one slot, copy CPU->GPU, update mappings, return transfer time.
    """
    global evict_ptr

    # Cache hit
    if page_id in page_to_slot:
        return page_to_slot[page_id], 0.0

    # Pick victim slot (round robin)
    slot = evict_ptr
    evict_ptr = (evict_ptr + 1) % hbm_capacity

    # If slot had a page, evict it (remove mapping)
    old_page = int(slot_to_page[slot])
    if old_page != -1:
        del page_to_slot[old_page]

    # Copy CPU->GPU; synchronize to measure transfer time
    t0 = time.time()
    k_hbm[slot].copy_(k_cpu[page_id], non_blocking=True)
    v_hbm[slot].copy_(v_cpu[page_id], non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.time()

    slot_to_page[slot] = page_id
    page_to_slot[page_id] = slot
    return slot, (t1 - t0) * 1e3  # ms

def build_page_table(logical_page_ids):
    """
    Given logical pages needed by the request:
    - Ensure each is in HBM via prefetch_page()
    - Build FlashInfer page table where indices are GPU SLOT ids (not logical page ids)

    Returns:
      indptr, indices, last_page_len, total_transfer_ms, miss_count
    """
    slots = []
    transfer_ms = 0.0
    miss = 0

    for pid in logical_page_ids:
        slot, t_ms = prefetch_page(int(pid))
        if t_ms > 0:
            miss += 1
            transfer_ms += t_ms
        slots.append(slot)

    indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, len(slots)], device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)
    return indptr, indices, last_page_len, transfer_ms, miss

def make_wrapper(workspace_bytes=256 * 1024 * 1024):
    """Create FlashInfer wrapper + GPU workspace."""
    workspace = torch.empty((workspace_bytes // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        backend="auto",
    )

def run_decode(wrapper, indptr, indices, last_page_len, iters=200, warmup=50):
    """
    Run many decode calls to get stable kernel time.
    NOTE: This measures kernel time AFTER pages are in HBM.
    """
    wrapper.begin_forward(
        indptr, indices, last_page_len,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        page_size=page_size,
        pos_encoding_mode="NONE",
        non_blocking=True,
    )

    # warmup
    for _ in range(warmup):
        _ = wrapper.forward(q, (k_hbm, v_hbm))
    torch.cuda.synchronize()

    # timed
    t0 = time.time()
    for _ in range(iters):
        _ = wrapper.forward(q, (k_hbm, v_hbm))
    torch.cuda.synchronize()
    t1 = time.time()

    wrapper.end_forward()
    return (t1 - t0) * 1e3 / iters

def main():
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[info] GPU free {free_b/1e9:.2f} GB / total {total_b/1e9:.2f} GB")

    # Request needs pages 0..active_pages-1
    logical_needed = torch.arange(active_pages, dtype=torch.int32)

    wrapper = make_wrapper()

    # This is the multi-level memory bottleneck:
    # build_page_table forces CPU->GPU transfers for missing pages.
    indptr, indices, last_page_len, transfer_ms, miss = build_page_table(logical_needed)

    # Kernel time once pages are resident in HBM slots
    attn_ms = run_decode(wrapper, indptr, indices, last_page_len)

    print(f"[result] active_pages={active_pages}, hbm_capacity={hbm_capacity}")
    print(f"[result] misses={miss}/{active_pages}  transfer_ms_total={transfer_ms:.2f} ms")
    print(f"[result] attention_latency_ms={attn_ms:.4f} ms (HBM-only kernel time, after pages loaded)")

if __name__ == "__main__":
    main()
