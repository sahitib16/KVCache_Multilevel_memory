import os, time
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
torch.set_grad_enabled(False)

device = "cuda"
dtype = torch.float16

# ----------------------------
# Config (keep small at first)
# ----------------------------
B = 1
Hq = 16
Hkv = 4
D = 64
page_size = 16

num_pages_total = 2048       # total logical KV pages (CPU holds all)
active_pages = 512           # pages needed for the request
hbm_capacity = 256           # how many pages fit in GPU "hot cache" (force misses)

# decode query: [B, Hq, D]
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# ----------------------------
# CPU tier: pinned KV store (all pages)
# ----------------------------
# Store the full logical KV on CPU pinned memory (page-locked)
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)

# ----------------------------
# GPU tier: HBM KV cache (only hbm_capacity pages)
# ----------------------------
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)

# Bookkeeping: which logical page id currently resides in each HBM slot
# -1 means empty
slot_to_page = torch.full((hbm_capacity,), -1, device="cpu", dtype=torch.int32)
page_to_slot = {}  # python dict for quick lookup

# Simple eviction pointer (round-robin)
evict_ptr = 0

def prefetch_page(page_id: int):
    """Ensure logical page_id is in HBM; return slot index."""
    global evict_ptr
    if page_id in page_to_slot:
        return page_to_slot[page_id], 0.0

    # Evict one slot (round-robin)
    slot = evict_ptr
    evict_ptr = (evict_ptr + 1) % hbm_capacity

    old_page = int(slot_to_page[slot])
    if old_page != -1:
        del page_to_slot[old_page]

    # Copy CPU->GPU (pinned -> cuda). Measure transfer time.
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
    Given logical page ids needed, ensure they are in HBM and build:
    - indices: HBM slot ids in the order the kernel should read
    - indptr / last_page_len for B=1
    Returns (indices_cuda, transfer_ms_total, miss_count)
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
    workspace = torch.empty((workspace_bytes // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        use_cuda_graph=False,
        use_tensor_cores=False,
        backend="auto",
    )

def run_decode(wrapper, indptr, indices, last_page_len, iters=200, warmup=50):
    wrapper.begin_forward(
        indptr, indices, last_page_len,
        num_qo_heads=Hq, num_kv_heads=Hkv, head_dim=D, page_size=page_size,
        pos_encoding_mode="NONE",
        non_blocking=True,
    )

    for _ in range(warmup):
        _ = wrapper.forward(q, (k_hbm, v_hbm))
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = wrapper.forward(q, (k_hbm, v_hbm))
    torch.cuda.synchronize()
    t1 = time.time()

    wrapper.end_forward()
    return (t1 - t0) * 1e3 / iters  # ms

def main():
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[info] GPU free {free_b/1e9:.2f} GB / total {total_b/1e9:.2f} GB")

    # Logical pages needed for this request (same as your HBM baseline, but now many will miss)
    logical_needed = torch.arange(active_pages, dtype=torch.int32)

    wrapper = make_wrapper()

    # Build page table by ensuring pages are resident in HBM slots
    indptr, indices, last_page_len, transfer_ms, miss = build_page_table(logical_needed)

    # Run decode attention with HBM cache + page table
    attn_ms = run_decode(wrapper, indptr, indices, last_page_len)

    print(f"[result] active_pages={active_pages}, hbm_capacity={hbm_capacity}")
    print(f"[result] misses={miss}/{active_pages}  transfer_ms_total={transfer_ms:.2f} ms")
    print(f"[result] attention_latency_ms={attn_ms:.4f} ms (HBM-only kernel time, after pages loaded)")

if __name__ == "__main__":
    main()
