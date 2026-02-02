"""
two_tier_decode_loop_commented.py

Goal:
- Simulate multiple decode steps with a sliding-window KV access pattern.
- Compare behavior with/without simple sequential prefetch.
- This is an "easy" workload: the next needed pages are predictable.

Useful as:
- sanity check that cache + prefetch behave as expected
- baseline before moving to irregular sparse access
"""

import time
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
torch.set_grad_enabled(False)

device = "cuda"
dtype = torch.float16

B = 1
Hq = 16
Hkv = 4
D = 64
page_size = 16

num_pages_total = 4096
hbm_capacity = 512

T = 40
W = 256
prefetch_k = 64  # set to 0 for "no prefetch"

q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# CPU pinned KV store
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)

# GPU hot cache buffers
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)

slot_to_page = torch.full((hbm_capacity,), -1, dtype=torch.int32)
page_to_slot = {}
evict_ptr = 0

def prefetch_page(page_id):
    """
    Ensure page_id is resident in GPU.
    Returns (slot, transfer_ms, miss_bool).
    Uses round-robin eviction.
    """
    global evict_ptr
    if page_id in page_to_slot:
        return page_to_slot[page_id], 0.0, False

    slot = evict_ptr
    evict_ptr = (evict_ptr + 1) % hbm_capacity

    old = int(slot_to_page[slot])
    if old != -1:
        del page_to_slot[old]

    t0 = time.time()
    k_hbm[slot].copy_(k_cpu[page_id], non_blocking=True)
    v_hbm[slot].copy_(v_cpu[page_id], non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.time()

    slot_to_page[slot] = page_id
    page_to_slot[page_id] = slot
    return slot, (t1 - t0) * 1e3, True

def build_page_table(logical_pages):
    """
    Load all pages needed for this step into HBM, then build page-table indices as slot IDs.
    Returns indptr/indices/last_page_len + miss counts/time for ON-DEMAND loads.
    """
    slots = []
    transfer_ms = 0.0
    misses = 0

    for pid in logical_pages:
        slot, t_ms, miss = prefetch_page(pid)
        slots.append(slot)
        if miss:
            misses += 1
            transfer_ms += t_ms

    indptr = torch.tensor([0, len(slots)], device="cuda", dtype=torch.int32)
    indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)
    return indptr, indices, last_page_len, misses, transfer_ms

def make_wrapper():
    workspace = torch.empty((256 * 1024 * 1024 // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(float_workspace_buffer=workspace, kv_layout="NHD", backend="auto")

def run_decode(wrapper, indptr, indices, last_page_len):
    """
    Run one decode attention call.
    """
    wrapper.begin_forward(indptr, indices, last_page_len,
                          num_qo_heads=Hq, num_kv_heads=Hkv, head_dim=D, page_size=page_size)
    _ = wrapper.forward(q, (k_hbm, v_hbm))
    torch.cuda.synchronize()
    wrapper.end_forward()

def main():
    wrapper = make_wrapper()

    total_transfer = 0.0
    total_misses = 0

    for t in range(T):
        # Sliding window pages: pages [t, t+W)
        needed = list(range(t, t + W))

        # Prefetch pages just beyond the window (predictable workload)
        if prefetch_k > 0:
            for pid in range(t + W, t + W + prefetch_k):
                _slot, _t_ms, _miss = prefetch_page(pid)

        indptr, indices, last_page_len, misses, transfer_ms = build_page_table(needed)

        t0 = time.time()
        run_decode(wrapper, indptr, indices, last_page_len)
        t1 = time.time()
        step_ms = (t1 - t0) * 1e3

        total_transfer += transfer_ms
        total_misses += misses

        print(f"step {t:02d} | misses {misses:4d} | transfer {transfer_ms:6.2f} ms | step time {step_ms:6.2f} ms")

    print("\nSUMMARY")
    print(f"Total misses: {total_misses}")
    print(f"Total transfer time: {total_transfer:.2f} ms")

if __name__ == "__main__":
    main()
