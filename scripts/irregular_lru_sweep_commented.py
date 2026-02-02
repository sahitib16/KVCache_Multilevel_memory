"""
irregular_lru_sweep_commented.py

Goal:
- Simulate a two-tier KV cache (CPU pinned memory = cold tier, GPU HBM = hot tier).
- Each decode step needs:
    local window pages (W) + sparse random pages (num_sparse)
- Use LRU eviction in HBM.
- Measure BOTH:
    (1) on-demand CPU->GPU transfers (happen because needed right now)
    (2) prefetch CPU->GPU transfers (happen proactively)
- Sweep prefetch_k in {0,4,8,16} and print a summary table.

This is the core experiment that shows the prefetch tradeoff.
"""

import time, random
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
random.seed(0)
torch.set_grad_enabled(False)

# ----------------------------
# Model-ish dimensions (small to fit shared GPU memory)
# ----------------------------
device = "cuda"
dtype = torch.float16
B = 1
Hq = 16
Hkv = 4
D = 64
page_size = 16

# ----------------------------
# KV sizes & cache capacity
# ----------------------------
num_pages_total = 4096     # total logical KV pages (exist on CPU)
hbm_capacity = 64          # how many pages can be resident on GPU

# ----------------------------
# Timeline & workload
# ----------------------------
T = 200                    # number of decode steps
W = 32                     # local working set window in pages
num_sparse = 8             # extra sparse pages per step (random long-range)

# Query tensor (decode q_len_per_req = 1): [B, Hq, D]
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# ----------------------------
# CPU tier: pinned KV store (cold)
# ----------------------------
# pin_memory=True makes CPU->GPU copies faster
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)

# ----------------------------
# GPU tier: HBM KV cache (hot)
# ----------------------------
# Only hbm_capacity pages live here at any time.
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)

# Mapping structures:
# slot_to_page[slot] = logical page id currently stored in that GPU slot, or -1 if empty
# page_to_slot[page] = which GPU slot contains that logical page
slot_to_page = torch.full((hbm_capacity,), -1, dtype=torch.int32)
page_to_slot = {}

# LRU state:
# last_used_step[page] = the decode step index when that page was last accessed
last_used_step = {}
current_step = 0

def make_wrapper():
    """Create FlashInfer decode wrapper with a GPU workspace."""
    workspace = torch.empty((256 * 1024 * 1024 // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        backend="auto",
    )

def choose_victim_slot():
    """
    LRU eviction:
    - if any empty slot exists, use it
    - else evict the page with the smallest last_used_step (least recently used)
    """
    # prefer empty slot
    for slot in range(hbm_capacity):
        if int(slot_to_page[slot]) == -1:
            return slot

    # otherwise evict LRU
    victim_slot = 0
    victim_time = None
    for slot in range(hbm_capacity):
        pid = int(slot_to_page[slot])
        t = last_used_step.get(pid, -1)
        if victim_time is None or t < victim_time:
            victim_time = t
            victim_slot = slot
    return victim_slot

def load_page_to_hbm(page_id):
    """
    Ensure page_id is resident in GPU HBM.
    Returns: (miss?, transfer_ms)

    - If already resident: miss=False and transfer_ms=0
    - If not resident: choose a victim slot (LRU), copy CPU->GPU, update mappings, miss=True
    """
    global current_step

    # cache hit
    if page_id in page_to_slot:
        last_used_step[page_id] = current_step
        return False, 0.0

    # cache miss: evict someone
    slot = choose_victim_slot()
    old_page = int(slot_to_page[slot])
    if old_page != -1:
        page_to_slot.pop(old_page, None)
        last_used_step.pop(old_page, None)

    # CPU->GPU transfer
    t0 = time.time()
    k_hbm[slot].copy_(k_cpu[page_id], non_blocking=True)
    v_hbm[slot].copy_(v_cpu[page_id], non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.time()

    # update mappings + LRU
    slot_to_page[slot] = page_id
    page_to_slot[page_id] = slot
    last_used_step[page_id] = current_step

    return True, (t1 - t0) * 1e3

def build_page_table(needed_pages):
    """
    Given a list of logical pages needed this step:
    - ensure all are resident in HBM
    - build the FlashInfer page-table indices as GPU slot IDs

    Returns:
      indptr, indices, last_page_len, misses, transfer_ms
    where misses/transfer_ms count ON-DEMAND loads.
    """
    slots = []
    misses = 0
    transfer_ms = 0.0

    for pid in needed_pages:
        miss, t_ms = load_page_to_hbm(pid)
        if miss:
            misses += 1
            transfer_ms += t_ms
        slots.append(page_to_slot[pid])

    indptr = torch.tensor([0, len(slots)], device="cuda", dtype=torch.int32)
    indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)
    return indptr, indices, last_page_len, misses, transfer_ms

def run_decode(wrapper, indptr, indices, last_page_len):
    """
    Run one decode attention call.
    Note: the kernel always reads from (k_hbm, v_hbm); indices tell it which slots to use.
    """
    wrapper.begin_forward(
        indptr, indices, last_page_len,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        page_size=page_size,
    )
    _ = wrapper.forward(q, (k_hbm, v_hbm))
    torch.cuda.synchronize()
    wrapper.end_forward()

def gen_needed_pages(t):
    """
    Workload model:
    - local pages: [t, t+W)
    - sparse pages: num_sparse random pages from the full KV space
    - remove any sparse pages that overlap with local window
    """
    local_pages = list(range(t, t + W))
    sparse_pages = random.sample(range(0, num_pages_total), num_sparse)
    sparse_pages = [p for p in sparse_pages if p < t or p >= t + W]
    return local_pages + sparse_pages

def reset_cache_state():
    """Reset cache mappings and LRU state for a clean run."""
    global page_to_slot, last_used_step, current_step
    page_to_slot = {}
    last_used_step = {}
    slot_to_page[:] = -1
    current_step = 0

def run_experiment(prefetch_k):
    """
    Run T steps with a fixed prefetch_k.
    Track:
      - on-demand misses/transfer (needed pages not in cache)
      - prefetch misses/transfer (pages loaded proactively)
    """
    global current_step

    wrapper = make_wrapper()
    reset_cache_state()

    ondemand_misses = 0
    ondemand_transfer = 0.0
    prefetch_misses = 0
    prefetch_transfer = 0.0

    for t in range(T):
        current_step = t

        # Prefetch next-K sequential pages beyond the local window
        # (This is a naive heuristic baseline)
        if prefetch_k > 0:
            for pid in range(t + W, t + W + prefetch_k):
                miss, t_ms = load_page_to_hbm(pid)
                if miss:
                    prefetch_misses += 1
                    prefetch_transfer += t_ms

        needed = gen_needed_pages(t)
        indptr, indices, last_page_len, misses, transfer_ms = build_page_table(needed)

        ondemand_misses += misses
        ondemand_transfer += transfer_ms

        # kernel call
        run_decode(wrapper, indptr, indices, last_page_len)

    total_transfer = ondemand_transfer + prefetch_transfer
    total_misses = ondemand_misses + prefetch_misses

    return {
        "prefetch_k": prefetch_k,
        "ondemand_misses": ondemand_misses,
        "ondemand_transfer_ms": ondemand_transfer,
        "prefetch_misses": prefetch_misses,
        "prefetch_transfer_ms": prefetch_transfer,
        "total_misses": total_misses,
        "total_transfer_ms": total_transfer,
    }

def main():
    print(f"[config] hbm_capacity={hbm_capacity}, T={T}, W={W}, num_sparse={num_sparse}")
    print("[note] total_transfer_ms includes BOTH on-demand and prefetch transfers")

    results = []
    for k in [0, 4, 8, 16]:
        results.append(run_experiment(k))

    print("\nRESULTS")
    print("k | ond_miss | ond_ms | pre_miss | pre_ms | total_miss | total_ms")
    for r in results:
        print(
            f"{r['prefetch_k']:>2} | "
            f"{r['ondemand_misses']:>8} | {r['ondemand_transfer_ms']:>6.2f} | "
            f"{r['prefetch_misses']:>8} | {r['prefetch_transfer_ms']:>6.2f} | "
            f"{r['total_misses']:>10} | {r['total_transfer_ms']:>7.2f}"
        )

if __name__ == "__main__":
    main()
