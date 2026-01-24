import time, random
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
random.seed(0)
torch.set_grad_enabled(False)

device = "cuda"
dtype = torch.float16

# ---- Model-ish dimensions (small but non-trivial) ----
B = 1
Hq = 16
Hkv = 4
D = 64
page_size = 16

# ---- KV sizes ----
num_pages_total = 4096

# ---- Cache ----
hbm_capacity = 64

# ---- Timeline ----
T = 200
W = 32
num_sparse = 8

# Query (decode): [B, Hq, D]
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# CPU pinned KV store (cold tier)
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D),
                    device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D),
                    device="cpu", dtype=dtype, pin_memory=True)

# GPU HBM KV cache (hot tier) — fixed number of slots
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)

# slot bookkeeping
slot_to_page = torch.full((hbm_capacity,), -1, dtype=torch.int32)
page_to_slot = {}

# LRU bookkeeping
# last_used_step[page_id] = step index when it was last accessed (needed or prefetched)
last_used_step = {}
current_step = 0

def make_wrapper():
    workspace = torch.empty((256 * 1024 * 1024 // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        backend="auto",
    )

def choose_victim_slot():
    """
    LRU eviction: evict the page in HBM with the smallest last_used_step.
    If a slot is empty, use it.
    """
    # empty slot?
    for slot in range(hbm_capacity):
        if int(slot_to_page[slot]) == -1:
            return slot

    # find least recently used page among slots
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
    Ensure page_id is in HBM. Returns (miss?, transfer_ms).
    Updates LRU state.
    """
    global current_step

    if page_id in page_to_slot:
        last_used_step[page_id] = current_step
        return False, 0.0

    slot = choose_victim_slot()
    old_page = int(slot_to_page[slot])
    if old_page != -1:
        # evict mapping
        page_to_slot.pop(old_page, None)
        last_used_step.pop(old_page, None)

    t0 = time.time()
    k_hbm[slot].copy_(k_cpu[page_id], non_blocking=True)
    v_hbm[slot].copy_(v_cpu[page_id], non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.time()

    slot_to_page[slot] = page_id
    page_to_slot[page_id] = slot
    last_used_step[page_id] = current_step
    return True, (t1 - t0) * 1e3

def build_page_table(needed_pages):
    """
    Ensure all needed_pages are resident in HBM.
    Returns page-table tensors + counts/time for ON-DEMAND loads (these loads happened because needed).
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
    Local window + sparse random pages from the full KV space (excluding overlap).
    """
    local_pages = list(range(t, t + W))
    sparse_pages = random.sample(range(0, num_pages_total), num_sparse)
    sparse_pages = [p for p in sparse_pages if p < t or p >= t + W]
    return local_pages + sparse_pages

def reset_cache_state():
    global page_to_slot, slot_to_page, last_used_step, current_step
    page_to_slot = {}
    last_used_step = {}
    slot_to_page[:] = -1
    current_step = 0

def run_experiment(prefetch_k):
    """
    Runs T steps. Counts:
      - misses + transfer time from on-demand loads
      - misses + transfer time from prefetch loads
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

        # Run kernel (not timing step_ms here; focusing on memory behavior first)
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
        r = run_experiment(k)
        results.append(r)

    # Print as a compact table
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
