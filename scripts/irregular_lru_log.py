import time, random, csv
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
random.seed(0)
torch.set_grad_enabled(False)

device = "cuda"
dtype = torch.float16

B = 1
Hq = 16
Hkv = 4
D = 64
page_size = 16

num_pages_total = 4096
hbm_capacity = 64

T = 200
W = 32
num_sparse = 8

# Query
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# CPU pinned KV store
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D),
                    device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D),
                    device="cpu", dtype=dtype, pin_memory=True)

# GPU cache
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)

slot_to_page = torch.full((hbm_capacity,), -1, dtype=torch.int32)
page_to_slot = {}

last_used_step = {}
current_step = 0

# Counters for evictions
evictions_this_step = 0

def make_wrapper():
    workspace = torch.empty((256 * 1024 * 1024 // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        backend="auto",
    )

def reset_state():
    global page_to_slot, last_used_step, current_step
    page_to_slot = {}
    last_used_step = {}
    slot_to_page[:] = -1
    current_step = 0

def choose_victim_slot():
    # Empty slot?
    for slot in range(hbm_capacity):
        if int(slot_to_page[slot]) == -1:
            return slot
    # LRU
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
    global evictions_this_step
    if page_id in page_to_slot:
        last_used_step[page_id] = current_step
        return False, 0.0

    slot = choose_victim_slot()
    old_page = int(slot_to_page[slot])
    if old_page != -1:
        evictions_this_step += 1
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
    local_pages = list(range(t, t + W))
    sparse_pages = random.sample(range(0, num_pages_total), num_sparse)
    sparse_pages = [p for p in sparse_pages if p < t or p >= t + W]
    return local_pages, sparse_pages, local_pages + sparse_pages

def run_with_prefetch(prefetch_k, csv_path):
    global current_step, evictions_this_step

    wrapper = make_wrapper()
    reset_state()

    fieldnames = [
        "t", "prefetch_k", "needed_local", "needed_sparse",
        "evictions", "ondemand_misses", "ondemand_transfer_ms",
        "prefetch_misses", "prefetch_transfer_ms"
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for t in range(T):
            current_step = t
            evictions_this_step = 0

            local_pages, sparse_pages, needed = gen_needed_pages(t)

            # Prefetch phase
            prefetch_misses = 0
            prefetch_transfer = 0.0
            if prefetch_k > 0:
                for pid in range(t + W, t + W + prefetch_k):
                    miss, t_ms = load_page_to_hbm(pid)
                    if miss:
                        prefetch_misses += 1
                        prefetch_transfer += t_ms

            # On-demand phase
            indptr, indices, last_page_len, ond_miss, ond_ms = build_page_table(needed)

            run_decode(wrapper, indptr, indices, last_page_len)

            w.writerow({
                "t": t,
                "prefetch_k": prefetch_k,
                "needed_local": len(local_pages),
                "needed_sparse": len(sparse_pages),
                "evictions": evictions_this_step,
                "ondemand_misses": ond_miss,
                "ondemand_transfer_ms": round(ond_ms, 6),
                "prefetch_misses": prefetch_misses,
                "prefetch_transfer_ms": round(prefetch_transfer, 6),
            })

def main():
    # Change this value to log different runs
    prefetch_k = 16
    out = f"results/step_log_prefetch{prefetch_k}.csv"
    print(f"[run] prefetch_k={prefetch_k} -> {out}")
    run_with_prefetch(prefetch_k, out)
    print("[done] wrote", out)

if __name__ == "__main__":
    main()
