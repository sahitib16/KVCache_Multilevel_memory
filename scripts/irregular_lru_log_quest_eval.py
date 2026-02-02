import os, sys, time, random, csv
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
random.seed(0)
torch.set_grad_enabled(False)

# ----------------------------
# Import QUEST repo eval selector
# ----------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUEST_ROOT = os.path.join(REPO_ROOT, "external", "Quest")
if QUEST_ROOT not in sys.path:
    sys.path.insert(0, QUEST_ROOT)

from evaluation.quest_attention import local_heavy_hitter_mask

# ----------------------------
# Configuration
# ----------------------------
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
topk_pages = 8
chunk_size = page_size
token_budget = topk_pages * page_size

def make_wrapper():
    workspace = torch.empty((256 * 1024 * 1024 // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(float_workspace_buffer=workspace, kv_layout="NHD", backend="auto")

# Query (decode)
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# CPU KV (pinned)
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)

# GPU cache
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)

slot_to_page = torch.full((hbm_capacity,), -1, dtype=torch.int32)
page_to_slot = {}
last_used_step = {}
current_step = 0
evictions_this_step = 0

def reset_cache_state():
    global page_to_slot, last_used_step, current_step
    page_to_slot = {}
    last_used_step = {}
    slot_to_page[:] = -1
    current_step = 0

def choose_victim_slot():
    for slot in range(hbm_capacity):
        if int(slot_to_page[slot]) == -1:
            return slot
    victim_slot = 0
    victim_time = None
    for slot in range(hbm_capacity):
        pid = int(slot_to_page[slot])
        t = last_used_step.get(pid, -1)
        if victim_time is None or t < victim_time:
            victim_time = t
            victim_slot = slot
    return victim_slot

def load_page_to_hbm(page_id: int):
    global current_step, evictions_this_step

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
    wrapper.begin_forward(indptr, indices, last_page_len,
                          num_qo_heads=Hq, num_kv_heads=Hkv, head_dim=D, page_size=page_size)
    _ = wrapper.forward(q, (k_hbm, v_hbm))
    torch.cuda.synchronize()
    wrapper.end_forward()

@torch.no_grad()
def quest_eval_selected_pages(t: int):
    local_lo = t
    local_hi = t + W

    k0 = k_cpu[:, :, 0, :].contiguous()
    S = k0.shape[0] * k0.shape[1]
    k_flat = k0.view(S, k0.shape[-1])

    q0 = q[0, 0, :].detach().to("cpu")
    scores = (k_flat @ q0).view(1, 1, 1, S)

    mask = local_heavy_hitter_mask(scores, token_budget=token_budget, chunk_size=chunk_size)
    sel_tokens = mask[0, 0, 0].nonzero(as_tuple=False).flatten()

    sel_pages = torch.unique(sel_tokens // page_size).tolist()
    sel_pages = sel_pages[:topk_pages]
    sel_pages = [p for p in sel_pages if (p < local_lo or p >= local_hi)]
    return sel_pages

def gen_needed_pages(t: int):
    local_pages = list(range(t, t + W))
    sparse_pages = quest_eval_selected_pages(t)
    return local_pages, sparse_pages, local_pages + sparse_pages

def run_log(prefetch_k: int, csv_path: str):
    global current_step, evictions_this_step

    wrapper = make_wrapper()
    reset_cache_state()

    fieldnames = [
        "t","prefetch_k","needed_local","needed_sparse",
        "evictions","ondemand_misses","ondemand_transfer_ms",
        "prefetch_misses","prefetch_transfer_ms"
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for t in range(T):
            current_step = t
            evictions_this_step = 0

            local_pages, sparse_pages, needed = gen_needed_pages(t)

            # QUEST-aware prefetch: prefetch next step selected pages
            pre_miss = 0
            pre_ms = 0.0
            if prefetch_k > 0 and (t + 1 < T):
                next_pages = quest_eval_selected_pages(t + 1)
                for pid in next_pages[:prefetch_k]:
                    miss, t_ms = load_page_to_hbm(pid)
                    if miss:
                        pre_miss += 1
                        pre_ms += t_ms

            # On-demand
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
                "prefetch_misses": pre_miss,
                "prefetch_transfer_ms": round(pre_ms, 6),
            })

def main():
    os.makedirs("results", exist_ok=True)
    for k in [0, 4, 8]:
        out = f"results/quest_eval_step_log_prefetch{k}.csv"
        print(f"[run] prefetch_k={k} -> {out}")
        run_log(k, out)
    print("[done] logs written.")

if __name__ == "__main__":
    main()
