import os, sys, time, random
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
random.seed(0)
torch.set_grad_enabled(False)

# ----------------------------
# Add QUEST repo to PYTHONPATH (so we can import evaluation/quest_attention.py)
# ----------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUEST_ROOT = os.path.join(REPO_ROOT, "external", "Quest")
if QUEST_ROOT not in sys.path:
    sys.path.insert(0, QUEST_ROOT)

from evaluation.quest_attention import local_heavy_hitter_mask  # QUEST repo code (pure torch)

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
hbm_capacity = 16

T = 200
W = 32

# How many pages we want the selector to return
topk_pages = 8

# Parameters for QUEST repo selection function (token-level, then convert -> pages)
# chunk_size should divide tokens logically; we set it equal to page_size so chunks align with pages
chunk_size = page_size
# token_budget roughly corresponds to how many tokens are kept by the selection mask
token_budget = topk_pages * page_size

# Evaluate a few prefetch choices. Here prefetch_k means "how many selected pages to prefetch for next step".
PREFETCH_K_VALUES = [0, 4, 8]

# Query tensor (decode): [B, Hq, D]
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# ----------------------------
# CPU tier: pinned KV store
# ----------------------------
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D), device="cpu", dtype=dtype, pin_memory=True)

# ----------------------------
# GPU tier: HBM KV cache
# ----------------------------
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D), device="cuda", dtype=dtype)

slot_to_page = torch.full((hbm_capacity,), -1, dtype=torch.int32)
page_to_slot = {}

# LRU bookkeeping
last_used_step = {}
current_step = 0

# ----------------------------
# FlashInfer wrapper
# ----------------------------
def make_wrapper():
    workspace = torch.empty((256 * 1024 * 1024 // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        backend="auto",
    )

# ----------------------------
# LRU cache management
# ----------------------------
def choose_victim_slot():
    # Prefer empty slot.
    for slot in range(hbm_capacity):
        if int(slot_to_page[slot]) == -1:
            return slot

    # Otherwise evict least recently used page.
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
    """
    Ensure page_id is in GPU HBM.
    Returns (miss?, transfer_ms).
    """
    global current_step

    if page_id in page_to_slot:
        last_used_step[page_id] = current_step
        return False, 0.0

    slot = choose_victim_slot()
    old_page = int(slot_to_page[slot])
    if old_page != -1:
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
    ON-DEMAND loads: ensure needed pages are resident and build page-table indices as GPU slot IDs.
    Returns (indptr, indices, last_page_len, ond_miss, ond_ms).
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

def reset_cache_state():
    global page_to_slot, last_used_step, current_step
    page_to_slot = {}
    last_used_step = {}
    slot_to_page[:] = -1
    current_step = 0

# ----------------------------
# QUEST repo evaluation-based selector (token mask -> pages)
# ----------------------------
@torch.no_grad()
def quest_eval_selected_pages(t: int):
    """
    Use QUEST repo's local_heavy_hitter_mask to select important tokens, then map to page IDs.

    We build a lightweight attention score vector over *all tokens* in CPU KV (head 0),
    using scores = q0 · k(token). Then QUEST repo function selects top chunks (token_budget/chunk_size),
    we convert selected tokens -> page IDs.

    Returns list[int] page IDs (length <= topk_pages), excluding overlap with local window [t, t+W).
    """
    local_lo = t
    local_hi = t + W

    # Flatten K for head 0 into token sequence [S, D] on CPU
    k0 = k_cpu[:, :, 0, :].contiguous()                  # [num_pages, page_size, D]
    S = k0.shape[0] * k0.shape[1]
    k_flat = k0.view(S, k0.shape[-1])                    # [S, D]

    # Compute scores on CPU (cheap) using q head 0
    q0 = q[0, 0, :].detach().to("cpu")                   # [D]
    scores = (k_flat @ q0).view(1, 1, 1, S)              # [BS=1, head=1, query=1, keys=S]

    # QUEST repo selection mask (boolean mask over tokens)
    mask = local_heavy_hitter_mask(scores, token_budget=token_budget, chunk_size=chunk_size)
    sel_tokens = mask[0, 0, 0].nonzero(as_tuple=False).flatten()  # token indices

    # Convert token indices to page IDs
    sel_pages = torch.unique(sel_tokens // page_size).tolist()

    # Keep only the first topk_pages pages (stable deterministic)
    sel_pages = sel_pages[:topk_pages]

    # Avoid overlap with local window
    sel_pages = [p for p in sel_pages if (p < local_lo or p >= local_hi)]
    return sel_pages

def gen_needed_pages(t: int):
    """
    Needed pages at step t:
      local window pages + QUEST-eval selected pages
    """
    local_pages = list(range(t, t + W))
    sparse_pages = quest_eval_selected_pages(t)
    return local_pages + sparse_pages

# ----------------------------
# Experiment loop with QUEST-aware prefetch
# ----------------------------
def run_experiment(prefetch_k: int):
    """
    Prefetch policy:
      - prefetch pages that QUEST-eval will select at step t+1 (up to prefetch_k)
    Track:
      - on-demand misses/transfer
      - prefetch misses/transfer
    """
    global current_step

    wrapper = make_wrapper()
    reset_cache_state()

    ond_miss = 0
    ond_ms = 0.0
    pre_miss = 0
    pre_ms = 0.0

    for t in range(T):
        current_step = t

        # Prefetch: pages selected for next step (QUEST-aware)
        if prefetch_k > 0 and (t + 1 < T):
            next_pages = quest_eval_selected_pages(t + 1)
            for pid in next_pages[:prefetch_k]:
                miss, t_ms = load_page_to_hbm(pid)
                if miss:
                    pre_miss += 1
                    pre_ms += t_ms

        # On-demand
        needed = gen_needed_pages(t)
        indptr, indices, last_page_len, m, ms = build_page_table(needed)
        ond_miss += m
        ond_ms += ms

        run_decode(wrapper, indptr, indices, last_page_len)

    return {
        "prefetch_k": prefetch_k,
        "ondemand_misses": ond_miss,
        "ondemand_transfer_ms": ond_ms,
        "prefetch_misses": pre_miss,
        "prefetch_transfer_ms": pre_ms,
        "total_misses": ond_miss + pre_miss,
        "total_transfer_ms": ond_ms + pre_ms,
    }

def main():
    print("[config] QUEST repo eval selection (local_heavy_hitter_mask) -> pages")
    print(f"[config] hbm_capacity={hbm_capacity}, T={T}, W={W}, topk_pages={topk_pages}, token_budget={token_budget}, chunk_size={chunk_size}")
    print("[note] total_transfer_ms includes BOTH on-demand and prefetch transfers")

    print("\nRESULTS")
    print("k | ond_miss | ond_ms | pre_miss | pre_ms | total_miss | total_ms")
    for k in PREFETCH_K_VALUES:
        r = run_experiment(k)
        print(
            f"{r['prefetch_k']:>2} | "
            f"{r['ondemand_misses']:>8} | {r['ondemand_transfer_ms']:>6.2f} | "
            f"{r['prefetch_misses']:>8} | {r['prefetch_transfer_ms']:>6.2f} | "
            f"{r['total_misses']:>10} | {r['total_transfer_ms']:>7.2f}"
        )

if __name__ == "__main__":
    main()
