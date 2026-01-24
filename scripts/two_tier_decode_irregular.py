import time, random
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
random.seed(0)
torch.set_grad_enabled(False)

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

""" T = 200
W = 32 """
T = 200
W = 32
num_sparse = 8           # how many long-range pages per step
prefetch_k = 64          # try 0 vs 64

# ----------------------------
# Query tensor
# ----------------------------
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# ----------------------------
# CPU tier (pinned)
# ----------------------------
k_cpu = torch.randn((num_pages_total, page_size, Hkv, D),
                    device="cpu", dtype=dtype, pin_memory=True)
v_cpu = torch.randn((num_pages_total, page_size, Hkv, D),
                    device="cpu", dtype=dtype, pin_memory=True)

# ----------------------------
# GPU tier (HBM cache)
# ----------------------------
k_hbm = torch.empty((hbm_capacity, page_size, Hkv, D),
                    device="cuda", dtype=dtype)
v_hbm = torch.empty((hbm_capacity, page_size, Hkv, D),
                    device="cuda", dtype=dtype)

slot_to_page = torch.full((hbm_capacity,), -1, dtype=torch.int32)
page_to_slot = {}
evict_ptr = 0

def prefetch_page(page_id):
    global evict_ptr
    if page_id in page_to_slot:
        return 0.0, False

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
    return (t1 - t0) * 1e3, True

def build_page_table(pages):
    slots = []
    transfer_ms = 0.0
    misses = 0

    for pid in pages:
        t_ms, miss = prefetch_page(pid)
        if miss:
            misses += 1
            transfer_ms += t_ms
        slots.append(page_to_slot[pid])

    indptr = torch.tensor([0, len(slots)], device="cuda", dtype=torch.int32)
    indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)

    return indptr, indices, last_page_len, misses, transfer_ms

def make_wrapper():
    workspace = torch.empty((256 * 1024 * 1024 // 4,),
                            device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        backend="auto",
    )

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

def main():
    wrapper = make_wrapper()

    total_misses = 0
    total_transfer = 0.0

    for t in range(T):
        local_pages = list(range(t, t + W))
        sparse_pages = random.sample(range(0, num_pages_total), num_sparse)
        sparse_pages = [p for p in sparse_pages if p < t or p >= t+W]  # avoid overlap with local
        needed = local_pages + sparse_pages

        # naive next-K prefetch
        if prefetch_k > 0:
            for pid in range(t + W, t + W + prefetch_k):
                _t, _ = prefetch_page(pid)

        indptr, indices, last_page_len, misses, transfer_ms = \
            build_page_table(needed)

        t0 = time.time()
        run_decode(wrapper, indptr, indices, last_page_len)
        t1 = time.time()

        step_ms = (t1 - t0) * 1e3
        total_misses += misses
        total_transfer += transfer_ms

        print(
            f"step {t:02d} | "
            f"misses {misses:4d} | "
            f"transfer {transfer_ms:6.2f} ms | "
            f"step time {step_ms:6.2f} ms"
        )

    print("\nSUMMARY")
    print(f"Total misses: {total_misses}")
    print(f"Total transfer time: {total_transfer:.2f} ms")

if __name__ == "__main__":
    main()
