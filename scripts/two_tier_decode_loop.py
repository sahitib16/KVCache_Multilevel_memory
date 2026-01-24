import time
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
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

num_pages_total = 4096        # logical KV size
hbm_capacity = 512            # GPU cache size

T = 40                        # number of decode steps
W = 256                       # pages needed per step (window size)
prefetch_k = 64                # set to >0 later

# ----------------------------
# Query tensor
# ----------------------------
q = torch.randn((B, Hq, D), device=device, dtype=dtype)

# ----------------------------
# CPU tier (pinned memory)
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

    total_transfer = 0.0
    total_misses = 0

    for t in range(T):
        start = t
        needed = list(range(start, start + W))

        # Prefetch next K pages (simple baseline)
        if prefetch_k > 0:
            prefetch_pages = list(range(start + W, start + W + prefetch_k))
            for pid in prefetch_pages:
                _slot, _t_ms, _miss = prefetch_page(pid)

        indptr, indices, last_page_len, misses, transfer_ms = \
            build_page_table(needed)

        t0 = time.time()
        run_decode(wrapper, indptr, indices, last_page_len)
        t1 = time.time()
        step_ms = (t1 - t0) * 1e3
        total_transfer += transfer_ms
        total_misses += misses

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
