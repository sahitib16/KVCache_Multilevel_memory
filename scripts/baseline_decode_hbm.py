import os, time, json
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(0)
torch.set_grad_enabled(False)

def make_wrapper(workspace_bytes=256 * 1024 * 1024):
    # workspace required by flashinfer wrapper
    workspace = torch.empty((workspace_bytes // 4,), device="cuda", dtype=torch.float32)
    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        use_cuda_graph=False,
        use_tensor_cores=False,
        backend="auto",
    )

def run_once(wrapper, q, k_pages, v_pages, indptr, indices, last_page_len,
             Hq, Hkv, D, page_size, warmup=50, iters=200):
    # bind page table
    wrapper.begin_forward(
        indptr, indices, last_page_len,
        num_qo_heads=Hq, num_kv_heads=Hkv, head_dim=D, page_size=page_size,
        pos_encoding_mode="NONE",
        non_blocking=True,
    )

    # warmup
    for _ in range(warmup):
        _ = wrapper.forward(q, (k_pages, v_pages))
    torch.cuda.synchronize()

    # timed
    t0 = time.time()
    for _ in range(iters):
        _ = wrapper.forward(q, (k_pages, v_pages))
    torch.cuda.synchronize()
    t1 = time.time()

    wrapper.end_forward()
    return (t1 - t0) * 1e3 / iters  # ms

def main():
    # ---------- configurable knobs ----------
    dtype = torch.float16
    B = 1
    Hq = 16
    Hkv = 4
    D = 64
    page_size = 16

    num_pages_total = 2048
    active_pages = 512

    # ----------------------------------------
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[info] GPU free {free_b/1e9:.2f} GB / total {total_b/1e9:.2f} GB")

    # KV pages in HBM
    k_pages = torch.randn((num_pages_total, page_size, Hkv, D), device="cuda", dtype=dtype)
    v_pages = torch.randn((num_pages_total, page_size, Hkv, D), device="cuda", dtype=dtype)

    # decode query (q_len_per_req=1): [B, Hq, D]
    q = torch.randn((B, Hq, D), device="cuda", dtype=dtype)

    # page table: one request uses active_pages pages
    all_ids = torch.arange(num_pages_total, device="cuda", dtype=torch.int32)
    indices = all_ids[:active_pages].contiguous()
    indptr = torch.tensor([0, active_pages], device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)

    wrapper = make_wrapper()

    ms = run_once(wrapper, q, k_pages, v_pages, indptr, indices, last_page_len,
                  Hq, Hkv, D, page_size)

    # write a small json record so you can track experiments cleanly
    out = {
        "B": B, "Hq": Hq, "Hkv": Hkv, "D": D, "page_size": page_size,
        "num_pages_total": num_pages_total, "active_pages": active_pages,
        "latency_ms": ms,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": torch.cuda.get_device_name(0),
    }
    print("[result]", out)

if __name__ == "__main__":
    main()
