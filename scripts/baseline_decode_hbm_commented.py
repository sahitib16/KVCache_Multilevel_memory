"""
baseline_decode_hbm_commented.py

Goal:
- Measure decode-time attention latency when *all KV pages already live in GPU HBM*.
- This is the "kernel-only" baseline: no CPU tier, no prefetch, no eviction.
- Updated: use_tensor_cores=True so FlashInfer uses fast kernels when possible.
"""

import os, time
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

# Make experiments reproducible
torch.manual_seed(0)
# Disable autograd (you are not training)
torch.set_grad_enabled(False)

def make_wrapper(workspace_bytes=256 * 1024 * 1024):
    """
    Create FlashInfer decode wrapper + workspace.

    FlashInfer needs a GPU scratch buffer (workspace) for internal computation.
    This wrapper consumes a paged KV cache via a page table:
      indptr, indices, last_page_len.
    """
    # float32 = 4 bytes; workspace_bytes//4 elements = workspace_bytes bytes total
    workspace = torch.empty((workspace_bytes // 4,), device="cuda", dtype=torch.float32)

    return BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,  # GPU scratch buffer
        kv_layout="NHD",                   # KV page layout convention
        use_cuda_graph=False,              # keep simple for now
        use_tensor_cores=True,             # IMPORTANT: enables faster kernel variants
        backend="auto",                    # let FlashInfer choose best backend
    )

def run_once(wrapper, q, k_pages, v_pages, indptr, indices, last_page_len,
             Hq, Hkv, D, page_size, warmup=50, iters=200):
    """
    Run decode attention many times to get stable latency.

    FlashInfer wrapper API pattern:
    1) begin_forward(...) sets up the page table and dimensions.
    2) forward(q, (k_pages, v_pages)) runs attention.
    3) end_forward() closes the session.
    """
    # Bind page table + dimensions
    wrapper.begin_forward(
        indptr, indices, last_page_len,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        page_size=page_size,
        pos_encoding_mode="NONE",
        non_blocking=True,
    )

    # Warmup (avoid first-run noise)
    for _ in range(warmup):
        _ = wrapper.forward(q, (k_pages, v_pages))
    torch.cuda.synchronize()

    # Timed loop
    t0 = time.time()
    for _ in range(iters):
        _ = wrapper.forward(q, (k_pages, v_pages))
    torch.cuda.synchronize()
    t1 = time.time()

    wrapper.end_forward()
    return (t1 - t0) * 1e3 / iters  # ms per call

def main():
    # Config (small but representative)
    dtype = torch.float16
    B = 1
    Hq = 16
    Hkv = 4
    D = 64
    page_size = 16

    # Total KV pages allocated on GPU; how many used by this request
    num_pages_total = 2048
    active_pages = 512

    # Shared machine sanity check
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[info] GPU free {free_b/1e9:.2f} GB / total {total_b/1e9:.2f} GB")

    # KV cache pages fully in GPU memory (HBM-only baseline)
    k_pages = torch.randn((num_pages_total, page_size, Hkv, D), device="cuda", dtype=dtype)
    v_pages = torch.randn((num_pages_total, page_size, Hkv, D), device="cuda", dtype=dtype)

    # Decode query for one token: [B, Hq, D]
    q = torch.randn((B, Hq, D), device="cuda", dtype=dtype)

    # Page table for B=1:
    # indices lists the page IDs used; indptr says indices[0:active_pages] belongs to request 0.
    all_ids = torch.arange(num_pages_total, device="cuda", dtype=torch.int32)
    indices = all_ids[:active_pages].contiguous()
    indptr = torch.tensor([0, active_pages], device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)

    wrapper = make_wrapper()
    ms = run_once(wrapper, q, k_pages, v_pages, indptr, indices, last_page_len,
                  Hq, Hkv, D, page_size)

    out = {
        "B": B, "Hq": Hq, "Hkv": Hkv, "D": D, "page_size": page_size,
        "num_pages_total": num_pages_total,
        "active_pages": active_pages,
        "latency_ms": ms,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": torch.cuda.get_device_name(0),
    }
    print("[result]", out)

if __name__ == "__main__":
    main()
