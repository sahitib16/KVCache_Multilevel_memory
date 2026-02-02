"""
quest_repo_selector.py

Uses QUEST repository's Python evaluation code (evaluation/quest_attention.py)
to create a sparse selection mask, then converts that into selected page IDs.

This avoids compiling quest._kernels while still using QUEST repo code.
"""

import os, sys
import torch

# Import from external/Quest/evaluation/quest_attention.py
QUEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "Quest"))
sys.path.insert(0, QUEST_ROOT)

from evaluation.quest_attention import local_heavy_hitter_mask  # uses only torch

@torch.no_grad()
def select_pages_via_quest_eval(
    q_cuda: torch.Tensor,
    k_cpu: torch.Tensor,
    page_size: int,
    topk_pages: int,
    chunk_size: int = 16,
    token_budget: int = 128,
) -> list[int]:
    """
    Inputs:
      q_cuda: [B, Hq, D] (decode query on GPU)
      k_cpu:  [num_pages, page_size, Hkv, D] (keys on CPU pinned; we'll pull a CPU view)
      page_size: tokens per page
      topk_pages: number of pages to return
      chunk_size/token_budget: parameters used by QUEST evaluation selection

    Output:
      list of selected logical page IDs
    """

    # ---- Build a simple attention score vector over tokens (CPU) ----
    # Flatten keys to token sequence: [S, D] for a single head.
    # We take head 0 to keep it simple and fast.
    k0 = k_cpu[:, :, 0, :].contiguous()            # [num_pages, page_size, D]
    S = k0.shape[0] * k0.shape[1]
    k_flat = k0.view(S, k0.shape[-1])              # [S, D]

    # Move q head 0 to CPU and compute dot(q, k) -> [S]
    q0 = q_cuda[0, 0, :].detach().to("cpu")        # [D]
    scores = (k_flat @ q0).view(1, 1, 1, S)        # [BS=1, head=1, query=1, keys=S]

    # ---- Use QUEST repo code to compute a mask of selected tokens ----
    # local_heavy_hitter_mask returns a boolean mask over tokens
    mask = local_heavy_hitter_mask(scores, token_budget=token_budget, chunk_size=chunk_size)
    # mask shape: [1, 1, 1, S] bool

    # Extract selected token indices
    sel_tokens = mask[0, 0, 0].nonzero(as_tuple=False).flatten()  # [num_selected_tokens]

    # Convert token index -> page id
    sel_pages = torch.unique(sel_tokens // page_size).tolist()

    # Cap to topk_pages (stable ordering: keep smallest page ids; you can refine later)
    sel_pages = sel_pages[:topk_pages]
    return sel_pages
