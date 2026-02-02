"""
quest_selector.py

A minimal QUEST-style page selector (Python-only) for your simulator.

Idea (from QUEST paper):
- For each KV page, precompute per-channel min and max of the Key vectors.
- Given a query vector q, compute an upper-bound score per page using q and min/max.
- Select Top-K pages by score.

This approximates the QUEST selection step without compiling their kernels.
"""

import torch

def compute_page_minmax(k_pages_cpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-page min/max over tokens within a page.

    k_pages_cpu shape: [num_pages, page_size, Hkv, D]
    returns:
      k_min shape: [num_pages, Hkv, D]
      k_max shape: [num_pages, Hkv, D]
    """
    # min/max over token dimension (dim=1)
    k_min = k_pages_cpu.amin(dim=1)
    k_max = k_pages_cpu.amax(dim=1)
    return k_min, k_max

def quest_page_scores(q: torch.Tensor, k_min: torch.Tensor, k_max: torch.Tensor) -> torch.Tensor:
    """
    Compute QUEST-style upper-bound scores for each page.

    q shape: [Hq, D]  (for decode, you can use q[0] or average heads)
    k_min/k_max shape: [num_pages, Hkv, D]

    We need a single score per page. Simplest approach:
    - Map query heads -> kv heads by modulo (rough GQA mapping), or just use the first head.
    - Use one head score; later you can sum/avg across heads.

    Scoring intuition from paper:
    For each dimension i, upper bound of q_i * k_i within page is:
      max(q_i * k_max_i, q_i * k_min_i)
    Then sum across i for a page score.
    """
    # Pick one query head (0) and map to one KV head (0) for simplicity
    q0 = q[0]                      # [D]
    kmin0 = k_min[:, 0, :]         # [num_pages, D]
    kmax0 = k_max[:, 0, :]         # [num_pages, D]

    # For each page and each dimension, compute max(q*d*min, q*d*max)
    a = q0 * kmin0
    b = q0 * kmax0
    per_dim = torch.maximum(a, b)  # [num_pages, D]

    # Sum across dimensions -> page score
    scores = per_dim.sum(dim=1)    # [num_pages]
    return scores

def select_topk_pages(q: torch.Tensor, k_min: torch.Tensor, k_max: torch.Tensor, topk: int) -> torch.Tensor:
    """
    Return Top-K page ids by QUEST-style score.

    returns: page_ids shape [topk] on CPU (int64)
    """
    scores = quest_page_scores(q, k_min, k_max)
    topk = min(topk, scores.numel())
    vals, idx = torch.topk(scores, k=topk, largest=True)
    return idx
