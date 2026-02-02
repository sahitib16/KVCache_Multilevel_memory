"""
select_pages.py

A tiny wrapper that lets you swap page-selection implementations:
- "minimal_quest": our current min/max + TopK selector (already working)
- "quest_repo": calls QUEST repo code (once we locate the function)

The rest of the simulator should call:
  select_topk_pages(q_cpu, k_min, k_max, topk, impl=...)
"""

from typing import Literal
import torch

from quest_selector import compute_page_minmax, select_topk_pages as minimal_select

Impl = Literal["minimal_quest", "quest_repo"]

def select_topk_pages(q_cpu: torch.Tensor,
                      k_min: torch.Tensor,
                      k_max: torch.Tensor,
                      topk: int,
                      impl: Impl = "minimal_quest") -> torch.Tensor:
    """
    Returns page ids (CPU tensor).
    """
    if impl == "minimal_quest":
        return minimal_select(q_cpu, k_min, k_max, topk=topk)

    # Placeholder until we wire in QUEST repo code.
    # We'll replace this block once we locate the right QUEST function.
    raise NotImplementedError(
        "quest_repo impl not wired yet. Run the quest_repo scan step and paste output."
    )
