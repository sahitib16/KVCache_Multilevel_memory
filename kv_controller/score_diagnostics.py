from __future__ import annotations

"""Diagnostics for replay-time head-weighted score quality.

The head-weighted scoring plan says the next high-value work is not just
"collect more traces," but also:
- understand score distributions
- measure how well high-ranked pages predict next-step usage
- compare scorer views on the exact same trace

This module keeps those diagnostics reusable so scripts can report them without
re-implementing the same bookkeeping repeatedly.
"""

import math

from .types import KVPageId, WorkloadStep


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _safe_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def score_distribution_summary(trace: list[WorkloadStep]) -> dict[str, float]:
    """Summarize the raw score distribution over a replay trace."""

    values = [
        float(score)
        for step in trace
        for score in step.head_weighted_scores.values()
    ]
    if not values:
        return {
            "count": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "mean_score": 0.0,
            "std_score": 0.0,
        }
    return {
        "count": float(len(values)),
        "min_score": min(values),
        "max_score": max(values),
        "mean_score": _safe_mean(values),
        "std_score": _safe_std(values),
    }


def topk_next_step_hit_rates(
    trace: list[WorkloadStep],
    *,
    ks: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, float]:
    """Measure whether top-ranked pages are actually used on the next step.

    For each step:
    - rank pages by score
    - take top-k pages
    - check whether those pages appear in the next step's required set

    This is a lightweight way to ask:
    "does a high score actually correspond to near-future control value?"
    """

    hit_counts = {k: 0 for k in ks}
    total_counts = {k: 0 for k in ks}

    next_for_request = _next_step_by_request(trace)
    for index, step in enumerate(trace):
        next_step = next_for_request.get(index)
        if next_step is None:
            continue
        next_required = set(next_step.required_pages)
        ranked_pages = sorted(
            step.head_weighted_scores,
            key=lambda page: (step.head_weighted_scores.get(page, 0.0), page.layer_id, page.page_id),
            reverse=True,
        )
        for k in ks:
            topk = ranked_pages[:k]
            total_counts[k] += len(topk)
            hit_counts[k] += sum(1 for page in topk if page in next_required)

    return {
        f"top{k}_next_step_hit_rate": (
            hit_counts[k] / max(1, total_counts[k])
        )
        for k in ks
    }


def reused_page_rank_summary(trace: list[WorkloadStep]) -> dict[str, float]:
    """Measure where next-step reused pages tend to rank in the current score order.

    Lower normalized rank is better:
    - `0.0` means reused pages tend to be at the very top
    - `1.0` means reused pages tend to be near the bottom
    """

    normalized_ranks: list[float] = []
    next_for_request = _next_step_by_request(trace)
    for index, step in enumerate(trace):
        next_step = next_for_request.get(index)
        if next_step is None:
            continue
        next_required = set(next_step.required_pages)
        ranked_pages = sorted(
            step.head_weighted_scores,
            key=lambda page: (step.head_weighted_scores.get(page, 0.0), page.layer_id, page.page_id),
            reverse=True,
        )
        rank_lookup = {page: rank for rank, page in enumerate(ranked_pages)}
        denom = max(1, len(ranked_pages) - 1)
        for page in next_required:
            if page in rank_lookup:
                normalized_ranks.append(rank_lookup[page] / denom)

    return {
        "mean_reused_page_rank": _safe_mean(normalized_ranks),
        "std_reused_page_rank": _safe_std(normalized_ranks),
    }


def diagnose_trace_scores(trace: list[WorkloadStep]) -> dict[str, float]:
    """Return the main score-quality diagnostics for one trace."""

    return {
        **score_distribution_summary(trace),
        **topk_next_step_hit_rates(trace),
        **reused_page_rank_summary(trace),
    }


def _next_step_by_request(trace: list[WorkloadStep]) -> dict[int, WorkloadStep]:
    """Map each step index to the next step for the same request.

    This matters for interleaved traces:
    - the next global step may belong to a different request
    - but score usefulness should be judged against the next decode step of the
      *same* request
    """

    by_request: dict[str, list[int]] = {}
    for index, step in enumerate(trace):
        by_request.setdefault(step.request_id, []).append(index)

    next_map: dict[int, WorkloadStep] = {}
    for indices in by_request.values():
        for pos in range(len(indices) - 1):
            current_index = indices[pos]
            next_index = indices[pos + 1]
            next_map[current_index] = trace[next_index]
    return next_map
