from __future__ import annotations

"""Transform replay traces into pressure-inducing controller workloads.

Why this file exists:
- Real dense causal traces are useful because they contain *real* attention and
  per-head activity signals.
- But if every historical block is always marked as required, replay becomes
  too easy: the final required set is basically the full working set, so once
  pages arrive they never need to be evicted.

This file provides a few offline transformations that keep the real head signal
but change the replay semantics so the controller has to make choices.

The current transformations are:
- recent-topk:
  keep a small recent mandatory window plus top-k older attended blocks
- recent-threshold:
  keep a small recent mandatory window plus older blocks until a cumulative
  score-mass target is reached
- round-robin interleave:
  combine several traces from different requests into one replay stream while
  remapping page ids so requests do not collide
"""

from dataclasses import replace

from .types import KVPageId, WorkloadStep


def _page_score(step: WorkloadStep, page: KVPageId) -> float:
    """Return the page score stored on a step, defaulting safely to zero."""

    return float(step.head_weighted_scores.get(page, 0.0))


def _pages_by_layer(pages: tuple[KVPageId, ...]) -> dict[int, list[KVPageId]]:
    grouped: dict[int, list[KVPageId]] = {}
    for page in pages:
        grouped.setdefault(page.layer_id, []).append(page)
    for layer_pages in grouped.values():
        layer_pages.sort(key=lambda page: page.page_id)
    return grouped


def _recent_pages_for_layer(
    step: WorkloadStep,
    layer_id: int,
    recent_block_window: int,
) -> list[KVPageId]:
    """Return the most recent logical blocks for one layer.

    We treat the newest few blocks as mandatory because dense causal decode
    almost always cares about recent context even when we later sparsify the
    older portion of the prefix.
    """

    blocks = list(step.layer_block_tables.get(layer_id, ()))
    if not blocks:
        return []
    recent_blocks = blocks[-recent_block_window:]
    return [KVPageId(layer_id=layer_id, page_id=block_id) for block_id in recent_blocks]


def _older_pages_sorted_by_score(
    step: WorkloadStep,
    layer_id: int,
    excluded_pages: set[KVPageId],
) -> list[KVPageId]:
    """Return older candidate pages for a layer, highest-score first."""

    layer_pages = [
        page for page in _pages_by_layer(step.required_pages).get(layer_id, [])
        if page not in excluded_pages
    ]
    layer_pages.sort(key=lambda page: (_page_score(step, page), page.page_id), reverse=True)
    return layer_pages


def _predicted_from_next_required(
    transformed_steps: list[WorkloadStep],
    step_index: int,
) -> tuple[KVPageId, ...]:
    """Use the next step's transformed required set as the current prediction."""

    if step_index + 1 >= len(transformed_steps):
        return ()
    current_required = set(transformed_steps[step_index].required_pages)
    next_required = transformed_steps[step_index + 1].required_pages
    return tuple(page for page in next_required if page not in current_required)


def convert_trace_recent_topk(
    trace: list[WorkloadStep],
    *,
    recent_block_window: int = 1,
    top_k_older_per_layer: int = 1,
) -> list[WorkloadStep]:
    """Offline converter: recent mandatory window + top-k older attended blocks.

    This is the simplest sparse reinterpretation of the dense trace:
    - always keep the newest few blocks per layer mandatory
    - select only the top-k older blocks per layer based on the real score
    """

    converted: list[WorkloadStep] = []
    for step in trace:
        required_pages: list[KVPageId] = []
        for layer_id in step.referenced_layers:
            recent_pages = _recent_pages_for_layer(step, layer_id, recent_block_window)
            required_pages.extend(recent_pages)
            older_candidates = _older_pages_sorted_by_score(step, layer_id, set(recent_pages))
            required_pages.extend(older_candidates[:top_k_older_per_layer])

        # Keep deterministic order for reproducible debugging and replay.
        required_pages = sorted(set(required_pages))
        converted.append(replace(step, required_pages=tuple(required_pages), predicted_pages=()))

    return [
        replace(step, predicted_pages=_predicted_from_next_required(converted, index))
        for index, step in enumerate(converted)
    ]


def convert_trace_recent_threshold(
    trace: list[WorkloadStep],
    *,
    recent_block_window: int = 1,
    score_mass_fraction: float = 0.5,
) -> list[WorkloadStep]:
    """Offline converter: recent mandatory window + enough old blocks to hit a score budget.

    This formulation keeps a variable number of old blocks:
    - recent blocks are always mandatory
    - then older blocks are added in descending score order until they cover a
      chosen fraction of older score mass
    """

    converted: list[WorkloadStep] = []
    for step in trace:
        required_pages: list[KVPageId] = []
        for layer_id in step.referenced_layers:
            recent_pages = _recent_pages_for_layer(step, layer_id, recent_block_window)
            required_pages.extend(recent_pages)

            older_candidates = _older_pages_sorted_by_score(step, layer_id, set(recent_pages))
            total_older_score = sum(_page_score(step, page) for page in older_candidates)
            target_score = total_older_score * score_mass_fraction

            accumulated = 0.0
            for page in older_candidates:
                if accumulated >= target_score and target_score > 0.0:
                    break
                required_pages.append(page)
                accumulated += _page_score(step, page)

        required_pages = sorted(set(required_pages))
        converted.append(replace(step, required_pages=tuple(required_pages), predicted_pages=()))

    return [
        replace(step, predicted_pages=_predicted_from_next_required(converted, index))
        for index, step in enumerate(converted)
    ]


def _remap_page(page: KVPageId, page_offset: int) -> KVPageId:
    return KVPageId(layer_id=page.layer_id, page_id=page.page_id + page_offset)


def _offset_trace_pages(
    trace: list[WorkloadStep],
    *,
    page_offset: int,
    request_suffix: str,
) -> list[WorkloadStep]:
    """Make a trace request-unique by shifting page ids.

    ``KVPageId`` does not currently include a request id, so if we want to
    interleave multiple requests we need to offset page ids so that request A's
    layer-3/page-5 is not mistaken for request B's layer-3/page-5.
    """

    shifted: list[WorkloadStep] = []
    for step in trace:
        required_pages = tuple(_remap_page(page, page_offset) for page in step.required_pages)
        predicted_pages = tuple(_remap_page(page, page_offset) for page in step.predicted_pages)
        scores = {
            _remap_page(page, page_offset): score
            for page, score in step.head_weighted_scores.items()
        }
        head_activity = {
            _remap_page(page, page_offset): values
            for page, values in step.per_page_head_activity.items()
        }
        features = {
            _remap_page(page, page_offset): page_features
            for page, page_features in step.per_page_features.items()
        }
        block_tables = {
            layer_id: tuple(block_id + page_offset for block_id in blocks)
            for layer_id, blocks in step.layer_block_tables.items()
        }
        shifted.append(
            replace(
                step,
                required_pages=required_pages,
                predicted_pages=predicted_pages,
                head_weighted_scores=scores,
                per_page_head_activity=head_activity,
                per_page_features=features,
                layer_block_tables=block_tables,
                request_id=f"{step.request_id}_{request_suffix}",
            )
        )
    return shifted


def interleave_traces_round_robin(
    traces: list[list[WorkloadStep]],
    *,
    page_stride: int = 1000,
) -> list[WorkloadStep]:
    """Pressure-inducing formulation: interleave multiple requests.

    This is the cleanest way to create controller pressure from real dense
    traces without pretending that one single request becomes sparse by itself.
    We keep each request's real head signal, but alternate decode steps across
    requests so they compete for the same HBM space.
    """

    shifted_traces = [
        _offset_trace_pages(trace, page_offset=index * page_stride, request_suffix=f"rr{index}")
        for index, trace in enumerate(traces)
    ]

    merged: list[WorkloadStep] = []
    max_len = max((len(trace) for trace in shifted_traces), default=0)
    for local_step in range(max_len):
        for trace in shifted_traces:
            if local_step < len(trace):
                merged.append(trace[local_step])

    return [replace(step, step_idx=index) for index, step in enumerate(merged)]
