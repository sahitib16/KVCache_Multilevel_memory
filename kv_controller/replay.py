from __future__ import annotations

"""Offline trace replay helpers for the KV controller simulator.

Why this file exists:
- Synthetic traces are useful for controlled development.
- But before live-engine work, we want a way to replay externally collected
  traces offline through the same simulator and controller interfaces.

This file defines a very small JSON trace format and helpers to:
- serialize a synthetic or instrumented trace
- reload that trace later
- replay it through the simulator without regenerating it

That gives us a stepping stone between pure simulation and real engine
integration.
"""

import json

from .types import KVPageId, WorkloadStep


def _page_to_json(page: KVPageId) -> dict[str, int]:
    """Serialize one page id into a JSON-friendly dictionary."""

    return {
        "layer_id": page.layer_id,
        "page_id": page.page_id,
    }


def _page_from_json(data: dict[str, int]) -> KVPageId:
    """Deserialize one page id from a JSON dictionary."""

    return KVPageId(
        layer_id=int(data["layer_id"]),
        page_id=int(data["page_id"]),
    )


def _scores_to_json(scores: dict[KVPageId, float]) -> list[dict[str, object]]:
    """Serialize page-keyed score dictionaries into a stable JSON list."""

    rows = []
    for page, score in sorted(scores.items()):
        rows.append(
            {
                "page": _page_to_json(page),
                "score": float(score),
            }
        )
    return rows


def _scores_from_json(rows: list[dict[str, object]]) -> dict[KVPageId, float]:
    """Deserialize score rows back into a page-keyed dictionary."""

    return {
        _page_from_json(row["page"]): float(row["score"])
        for row in rows
    }


def _features_to_json(features: dict[KVPageId, dict[str, float]]) -> list[dict[str, object]]:
    """Serialize per-page features into a JSON-friendly list."""

    rows = []
    for page, page_features in sorted(features.items()):
        rows.append(
            {
                "page": _page_to_json(page),
                "features": {key: float(value) for key, value in page_features.items()},
            }
        )
    return rows


def _head_activity_to_json(activity: dict[KVPageId, tuple[float, ...]]) -> list[dict[str, object]]:
    """Serialize per-page per-head activity into JSON-friendly rows."""

    rows = []
    for page, values in sorted(activity.items()):
        rows.append(
            {
                "page": _page_to_json(page),
                "head_activity": [float(value) for value in values],
            }
        )
    return rows


def _head_activity_from_json(rows: list[dict[str, object]]) -> dict[KVPageId, tuple[float, ...]]:
    """Deserialize per-page head activity from replay JSON."""

    return {
        _page_from_json(row["page"]): tuple(float(value) for value in row["head_activity"])
        for row in rows
    }


def _features_from_json(rows: list[dict[str, object]]) -> dict[KVPageId, dict[str, float]]:
    """Deserialize per-page features from the replay format."""

    return {
        _page_from_json(row["page"]): {
            str(key): float(value)
            for key, value in dict(row["features"]).items()
        }
        for row in rows
    }


def trace_to_json_rows(trace: list[WorkloadStep]) -> list[dict[str, object]]:
    """Convert a trace into the replay JSON structure.

    The format is intentionally explicit rather than clever so it is easy to:
    - inspect by hand
    - generate from future instrumentation code
    - evolve carefully as the project grows
    """

    rows = []
    for step in trace:
        rows.append(
            {
                "step_idx": step.step_idx,
                "required_pages": [_page_to_json(page) for page in step.required_pages],
                "predicted_pages": [_page_to_json(page) for page in step.predicted_pages],
                "head_weighted_scores": _scores_to_json(step.head_weighted_scores),
                "query_head_weights": {str(layer_id): list(weights) for layer_id, weights in step.query_head_weights.items()},
                "per_page_head_activity": _head_activity_to_json(step.per_page_head_activity),
                "per_page_features": _features_to_json(step.per_page_features),
                "referenced_layers": list(step.referenced_layers),
                "request_id": step.request_id,
                "decode_position": step.decode_position,
                "sequence_length": step.sequence_length,
                "kv_block_size_tokens": step.kv_block_size_tokens,
                "layer_block_tables": {
                    str(layer_id): list(blocks)
                    for layer_id, blocks in step.layer_block_tables.items()
                },
            }
        )
    return rows


def save_trace_json(path: str, trace: list[WorkloadStep]) -> None:
    """Write a replayable trace to disk as JSON."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace_to_json_rows(trace), f, indent=2)


def load_trace_json(path: str) -> list[WorkloadStep]:
    """Load a replay trace from JSON into `WorkloadStep` objects."""

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    trace: list[WorkloadStep] = []
    for row in rows:
        trace.append(
            WorkloadStep(
                step_idx=int(row["step_idx"]),
                required_pages=tuple(_page_from_json(page) for page in row["required_pages"]),
                predicted_pages=tuple(_page_from_json(page) for page in row["predicted_pages"]),
                head_weighted_scores=_scores_from_json(row["head_weighted_scores"]),
                query_head_weights={
                    int(layer_id): tuple(float(value) for value in weights)
                    for layer_id, weights in dict(row["query_head_weights"]).items()
                },
                per_page_head_activity=_head_activity_from_json(row.get("per_page_head_activity", [])),
                per_page_features=_features_from_json(row.get("per_page_features", [])),
                referenced_layers=tuple(int(layer_id) for layer_id in row.get("referenced_layers", [])),
                request_id=str(row.get("request_id", "")),
                decode_position=int(row.get("decode_position", row.get("step_idx", 0))),
                sequence_length=int(row.get("sequence_length", 0)),
                kv_block_size_tokens=int(row.get("kv_block_size_tokens", 16)),
                layer_block_tables={
                    int(layer_id): tuple(int(block_id) for block_id in blocks)
                    for layer_id, blocks in dict(row.get("layer_block_tables", {})).items()
                },
            )
        )
    return trace
