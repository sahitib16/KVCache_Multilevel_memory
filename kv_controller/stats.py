from __future__ import annotations

"""Page-wise and tile-wise statistics for simulator runs.

These helpers turn one simulator run into the kind of low-level residency and
reuse summaries that are useful for validating whether the project is really
behaving like a page-wise simulator rather than only a policy benchmark.
"""

from collections import defaultdict
import csv
from statistics import mean

from .types import KVPageId, StepMetrics, WorkloadStep


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _group_to_tile(page: KVPageId, tile_size_pages: int) -> tuple[int, int]:
    return (page.layer_id, page.page_id // max(1, tile_size_pages))


def summarize_page_tile_stats(
    trace: list[WorkloadStep],
    metrics: list[StepMetrics],
    tile_size_pages: int = 4,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Build per-page, per-layer, and per-tile summaries from one run."""

    page_access_steps: dict[KVPageId, list[int]] = defaultdict(list)
    page_stats: dict[KVPageId, dict[str, object]] = defaultdict(
        lambda: {
            "layer_id": 0,
            "page_id": 0,
            "access_count": 0,
            "demand_miss_count": 0,
            "prefetch_submit_count": 0,
            "prefetch_hit_count": 0,
            "prefetch_wasted_count": 0,
            "eviction_count": 0,
            "resident_steps": 0,
            "first_access_step": None,
            "last_access_step": None,
        }
    )
    prefetched_waiting: set[KVPageId] = set()

    for step, step_metrics in zip(trace, metrics, strict=True):
        resident_end = set(step_metrics.resident_pages_end)

        for page in resident_end:
            row = page_stats[page]
            row["layer_id"] = page.layer_id
            row["page_id"] = page.page_id
            row["resident_steps"] += 1

        for page in step_metrics.prefetched_pages:
            row = page_stats[page]
            row["layer_id"] = page.layer_id
            row["page_id"] = page.page_id
            row["prefetch_submit_count"] += 1
            prefetched_waiting.add(page)

        for page in step_metrics.demand_miss_pages:
            row = page_stats[page]
            row["layer_id"] = page.layer_id
            row["page_id"] = page.page_id
            row["demand_miss_count"] += 1

        for page in step_metrics.evicted_pages:
            row = page_stats[page]
            row["layer_id"] = page.layer_id
            row["page_id"] = page.page_id
            row["eviction_count"] += 1
            if page in prefetched_waiting:
                row["prefetch_wasted_count"] += 1
                prefetched_waiting.discard(page)

        for page in step.required_pages:
            row = page_stats[page]
            row["layer_id"] = page.layer_id
            row["page_id"] = page.page_id
            row["access_count"] += 1
            page_access_steps[page].append(step.step_idx)
            if row["first_access_step"] is None:
                row["first_access_step"] = step.step_idx
            row["last_access_step"] = step.step_idx
            if page in prefetched_waiting:
                row["prefetch_hit_count"] += 1
                prefetched_waiting.discard(page)

    for page in list(prefetched_waiting):
        page_stats[page]["prefetch_wasted_count"] += 1

    page_rows: list[dict[str, object]] = []
    for page, row in sorted(page_stats.items()):
        access_steps = page_access_steps.get(page, [])
        reuse_distances = [
            float(current - previous)
            for previous, current in zip(access_steps, access_steps[1:], strict=False)
        ]
        page_rows.append(
            {
                "layer_id": row["layer_id"],
                "page_id": row["page_id"],
                "access_count": row["access_count"],
                "demand_miss_count": row["demand_miss_count"],
                "prefetch_submit_count": row["prefetch_submit_count"],
                "prefetch_hit_count": row["prefetch_hit_count"],
                "prefetch_wasted_count": row["prefetch_wasted_count"],
                "eviction_count": row["eviction_count"],
                "resident_steps": row["resident_steps"],
                "first_access_step": -1 if row["first_access_step"] is None else row["first_access_step"],
                "last_access_step": -1 if row["last_access_step"] is None else row["last_access_step"],
                "mean_reuse_distance": _safe_mean(reuse_distances),
                "short_reuse_count": sum(1 for value in reuse_distances if value <= 2.0),
                "long_reuse_count": sum(1 for value in reuse_distances if value > 2.0),
            }
        )

    layer_agg: dict[int, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    tile_agg: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for row in page_rows:
        layer_id = int(row["layer_id"])
        tile_id = _group_to_tile(KVPageId(layer_id=layer_id, page_id=int(row["page_id"])), tile_size_pages)[1]
        layer_row = layer_agg[layer_id]
        tile_row = tile_agg[(layer_id, tile_id)]
        for key in [
            "access_count",
            "demand_miss_count",
            "prefetch_submit_count",
            "prefetch_hit_count",
            "prefetch_wasted_count",
            "eviction_count",
            "resident_steps",
            "short_reuse_count",
            "long_reuse_count",
        ]:
            layer_row[key] += float(row[key])
            tile_row[key] += float(row[key])
        layer_row["page_count"] += 1.0
        tile_row["page_count"] += 1.0
        if float(row["mean_reuse_distance"]) > 0.0:
            layer_row["reuse_distance_sum"] += float(row["mean_reuse_distance"])
            layer_row["reuse_distance_count"] += 1.0
            tile_row["reuse_distance_sum"] += float(row["mean_reuse_distance"])
            tile_row["reuse_distance_count"] += 1.0

    layer_rows = []
    for layer_id, row in sorted(layer_agg.items()):
        layer_rows.append(
            {
                "layer_id": layer_id,
                "page_count": int(row["page_count"]),
                "access_count": int(row["access_count"]),
                "demand_miss_count": int(row["demand_miss_count"]),
                "prefetch_submit_count": int(row["prefetch_submit_count"]),
                "prefetch_hit_count": int(row["prefetch_hit_count"]),
                "prefetch_wasted_count": int(row["prefetch_wasted_count"]),
                "eviction_count": int(row["eviction_count"]),
                "resident_steps": int(row["resident_steps"]),
                "mean_page_reuse_distance": (
                    row["reuse_distance_sum"] / row["reuse_distance_count"]
                    if row["reuse_distance_count"] > 0
                    else 0.0
                ),
                "short_reuse_count": int(row["short_reuse_count"]),
                "long_reuse_count": int(row["long_reuse_count"]),
            }
        )

    tile_rows = []
    for (layer_id, tile_id), row in sorted(tile_agg.items()):
        tile_rows.append(
            {
                "layer_id": layer_id,
                "tile_id": tile_id,
                "tile_size_pages": tile_size_pages,
                "page_count": int(row["page_count"]),
                "access_count": int(row["access_count"]),
                "demand_miss_count": int(row["demand_miss_count"]),
                "prefetch_submit_count": int(row["prefetch_submit_count"]),
                "prefetch_hit_count": int(row["prefetch_hit_count"]),
                "prefetch_wasted_count": int(row["prefetch_wasted_count"]),
                "eviction_count": int(row["eviction_count"]),
                "resident_steps": int(row["resident_steps"]),
                "mean_page_reuse_distance": (
                    row["reuse_distance_sum"] / row["reuse_distance_count"]
                    if row["reuse_distance_count"] > 0
                    else 0.0
                ),
                "short_reuse_count": int(row["short_reuse_count"]),
                "long_reuse_count": int(row["long_reuse_count"]),
            }
        )

    return page_rows, layer_rows, tile_rows


def write_rows_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
