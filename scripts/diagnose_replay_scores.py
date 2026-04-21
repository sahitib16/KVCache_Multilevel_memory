#!/usr/bin/env python3
"""Print replay-time diagnostics for scorer quality.

This script complements `evaluate_replay_scores.py`.

`evaluate_replay_scores.py` answers:
- which controller did better under a scorer?

This script answers:
- what does the scorer signal itself look like?
- does it correlate with next-step usage?
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kv_controller import (
    attach_reuse_distance_features,
    HeadActivityRecomputedScorer,
    LayerNormalizedHeadActivityScorer,
    NormalizedHeadWeightedScorer,
    PageStatsHybridScorer,
    PassthroughHeadWeightedScorer,
    PredictedBoostedHeadActivityScorer,
    ReuseDistanceHybridScorer,
    apply_scorer_to_trace,
    diagnose_trace_scores,
    load_trace_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose replay-time score quality.")
    parser.add_argument(
        "--trace-jsons",
        type=str,
        required=True,
        help="Comma-separated replay traces to diagnose.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    traces = [attach_reuse_distance_features(load_trace_json(path.strip())) for path in args.trace_jsons.split(",") if path.strip()]

    scorers = {
        "passthrough": PassthroughHeadWeightedScorer(),
        "normalized": NormalizedHeadWeightedScorer(),
        "recomputed": HeadActivityRecomputedScorer(),
        "layer_normalized": LayerNormalizedHeadActivityScorer(),
        "predicted_boosted": PredictedBoostedHeadActivityScorer(),
        "reuse_hybrid": ReuseDistanceHybridScorer(),
        "page_stats_hybrid": PageStatsHybridScorer(),
    }

    for scorer_name, scorer in scorers.items():
        rows = [
            diagnose_trace_scores(apply_scorer_to_trace(trace, scorer))
            for trace in traces
        ]
        avg = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in rows[0]
        }
        print(f"\nSCORER: {scorer_name}")
        for key, value in avg.items():
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
