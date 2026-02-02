"""
summarize_step_logs_commented.py

Goal:
- Read step-level CSV logs from irregular_lru_log.py
- Compute summary statistics that explain behavior:
    mean/p95 on-demand misses
    mean/p95 evictions
    mean transfer costs
    fraction of "bad" steps
    correlation of prefetch activity vs evictions
"""

import sys
import pandas as pd

def summarize(csv_path):
    # Load the step-level log
    df = pd.read_csv(csv_path)

    out = {}
    out["steps"] = len(df)

    # Average and tail behavior for on-demand misses and evictions
    out["mean_ondemand_misses"] = df["ondemand_misses"].mean()
    out["p95_ondemand_misses"] = df["ondemand_misses"].quantile(0.95)
    out["mean_evictions"] = df["evictions"].mean()
    out["p95_evictions"] = df["evictions"].quantile(0.95)

    # Prefetch activity averages
    out["mean_prefetch_misses"] = df["prefetch_misses"].mean()

    # Average transfer costs
    out["mean_ondemand_ms"] = df["ondemand_transfer_ms"].mean()
    out["mean_prefetch_ms"] = df["prefetch_transfer_ms"].mean()

    # "Bad step" heuristic:
    # If on-demand misses are larger than the sparse component, the cache is doing poorly.
    num_sparse = df["needed_sparse"].iloc[0]
    out["frac_steps_high_miss"] = (df["ondemand_misses"] > num_sparse).mean()

    # Correlation: does prefetch coincide with evictions?
    if df["prefetch_misses"].std() > 0 and df["evictions"].std() > 0:
        out["corr_prefetch_eviction"] = df["prefetch_misses"].corr(df["evictions"])
    else:
        out["corr_prefetch_eviction"] = 0.0

    return out

def main():
    # Loop over all CSVs passed on the command line
    for path in sys.argv[1:]:
        stats = summarize(path)
        print("\nFILE:", path)
        for k, v in stats.items():
            # Pretty-print floats with 4 decimals
            if isinstance(v, float):
                print(f"{k:>24}: {v:.4f}")
            else:
                print(f"{k:>24}: {v}")

if __name__ == "__main__":
    main()
