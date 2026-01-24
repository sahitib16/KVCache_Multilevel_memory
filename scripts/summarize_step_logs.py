import sys
import pandas as pd

def summarize(csv_path):
    df = pd.read_csv(csv_path)

    out = {}
    out["steps"] = len(df)
    out["mean_ondemand_misses"] = df["ondemand_misses"].mean()
    out["p95_ondemand_misses"] = df["ondemand_misses"].quantile(0.95)
    out["mean_evictions"] = df["evictions"].mean()
    out["p95_evictions"] = df["evictions"].quantile(0.95)
    out["mean_prefetch_misses"] = df["prefetch_misses"].mean()
    out["mean_ondemand_ms"] = df["ondemand_transfer_ms"].mean()
    out["mean_prefetch_ms"] = df["prefetch_transfer_ms"].mean()

    # how often are we doing worse than "just sparse misses"?
    num_sparse = df["needed_sparse"].iloc[0]
    out["frac_steps_high_miss"] = (df["ondemand_misses"] > num_sparse).mean()

    # correlation: does prefetch cause eviction?
    if df["prefetch_misses"].std() > 0 and df["evictions"].std() > 0:
        out["corr_prefetch_eviction"] = df["prefetch_misses"].corr(df["evictions"])
    else:
        out["corr_prefetch_eviction"] = 0.0

    return out

def main():
    for path in sys.argv[1:]:
        stats = summarize(path)
        print("\nFILE:", path)
        for k, v in stats.items():
            print(f"{k:>24}: {v:.4f}" if isinstance(v, float) else f"{k:>24}: {v}")

if __name__ == "__main__":
    main()
