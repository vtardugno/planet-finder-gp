"""Pool multiple injection_recovery.py output CSVs into one combined dataset.

Different runs (different grids, different machines) have period_idx/k_idx
columns that are NOT comparable to each other -- they're just positions into
that run's own grid array. This ignores those columns entirely and re-derives
a single unified grid from the actual injected period_inj/k_inj values across
all inputs, clustering values that agree to within --rel-tol (so the same
physical grid point computed by two different runs/machines merges into one
cell, pooling their phase trials together, while genuinely different grid
points from different runs are kept as separate cells). The output has the
same schema as a single injection_recovery.py CSV, so it's a drop-in input
for plot_injection_recovery.py.
"""

import argparse
import glob
import numpy as np
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(description="Pool multiple injection-recovery CSVs into one merged grid.")
    parser.add_argument("inputs", nargs="+", help="CSV file paths or glob patterns to pool together.")
    parser.add_argument("--output", default="results/injection_recovery_pooled.csv")
    parser.add_argument("--rel-tol", type=float, default=1e-3,
                         help="Relative tolerance for treating two period/K values from "
                              "different runs as the same grid point.")
    return parser


def cluster_values(values, rel_tol):
    """Map each value to a canonical (cluster-mean) value, merging consecutive
    sorted values that are within rel_tol of the cluster's first member. Grid
    points are always spaced far more than rel_tol apart by construction (even
    interleaved local+Glamdring grids differ by ~40%), so this only merges
    values that are meant to represent the same physical grid point."""
    order = np.argsort(values)
    sorted_vals = values[order]

    canonical = np.empty_like(sorted_vals, dtype=float)
    cluster_start = 0
    for i in range(1, len(sorted_vals) + 1):
        at_end = i == len(sorted_vals)
        if at_end or abs(sorted_vals[i] - sorted_vals[cluster_start]) > rel_tol * sorted_vals[cluster_start]:
            canonical[cluster_start:i] = np.mean(sorted_vals[cluster_start:i])
            cluster_start = i

    result = np.empty_like(canonical)
    result[order] = canonical
    return result


def main():
    args = build_parser().parse_args()

    paths = []
    for pattern in args.inputs:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])

    if not paths:
        raise SystemExit("No input files matched.")

    print(f"Pooling {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    n_rows_in = len(df)

    df["period_inj"] = cluster_values(df["period_inj"].to_numpy(dtype=float), args.rel_tol)
    df["k_inj"] = cluster_values(df["k_inj"].to_numpy(dtype=float), args.rel_tol)

    unique_periods = np.sort(df["period_inj"].unique())
    unique_ks = np.sort(df["k_inj"].unique())
    period_to_idx = {p: i for i, p in enumerate(unique_periods)}
    k_to_idx = {k: i for i, k in enumerate(unique_ks)}

    df["period_idx"] = df["period_inj"].map(period_to_idx)
    df["k_idx"] = df["k_inj"].map(k_to_idx)
    df["phase_idx"] = df.groupby(["period_idx", "k_idx", "mode"]).cumcount()

    df.to_csv(args.output, index=False)
    n_per_cell = df[df["mode"] == "cyc"].groupby(["period_idx", "k_idx"]).size()
    print(f"Pooled {n_rows_in} input rows -> {len(df)} rows, "
          f"{len(unique_periods)} periods x {len(unique_ks)} K values.")
    print(f"Phase trials per cell: min={n_per_cell.min()}, max={n_per_cell.max()}, "
          f"median={n_per_cell.median():.1f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
