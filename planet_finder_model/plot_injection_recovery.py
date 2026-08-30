"""Plot the injection-recovery grid produced by injection_recovery.py.

Reads the results CSV and draws period x semi-amplitude recovery-fraction
heatmaps for the cyc and nonstat models, plus their difference.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser():
    parser = argparse.ArgumentParser(description="Plot the injection-recovery grid comparing cyc vs nonstat.")
    parser.add_argument("--input-csv", default="results/injection_recovery.csv")
    parser.add_argument("--output", default="results/injection_recovery_grid.png")
    return parser


def recovery_grid(df, mode, n_period, n_k):
    sub = df[df["mode"] == mode]
    grid = np.full((n_k, n_period), np.nan)
    for (pi, ki), frac in sub.groupby(["period_idx", "k_idx"])["recovered"].mean().items():
        grid[ki, pi] = frac
    return grid


def log_edges(centers):
    """Cell boundaries for a log-spaced axis, computed in log-space so
    pcolormesh cells line up correctly under set_xscale/set_yscale('log')
    (its default 'nearest'/'auto' shading interpolates edges linearly, which
    badly distorts cells when the centers are log-spaced)."""
    log_c = np.log10(centers)
    mid = (log_c[:-1] + log_c[1:]) / 2.0
    first = log_c[0] - (mid[0] - log_c[0])
    last = log_c[-1] + (log_c[-1] - mid[-1])
    edges = np.concatenate([[first], mid, [last]])
    return 10 ** edges


def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input_csv)

    n_period = int(df["period_idx"].max()) + 1
    n_k = int(df["k_idx"].max()) + 1

    periods = df.drop_duplicates("period_idx").sort_values("period_idx")["period_inj"].to_numpy()
    ks_m_s = df.drop_duplicates("k_idx").sort_values("k_idx")["k_inj"].to_numpy() * 1000.0
    period_edges = log_edges(periods)
    k_edges = log_edges(ks_m_s)

    grid_cyc = recovery_grid(df, "cyc", n_period, n_k)
    grid_nonstat = recovery_grid(df, "nonstat", n_period, n_k)
    grid_diff = grid_cyc - grid_nonstat

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    panels = [
        (axs[0], grid_cyc, "cyc", "viridis", 0, 1),
        (axs[1], grid_nonstat, "nonstat", "viridis", 0, 1),
        (axs[2], grid_diff, "cyc - nonstat", "RdBu_r", -1, 1),
    ]
    for ax, grid, title, cmap, vmin, vmax in panels:
        im = ax.pcolormesh(period_edges, k_edges, grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Injected period (days)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="recovery fraction" if title != "cyc - nonstat" else "cyc - nonstat")

    axs[0].set_ylabel("Injected semi-amplitude K (m/s)")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
