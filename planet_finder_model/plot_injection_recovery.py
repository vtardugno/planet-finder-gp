"""Plot the injection-recovery grid produced by injection_recovery.py.

Reads the results CSV and draws period x semi-amplitude recovery-fraction
heatmaps for the cyc and nonstat models. Points are binned into a coarser
log-spaced grid (rather than plotted at their exact, irregularly-spaced
injected values) so results pooled from multiple runs/machines render as a
clean heatmap, with each cell annotated with its recovered percentage and the
underlying fraction (n_recovered/n_total).
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
    parser.add_argument("--n-period-bins", type=int, default=6)
    parser.add_argument("--n-k-bins", type=int, default=6)
    parser.add_argument("--period-bin-edges", type=str, default=None,
                         help="Comma-separated explicit period bin edges (days), overrides --n-period-bins.")
    parser.add_argument("--k-bin-edges", type=str, default=None,
                         help="Comma-separated explicit K bin edges (m/s), overrides --n-k-bins. Use this "
                              "to concentrate resolution in a region of interest, e.g. near the detection "
                              "threshold, rather than uniform log-spacing across the whole range.")
    return parser


def bin_edges(values, n_bins):
    lo, hi = values.min(), values.max()
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


def binned_recovery(df, mode, period_edges, k_edges):
    sub = df[df["mode"] == mode].copy()
    n_pb, n_kb = len(period_edges) - 1, len(k_edges) - 1

    sub["period_bin"] = np.clip(np.digitize(sub["period_inj"], period_edges) - 1, 0, n_pb - 1)
    sub["k_bin"] = np.clip(np.digitize(sub["k_inj_ms"], k_edges) - 1, 0, n_kb - 1)

    frac = np.full((n_kb, n_pb), np.nan)
    n_recovered = np.zeros((n_kb, n_pb), dtype=int)
    n_total = np.zeros((n_kb, n_pb), dtype=int)

    grouped = sub.groupby(["period_bin", "k_bin"])["recovered"].agg(["sum", "count"])
    for (pb, kb), row in grouped.iterrows():
        frac[kb, pb] = row["sum"] / row["count"]
        n_recovered[kb, pb] = row["sum"]
        n_total[kb, pb] = row["count"]

    return frac, n_recovered, n_total


def plot_panel(ax, frac, n_recovered, n_total, period_edges, k_edges, title):
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("white")
    masked = np.ma.masked_invalid(frac)

    im = ax.pcolormesh(period_edges, k_edges, masked, cmap=cmap, vmin=0, vmax=1,
                        edgecolors="white", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Period (days)")
    ax.set_title(title)

    n_kb, n_pb = frac.shape
    for ki in range(n_kb):
        for pi in range(n_pb):
            if n_total[ki, pi] == 0:
                continue
            x_c = np.sqrt(period_edges[pi] * period_edges[pi + 1])
            y_c = np.sqrt(k_edges[ki] * k_edges[ki + 1])
            r, g, b, _ = cmap(frac[ki, pi])
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if lum < 0.5 else "black"
            pct = frac[ki, pi] * 100
            label = f"{pct:.0f}%\n({n_recovered[ki, pi]}/{n_total[ki, pi]})"
            ax.text(x_c, y_c, label, ha="center", va="center",
                     color=color, fontsize=6.5, fontweight="bold", linespacing=1.3)

    return im


def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input_csv)
    df["k_inj_ms"] = df["k_inj"] * 1000.0

    if args.period_bin_edges:
        period_edges = np.array([float(x) for x in args.period_bin_edges.split(",")])
    else:
        period_edges = bin_edges(df["period_inj"].to_numpy(), args.n_period_bins)

    if args.k_bin_edges:
        k_edges = np.array([float(x) for x in args.k_bin_edges.split(",")])
    else:
        k_edges = bin_edges(df["k_inj_ms"].to_numpy(), args.n_k_bins)

    mode_titles = {"cyc": "cyc", "nonstat": "nonstat", "no_cycle": "no cycle"}
    present_modes = [m for m in ("cyc", "nonstat", "no_cycle") if m in df["mode"].unique()]

    fig, axs = plt.subplots(1, len(present_modes), figsize=(7 * len(present_modes), 11), sharey=True)
    if len(present_modes) == 1:
        axs = [axs]

    for ax, mode in zip(axs, present_modes):
        frac, n_recovered, n_total = binned_recovery(df, mode, period_edges, k_edges)
        im = plot_panel(ax, frac, n_recovered, n_total, period_edges, k_edges, mode_titles[mode])

    fig.colorbar(im, ax=axs, label="Recovered fraction", fraction=0.046, pad=0.02)
    axs[0].set_ylabel("K (m/s)")
    fig.suptitle("Injection Recovery Grid (within 10% in P and 15% in K)")

    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
