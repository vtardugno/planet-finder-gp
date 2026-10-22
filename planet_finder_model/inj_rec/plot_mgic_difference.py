"""Plot how much injection recovery the dMGIC cut costs, per grid cell.

Difference = (param match only) - (param match AND dMGIC > threshold), i.e. the
fraction of injections whose P/K were recovered but which fail the MGIC check.
Uses the same binning as plot_injection_recovery.py so cells line up.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_injection_recovery import bin_edges, binned_recovery

plt.rcParams["font.size"] = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--modes", type=str, default=None)
    parser.add_argument("--output", default="inj_rec/injection_recovery_mgic_diff.png")
    parser.add_argument("--n-period-bins", type=int, default=6)
    parser.add_argument("--n-k-bins", type=int, default=6)
    parser.add_argument("--period-bin-edges", type=str, default=None)
    parser.add_argument("--k-bin-edges", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.modes:
        df = df[df["mode"].isin(args.modes.split(","))]
    df["k_inj_ms"] = df["k_inj"] * 1000.0
    df_nomgic = df.copy()
    df_nomgic["recovered"] = (df["param_match_pass"].astype(str) == "True").astype(int)

    if args.period_bin_edges:
        period_edges = np.array([float(x) for x in args.period_bin_edges.split(",")])
    else:
        period_edges = bin_edges(df["period_inj"].to_numpy(), args.n_period_bins)
    if args.k_bin_edges:
        k_edges = np.array([float(x) for x in args.k_bin_edges.split(",")])
    else:
        k_edges = bin_edges(df["k_inj_ms"].to_numpy(), args.n_k_bins)

    mode_titles = {"cyc": "Multi-GP + f_c", "nonstat2": "Nonstat-GP", "no_cycle": "Multi-GP"}
    mode_order = args.modes.split(",") if args.modes else ("cyc", "nonstat2", "no_cycle")
    present_modes = [m for m in mode_order if m in df["mode"].unique()]

    results = {}
    for mode in present_modes:
        frac_m, n_rec_m, n_total = binned_recovery(df, mode, period_edges, k_edges)
        frac_n, n_rec_n, _ = binned_recovery(df_nomgic, mode, period_edges, k_edges)
        results[mode] = (frac_n - frac_m, n_rec_n - n_rec_m, n_total)
    vmax = max(np.nanmax(d) for d, _, _ in results.values())

    fig, axs = plt.subplots(1, len(present_modes), figsize=(7 * len(present_modes), 9), sharey=True)
    if len(present_modes) == 1:
        axs = [axs]

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")
    for ax, mode in zip(axs, present_modes):
        diff, n_lost, n_total = results[mode]
        im = ax.pcolormesh(period_edges, k_edges, np.ma.masked_invalid(diff), cmap=cmap,
                           vmin=0, vmax=vmax, edgecolors="white", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="both", labelsize=12)
        ax.set_xlabel("Period (days)")
        title = mode_titles[mode]
        if "mgic_threshold" in df.columns:
            title += f" (dMGIC > {df.loc[df['mode'] == mode, 'mgic_threshold'].iloc[0]:.1f})"
        ax.set_title(title)

        for ki in range(diff.shape[0]):
            for pi in range(diff.shape[1]):
                if n_total[ki, pi] == 0:
                    continue
                x_c = np.sqrt(period_edges[pi] * period_edges[pi + 1])
                y_c = np.sqrt(k_edges[ki] * k_edges[ki + 1])
                color = "white" if diff[ki, pi] > 0.6 * vmax else "black"
                pct = f"-{diff[ki, pi] * 100:.0f}" if n_lost[ki, pi] else "0"
                ax.text(x_c, y_c, f"{pct} pp\n({n_lost[ki, pi]}/{n_total[ki, pi]})",
                        ha="center", va="center", color=color, fontsize=9,
                        fontweight="bold", linespacing=1.3)

    fig.colorbar(im, ax=axs, label="Recovered fraction lost to dMGIC cut",
                 fraction=0.046, pad=0.02)
    axs[0].set_ylabel("K (m/s)")
    fig.suptitle("Recovery lost to dMGIC cut (param match only - param match AND dMGIC)")

    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
