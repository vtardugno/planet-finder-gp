"""Injection-recovery sweep for the no_cycle baseline (bare GP, no cycle term
at all -- functions.py's mf.optimise_params / mf.negloglike_nocyc).

Rather than re-sweeping its own (period, K, phase) grid, this sources the
exact injected combos already present in an existing injection_recovery.py
results CSV (e.g. the pooled cyc/nonstat CSV) and fits only the no_cycle
model against each one. This means:
  - cyc/nonstat results already computed are never redone.
  - the no_cycle rows use the *identical* injected signal (same period_inj,
    k_inj, phase_inj, hence same A_inj/B_inj) as the existing cyc/nonstat
    rows for that combo, and inherit their exact period_idx/k_idx/phase_idx,
    so the output drops straight into pool_results.py / plot_injection_recovery.py
    alongside the existing data with no re-clustering ambiguity.

Resumable: re-running the same command skips combos that already have a
no_cycle row in --output-csv.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from spleaf import cov, term

import functions as mf
import injection_recovery as ir


def build_parser():
    parser = ir.build_parser()
    parser.description = ("Fit the no_cycle baseline model against the injected combos "
                           "already present in an existing injection_recovery.py results CSV.")
    parser.add_argument("--source-csv", default="results/injection_recovery_pooled.csv",
                         help="Existing injection_recovery.py-format CSV to source injected "
                              "(period, K, phase) combos from.")
    parser.set_defaults(output_csv="results/injection_recovery_nocycle.csv")

    # main.py's build_bounds_list() widens rho/eta and delta1 when --no-fit-cycle
    # is passed (no cycle term to absorb long-period variability, so the GP's own
    # rho/eta need more room), and always uses its own delta0 default regardless
    # of --fit-cycle. These are the --no-fit-cycle values, not the cyc/nonstat
    # ones inherited from ir.build_parser() above.
    parser.set_defaults(rho_max=100.0, eta_max=3.0, delta0_min=-0.5, delta0_max=0.5,
                         delta1_min=-5.0, delta1_max=5.0)
    return parser


def build_bounds_list_nocyc(args, stds, period_bounds, amp_bound):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]

    return [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (args.prot_min, args.prot_max),
        (args.rho_min, args.rho_max),
        (args.eta_min, args.eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (args.delta0_min, args.delta0_max),
        (args.delta1_min, args.delta1_max),
        period_bounds,
        (-amp_bound, amp_bound),
        (-amp_bound, amp_bound),
    ]


def fit_nocyc(t_full, y_full, yerr_full, series_index, stds, args, rv_offset0, rhk_offset0, planet_guess, amp_bound, rv_std):
    best_loglike = -np.inf
    xbest_all = None

    for p in planet_guess:
        period_bounds = (max(p - 10, 1.1), min(p + 10, args.period_max))
        bounds_list = build_bounds_list_nocyc(args, stds, period_bounds, amp_bound)

        for Afrac in ir.A_INIT_FRACS:
            for Bfrac in ir.B_INIT_FRACS:
                C = cov.Cov(
                    t_full,
                    err=term.Error(yerr_full),
                    rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
                    rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
                    rot=ir.MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                             np.array([stds[0], stds[1]]),
                                             np.array([stds[0], 0.0])),
                )

                xbest, C = mf.optimise_params(
                    t_full, y_full, series_index, C, bounds_list,
                    delta_0=rv_offset0, delta_1=rhk_offset0,
                    planet_p=p, planet_A=args.planet_A_fit * Afrac, planet_B=args.planet_B_fit * Bfrac,
                    fit_planet=True, change_C=True,
                )

                loglike = -1 * mf.negloglike_nocyc(xbest, t_full, y_full, series_index, C, rv_std, inject_planet=True)[0]

                if loglike > best_loglike:
                    best_loglike = loglike
                    xbest_all = xbest

    return xbest_all, best_loglike


def run_one_combo(args_dict, period_idx, k_idx, phase_idx, period, k, phase):
    args = argparse.Namespace(**args_dict)

    A_inj = k * np.cos(phase)
    B_inj = k * np.sin(phase)

    load_result = mf.load_and_norm_data(
        args.path, args.star_name, normalise=args.normalise,
        inject_planet=True, planet_params=(period, A_inj, B_inj),
    )
    if args.normalise:
        t_full, y_full, yerr_full, series_index, rv_std = load_result
    else:
        t_full, y_full, yerr_full, series_index = load_result
        rv_std = 1.0

    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]
    amp_bound = args.planet_amp_max if args.planet_amp_max is not None else 1.5 * args.k_max

    # main.py's --no-fit-cycle path never fits a cycle (nothing to warm-start),
    # and seeds delta_0/delta_1 straight from the raw data means.
    rv_offset0 = float(np.mean(y_full[series_index[0]]))
    rhk_offset0 = float(np.mean(y_full[series_index[1]]))
    planet_guess, _ = mf.period_guess(
        t_full, y_full, yerr_full, series_index,
        PMIN=1.1, PMAX=args.period_max, MAX_FAP=1e-5, MAX_NPL=2, plot=False,
    )

    try:
        xbest, loglike = fit_nocyc(t_full, y_full, yerr_full, series_index, stds, args,
                                    rv_offset0, rhk_offset0, planet_guess, amp_bound, rv_std)
    except Exception as exc:
        print(f"[combo p_idx={period_idx} k_idx={k_idx} phase_idx={phase_idx} mode=no_cycle] "
              f"failed: {exc}", flush=True)
        xbest, loglike = None, float("nan")

    return [ir.make_row(period_idx, k_idx, phase_idx, period, k, phase, A_inj, B_inj, "no_cycle", xbest, loglike)]


def _worker(task):
    return run_one_combo(*task)


def load_source_combos(source_csv):
    df = pd.read_csv(source_csv)
    sub = df.drop_duplicates(subset=["period_idx", "k_idx", "phase_idx"])
    return sub[["period_idx", "k_idx", "phase_idx", "period_inj", "k_inj", "phase_inj"]]


def load_completed_combos(csv_path):
    if not os.path.exists(csv_path):
        return set()

    done = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["mode"] == "no_cycle":
                done.add((int(row["period_idx"]), int(row["k_idx"]), int(row["phase_idx"])))
    return done


def main():
    args = build_parser().parse_args()

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.force_fresh and os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    combos = load_source_combos(args.source_csv)
    completed = load_completed_combos(args.output_csv)

    args_dict = vars(args)
    tasks = []
    for row in combos.itertuples(index=False):
        key = (int(row.period_idx), int(row.k_idx), int(row.phase_idx))
        if key in completed:
            continue
        tasks.append((args_dict, key[0], key[1], key[2],
                      float(row.period_inj), float(row.k_inj), float(row.phase_inj)))

    print(f"{len(combos)} source combos, {len(completed)} already done, {len(tasks)} remaining.", flush=True)

    if not tasks:
        print("Nothing to do.")
        return

    n_workers = args.n_workers or max(1, min(9, (os.cpu_count() or 2) - 1))
    print(f"Using {n_workers} worker processes.", flush=True)

    file_is_new = not os.path.exists(args.output_csv)
    csv_file = open(args.output_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=ir.CSV_FIELDS)
    if file_is_new:
        writer.writeheader()
        csv_file.flush()

    start_time = time.time()
    n_done = 0

    try:
        with mp.Pool(n_workers) as pool:
            for rows in pool.imap_unordered(_worker, tasks):
                for row in rows:
                    writer.writerow(row)
                csv_file.flush()

                n_done += 1
                elapsed = time.time() - start_time
                eta_min = (elapsed / n_done) * (len(tasks) - n_done) / 60.0
                print(f"[{n_done}/{len(tasks)}] combo done "
                      f"(elapsed={elapsed / 60.0:.1f} min, ETA={eta_min:.1f} min)", flush=True)
    finally:
        csv_file.close()

    print("Done.")


if __name__ == "__main__":
    main()
