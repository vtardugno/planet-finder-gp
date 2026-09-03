"""Empirically calibrate the per-GP-model likelihood-ratio detection threshold.

For each GP model (cyc, nonstat, no_cycle), this script:
  1. Fits that model (GP-only, no planet) once to the real, un-injected Solar
     data -- the fitted hyperparameters + mean-term coefficients become the
     fixed "generative truth" for step 2 (results/null_calibration/reference_fit_<mode>.json).
  2. Draws --n-null synthetic no-planet datasets at the real observing times
     and measurement uncertainties, using spleaf's Cov.sample() to realize the
     GP+noise covariance exactly. Each null dataset is fit with GP-only and
     GP+planet using the *same* fit_cyc/fit_nonstat/fit_nocyc functions (and
     the same period-search/multi-start/bounds machinery) that
     injection_recovery.py uses in production, imported directly from there
     so calibration and injection-recovery are provably the same pipeline.
  3. Computes T = 2*(logL_max(GP+planet) - logL_max(GP only)) for every null
     realization, and T_crit = the empirical (1-alpha) quantile of that null
     distribution -- no chi-square assumption, no fixed threshold.

Output (per mode, under --null-output-dir):
  reference_fit_<mode>.json   -- the one-time reference (no-planet) fit
  null_fits_<mode>.csv        -- every null realization's paired fits + diagnostics
  null_fits_<mode>.meta.json  -- calibration args snapshot, guards --rerun against
                                  silently mixing incompatible null draws
  tcrit.json                  -- combined per-mode T_crit + calibration summary
  summary_table.csv           -- gp_model, n_null, fpr_target, t_crit, valid/failed counts

Run directly on glamdring, e.g.:
  python null_calibration.py --models cyc,nonstat,no_cycle --n-null 500 --alpha 0.05
Then run injection_recovery.py, which reads tcrit.json to apply the calibrated
T_inj > T_crit criterion on top of the existing period/K parameter-match test.
"""

import os

# Must happen before numpy/scipy/spleaf are imported, same reasoning as
# injection_recovery.py: avoid each worker process's BLAS backend spawning its
# own multi-threaded thread pool under multiprocessing.Pool.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from spleaf import cov, term

import functions as mf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nonstationary"))
import functions_nonstat as mfn

import injection_recovery as ir

MODE_SEED_OFFSET = {"cyc": 0, "nonstat": 1_000_000, "no_cycle": 2_000_000}

NULL_CSV_FIELDS = [
    "null_idx", "seed", "mode",
    "loglike_gp_only", "loglike_gp_planet", "T_stat",
    "k_gp_only", "k_gp_planet", "n_obs",
    "bic_gp_only", "bic_gp_planet", "delta_bic",
    "gp_only_at_bounds", "gp_planet_at_bounds",
    "loglike_nonfinite", "nested_model_violation",
    "best_period_guess", "valid", "diagnostic_flags",
]


def build_parser():
    parser = ir.build_parser()
    parser.description = ("Empirically calibrate the per-GP-model likelihood-ratio detection "
                           "threshold T_crit from no-planet null simulations.")
    # ir.build_parser() defines --alpha as an optional injection-recovery-side cross-check
    # (default None); here it IS the target false-positive rate driving the T_crit quantile.
    for action in parser._actions:
        if action.dest == "alpha":
            action.default = 0.05
            action.help = "Target false-positive rate for the T_crit quantile (shared across GP models)."
            break

    parser.add_argument("--n-null", type=int, default=500,
                         help="Number of no-planet null realizations per GP model.")
    parser.add_argument("--min-valid-null-fits", type=int, default=100,
                         help="Minimum valid null fits required before accepting a T_crit.")
    parser.add_argument("--null-output-dir", default="results/null_calibration")
    parser.add_argument("--rerun", action="store_true",
                         help="Discard any existing reference fit / null fits for the requested "
                              "modes and start fresh, instead of resuming.")
    parser.add_argument("--high-variance-multiplier", type=float, default=4.0,
                         help="Flag a mode's T_null distribution as high-variance if its variance "
                              "exceeds this multiple of the naive chi-square(3) variance (~6).")
    return parser


def _build_mode_cov(mode, t_full, yerr_full, series_index, stds, args):
    if mode in ("cyc", "no_cycle"):
        return cov.Cov(
            t_full,
            err=term.Error(yerr_full),
            rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
            rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
            rot=ir.MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                      np.array([stds[0], stds[1]]),
                                      np.array([stds[0], 0.0])),
        )
    if mode == "nonstat":
        return cov.Cov(
            t_full,
            err=term.Error(yerr_full),
            rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
            rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
            rot=term.SimpleProductKernel(
                nonstat=mfn.NonStationaryKernel(mfn.alpha_cyc, mfn.alpha_cyc_grad,
                                                 mu=args.mu0, b=args.cycle_b0, P=args.cycle_P0, phi=args.cycle_phi0),
                qp=ir.MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                         np.array([stds[0], stds[1]]),
                                         np.array([stds[0], 0.0])),
            ),
        )
    raise ValueError(f"unknown mode {mode!r}")


def _params_for_mode(mode, C):
    if mode in ("cyc", "no_cycle"):
        return mf.get_opt_params(C)[0]
    return mfn.get_opt_params_nonstat(C)[0]


def fit_reference(mode, args):
    """One-time GP-only fit of `mode` to the real, un-injected Solar data,
    via injection_recovery.py's own fit_cyc/fit_nonstat/fit_nocyc (fit_planet=False)
    -- the exact production fitting code. The result is the fixed generative
    truth used for every null draw of this mode."""
    t_full, y_full, yerr_full, series_index = mf.load_and_norm_data(
        args.path, args.star_name, normalise=args.normalise, inject_planet=False,
    )
    rv_std = 1.0
    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]
    amp_bound = args.planet_amp_max if args.planet_amp_max is not None else 1.5 * args.k_max

    if mode == "cyc":
        try:
            _, _, cycle_params = mf.fit_cycle(t_full, y_full, series_index, b0=args.cycle_b0, P0=args.cycle_P0,
                                               phi0=args.cycle_phi0, plot=False, print_results=False,
                                               return_fit=True)
        except Exception as exc:
            print(f"[reference mode=cyc] cycle warm-start fit failed, falling back to raw seed: {exc}",
                  flush=True)
            cycle_params = ir.fallback_cycle_seed(y_full, series_index, args)
        xbest, loglike, diag = ir.fit_cyc(t_full, y_full, yerr_full, series_index, stds, args,
                                           [None], cycle_params, amp_bound, rv_std, fit_planet=False)
    elif mode == "nonstat":
        try:
            _, _, cycle_params = mf.fit_cycle(t_full, y_full, series_index, b0=args.cycle_b0, P0=args.cycle_P0,
                                               phi0=args.cycle_phi0, plot=False, print_results=False,
                                               return_fit=True)
        except Exception as exc:
            print(f"[reference mode=nonstat] cycle warm-start fit failed, falling back to raw seed: {exc}",
                  flush=True)
            cycle_params = ir.fallback_cycle_seed(y_full, series_index, args)
        xbest, loglike, diag = ir.fit_nonstat(t_full, y_full, yerr_full, series_index, stds, args,
                                               [None], cycle_params, amp_bound, rv_std, fit_planet=False)
    elif mode == "no_cycle":
        rv_offset0 = float(np.mean(y_full[series_index[0]]))
        rhk_offset0 = float(np.mean(y_full[series_index[1]]))
        xbest, loglike, diag = ir.fit_nocyc(t_full, y_full, yerr_full, series_index, stds, args,
                                             [None], rv_offset0, rhk_offset0, amp_bound, rv_std, fit_planet=False)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    C = _build_mode_cov(mode, t_full, yerr_full, series_index, stds, args)
    params = _params_for_mode(mode, C)

    ref = {
        "mode": mode,
        "params": params,
        "theta": [float(v) for v in xbest],
        "stds": [float(s) for s in stds],
        "loglike_ref": float(loglike),
        "at_bounds": diag.get("at_bounds", []),
        "n": int(len(t_full)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return ref, t_full, yerr_full, series_index


def build_reference_cov(mode, t_full, yerr_full, series_index, args, ref):
    C = _build_mode_cov(mode, t_full, yerr_full, series_index, ref["stds"], args)
    params = ref["params"]
    theta = np.array(ref["theta"])
    C.set_param(theta[:len(params)], params)
    return C


def draw_null_dataset(mode, C_ref, t_full, series_index, ref, seed):
    """Draws one no-planet realization at the real observing times: a GP+noise
    sample from C_ref (fixed at the reference hyperparameters) plus the
    reference fit's deterministic mean -- never a planet term."""
    np.random.seed(seed)
    y_null = C_ref.sample()

    theta = np.array(ref["theta"])
    if mode == "no_cycle":
        d0, d1 = theta[-2], theta[-1]
        y_null[series_index[0]] += d0
        y_null[series_index[1]] += d1
    elif mode == "cyc":
        b, P, phi, a0, a1, d0, d1 = theta[-7:]
        y_null[series_index[0]] += a0 * mf.shared_core(t_full[series_index[0]], b, P, phi) + d0
        y_null[series_index[1]] += a1 * mf.shared_core(t_full[series_index[1]], b, P, phi) + d1
    elif mode == "nonstat":
        a0, a1, d0, d1 = theta[-4:]
        b, P, phi = C_ref.get_param(['rot.nonstat_b', 'rot.nonstat_P', 'rot.nonstat_phi'])
        y_null[series_index[0]] += a0 * mfn.shared_core(t_full[series_index[0]], b, P, phi) + d0
        y_null[series_index[1]] += a1 * mfn.shared_core(t_full[series_index[1]], b, P, phi) + d1
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return y_null


def reference_warm_start(mode, ref, args):
    """Uses the one-time reference fit's own converged hyperparameters
    (kernel shape, cycle timing/modulation strength, mean-term amplitude and
    offset) as the warm start for a per-null-realization refit, instead of
    generic CLI defaults plus a fresh mf.fit_cycle curve_fit on that specific
    noisy draw. The reference fit is a reliable, well-converged estimate of
    the true generative parameters (it's fit to the full real dataset, not
    one noisy realization of it); starting each per-realization refit there
    rather than from scratch avoids re-discovering that same structure from
    an uninformed default every time, which is what was making the
    per-realization refits land on hard bounds so much more often than the
    reference fit itself does."""
    params = ref["params"]
    theta = ref["theta"]

    def p(name):
        return theta[params.index(name)]

    overrides = {}
    if mode in ("cyc", "no_cycle"):
        overrides.update(prot=p("rot.P"), rho=p("rot.rho"), eta=p("rot.eta"))
    elif mode == "nonstat":
        overrides.update(mu0=p("rot.nonstat_mu"),
                          prot=p("rot.qp_P"), rho=p("rot.qp_rho"), eta=p("rot.qp_eta"))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    warm_args = argparse.Namespace(**{**vars(args), **overrides})

    cycle_seed = None
    if mode == "cyc":
        b, P, phi, a0, a1, d0, d1 = theta[-7:]
        cycle_seed = (d0, a0, d1, a1, b, P, phi)
    elif mode == "nonstat":
        a0, a1, d0, d1 = theta[-4:]
        b, P, phi = p("rot.nonstat_b"), p("rot.nonstat_P"), p("rot.nonstat_phi")
        cycle_seed = (d0, a0, d1, a1, b, P, phi)

    return warm_args, cycle_seed


def run_one_null(args_dict, mode, null_idx, seed, ref, t_full, yerr_full, series_index):
    args = argparse.Namespace(**args_dict)
    rv_std = 1.0
    stds = ref["stds"]

    warm_args, cycle_seed = reference_warm_start(mode, ref, args)
    amp_bound = warm_args.planet_amp_max if warm_args.planet_amp_max is not None else 1.5 * warm_args.k_max

    C_ref = build_reference_cov(mode, t_full, yerr_full, series_index, args, ref)
    y_null = draw_null_dataset(mode, C_ref, t_full, series_index, ref, seed)

    planet_guess, _ = mf.period_guess(t_full, y_null, yerr_full, series_index,
                                       PMIN=1.1, PMAX=warm_args.period_max, MAX_FAP=1e-5, MAX_NPL=2, plot=False)

    if mode == "cyc":
        fit_fn = ir.fit_cyc
        fit_args = (t_full, y_null, yerr_full, series_index, stds, warm_args,
                    planet_guess, cycle_seed, amp_bound, rv_std)
    elif mode == "nonstat":
        fit_fn = ir.fit_nonstat
        fit_args = (t_full, y_null, yerr_full, series_index, stds, warm_args,
                    planet_guess, cycle_seed, amp_bound, rv_std)
    elif mode == "no_cycle":
        rv_offset0 = float(np.mean(y_null[series_index[0]]))
        rhk_offset0 = float(np.mean(y_null[series_index[1]]))
        fit_fn = ir.fit_nocyc
        fit_args = (t_full, y_null, yerr_full, series_index, stds, warm_args,
                    planet_guess, rv_offset0, rhk_offset0, amp_bound, rv_std)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    xbest1 = xbest0 = None
    loglike1 = loglike0 = float("nan")
    diag1 = diag0 = {"at_bounds": [], "best_period_guess": None}
    errors = []

    try:
        xbest1, loglike1, diag1 = fit_fn(*fit_args, fit_planet=True)
    except Exception as exc:
        errors.append(f"gp_planet_exception:{exc!r}")
        print(f"[null mode={mode} idx={null_idx}] GP+planet fit failed: {exc}", flush=True)

    try:
        xbest0, loglike0, diag0 = fit_fn(*fit_args, fit_planet=False)
    except Exception as exc:
        errors.append(f"gp_only_exception:{exc!r}")
        print(f"[null mode={mode} idx={null_idx}] GP-only fit failed: {exc}", flush=True)

    n_obs = len(t_full)
    loglike_nonfinite = int(not (xbest1 is not None and xbest0 is not None
                                  and np.isfinite(loglike1) and np.isfinite(loglike0)))

    if loglike_nonfinite:
        return {
            "null_idx": null_idx, "seed": seed, "mode": mode,
            "loglike_gp_only": loglike0 if xbest0 is not None else float("nan"),
            "loglike_gp_planet": loglike1 if xbest1 is not None else float("nan"),
            "T_stat": float("nan"),
            "k_gp_only": len(xbest0) if xbest0 is not None else float("nan"),
            "k_gp_planet": len(xbest1) if xbest1 is not None else float("nan"),
            "n_obs": n_obs,
            "bic_gp_only": float("nan"), "bic_gp_planet": float("nan"), "delta_bic": float("nan"),
            "gp_only_at_bounds": ",".join(map(str, diag0.get("at_bounds", []))) if diag0 else "",
            "gp_planet_at_bounds": ",".join(map(str, diag1.get("at_bounds", []))) if diag1 else "",
            "loglike_nonfinite": 1, "nested_model_violation": 0,
            "best_period_guess": diag1.get("best_period_guess") if diag1 else None,
            "valid": 0,
            "diagnostic_flags": ";".join(errors) if errors else "loglike_nonfinite",
        }

    T_stat = 2.0 * (loglike1 - loglike0)
    k1, k0 = len(xbest1), len(xbest0)
    bic1 = k1 * np.log(n_obs) - 2 * loglike1
    bic0 = k0 * np.log(n_obs) - 2 * loglike0
    delta_bic = bic0 - bic1
    nested_model_violation = int(T_stat < 0)

    flags = []
    if nested_model_violation:
        flags.append("nested_model_violation")
    if diag1.get("at_bounds"):
        flags.append("gp_planet_at_bounds")
    if diag0.get("at_bounds"):
        flags.append("gp_only_at_bounds")

    return {
        "null_idx": null_idx, "seed": seed, "mode": mode,
        "loglike_gp_only": loglike0, "loglike_gp_planet": loglike1, "T_stat": T_stat,
        "k_gp_only": k0, "k_gp_planet": k1, "n_obs": n_obs,
        "bic_gp_only": bic0, "bic_gp_planet": bic1, "delta_bic": delta_bic,
        "gp_only_at_bounds": ",".join(map(str, diag0.get("at_bounds", []))),
        "gp_planet_at_bounds": ",".join(map(str, diag1.get("at_bounds", []))),
        "loglike_nonfinite": 0, "nested_model_violation": nested_model_violation,
        "best_period_guess": diag1.get("best_period_guess"),
        "valid": int(not nested_model_violation),
        "diagnostic_flags": ",".join(flags),
    }


def _null_worker(task):
    args_dict, mode, null_idx, seed, ref, t_full, yerr_full, series_index = task
    return [run_one_null(args_dict, mode, null_idx, seed, ref, t_full, yerr_full, series_index)]


def calibrate_mode(mode, args):
    null_dir = args.null_output_dir
    os.makedirs(null_dir, exist_ok=True)
    ref_path = os.path.join(null_dir, f"reference_fit_{mode}.json")
    null_csv_path = os.path.join(null_dir, f"null_fits_{mode}.csv")
    meta_path = os.path.join(null_dir, f"null_fits_{mode}.meta.json")

    calibration_args = {k: v for k, v in vars(args).items() if k not in ("rerun", "n_workers")}

    if args.rerun:
        for path in (ref_path, null_csv_path, meta_path):
            if os.path.exists(path):
                os.remove(path)

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prev_snapshot = json.load(f)
        if prev_snapshot != calibration_args:
            raise SystemExit(
                f"{null_csv_path} was calibrated with different arguments than the current run "
                "(bounds/model config/n_null/seed/etc. changed). Pass --rerun to start fresh, or "
                "restore the original arguments."
            )
    else:
        with open(meta_path, "w") as f:
            json.dump(calibration_args, f, indent=2)

    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)
        t_full, y_full, yerr_full, series_index = mf.load_and_norm_data(
            args.path, args.star_name, normalise=args.normalise, inject_planet=False,
        )
    else:
        ref, t_full, yerr_full, series_index = fit_reference(mode, args)
        with open(ref_path, "w") as f:
            json.dump(ref, f, indent=2)
        print(f"[{mode}] reference fit: loglike={ref['loglike_ref']:.4f}, saved to {ref_path}", flush=True)

    completed_idx = set()
    if os.path.exists(null_csv_path):
        with open(null_csv_path, newline="") as f:
            for row in csv.DictReader(f):
                completed_idx.add(int(row["null_idx"]))

    remaining_idx = [i for i in range(args.n_null) if i not in completed_idx]
    print(f"[{mode}] {args.n_null} total null realizations, {len(completed_idx)} already done, "
          f"{len(remaining_idx)} remaining.", flush=True)

    args_dict = vars(args).copy()
    tasks = [
        (args_dict, mode, i, args.seed + MODE_SEED_OFFSET[mode] + i, ref, t_full, yerr_full, series_index)
        for i in remaining_idx
    ]

    ir.run_resumable_pool(null_csv_path, NULL_CSV_FIELDS, tasks, _null_worker, args.n_workers, force_fresh=False)


def compute_t_crit(null_csv_path, alpha, min_valid_null_fits, high_variance_multiplier=4.0):
    df = pd.read_csv(null_csv_path)
    valid = df[df["valid"] == 1]
    n_valid = len(valid)
    n_failed = len(df) - n_valid
    if n_valid < min_valid_null_fits:
        raise RuntimeError(
            f"only {n_valid} valid null fits in {null_csv_path} "
            f"(< min_valid_null_fits={min_valid_null_fits})"
        )

    t_crit = float(np.quantile(valid["T_stat"].to_numpy(), 1 - alpha))
    t_var = float(valid["T_stat"].var())
    naive_chi2_3_var = 6.0  # Var[chi2(k)] = 2k; informal reference point only, not an assumption on T.
    high_variance_flag = bool(t_var > high_variance_multiplier * naive_chi2_3_var)
    return t_crit, n_valid, n_failed, t_var, high_variance_flag


def finalize_tcrit(args):
    modes = [m.strip() for m in args.models.split(",")]
    unknown = set(modes) - {"cyc", "nonstat", "no_cycle"}
    if unknown:
        raise SystemExit(f"--models: unknown model(s) {sorted(unknown)}; choose from cyc,nonstat,no_cycle")

    result_modes = {}
    summary_rows = []
    for mode in modes:
        null_csv_path = os.path.join(args.null_output_dir, f"null_fits_{mode}.csv")
        if not os.path.exists(null_csv_path):
            raise SystemExit(f"{null_csv_path} not found; run calibrate_mode for {mode!r} first.")

        t_crit, n_valid, n_failed, t_var, high_var_flag = compute_t_crit(
            null_csv_path, args.alpha, args.min_valid_null_fits, args.high_variance_multiplier,
        )
        result_modes[mode] = {
            "t_crit": t_crit, "n_null_requested": args.n_null,
            "n_valid": n_valid, "n_failed": n_failed,
            "t_null_var": t_var, "high_variance_flag": high_var_flag,
            "min_valid_required": args.min_valid_null_fits,
        }
        summary_rows.append({
            "gp_model": mode, "n_null": args.n_null, "fpr_target": args.alpha,
            "t_crit": t_crit, "n_valid_null_fits": n_valid, "n_failed_null_fits": n_failed,
        })
        print(f"[{mode}] T_crit={t_crit:.4f} (alpha={args.alpha}, n_valid={n_valid}, n_failed={n_failed}, "
              f"T_null var={t_var:.4f}{' [HIGH VARIANCE]' if high_var_flag else ''})", flush=True)

    tcrit_path = os.path.join(args.null_output_dir, "tcrit.json")
    with open(tcrit_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "alpha": args.alpha, "seed": args.seed,
            "modes": result_modes,
        }, f, indent=2)
    print(f"Wrote {tcrit_path}", flush=True)

    summary_path = os.path.join(args.null_output_dir, "summary_table.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}", flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False))


def main():
    args = build_parser().parse_args()
    modes = [m.strip() for m in args.models.split(",")]
    unknown = set(modes) - {"cyc", "nonstat", "no_cycle"}
    if unknown:
        raise SystemExit(f"--models: unknown model(s) {sorted(unknown)}; choose from cyc,nonstat,no_cycle")

    for mode in modes:
        calibrate_mode(mode, args)

    finalize_tcrit(args)


if __name__ == "__main__":
    main()
