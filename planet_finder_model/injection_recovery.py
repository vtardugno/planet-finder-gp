"""Injection-recovery grid sweep comparing the cyc, nonstat, and no_cycle models.

For each (period, semi-amplitude K, orbital phase) combination on a log-spaced
grid, injects a synthetic planet into the Solar RV data and fits it with
whichever of the three models --models selects: cyc (functions.py),
nonstat (nonstationary/functions_nonstat.py), and no_cycle (functions.py's
bare-GP branch), using the same blind periodogram + multi-start L-BFGS-B
search each existing single-injection script (main_cycle_likelihood.py /
nonstationary/main_nonstat.py / main.py --no-fit-cycle) already uses.
Results are appended to a CSV as they complete, so the sweep can be
interrupted and resumed by simply re-running the same command. Resumability
is tracked per (combo, mode): re-running with a different --models against
an existing --output-csv only computes whichever modes are still missing for
each combo, so e.g. adding no_cycle to a CSV that already has cyc/nonstat
never recomputes those.

A planet is considered "recovered" only if BOTH:
  1. The best-fit period/K land within tolerance of the injected truth
     (the original parameter-matching criterion), AND
  2. The detection statistic T = 2*(logL_max(GP+planet) - logL_max(GP only))
     exceeds a per-mode threshold T_crit, empirically calibrated ahead of
     time by null_calibration.py at a target false-positive rate and saved
     to --tcrit-path.
See null_calibration.py for how T_crit is calibrated.
"""

import os

# Must happen before numpy/scipy/spleaf are imported (below, and transitively via
# functions.py) so each worker process's BLAS backend doesn't spawn its own
# multi-threaded thread pool. Without this, N worker processes each try to use
# every core for linear algebra, oversubscribing the machine N-fold and making
# the whole sweep dramatically slower than running one fit in isolation.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib
matplotlib.use("Agg")

import functions as mf
import numpy as np
from spleaf import cov, term
import argparse
import sys
import csv
import json
import time
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nonstationary"))
import functions_nonstat as mfn


class MultiSeriesKernel(term.MultiSeriesKernel):
  def _grad_param(self, grad_dU=None, grad_dV=None):
    if grad_dU is not None or grad_dV is not None:
      raise NotImplementedError()
    return super()._grad_param()


A_INIT_FRACS = [1.0, 0.1, 0.01, 0.001]
B_INIT_FRACS = [1.0, 0.1, 0.01, 0.001]

CSV_FIELDS = [
    "period_idx", "k_idx", "phase_idx",
    "period_inj", "k_inj", "phase_inj", "A_inj", "B_inj",
    "mode", "P_rec", "K_rec", "A_rec", "B_rec",
    "best_loglike", "recovered",
    # --- likelihood-ratio detection (additive) ---
    "loglike_gp_only", "T_stat", "t_crit", "alpha_fpr",
    "likelihood_ratio_pass", "param_match_pass",
    "period_tolerance", "k_tolerance",
    # --- BIC diagnostics only, not used for detection (additive) ---
    "bic_gp_only", "bic_gp_planet", "delta_bic",
    # --- fit-quality diagnostics, recorded not discarded (additive) ---
    "gp_planet_at_bounds", "gp_only_at_bounds",
    "loglike_nonfinite", "nested_model_violation",
    "diagnostic_flags",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Injection-recovery grid sweep comparing the cyc, nonstat, and no_cycle models."
    )

    parser.add_argument("--path", default="data/Solar_Data")
    parser.add_argument("--star-name", default="Sun")
    parser.add_argument("--normalise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--models", type=str, default="cyc,nonstat,no_cycle",
                         help="Comma-separated subset of cyc,nonstat,no_cycle to fit per combo.")

    # Grid
    parser.add_argument("--n-period", type=int, default=10)
    parser.add_argument("--n-k", type=int, default=10)
    parser.add_argument("--n-phases", type=int, default=3)
    parser.add_argument("--period-min", type=float, default=1.1)
    parser.add_argument("--period-max", type=float, default=400.0)
    parser.add_argument("--k-min", type=float, default=0.00003)
    parser.add_argument("--k-max", type=float, default=0.2)
    parser.add_argument("--period-values", type=str, default=None,
                         help="Comma-separated explicit period values (days); overrides "
                              "--period-min/--period-max/--n-period. Use this to run a grid "
                              "complementary to another run's, e.g. interleaved midpoints.")
    parser.add_argument("--k-values", type=str, default=None,
                         help="Comma-separated explicit K values (km/s); overrides "
                              "--k-min/--k-max/--n-k.")

    # Model / covariance settings (defaults match main_cycle_likelihood.py / main_nonstat.py)
    parser.add_argument("--sig", type=float, default=1.0)
    parser.add_argument("--prot", type=float, default=27.0)
    parser.add_argument("--rho", type=float, default=20.0)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--rvjit-frac", type=float, default=0.1)
    parser.add_argument("--rhkjit-frac", type=float, default=0.1)

    parser.add_argument("--prot-min", type=float, default=20.0)
    parser.add_argument("--prot-max", type=float, default=32.0)
    parser.add_argument("--rho-min", type=float, default=10.0)
    parser.add_argument("--rho-max", type=float, default=65.0)
    parser.add_argument("--eta-min", type=float, default=0.1)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--rvjit-max-frac", type=float, default=5.0)
    parser.add_argument("--rhkjit-max-frac", type=float, default=5.0)
    parser.add_argument("--alpha0-max-frac", type=float, default=10.0)
    parser.add_argument("--alpha1-max-frac", type=float, default=5.0)
    parser.add_argument("--beta0-max-frac", type=float, default=10.0)

    # cyc / nonstat shared cycle-term bounds
    parser.add_argument("--b-min", type=float, default=-0.5)
    parser.add_argument("--b-max", type=float, default=0.5)
    parser.add_argument("--Pcyc-min", type=float, default=0.5)
    parser.add_argument("--Pcyc-max", type=float, default=10000.0)
    parser.add_argument("--phi-min", type=float, default=-np.pi)
    parser.add_argument("--phi-max", type=float, default=np.pi)
    parser.add_argument("--a0-min", type=float, default=-0.1)
    parser.add_argument("--a0-max", type=float, default=0.1)
    parser.add_argument("--a1-min", type=float, default=-5)
    parser.add_argument("--a1-max", type=float, default=5)
    parser.add_argument("--delta0-min", type=float, default=-0.1)
    parser.add_argument("--delta0-max", type=float, default=1)
    parser.add_argument("--delta1-min", type=float, default=-5)
    parser.add_argument("--delta1-max", type=float, default=5)

    # nonstat-only bounds (main_nonstat.py uses its own delta0/delta1 bounds,
    # different from main_cycle_likelihood.py's above)
    parser.add_argument("--mu-min", type=float, default=-5.0)
    parser.add_argument("--mu-max", type=float, default=5.0)
    parser.add_argument("--nonstat-delta0-min", type=float, default=-0.5)
    parser.add_argument("--nonstat-delta0-max", type=float, default=0.5)
    parser.add_argument("--nonstat-delta1-min", type=float, default=-2.0)
    parser.add_argument("--nonstat-delta1-max", type=float, default=2.0)

    # no_cycle-only bounds (main.py --no-fit-cycle widens rho/eta -- no cycle
    # term to absorb long-period variability, so the GP's own rho/eta need
    # more room -- and uses its own delta0/delta1 bounds; rho_min/eta_min and
    # everything else above stay shared with cyc/nonstat)
    parser.add_argument("--nocyc-rho-max", type=float, default=100.0)
    parser.add_argument("--nocyc-eta-max", type=float, default=3.0)
    parser.add_argument("--nocyc-delta0-min", type=float, default=-0.5)
    parser.add_argument("--nocyc-delta0-max", type=float, default=0.5)
    parser.add_argument("--nocyc-delta1-min", type=float, default=-5.0)
    parser.add_argument("--nocyc-delta1-max", type=float, default=5.0)

    # Planet amplitude fit bound: since injected phase spans the full circle,
    # A_inj/B_inj can be negative, so the fit bound must be symmetric
    # (+/- planet_amp_max), not the (small-positive, max) bound the
    # single-injection scripts use for their fixed-positive default test
    # amplitudes. Defaults to 1.5x the grid's K_max unless overridden, so the
    # hot-Jupiter end of the grid stays inside the fit's search space.
    parser.add_argument("--planet-amp-max", type=float, default=None)

    # Multi-start initial magnitudes for planet A/B (same scale-down pattern
    # as main_cycle_likelihood.py / main_nonstat.py)
    parser.add_argument("--planet-A-fit", type=float, default=0.01)
    parser.add_argument("--planet-B-fit", type=float, default=0.01)

    # Cycle warm-start seeds
    parser.add_argument("--cycle-b0", type=float, default=0.0)
    parser.add_argument("--cycle-P0", type=float, default=4000.0)
    parser.add_argument("--cycle-phi0", type=float, default=0.0)
    parser.add_argument("--mu0", type=float, default=0.0)

    parser.add_argument("--output-csv", default="results/injection_recovery.csv")
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)

    # Likelihood-ratio detection criterion
    parser.add_argument("--period-tolerance", type=float, default=0.10,
                         help="Relative tolerance for the recovered-period parameter-match test.")
    parser.add_argument("--k-tolerance", type=float, default=0.15,
                         help="Relative tolerance for the recovered-K parameter-match test.")
    parser.add_argument("--tcrit-path", default="results/null_calibration/tcrit.json",
                         help="Path to the tcrit.json produced by null_calibration.py, giving each "
                              "mode's empirically calibrated T_crit detection threshold.")
    parser.add_argument("--alpha", type=float, default=None,
                         help="Optional cross-check: if given, must match the false-positive rate "
                              "baked into --tcrit-path, or the run aborts.")

    return parser


def build_bounds_list_cyc(args, stds, period_bounds, amp_bound, fit_planet=True):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]

    bounds = [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (args.prot_min, args.prot_max),
        (args.rho_min, args.rho_max),
        (args.eta_min, args.eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (args.b_min, args.b_max),
        (args.Pcyc_min, args.Pcyc_max),
        (args.phi_min, args.phi_max),
        (args.a0_min, args.a0_max),
        (args.a1_min, args.a1_max),
        (args.delta0_min, args.delta0_max),
        (args.delta1_min, args.delta1_max),
    ]
    if fit_planet:
        bounds += [period_bounds, (-amp_bound, amp_bound), (-amp_bound, amp_bound)]
    return bounds


def build_bounds_list_nonstat(args, stds, period_bounds, amp_bound, fit_planet=True):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]

    bounds = [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (args.mu_min, args.mu_max),
        (args.b_min, args.b_max),
        (args.Pcyc_min, args.Pcyc_max),
        (args.phi_min, args.phi_max),
        (args.prot_min, args.prot_max),
        (args.rho_min, args.rho_max),
        (args.eta_min, args.eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (args.a0_min, args.a0_max),
        (args.a1_min, args.a1_max),
        (args.nonstat_delta0_min, args.nonstat_delta0_max),
        (args.nonstat_delta1_min, args.nonstat_delta1_max),
    ]
    if fit_planet:
        bounds += [period_bounds, (-amp_bound, amp_bound), (-amp_bound, amp_bound)]
    return bounds


def build_bounds_list_nocyc(args, stds, period_bounds, amp_bound, fit_planet=True):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]

    bounds = [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (args.prot_min, args.prot_max),
        (args.rho_min, args.nocyc_rho_max),
        (args.eta_min, args.nocyc_eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (args.nocyc_delta0_min, args.nocyc_delta0_max),
        (args.nocyc_delta1_min, args.nocyc_delta1_max),
    ]
    if fit_planet:
        bounds += [period_bounds, (-amp_bound, amp_bound), (-amp_bound, amp_bound)]
    return bounds


def at_bounds_params(xbest, bounds_list, rtol=1e-6):
    """Indices where xbest sits within rtol of one of its hard bounds."""
    flagged = []
    for i, (lo, hi) in enumerate(bounds_list):
        span = hi - lo
        tol = rtol * max(abs(span), 1.0)
        if abs(xbest[i] - lo) <= tol or abs(xbest[i] - hi) <= tol:
            flagged.append(i)
    return flagged


def fallback_cycle_seed(y_full, series_index, args):
    """Used when mf.fit_cycle's curve_fit warm-start fails to converge --
    can happen especially on synthetic null datasets, which unlike the real
    Solar data have no guaranteed periodic cycle signal for curve_fit to lock
    onto. Falls back to raw data means/stds plus the args-provided initial
    cycle guess, in the same (rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0,
    P0, phi0) order fit_cycle would have returned, so the subsequent L-BFGS-B
    optimization can still proceed from a valid (if less-informed) start."""
    rv_offset0 = float(np.mean(y_full[series_index[0]]))
    rv_amp0 = float(np.std(y_full[series_index[0]]))
    rhk_offset0 = float(np.mean(y_full[series_index[1]]))
    rhk_amp0 = float(np.std(y_full[series_index[1]]))
    return (rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, args.cycle_b0, args.cycle_P0, args.cycle_phi0)


def fit_cyc(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, cycle_seed, amp_bound, rv_std,
            fit_planet=True):
    rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_seed

    best_loglike = -np.inf
    xbest_all = None
    best_bounds_list = None
    best_period_guess = None

    guesses = planet_guess if fit_planet else [None]
    a_fracs = A_INIT_FRACS if fit_planet else [1.0]
    b_fracs = B_INIT_FRACS if fit_planet else [1.0]

    for p in guesses:
        period_bounds = (max(p - 10, 1.1), min(p + 10, args.period_max)) if fit_planet else None
        bounds_list = build_bounds_list_cyc(args, stds, period_bounds, amp_bound, fit_planet=fit_planet)

        for Afrac in a_fracs:
            for Bfrac in b_fracs:
                C = cov.Cov(
                    t_full,
                    err=term.Error(yerr_full),
                    rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
                    rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
                    rot=MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                          np.array([stds[0], stds[1]]),
                                          np.array([stds[0], 0.0])),
                )

                xbest, C = mf.optimise_params_cyc(
                    t_full, y_full, series_index, C, bounds_list,
                    b=b0, P=P0, phi=phi0,
                    a0=rv_amp0, a1=rhk_amp0, d0=rv_offset0, d1=rhk_offset0,
                    planet_p=(p if fit_planet else args.cycle_P0),
                    planet_A=args.planet_A_fit * Afrac, planet_B=args.planet_B_fit * Bfrac,
                    fit_planet=fit_planet, change_C=True,
                )

                loglike = -1 * mf.negloglike_cyc(xbest, t_full, y_full, series_index, C, rv_std,
                                                  inject_planet=fit_planet)[0]

                if loglike > best_loglike:
                    best_loglike = loglike
                    xbest_all = xbest
                    best_bounds_list = bounds_list
                    best_period_guess = p

    diagnostics = {
        "at_bounds": at_bounds_params(xbest_all, best_bounds_list) if xbest_all is not None else [],
        "best_period_guess": best_period_guess,
    }
    return xbest_all, best_loglike, diagnostics


def fit_nonstat(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, cycle_seed, amp_bound, rv_std,
                 fit_planet=True):
    rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_seed

    best_loglike = -np.inf
    xbest_all = None
    best_bounds_list = None
    best_period_guess = None

    guesses = planet_guess if fit_planet else [None]
    a_fracs = A_INIT_FRACS if fit_planet else [1.0]
    b_fracs = B_INIT_FRACS if fit_planet else [1.0]

    for p in guesses:
        period_bounds = (max(p - 10, 1.1), min(p + 10, args.period_max)) if fit_planet else None
        bounds_list = build_bounds_list_nonstat(args, stds, period_bounds, amp_bound, fit_planet=fit_planet)

        for Afrac in a_fracs:
            for Bfrac in b_fracs:
                C = cov.Cov(
                    t_full,
                    err=term.Error(yerr_full),
                    rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
                    rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
                    rot=term.SimpleProductKernel(
                        nonstat=mfn.NonStationaryKernel(mfn.alpha_cyc, mfn.alpha_cyc_grad, mu=args.mu0, b=b0, P=P0, phi=phi0),
                        qp=MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                             np.array([stds[0], stds[1]]),
                                             np.array([stds[0], 0.0])),
                    ),
                )

                xbest, C = mfn.optimise_params_nonstat(
                    t_full, y_full, series_index, C, bounds_list,
                    a0=rv_amp0, a1=rhk_amp0, d0=rv_offset0, d1=rhk_offset0,
                    planet_p=(p if fit_planet else args.cycle_P0),
                    planet_A=args.planet_A_fit * Afrac, planet_B=args.planet_B_fit * Bfrac,
                    fit_planet=fit_planet, change_C=True,
                )

                loglike = -1 * mfn.negloglike_nonstat(xbest, t_full, y_full, series_index, C, rv_std,
                                                       inject_planet=fit_planet)[0]

                if loglike > best_loglike:
                    best_loglike = loglike
                    xbest_all = xbest
                    best_bounds_list = bounds_list
                    best_period_guess = p

    diagnostics = {
        "at_bounds": at_bounds_params(xbest_all, best_bounds_list) if xbest_all is not None else [],
        "best_period_guess": best_period_guess,
    }
    return xbest_all, best_loglike, diagnostics


def fit_nocyc(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, rv_offset0, rhk_offset0, amp_bound,
              rv_std, fit_planet=True):
    best_loglike = -np.inf
    xbest_all = None
    best_bounds_list = None
    best_period_guess = None

    guesses = planet_guess if fit_planet else [None]
    a_fracs = A_INIT_FRACS if fit_planet else [1.0]
    b_fracs = B_INIT_FRACS if fit_planet else [1.0]

    for p in guesses:
        period_bounds = (max(p - 10, 1.1), min(p + 10, args.period_max)) if fit_planet else None
        bounds_list = build_bounds_list_nocyc(args, stds, period_bounds, amp_bound, fit_planet=fit_planet)

        for Afrac in a_fracs:
            for Bfrac in b_fracs:
                C = cov.Cov(
                    t_full,
                    err=term.Error(yerr_full),
                    rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
                    rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
                    rot=MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                          np.array([stds[0], stds[1]]),
                                          np.array([stds[0], 0.0])),
                )

                xbest, C = mf.optimise_params(
                    t_full, y_full, series_index, C, bounds_list,
                    delta_0=rv_offset0, delta_1=rhk_offset0,
                    planet_p=(p if fit_planet else 40.05),
                    planet_A=args.planet_A_fit * Afrac, planet_B=args.planet_B_fit * Bfrac,
                    fit_planet=fit_planet, change_C=True,
                )

                loglike = -1 * mf.negloglike_nocyc(xbest, t_full, y_full, series_index, C, rv_std,
                                                    inject_planet=fit_planet)[0]

                if loglike > best_loglike:
                    best_loglike = loglike
                    xbest_all = xbest
                    best_bounds_list = bounds_list
                    best_period_guess = p

    diagnostics = {
        "at_bounds": at_bounds_params(xbest_all, best_bounds_list) if xbest_all is not None else [],
        "best_period_guess": best_period_guess,
    }
    return xbest_all, best_loglike, diagnostics


def make_row(period_idx, k_idx, phase_idx, period_inj, k_inj, phase_inj, A_inj, B_inj, mode,
             xbest1, loglike1, diag1, xbest0, loglike0, diag0,
             n_obs, t_crit, alpha_fpr, period_tolerance, k_tolerance, fit_error):
    base = {
        "period_idx": period_idx, "k_idx": k_idx, "phase_idx": phase_idx,
        "period_inj": period_inj, "k_inj": k_inj, "phase_inj": phase_inj,
        "A_inj": A_inj, "B_inj": B_inj, "mode": mode,
        "t_crit": t_crit, "alpha_fpr": alpha_fpr,
        "period_tolerance": period_tolerance, "k_tolerance": k_tolerance,
    }

    loglike_nonfinite = int(not (xbest1 is not None and xbest0 is not None
                                  and np.isfinite(loglike1) and np.isfinite(loglike0)))

    if xbest1 is None or xbest0 is None or loglike_nonfinite:
        base.update({
            "P_rec": float("nan"), "K_rec": float("nan"), "A_rec": float("nan"), "B_rec": float("nan"),
            "best_loglike": loglike1 if xbest1 is not None else float("nan"),
            "loglike_gp_only": loglike0 if xbest0 is not None else float("nan"),
            "T_stat": float("nan"),
            "bic_gp_only": float("nan"), "bic_gp_planet": float("nan"), "delta_bic": float("nan"),
            "gp_planet_at_bounds": ",".join(map(str, diag1.get("at_bounds", []))) if diag1 else "",
            "gp_only_at_bounds": ",".join(map(str, diag0.get("at_bounds", []))) if diag0 else "",
            "loglike_nonfinite": 1, "nested_model_violation": 0,
            "likelihood_ratio_pass": False, "param_match_pass": False, "recovered": 0,
            "diagnostic_flags": fit_error if fit_error else "loglike_nonfinite",
        })
        return base

    P_rec, A_rec, B_rec = xbest1[-3], xbest1[-2], xbest1[-1]
    K_rec = float(np.sqrt(A_rec ** 2 + B_rec ** 2))

    T_stat = 2.0 * (loglike1 - loglike0)
    k1, k0 = len(xbest1), len(xbest0)
    bic1 = k1 * np.log(n_obs) - 2 * loglike1
    bic0 = k0 * np.log(n_obs) - 2 * loglike0
    delta_bic = bic0 - bic1

    nested_model_violation = int(T_stat < 0)
    likelihood_ratio_pass = bool(t_crit is not None and T_stat > t_crit)
    param_match_pass = bool(
        abs(P_rec - period_inj) / period_inj <= period_tolerance
        and abs(K_rec - k_inj) / k_inj <= k_tolerance
    )
    recovered = int(likelihood_ratio_pass and param_match_pass)

    flags = []
    if nested_model_violation:
        flags.append("nested_model_violation")
    if diag1.get("at_bounds"):
        flags.append("gp_planet_at_bounds")
    if diag0.get("at_bounds"):
        flags.append("gp_only_at_bounds")
    if fit_error:
        flags.append(fit_error)

    base.update({
        "P_rec": P_rec, "K_rec": K_rec, "A_rec": A_rec, "B_rec": B_rec,
        "best_loglike": loglike1, "loglike_gp_only": loglike0, "T_stat": T_stat,
        "bic_gp_only": bic0, "bic_gp_planet": bic1, "delta_bic": delta_bic,
        "gp_planet_at_bounds": ",".join(map(str, diag1.get("at_bounds", []))),
        "gp_only_at_bounds": ",".join(map(str, diag0.get("at_bounds", []))),
        "loglike_nonfinite": 0, "nested_model_violation": nested_model_violation,
        "likelihood_ratio_pass": likelihood_ratio_pass, "param_match_pass": param_match_pass,
        "recovered": recovered,
        "diagnostic_flags": ",".join(flags),
    })
    return base


def run_one_combo(args_dict, period_idx, k_idx, phase_idx, period, k, seed, modes_needed):
    args = argparse.Namespace(**{key: val for key, val in args_dict.items() if key != "_tcrit"})
    tcrit_by_mode = args_dict.get("_tcrit", {})

    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0, 2 * np.pi))
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

    n_obs = len(t_full)
    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]
    amp_bound = args.planet_amp_max if args.planet_amp_max is not None else 1.5 * args.k_max

    # cyc/nonstat share the cycle-fit warm start; no_cycle never fits a cycle
    # at all (matches main.py --no-fit-cycle) and warm-starts its additive
    # offsets from the raw data means instead.
    cycle_params = None
    if "cyc" in modes_needed or "nonstat" in modes_needed:
        try:
            _, _, cycle_params = mf.fit_cycle(
                t_full, y_full, series_index,
                b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0,
                plot=False, print_results=False, return_fit=True,
            )
        except Exception as exc:
            print(f"[combo p_idx={period_idx} k_idx={k_idx} phase_idx={phase_idx}] "
                  f"cycle warm-start fit failed, falling back to raw seed: {exc}", flush=True)
            cycle_params = fallback_cycle_seed(y_full, series_index, args)

    planet_guess, _ = mf.period_guess(
        t_full, y_full, yerr_full, series_index,
        PMIN=1.1, PMAX=args.period_max, MAX_FAP=1e-5, MAX_NPL=2, plot=False,
    )

    rows = []
    for mode in modes_needed:
        if mode == "cyc":
            fit_fn = fit_cyc
            fit_args = (t_full, y_full, yerr_full, series_index, stds, args,
                        planet_guess, cycle_params, amp_bound, rv_std)
        elif mode == "nonstat":
            fit_fn = fit_nonstat
            fit_args = (t_full, y_full, yerr_full, series_index, stds, args,
                        planet_guess, cycle_params, amp_bound, rv_std)
        elif mode == "no_cycle":
            rv_offset0 = float(np.mean(y_full[series_index[0]]))
            rhk_offset0 = float(np.mean(y_full[series_index[1]]))
            fit_fn = fit_nocyc
            fit_args = (t_full, y_full, yerr_full, series_index, stds, args,
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
            print(f"[combo p_idx={period_idx} k_idx={k_idx} phase_idx={phase_idx} mode={mode}] "
                  f"GP+planet fit failed: {exc}", flush=True)

        try:
            xbest0, loglike0, diag0 = fit_fn(*fit_args, fit_planet=False)
        except Exception as exc:
            errors.append(f"gp_only_exception:{exc!r}")
            print(f"[combo p_idx={period_idx} k_idx={k_idx} phase_idx={phase_idx} mode={mode}] "
                  f"GP-only fit failed: {exc}", flush=True)

        t_crit_entry = tcrit_by_mode.get(mode, {})
        rows.append(make_row(
            period_idx, k_idx, phase_idx, period, k, phase, A_inj, B_inj, mode,
            xbest1, loglike1, diag1, xbest0, loglike0, diag0,
            n_obs, t_crit_entry.get("t_crit"), t_crit_entry.get("alpha"),
            args.period_tolerance, args.k_tolerance,
            ";".join(errors) if errors else None,
        ))

    return rows


def _worker(task):
    return run_one_combo(*task)


def build_grid(args):
    if args.period_values:
        periods = np.array([float(x) for x in args.period_values.split(",")])
    else:
        periods = np.logspace(np.log10(args.period_min), np.log10(args.period_max), args.n_period)

    if args.k_values:
        ks = np.array([float(x) for x in args.k_values.split(",")])
    else:
        ks = np.logspace(np.log10(args.k_min), np.log10(args.k_max), args.n_k)

    return periods, ks


def load_completed_combos(csv_path):
    """Returns {(period_idx, k_idx, phase_idx): {modes already present}}."""
    if not os.path.exists(csv_path):
        return {}

    modes_by_combo = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["period_idx"]), int(row["k_idx"]), int(row["phase_idx"]))
            modes_by_combo.setdefault(key, set()).add(row["mode"])

    return modes_by_combo


def load_tcrit(tcrit_path, requested_models):
    """Loads {mode: {"t_crit":..., "alpha":...}} for the requested modes from
    tcrit.json (written by null_calibration.py). Errors loudly rather than
    falling back to any guessed/fixed threshold."""
    if not os.path.exists(tcrit_path):
        raise SystemExit(
            f"--tcrit-path {tcrit_path} not found. Run null_calibration.py first to produce a "
            f"calibrated detection threshold for mode(s) {sorted(requested_models)} before "
            "running the injection-recovery sweep."
        )
    with open(tcrit_path) as f:
        tcrit_data = json.load(f)

    modes = tcrit_data.get("modes", {})
    missing = requested_models - set(modes)
    if missing:
        raise SystemExit(
            f"{tcrit_path} has no calibrated threshold for mode(s) {sorted(missing)}. "
            "Run null_calibration.py for those modes first."
        )
    return {
        mode: {"t_crit": modes[mode]["t_crit"], "alpha": tcrit_data.get("alpha")}
        for mode in requested_models
    }


def run_resumable_pool(csv_path, csv_fields, tasks, worker_fn, n_workers, force_fresh=False):
    """Shared driver for a multiprocessing.Pool sweep whose per-task results
    (lists of row dicts) are streamed to a resumable, flush-per-row CSV.
    Used by both injection_recovery.py and null_calibration.py so both scripts
    share identical multiprocessing/resumability behaviour."""
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if force_fresh and os.path.exists(csv_path):
        os.remove(csv_path)

    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header is not None and existing_header != csv_fields:
            raise SystemExit(
                f"{csv_path} has a different column schema than expected and cannot be safely "
                "resumed into (old rows can't be upgraded in place). Move the old file aside or "
                "pass --force-fresh."
            )

    if not tasks:
        print("Nothing to do.")
        return

    n_workers = n_workers or max(1, min(9, (os.cpu_count() or 2) - 1))
    print(f"Using {n_workers} worker processes.", flush=True)

    file_is_new = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if file_is_new:
        writer.writeheader()
        csv_file.flush()

    start_time = time.time()
    n_done = 0

    try:
        with mp.Pool(n_workers) as pool:
            for rows in pool.imap_unordered(worker_fn, tasks):
                for row in rows:
                    writer.writerow(row)
                csv_file.flush()

                n_done += 1
                elapsed = time.time() - start_time
                eta_min = (elapsed / n_done) * (len(tasks) - n_done) / 60.0
                print(f"[{n_done}/{len(tasks)}] task done "
                      f"(elapsed={elapsed / 60.0:.1f} min, ETA={eta_min:.1f} min)", flush=True)
    finally:
        csv_file.close()

    print("Done.")


def main_from_args(args):
    requested_models = set(m.strip() for m in args.models.split(","))
    unknown = requested_models - {"cyc", "nonstat", "no_cycle"}
    if unknown:
        raise SystemExit(f"--models: unknown model(s) {sorted(unknown)}; choose from cyc,nonstat,no_cycle")

    tcrit_by_mode = load_tcrit(args.tcrit_path, requested_models)
    if args.alpha is not None:
        for mode, entry in tcrit_by_mode.items():
            if entry["alpha"] is not None and not np.isclose(entry["alpha"], args.alpha):
                raise SystemExit(
                    f"--alpha {args.alpha} does not match the alpha={entry['alpha']} baked into "
                    f"{args.tcrit_path} for mode {mode!r}."
                )

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.force_fresh and os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    periods, ks = build_grid(args)

    all_combos = [
        (pi, ki, phi)
        for pi in range(len(periods))
        for ki in range(len(ks))
        for phi in range(args.n_phases)
    ]
    completed = load_completed_combos(args.output_csv)

    remaining = []
    for c in all_combos:
        modes_needed = requested_models - completed.get(c, set())
        if modes_needed:
            remaining.append((c, modes_needed))

    n_fully_done = len(all_combos) - len(remaining)
    print(f"{len(all_combos)} total combos, {n_fully_done} already fully done, "
          f"{len(remaining)} with outstanding modes.", flush=True)

    args_dict = vars(args).copy()
    args_dict["_tcrit"] = tcrit_by_mode
    tasks = [
        (args_dict, pi, ki, phi, float(periods[pi]), float(ks[ki]),
         args.seed + pi * 1000 + ki * 10 + phi, modes_needed)
        for ((pi, ki, phi), modes_needed) in remaining
    ]

    run_resumable_pool(args.output_csv, CSV_FIELDS, tasks, _worker, args.n_workers, force_fresh=False)


def main():
    main_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
