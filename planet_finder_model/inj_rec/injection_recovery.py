"""Injection-recovery grid sweep comparing the cyc, nonstat, and no_cycle models.

For each (period, semi-amplitude K, orbital phase) combination on a log-spaced
grid, injects a synthetic planet into the Solar RV data and fits it with
whichever models --models selects: cyc (cyc_pipeline2/functions.py),
nonstat (nonstationary/nonstationary_pipeline2/functions_nonstat.py),
nonstat2 (nonstationary_2/functions_nonstat.py), and no_cycle
(cyc_pipeline2/functions.py's bare-GP branch), using the same pipeline-2
search each single-injection script (cyc_pipeline2/main_cycle_likelihood.py
[--no-fit-cycle] / nonstationary/nonstationary_pipeline2/main_nonstat.py /
nonstationary_2/main_nonstat.py) uses: a blind periodogram, then one
L-BFGS-B start per period guess with planet A/B seeded by generalised least
squares, optimised on rescaled parameters.
Results are appended to a CSV as they complete, so the sweep can be
interrupted and resumed by simply re-running the same command. Resumability
is tracked per (combo, mode): re-running with a different --models against
an existing --output-csv only computes whichever modes are still missing for
each combo, so e.g. adding no_cycle to a CSV that already has cyc/nonstat
never recomputes those.

One planet is assumed present: a planet is considered "recovered" if the
best-fit period/K land within tolerance of the injected truth. See
injection_recovery_mgic.py for the version that also requires the planet
model to be preferred over a planet-less one.
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

# the model dirs (cyc_pipeline2, nonstationary*) live in the parent folder
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from spleaf import cov, term
import argparse
import csv
import time
import importlib.util
import multiprocessing as mp


def _load_module(name, relpath):
    # The pipeline-2 modules share their names (functions / functions_nonstat)
    # with the old ones, so load each explicitly from file under a distinct name
    spec = importlib.util.spec_from_file_location(name, os.path.join(_PARENT, relpath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mf = _load_module("functions_pipeline2", os.path.join("cyc_pipeline2", "functions.py"))
mfn = _load_module("functions_nonstat_pipeline2",
                   os.path.join("nonstationary", "nonstationary_pipeline2", "functions_nonstat.py"))
mfn2 = _load_module("functions_nonstat_2", os.path.join("nonstationary_2", "functions_nonstat.py"))


class MultiSeriesKernel(term.MultiSeriesKernel):
  def _grad_param(self, grad_dU=None, grad_dV=None):
    if grad_dU is not None or grad_dV is not None:
      raise NotImplementedError()
    return super()._grad_param()


CSV_FIELDS = [
    "period_idx", "k_idx", "phase_idx",
    "period_inj", "k_inj", "phase_inj", "A_inj", "B_inj",
    "mode", "P_rec", "K_rec", "A_rec", "B_rec",
    "best_loglike", "recovered",
    "param_match_pass", "period_tolerance", "k_tolerance", "k_sigma", "sigma_K_rec",
    # --- fit-quality diagnostics, recorded not discarded ---
    "gp_planet_at_bounds", "loglike_nonfinite",
    "period_search", "period_guesses",
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
                         help="Comma-separated subset of cyc,nonstat,nonstat2,no_cycle to fit per combo.")

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

    # cyc / nonstat / nonstat2 shared cycle-term bounds, data-relative as in
    # cyc_pipeline2/main_cycle_likelihood.py / nonstationary_pipeline2/main_nonstat.py:
    # |b| <= drift/T, |a| <= frac*std, delta within mean +/- frac*std
    parser.add_argument("--cycle-b-max-drift", type=float, default=5.0)
    parser.add_argument("--Pcyc-min", type=float, default=3000.0)
    parser.add_argument("--Pcyc-max", type=float, default=5500.0)
    parser.add_argument("--phi-min", type=float, default=-np.pi)
    parser.add_argument("--phi-max", type=float, default=np.pi)
    parser.add_argument("--a0-max-frac", type=float, default=10.0)
    parser.add_argument("--a1-max-frac", type=float, default=10.0)
    parser.add_argument("--delta0-max-frac", type=float, default=10.0)
    parser.add_argument("--delta1-max-frac", type=float, default=10.0)

    # nonstat-only bounds
    parser.add_argument("--mu-min", type=float, default=-5.0)
    parser.add_argument("--mu-max", type=float, default=5.0)

    # nonstat2-only bounds / warm start (same as CV/CV_nonstat_2.py)
    parser.add_argument("--cycle-c0", type=float, default=None)
    parser.add_argument("--nonstat2-Pcyc-min", type=float, default=3000.0)
    parser.add_argument("--cycle-c-min", type=float, default=0.0)
    parser.add_argument("--cycle-c-max", type=float, default=10.0)

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

    # Cycle warm-start seeds
    parser.add_argument("--cycle-b0", type=float, default=0.0)
    parser.add_argument("--cycle-P0", type=float, default=4000.0)
    parser.add_argument("--cycle-phi0", type=float, default=0.0)
    parser.add_argument("--mu0", type=float, default=0.0)

    parser.add_argument("--output-csv", default="inj_rec/injection_recovery.csv")
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)

    # Planet period search: "periodogram" = Lomb-Scargle on the RVs (mf.period_guess),
    # the same for every model; "gp" = scan of the likelihood gain of a sinusoid
    # under each model's own GP-only fit, so the search uses the activity model
    parser.add_argument("--period-search", choices=["periodogram", "gp"], default="periodogram")
    parser.add_argument("--search-period-min", type=float, default=1.1,
                         help="Shortest planet period searched (periodogram / GP scan) and allowed in "
                              "the planet fit. The longest is --period-max. Independent of the "
                              "injected grid's --period-min.")
    parser.add_argument("--n-period-guesses", type=int, default=2,
                         help="--period-search gp: number of periods (successively prewhitened "
                              "scan peaks) used as planet-fit starts.")
    parser.add_argument("--scan-oversample", type=float, default=5.0,
                         help="--period-search gp: frequency samples per 1/baseline.")

    # Recovery criterion
    parser.add_argument("--period-tolerance", type=float, default=0.10,
                         help="Relative tolerance for the recovered-period parameter-match test.")
    parser.add_argument("--k-tolerance", type=float, default=0.15,
                         help="Relative tolerance for the recovered-K parameter-match test.")
    parser.add_argument("--k-sigma", type=float, default=None,
                         help="If given, the K test also passes when |K_rec - K_inj| <= k_sigma * "
                              "sigma_K (sigma_K of the planet's linear A/B fit at the best-fit P and "
                              "GP), i.e. the looser of this and --k-tolerance.")

    return parser


def build_bounds_list_cyc(args, stds, means, T, period_bounds, amp_bound, fit_planet=True):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]
    b_max = args.cycle_b_max_drift / T
    a0_max = args.a0_max_frac * stds[0]
    a1_max = args.a1_max_frac * stds[1]
    delta0_half = args.delta0_max_frac * stds[0]
    delta1_half = args.delta1_max_frac * stds[1]

    bounds = [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (args.prot_min, args.prot_max),
        (args.rho_min, args.rho_max),
        (args.eta_min, args.eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (-b_max, b_max),
        (args.Pcyc_min, args.Pcyc_max),
        (args.phi_min, args.phi_max),
        (-a0_max, a0_max),
        (-a1_max, a1_max),
        (means[0] - delta0_half, means[0] + delta0_half),
        (means[1] - delta1_half, means[1] + delta1_half),
    ]
    if fit_planet:
        bounds += [period_bounds, (-amp_bound, amp_bound), (-amp_bound, amp_bound)]
    return bounds


def build_bounds_list_nonstat(args, stds, means, T, period_bounds, amp_bound, fit_planet=True):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]
    b_max = args.cycle_b_max_drift / T
    a0_max = args.a0_max_frac * stds[0]
    a1_max = args.a1_max_frac * stds[1]
    delta0_half = args.delta0_max_frac * stds[0]
    delta1_half = args.delta1_max_frac * stds[1]

    bounds = [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (args.mu_min, args.mu_max),
        (-b_max, b_max),
        (args.Pcyc_min, args.Pcyc_max),
        (args.phi_min, args.phi_max),
        (args.prot_min, args.prot_max),
        (args.rho_min, args.rho_max),
        (args.eta_min, args.eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (-a0_max, a0_max),
        (-a1_max, a1_max),
        (means[0] - delta0_half, means[0] + delta0_half),
        (means[1] - delta1_half, means[1] + delta1_half),
    ]
    if fit_planet:
        bounds += [period_bounds, (-amp_bound, amp_bound), (-amp_bound, amp_bound)]
    return bounds


def build_bounds_list_nonstat2(args, stds, means, T, period_bounds, amp_bound, fit_planet=True):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]
    b_max = args.cycle_b_max_drift / T
    a0_max = args.a0_max_frac * stds[0]
    a1_max = args.a1_max_frac * stds[1]
    delta0_half = args.delta0_max_frac * stds[0]
    delta1_half = args.delta1_max_frac * stds[1]

    bounds = [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (-b_max, b_max),
        (args.nonstat2_Pcyc_min, args.Pcyc_max),
        (args.phi_min, args.phi_max),
        (args.cycle_c_min, args.cycle_c_max),
        (args.prot_min, args.prot_max),
        (args.rho_min, args.rho_max),
        (args.eta_min, args.eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (-a0_max, a0_max),
        (-a1_max, a1_max),
        (means[0] - delta0_half, means[0] + delta0_half),
        (means[1] - delta1_half, means[1] + delta1_half),
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
    means = [np.mean(y_full[series_index[0]]), np.mean(y_full[series_index[1]])]
    T = np.max(t_full)

    # Keep the warm start inside the (data-relative) bounds
    ref_bounds = build_bounds_list_cyc(args, stds, means, T, None, amp_bound, fit_planet=False)
    phi0 = (phi0 + np.pi) % (2 * np.pi) - np.pi
    b0 = np.clip(b0, ref_bounds[8][0], ref_bounds[8][1])
    P0 = np.clip(P0, ref_bounds[9][0], ref_bounds[9][1])

    best_loglike = -np.inf
    xbest_all = None
    best_bounds_list = None
    best_period_guess = None
    best_C = None

    # One start per period guess: the planet A/B start at their generalised
    # least-squares values given the starting GP and cycle mean (the fit itself
    # runs on rescaled parameters, see mf.optimise_params_cyc)
    guesses = planet_guess if fit_planet else [None]

    for p in guesses:
        period_bounds = (max(p - 10, args.search_period_min), min(p + 10, args.period_max)) if fit_planet else None
        bounds_list = build_bounds_list_cyc(args, stds, means, T, period_bounds, amp_bound, fit_planet=fit_planet)

        C = cov.Cov(
            t_full,
            err=term.Error(yerr_full),
            rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
            rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
            rot=MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                  np.array([stds[0], stds[1]]),
                                  np.array([stds[0], 0.0])),
        )
        A0 = B0 = 0.0
        if fit_planet:
            resid = y_full.copy()
            resid[series_index[0]] -= rv_amp0 * mf.shared_core(t_full[series_index[0]], b0, P0, phi0) + rv_offset0
            resid[series_index[1]] -= rhk_amp0 * mf.shared_core(t_full[series_index[1]], b0, P0, phi0) + rhk_offset0
            A0, B0 = np.clip(mf.gls_planet_amplitudes(C, t_full, resid, series_index, p, rv_std),
                             -amp_bound, amp_bound)

        xbest, C = mf.optimise_params_cyc(
            t_full, y_full, series_index, C, bounds_list,
            b=b0, P=P0, phi=phi0,
            a0=rv_amp0, a1=rhk_amp0, d0=rv_offset0, d1=rhk_offset0,
            planet_p=(p if fit_planet else args.cycle_P0), planet_A=A0, planet_B=B0,
            fit_planet=fit_planet, change_C=True,
        )

        loglike = -1 * mf.negloglike_cyc(xbest, t_full, y_full, series_index, C, rv_std,
                                          inject_planet=fit_planet)[0]

        if loglike > best_loglike:
            best_loglike = loglike
            xbest_all = xbest
            best_bounds_list = bounds_list
            best_period_guess = p
            best_C = C

    diagnostics = {
        "at_bounds": at_bounds_params(xbest_all, best_bounds_list) if xbest_all is not None else [],
        "best_period_guess": best_period_guess,
        "cov": best_C,
    }
    return xbest_all, best_loglike, diagnostics


def fit_nonstat(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, cycle_seed, amp_bound, rv_std,
                 fit_planet=True):
    rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_seed
    means = [np.mean(y_full[series_index[0]]), np.mean(y_full[series_index[1]])]
    T = np.max(t_full)

    # Keep the warm start inside the (data-relative) bounds
    ref_bounds = build_bounds_list_nonstat(args, stds, means, T, None, amp_bound, fit_planet=False)
    phi0 = (phi0 + np.pi) % (2 * np.pi) - np.pi
    b0 = np.clip(b0, ref_bounds[3][0], ref_bounds[3][1])
    P0 = np.clip(P0, ref_bounds[4][0], ref_bounds[4][1])

    best_loglike = -np.inf
    xbest_all = None
    best_bounds_list = None
    best_period_guess = None
    best_C = None

    # One start per period guess: the planet A/B start at their generalised
    # least-squares values given the starting GP and cycle mean (the fit itself
    # runs on rescaled parameters, see mfn.optimise_params_nonstat)
    guesses = planet_guess if fit_planet else [None]

    for p in guesses:
        period_bounds = (max(p - 10, args.search_period_min), min(p + 10, args.period_max)) if fit_planet else None
        bounds_list = build_bounds_list_nonstat(args, stds, means, T, period_bounds, amp_bound, fit_planet=fit_planet)

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
        A0 = B0 = 0.0
        if fit_planet:
            resid = y_full.copy()
            resid[series_index[0]] -= rv_amp0 * mfn.shared_core(t_full[series_index[0]], b0, P0, phi0) + rv_offset0
            resid[series_index[1]] -= rhk_amp0 * mfn.shared_core(t_full[series_index[1]], b0, P0, phi0) + rhk_offset0
            A0, B0 = np.clip(mfn.gls_planet_amplitudes(C, t_full, resid, series_index, p, rv_std),
                             -amp_bound, amp_bound)

        xbest, C = mfn.optimise_params_nonstat(
            t_full, y_full, series_index, C, bounds_list,
            a0=rv_amp0, a1=rhk_amp0, d0=rv_offset0, d1=rhk_offset0,
            planet_p=(p if fit_planet else args.cycle_P0), planet_A=A0, planet_B=B0,
            fit_planet=fit_planet, change_C=True,
        )

        loglike = -1 * mfn.negloglike_nonstat(xbest, t_full, y_full, series_index, C, rv_std,
                                               inject_planet=fit_planet)[0]

        if loglike > best_loglike:
            best_loglike = loglike
            xbest_all = xbest
            best_bounds_list = bounds_list
            best_period_guess = p
            best_C = C

    diagnostics = {
        "at_bounds": at_bounds_params(xbest_all, best_bounds_list) if xbest_all is not None else [],
        "best_period_guess": best_period_guess,
        "cov": best_C,
    }
    return xbest_all, best_loglike, diagnostics


def fit_nonstat2(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, cycle_seed, amp_bound, rv_std,
                 fit_planet=True):
    rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_seed
    means = [np.mean(y_full[series_index[0]]), np.mean(y_full[series_index[1]])]
    T = np.max(t_full)

    # Warm start, same as nonstationary_2/main_nonstat.py. The covariance
    # envelope is core(t)*core(t'), so orient the core to rise with activity (RHK)
    ref_bounds = build_bounds_list_nonstat2(args, stds, means, T, None, amp_bound, fit_planet=False)
    if rhk_amp0 < 0:
        b0, phi0, rv_amp0, rhk_amp0 = -b0, phi0 + np.pi, -rv_amp0, -rhk_amp0
    phi0 = (phi0 + np.pi) % (2 * np.pi) - np.pi
    b0 = np.clip(b0, ref_bounds[2][0], ref_bounds[2][1])
    P0 = np.clip(P0, ref_bounds[3][0], ref_bounds[3][1])

    # c0 puts the core's minimum 1.0 above 0; then map fit_cycle's
    # offset + amp*(b t + sin) onto d + a*(b t + sin + c)/N exactly
    if args.cycle_c0 is None:
        c0 = -np.min(b0 * t_full + np.sin(2 * np.pi * t_full / P0 + phi0)) + 1.0
    else:
        c0 = args.cycle_c0
    N0 = 1 + c0 + np.abs(b0) * T
    rv_amp0, rhk_amp0 = rv_amp0 * N0, rhk_amp0 * N0
    rv_offset0, rhk_offset0 = rv_offset0 - rv_amp0 * c0 / N0, rhk_offset0 - rhk_amp0 * c0 / N0

    # QP amplitudes scaled so the initial GP variance matches the data despite
    # the envelope being < 1
    core_rms0 = np.sqrt(np.mean(mfn2.shared_core_pos(t_full, b0, P0, phi0, c0, T) ** 2))
    qp_amps0 = np.array([stds[0], stds[1]]) / core_rms0
    qp_beta0 = np.array([stds[0], 0.0]) / core_rms0

    def make_C():
        return cov.Cov(
            t_full,
            err=term.Error(yerr_full),
            rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
            rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
            rot=term.SimpleProductKernel(
                nonstat=mfn2.make_nonstat_kernel(T, b0, P0, phi0, c0),
                qp=MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                     qp_amps0, qp_beta0),
            ),
        )

    best_loglike = -np.inf
    xbest_all = None
    best_bounds_list = None
    best_period_guess = None
    best_C = None

    # One start per period guess (instead of the fixed A/B multi-start): the
    # planet A/B start at their generalised-least-squares values given the
    # starting GP and cycle mean (the fit itself runs on rescaled parameters,
    # see mfn2.optimise_params_nonstat)
    guesses = planet_guess if fit_planet else [None]

    for p in guesses:
        period_bounds = (max(p - 10, args.search_period_min), min(p + 10, args.period_max)) if fit_planet else None
        bounds_list = build_bounds_list_nonstat2(args, stds, means, T, period_bounds, amp_bound, fit_planet=fit_planet)

        C = make_C()
        A0 = B0 = 0.0
        if fit_planet:
            resid = y_full.copy()
            resid[series_index[0]] -= (rv_amp0 * mfn2.shared_core_pos(t_full[series_index[0]], b0, P0, phi0, c0, T)
                                       + rv_offset0)
            resid[series_index[1]] -= (rhk_amp0 * mfn2.shared_core_pos(t_full[series_index[1]], b0, P0, phi0, c0, T)
                                       + rhk_offset0)
            A0, B0 = np.clip(mfn2.gls_planet_amplitudes(C, t_full, resid, series_index, p, rv_std),
                             -amp_bound, amp_bound)

        xbest, C = mfn2.optimise_params_nonstat(
            t_full, y_full, series_index, C, bounds_list,
            a0=rv_amp0, a1=rhk_amp0, d0=rv_offset0, d1=rhk_offset0,
            planet_p=(p if fit_planet else args.cycle_P0), planet_A=A0, planet_B=B0,
            fit_planet=fit_planet, change_C=True,
        )
        loglike = -1 * mfn2.negloglike_nonstat(xbest, t_full, y_full, series_index, C, rv_std,
                                                inject_planet=fit_planet)[0]

        if loglike > best_loglike:
            best_loglike = loglike
            xbest_all = xbest
            best_bounds_list = bounds_list
            best_period_guess = p
            best_C = C

    diagnostics = {
        "at_bounds": at_bounds_params(xbest_all, best_bounds_list) if xbest_all is not None else [],
        "best_period_guess": best_period_guess,
        "cov": best_C,
    }
    return xbest_all, best_loglike, diagnostics


def fit_nocyc(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, rv_offset0, rhk_offset0, amp_bound,
              rv_std, fit_planet=True):
    best_loglike = -np.inf
    xbest_all = None
    best_bounds_list = None
    best_period_guess = None
    best_C = None

    # One start per period guess: the planet A/B start at their generalised
    # least-squares values given the starting GP and offsets (the fit itself
    # runs on rescaled parameters, see mf.optimise_params)
    guesses = planet_guess if fit_planet else [None]

    for p in guesses:
        period_bounds = (max(p - 10, args.search_period_min), min(p + 10, args.period_max)) if fit_planet else None
        bounds_list = build_bounds_list_nocyc(args, stds, period_bounds, amp_bound, fit_planet=fit_planet)

        C = cov.Cov(
            t_full,
            err=term.Error(yerr_full),
            rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
            rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
            rot=MultiSeriesKernel(term.MEPKernel(args.sig, args.prot, args.rho, args.eta), series_index,
                                  np.array([stds[0], stds[1]]),
                                  np.array([stds[0], 0.0])),
        )
        A0 = B0 = 0.0
        if fit_planet:
            resid = y_full.copy()
            resid[series_index[0]] -= rv_offset0
            resid[series_index[1]] -= rhk_offset0
            A0, B0 = np.clip(mf.gls_planet_amplitudes(C, t_full, resid, series_index, p, rv_std),
                             -amp_bound, amp_bound)

        xbest, C = mf.optimise_params(
            t_full, y_full, series_index, C, bounds_list,
            delta_0=rv_offset0, delta_1=rhk_offset0,
            planet_p=(p if fit_planet else 40.05), planet_A=A0, planet_B=B0,
            fit_planet=fit_planet, change_C=True,
        )

        loglike = -1 * mf.negloglike_nocyc(xbest, t_full, y_full, series_index, C, rv_std,
                                            inject_planet=fit_planet)[0]

        if loglike > best_loglike:
            best_loglike = loglike
            xbest_all = xbest
            best_bounds_list = bounds_list
            best_period_guess = p
            best_C = C

    diagnostics = {
        "at_bounds": at_bounds_params(xbest_all, best_bounds_list) if xbest_all is not None else [],
        "best_period_guess": best_period_guess,
        "cov": best_C,
    }
    return xbest_all, best_loglike, diagnostics


def residual(mode, x, C, prep, fit_planet):
    """y minus the mode's mean model at x (mirrors the negloglike_* mean blocks)."""
    t_full, series_index = prep["t_full"], prep["series_index"]
    t_rv, t_act = t_full[series_index[0]], t_full[series_index[1]]
    resid = prep["y_full"].copy()

    if fit_planet:
        resid[series_index[0]] -= mf.planet_injection(t_rv, x[-3], x[-2], x[-1]) / prep["rv_std"]

    if mode == "no_cycle":
        resid[series_index[0]] -= x[8]
        resid[series_index[1]] -= x[9]
        return resid

    if mode == "cyc":
        b, P, phi = x[8], x[9], x[10]
        n = 11
        core_rv, core_act = mf.shared_core(t_rv, b, P, phi), mf.shared_core(t_act, b, P, phi)
    elif mode == "nonstat":
        b, P, phi = C.get_param(["rot.nonstat_b", "rot.nonstat_P", "rot.nonstat_phi"])
        n = len(mfn.get_opt_params_nonstat(C)[0])
        core_rv, core_act = mfn.shared_core(t_rv, b, P, phi), mfn.shared_core(t_act, b, P, phi)
    elif mode == "nonstat2":
        b, P, phi, c = C.get_param(["rot.nonstat_b", "rot.nonstat_P", "rot.nonstat_phi", "rot.nonstat_c"])
        n = len(mfn2.get_opt_params_nonstat(C)[0])
        T = np.max(t_full)
        core_rv = mfn2.shared_core_pos(t_rv, b, P, phi, c, T)
        core_act = mfn2.shared_core_pos(t_act, b, P, phi, c, T)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    a0, a1, d0, d1 = x[n], x[n + 1], x[n + 2], x[n + 3]
    resid[series_index[0]] -= a0 * core_rv + d0
    resid[series_index[1]] -= a1 * core_act + d1
    return resid


def whiten(C, v):
    """C's Cholesky factor applied as L^-1 v / sqrt(D), as in mf.gls_planet_amplitudes."""
    return C.solveL(v, copy=True) / C.sqD()


def whitened_sinusoid(C, t_full, series_index, rv_std, p):
    """Whitened sin and cos design columns (RV series only) of a planet at period p."""
    arg = 2 * np.pi * t_full[series_index[0]] / (p + 0.000001)
    x = np.zeros(len(t_full))
    x[series_index[0]] = np.sin(arg) / rv_std
    xs = whiten(C, x)
    x[series_index[0]] = np.cos(arg) / rv_std
    return xs, whiten(C, x)


def gp_period_scan(C, t_full, resid, series_index, rv_std, periods):
    """Likelihood gain (delta chi^2) of the best-fit sinusoid A sin + B cos at
    each period, under covariance C: whiten with C's Cholesky factor, then
    generalised least squares."""
    rw = whiten(C, resid)
    dchi2 = np.empty(len(periods))
    for i, p in enumerate(periods):
        xs, xc = whitened_sinusoid(C, t_full, series_index, rv_std, p)
        G = np.array([[xs @ xs, xs @ xc], [xs @ xc, xc @ xc]])
        b = np.array([xs @ rw, xc @ rw])
        dchi2[i] = b @ np.linalg.solve(G, b)
    return dchi2


def planet_sigma_K(fit, prep):
    """Standard error of K_rec = hypot(A, B) for a GP+planet fit_* result, from
    the A/B generalised-least-squares covariance (X^T K^-1 X)^-1 at the
    best-fit P and GP hyperparameters (so it ignores their uncertainty)."""
    if fit is None or fit[0] is None or not np.isfinite(fit[1]):
        return float("nan")
    xbest, _, diag = fit
    P, A, B = xbest[-3], xbest[-2], xbest[-1]
    xs, xc = whitened_sinusoid(diag["cov"], prep["t_full"], prep["series_index"], prep["rv_std"], P)
    cov_AB = np.linalg.inv(np.array([[xs @ xs, xs @ xc], [xs @ xc, xc @ xc]]))
    g = np.array([A, B]) / np.hypot(A, B)
    return float(np.sqrt(g @ cov_AB @ g))


def gp_period_guesses(C, resid, prep, args):
    """Planet-fit starting periods from successively prewhitened gp_period_scan
    peaks, C being a GP-only fit and resid its mean-model residual."""
    t_full, series_index, rv_std = prep["t_full"], prep["series_index"], prep["rv_std"]
    baseline = np.ptp(t_full)
    freqs = np.arange(1.0 / args.period_max, 1.0 / args.search_period_min, 1.0 / (args.scan_oversample * baseline))
    periods = 1.0 / freqs

    resid = resid.copy()
    guesses = []
    for _ in range(args.n_period_guesses):
        p = float(periods[np.argmax(gp_period_scan(C, t_full, resid, series_index, rv_std, periods))])
        guesses.append(p)
        A, B = mf.gls_planet_amplitudes(C, t_full, resid, series_index, p, rv_std)
        resid[series_index[0]] -= mf.planet_injection(t_full[series_index[0]], p, A, B) / rv_std
    return guesses


def make_row(period_idx, k_idx, phase_idx, period_inj, k_inj, phase_inj, A_inj, B_inj, mode,
             xbest, loglike, diag, period_tolerance, k_tolerance, fit_error,
             sigma_K=float("nan"), k_sigma=None):
    base = {
        "period_idx": period_idx, "k_idx": k_idx, "phase_idx": phase_idx,
        "period_inj": period_inj, "k_inj": k_inj, "phase_inj": phase_inj,
        "A_inj": A_inj, "B_inj": B_inj, "mode": mode,
        "period_tolerance": period_tolerance, "k_tolerance": k_tolerance,
        "k_sigma": k_sigma, "sigma_K_rec": sigma_K,
        "gp_planet_at_bounds": ",".join(map(str, diag.get("at_bounds", []))) if diag else "",
    }

    if xbest is None or not np.isfinite(loglike):
        base.update({
            "P_rec": float("nan"), "K_rec": float("nan"), "A_rec": float("nan"), "B_rec": float("nan"),
            "best_loglike": loglike if xbest is not None else float("nan"),
            "loglike_nonfinite": 1, "param_match_pass": False, "recovered": 0,
            "diagnostic_flags": fit_error if fit_error else "loglike_nonfinite",
        })
        return base

    P_rec, A_rec, B_rec = xbest[-3], xbest[-2], xbest[-1]
    K_rec = float(np.sqrt(A_rec ** 2 + B_rec ** 2))

    # with k_sigma, whichever of the two K bounds is looser: the sigma_K one near
    # the detection limit, the fractional one at high K where sigma_K/K is tiny
    k_match = abs(K_rec - k_inj) / k_inj <= k_tolerance
    if k_sigma is not None:
        k_match = k_match or abs(K_rec - k_inj) <= k_sigma * sigma_K
    param_match_pass = bool(abs(P_rec - period_inj) / period_inj <= period_tolerance and k_match)

    flags = []
    if diag.get("at_bounds"):
        flags.append("gp_planet_at_bounds")
    if fit_error:
        flags.append(fit_error)

    base.update({
        "P_rec": P_rec, "K_rec": K_rec, "A_rec": A_rec, "B_rec": B_rec,
        "best_loglike": loglike, "loglike_nonfinite": 0,
        "param_match_pass": param_match_pass, "recovered": int(param_match_pass),
        "diagnostic_flags": ",".join(flags),
    })
    return base


def prepare_injection(args, period, k, seed, modes_needed, log_prefix=""):
    """Injects the planet into the data and computes everything the per-mode
    fits share (warm starts, periodogram guesses)."""
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

    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]
    amp_bound = args.planet_amp_max if args.planet_amp_max is not None else 1.5 * args.k_max

    # cyc/nonstat share the cycle-fit warm start; no_cycle never fits a cycle
    # at all (matches main.py --no-fit-cycle) and warm-starts its additive
    # offsets from the raw data means instead.
    cycle_params = None
    if "cyc" in modes_needed or "nonstat" in modes_needed or "nonstat2" in modes_needed:
        try:
            _, _, cycle_params = mf.fit_cycle(
                t_full, y_full, series_index,
                b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0,
                plot=False, print_results=False, return_fit=True,
            )
        except Exception as exc:
            print(f"{log_prefix} cycle warm-start fit failed, falling back to raw seed: {exc}", flush=True)
            cycle_params = fallback_cycle_seed(y_full, series_index, args)

    planet_guess = None
    if args.period_search == "periodogram":
        planet_guess, _ = mf.period_guess(
            t_full, y_full, yerr_full, series_index,
            PMIN=args.search_period_min, PMAX=args.period_max, MAX_FAP=1e-5, MAX_NPL=2, plot=False,
        )

    return {
        "phase": phase, "A_inj": A_inj, "B_inj": B_inj,
        "t_full": t_full, "y_full": y_full, "yerr_full": yerr_full, "series_index": series_index,
        "rv_std": rv_std, "stds": stds, "amp_bound": amp_bound,
        "cycle_params": cycle_params, "planet_guess": planet_guess,
    }


def fit_fn_and_args(mode, prep, args, planet_guess=None):
    """The fit_* function for mode and its positional args (fit_planet is passed separately)."""
    if planet_guess is None:
        planet_guess = prep["planet_guess"]
    common = (prep["t_full"], prep["y_full"], prep["yerr_full"], prep["series_index"], prep["stds"], args,
              planet_guess)
    if mode in ("cyc", "nonstat", "nonstat2"):
        fit_fn = {"cyc": fit_cyc, "nonstat": fit_nonstat, "nonstat2": fit_nonstat2}[mode]
        return fit_fn, common + (prep["cycle_params"], prep["amp_bound"], prep["rv_std"])
    if mode == "no_cycle":
        rv_offset0 = float(np.mean(prep["y_full"][prep["series_index"][0]]))
        rhk_offset0 = float(np.mean(prep["y_full"][prep["series_index"][1]]))
        return fit_nocyc, common + (rv_offset0, rhk_offset0, prep["amp_bound"], prep["rv_std"])
    raise ValueError(f"unknown mode {mode!r}")


def fit_mode(mode, prep, args, gp_only=False, log_prefix=""):
    """GP+planet fit of mode (and its GP-only fit, if gp_only or the period
    search needs it). Returns (fit_planet, fit_gp_only, errors, period_guesses),
    each fit being a fit_* (xbest, loglike, diag) tuple or None if it failed."""
    fit_fn, fit_args = fit_fn_and_args(mode, prep, args)
    errors = []
    fit0 = fit1 = None

    if gp_only or args.period_search == "gp":
        try:
            fit0 = fit_fn(*fit_args, fit_planet=False)
        except Exception as exc:
            errors.append(f"gp_only_exception:{exc!r}")
            print(f"{log_prefix} mode={mode} GP-only fit failed: {exc}", flush=True)

    guesses = prep["planet_guess"]
    if args.period_search == "gp":
        if fit0 is None:
            return None, None, errors, None
        xbest0, _, diag0 = fit0
        resid0 = residual(mode, xbest0, diag0["cov"], prep, fit_planet=False)
        guesses = gp_period_guesses(diag0["cov"], resid0, prep, args)
        fit_fn, fit_args = fit_fn_and_args(mode, prep, args, guesses)

    try:
        fit1 = fit_fn(*fit_args, fit_planet=True)
    except Exception as exc:
        errors.append(f"gp_planet_exception:{exc!r}")
        print(f"{log_prefix} mode={mode} GP+planet fit failed: {exc}", flush=True)

    return fit1, fit0, errors, guesses


def run_one_combo(args_dict, period_idx, k_idx, phase_idx, period, k, seed, modes_needed):
    args = argparse.Namespace(**args_dict)
    log_prefix = f"[combo p_idx={period_idx} k_idx={k_idx} phase_idx={phase_idx}]"
    prep = prepare_injection(args, period, k, seed, modes_needed, log_prefix)

    rows = []
    for mode in modes_needed:
        fit1, _, errors, guesses = fit_mode(mode, prep, args, log_prefix=log_prefix)
        xbest, loglike, diag = fit1 or (None, float("nan"), {"at_bounds": []})

        row = make_row(
            period_idx, k_idx, phase_idx, period, k, prep["phase"], prep["A_inj"], prep["B_inj"], mode,
            xbest, loglike, diag, args.period_tolerance, args.k_tolerance,
            ";".join(errors) if errors else None,
            sigma_K=planet_sigma_K(fit1, prep), k_sigma=args.k_sigma,
        )
        row.update(period_search_columns(args, guesses))
        rows.append(row)

    return rows


def period_search_columns(args, guesses):
    return {"period_search": args.period_search,
            "period_guesses": ";".join(f"{p:.4f}" for p in guesses) if guesses is not None else ""}


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


def run_resumable_pool(csv_path, csv_fields, tasks, worker_fn, n_workers, force_fresh=False):
    """Shared driver for a multiprocessing.Pool sweep whose per-task results
    (lists of row dicts) are streamed to a resumable, flush-per-row CSV.
    Used by both injection_recovery.py and injection_recovery_mgic.py so both scripts
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


def main_from_args(args, csv_fields=CSV_FIELDS, worker_fn=None):
    requested_models = set(m.strip() for m in args.models.split(","))
    unknown = requested_models - {"cyc", "nonstat", "nonstat2", "no_cycle"}
    if unknown:
        raise SystemExit(f"--models: unknown model(s) {sorted(unknown)}; choose from cyc,nonstat,nonstat2,no_cycle")

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
    tasks = [
        (args_dict, pi, ki, phi, float(periods[pi]), float(ks[ki]),
         args.seed + pi * 1000 + ki * 10 + phi, modes_needed)
        for ((pi, ki, phi), modes_needed) in remaining
    ]

    run_resumable_pool(args.output_csv, csv_fields, tasks, worker_fn or _worker, args.n_workers, force_fresh=False)


def main():
    main_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
