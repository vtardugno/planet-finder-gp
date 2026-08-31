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

    return parser


def build_bounds_list_cyc(args, stds, period_bounds, amp_bound):
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
        (args.b_min, args.b_max),
        (args.Pcyc_min, args.Pcyc_max),
        (args.phi_min, args.phi_max),
        (args.a0_min, args.a0_max),
        (args.a1_min, args.a1_max),
        (args.delta0_min, args.delta0_max),
        (args.delta1_min, args.delta1_max),
        period_bounds,
        (-amp_bound, amp_bound),
        (-amp_bound, amp_bound),
    ]


def build_bounds_list_nonstat(args, stds, period_bounds, amp_bound):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]

    return [
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
        period_bounds,
        (-amp_bound, amp_bound),
        (-amp_bound, amp_bound),
    ]


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
        (args.rho_min, args.nocyc_rho_max),
        (args.eta_min, args.nocyc_eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (args.nocyc_delta0_min, args.nocyc_delta0_max),
        (args.nocyc_delta1_min, args.nocyc_delta1_max),
        period_bounds,
        (-amp_bound, amp_bound),
        (-amp_bound, amp_bound),
    ]


def fit_cyc(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, cycle_seed, amp_bound, rv_std):
    rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_seed

    best_loglike = -np.inf
    xbest_all = None

    for p in planet_guess:
        period_bounds = (max(p - 10, 1.1), min(p + 10, args.period_max))
        bounds_list = build_bounds_list_cyc(args, stds, period_bounds, amp_bound)

        for Afrac in A_INIT_FRACS:
            for Bfrac in B_INIT_FRACS:
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
                    planet_p=p, planet_A=args.planet_A_fit * Afrac, planet_B=args.planet_B_fit * Bfrac,
                    fit_planet=True, change_C=True,
                )

                loglike = -1 * mf.negloglike_cyc(xbest, t_full, y_full, series_index, C, rv_std, inject_planet=True)[0]

                if loglike > best_loglike:
                    best_loglike = loglike
                    xbest_all = xbest

    return xbest_all, best_loglike


def fit_nonstat(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, cycle_seed, amp_bound, rv_std):
    rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_seed

    best_loglike = -np.inf
    xbest_all = None

    for p in planet_guess:
        period_bounds = (max(p - 10, 1.1), min(p + 10, args.period_max))
        bounds_list = build_bounds_list_nonstat(args, stds, period_bounds, amp_bound)

        for Afrac in A_INIT_FRACS:
            for Bfrac in B_INIT_FRACS:
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
                    planet_p=p, planet_A=args.planet_A_fit * Afrac, planet_B=args.planet_B_fit * Bfrac,
                    fit_planet=True, change_C=True,
                )

                loglike = -1 * mfn.negloglike_nonstat(xbest, t_full, y_full, series_index, C, rv_std, inject_planet=True)[0]

                if loglike > best_loglike:
                    best_loglike = loglike
                    xbest_all = xbest

    return xbest_all, best_loglike


def fit_nocyc(t_full, y_full, yerr_full, series_index, stds, args, planet_guess, rv_offset0, rhk_offset0, amp_bound, rv_std):
    best_loglike = -np.inf
    xbest_all = None

    for p in planet_guess:
        period_bounds = (max(p - 10, 1.1), min(p + 10, args.period_max))
        bounds_list = build_bounds_list_nocyc(args, stds, period_bounds, amp_bound)

        for Afrac in A_INIT_FRACS:
            for Bfrac in B_INIT_FRACS:
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
                    planet_p=p, planet_A=args.planet_A_fit * Afrac, planet_B=args.planet_B_fit * Bfrac,
                    fit_planet=True, change_C=True,
                )

                loglike = -1 * mf.negloglike_nocyc(xbest, t_full, y_full, series_index, C, rv_std, inject_planet=True)[0]

                if loglike > best_loglike:
                    best_loglike = loglike
                    xbest_all = xbest

    return xbest_all, best_loglike


def make_row(period_idx, k_idx, phase_idx, period_inj, k_inj, phase_inj, A_inj, B_inj, mode, xbest, loglike):
    if xbest is None:
        return {
            "period_idx": period_idx, "k_idx": k_idx, "phase_idx": phase_idx,
            "period_inj": period_inj, "k_inj": k_inj, "phase_inj": phase_inj,
            "A_inj": A_inj, "B_inj": B_inj, "mode": mode,
            "P_rec": float("nan"), "K_rec": float("nan"), "A_rec": float("nan"), "B_rec": float("nan"),
            "best_loglike": float("nan"), "recovered": 0,
        }

    P_rec, A_rec, B_rec = xbest[-3], xbest[-2], xbest[-1]
    K_rec = float(np.sqrt(A_rec ** 2 + B_rec ** 2))
    recovered = int(
        abs(P_rec - period_inj) / period_inj <= 0.10
        and abs(K_rec - k_inj) / k_inj <= 0.15
    )
    return {
        "period_idx": period_idx, "k_idx": k_idx, "phase_idx": phase_idx,
        "period_inj": period_inj, "k_inj": k_inj, "phase_inj": phase_inj,
        "A_inj": A_inj, "B_inj": B_inj, "mode": mode,
        "P_rec": P_rec, "K_rec": K_rec, "A_rec": A_rec, "B_rec": B_rec,
        "best_loglike": loglike, "recovered": recovered,
    }


def run_one_combo(args_dict, period_idx, k_idx, phase_idx, period, k, seed, modes_needed):
    args = argparse.Namespace(**args_dict)

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
    if "cyc" in modes_needed or "nonstat" in modes_needed:
        _, _, cycle_params = mf.fit_cycle(
            t_full, y_full, series_index,
            b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0,
            plot=False, print_results=False, return_fit=True,
        )

    planet_guess, _ = mf.period_guess(
        t_full, y_full, yerr_full, series_index,
        PMIN=1.1, PMAX=args.period_max, MAX_FAP=1e-5, MAX_NPL=2, plot=False,
    )

    rows = []
    for mode in modes_needed:
        try:
            if mode == "cyc":
                xbest, loglike = fit_cyc(t_full, y_full, yerr_full, series_index, stds, args,
                                          planet_guess, cycle_params, amp_bound, rv_std)
            elif mode == "nonstat":
                xbest, loglike = fit_nonstat(t_full, y_full, yerr_full, series_index, stds, args,
                                              planet_guess, cycle_params, amp_bound, rv_std)
            elif mode == "no_cycle":
                rv_offset0 = float(np.mean(y_full[series_index[0]]))
                rhk_offset0 = float(np.mean(y_full[series_index[1]]))
                xbest, loglike = fit_nocyc(t_full, y_full, yerr_full, series_index, stds, args,
                                            planet_guess, rv_offset0, rhk_offset0, amp_bound, rv_std)
            else:
                raise ValueError(f"unknown mode {mode!r}")
        except Exception as exc:
            print(f"[combo p_idx={period_idx} k_idx={k_idx} phase_idx={phase_idx} mode={mode}] "
                  f"failed: {exc}", flush=True)
            xbest, loglike = None, float("nan")

        rows.append(make_row(period_idx, k_idx, phase_idx, period, k, phase, A_inj, B_inj, mode, xbest, loglike))

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


def main():
    args = build_parser().parse_args()

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.force_fresh and os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    requested_models = set(m.strip() for m in args.models.split(","))
    unknown = requested_models - {"cyc", "nonstat", "no_cycle"}
    if unknown:
        raise SystemExit(f"--models: unknown model(s) {sorted(unknown)}; choose from cyc,nonstat,no_cycle")

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

    if not remaining:
        print("Nothing to do.")
        return

    args_dict = vars(args)
    tasks = [
        (args_dict, pi, ki, phi, float(periods[pi]), float(ks[ki]),
         args.seed + pi * 1000 + ki * 10 + phi, modes_needed)
        for ((pi, ki, phi), modes_needed) in remaining
    ]

    n_workers = args.n_workers or max(1, min(9, (os.cpu_count() or 2) - 1))
    print(f"Using {n_workers} worker processes.", flush=True)

    file_is_new = not os.path.exists(args.output_csv)
    csv_file = open(args.output_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
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
