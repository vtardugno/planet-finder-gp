import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit
from spleaf import cov, term

import functions as mf


class MultiSeriesKernel(term.MultiSeriesKernel):
    def _grad_param(self, grad_dU=None, grad_dV=None):
        if grad_dU is not None or grad_dV is not None:
            raise NotImplementedError()
        return super()._grad_param()


@dataclass
class FoldData:
    t_full: np.ndarray
    y_full: np.ndarray
    yerr_full: np.ndarray
    series_index: List[np.ndarray]
    rv_std_raw: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="12-fold cross-validation for the RV/RHK model.")

    parser.add_argument("--path", default="Solar Data", help="Directory containing Analyse_summary.csv and Analyse_ccf.p")
    parser.add_argument("--star-name", default="Sun", help="Star name key used inside Analyse_ccf.p")
    parser.add_argument("--n-folds", type=int, default=12, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used only if shuffling is enabled")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True, help="Shuffle observation order before splitting into folds")

    parser.add_argument("--inject-planet", action=argparse.BooleanOptionalAction, default=True, help="Inject a synthetic planet signal into RV before CV")
    parser.add_argument("--planet-period", type=float, default=40.0, help="Injected planet period")
    parser.add_argument("--planet-A", type=float, default=0.001, help="Injected planet semi-amplitude")
    parser.add_argument("--planet-B", type=float, default=0.001, help="Injected planet phase")

    parser.add_argument("--fit-planet", action=argparse.BooleanOptionalAction, default=True, help="Include planet parameters in the optimisation")

    parser.add_argument("--cycle-b0", type=float, default=0.0, help="Initial shared linear coefficient for cycle fit")
    parser.add_argument("--cycle-P0", type=float, default=4000.0, help="Initial cycle period")
    parser.add_argument("--cycle-phi0", type=float, default=0.0, help="Initial cycle phase")

    parser.add_argument("--sig", type=float, default=1.0, help="MEPKernel sigma")
    parser.add_argument("--prot", type=float, default=27.0, help="Initial rotation period")
    parser.add_argument("--rho", type=float, default=20.0, help="Initial rho")
    parser.add_argument("--eta", type=float, default=0.25, help="Initial eta")

    parser.add_argument("--rvjit-frac", type=float, default=0.1, help="RV jitter as a fraction of RV std")
    parser.add_argument("--rhkjit-frac", type=float, default=0.1, help="RHK jitter as a fraction of RHK std")

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
    parser.add_argument("--planet-p-min", type=float, default=None)
    parser.add_argument("--planet-p-max", type=float, default=None)
    parser.add_argument("--planet-A-min", type=float, default=1e-6)
    parser.add_argument("--planet-A-max", type=float, default=0.1)
    parser.add_argument("--planet-B-min", type=float, default=1e-6)
    parser.add_argument("--planet-B-max", type=float, default=0.1)
    parser.add_argument("--delta0-min", type=float, default=-0.5)
    parser.add_argument("--delta0-max", type=float, default=0.5)
    parser.add_argument("--delta1-min", type=float, default=-2.0)
    parser.add_argument("--delta1-max", type=float, default=2.0)

    parser.add_argument("--delta-0", type=float, default=-0.001, help="Initial delta_0")
    parser.add_argument("--delta-1", type=float, default=0.001, help="Initial delta_1")
    parser.add_argument("--planet-p", type=float, default=None, help="Initial planet period")
    parser.add_argument("--planet-A-fit", type=float, default=0.01, help="Initial planet A")
    parser.add_argument("--planet-B-fit", type=float, default=0.01, help="Initial planet B")

    parser.add_argument("--output-csv", default="cv_12fold_results.csv", help="Where to save per-fold results")
    return parser


def build_bounds_list_for_cv(args, stds, fit_planet: bool):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]

    base = [
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
    ]

    if not fit_planet:
        return base

    return base + [
        (args.planet_p_min, args.planet_p_max),
        (args.planet_A_min, args.planet_A_max),
        (args.planet_B_min, args.planet_B_max),
    ]


def build_covariance(t_full, yerr_full, series_index, args, stds):
    return cov.Cov(
        t_full,
        err=term.Error(yerr_full),
        rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
        rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
        rot=MultiSeriesKernel(
            term.MEPKernel(args.sig, args.prot, args.rho, args.eta),
            series_index,
            np.array([stds[0], stds[1]]),
            np.array([stds[0], 0.0]),
        ),
    )


def fit_cycle_params(t, rv, rhk, b0, P0, phi0):
    x_concat = np.concatenate([t, t])
    y_concat = np.concatenate([rv, rhk])

    p0 = [
        float(np.mean(rv)), float(np.std(rv)),
        float(np.mean(rhk)), float(np.std(rhk)),
        b0, P0, phi0,
    ]

    params, _ = curve_fit(mf.joint_model, x_concat, y_concat, p0=p0, maxfev=20000)
    return params


def apply_cycle_params(t, rv, rhk, params):
    a1, c1, a2, c2, b, P, phi = params
    rv_fit = mf.model_rv(t, a1, c1, b, P, phi)
    rhk_fit = mf.model_rhk(t, a2, c2, b, P, phi)
    return rv - rv_fit, rhk - rhk_fit


def prepare_fold_data(times, rv, rv_err, rhk, rhk_err, train_idx, test_idx, cycle_fit: bool, args):
    train_t = times[train_idx]
    test_t = times[test_idx]

    train_rv = rv[train_idx].copy()
    train_rhk = rhk[train_idx].copy()
    train_rv_err = rv_err[train_idx].copy()
    train_rhk_err = rhk_err[train_idx].copy()

    test_rv = rv[test_idx].copy()
    test_rhk = rhk[test_idx].copy()
    test_rv_err = rv_err[test_idx].copy()
    test_rhk_err = rhk_err[test_idx].copy()

    if cycle_fit:
        cycle_params = fit_cycle_params(train_t, train_rv, train_rhk, args.cycle_b0, args.cycle_P0, args.cycle_phi0)
        train_rv, train_rhk = apply_cycle_params(train_t, train_rv, train_rhk, cycle_params)
        test_rv, test_rhk = apply_cycle_params(test_t, test_rv, test_rhk, cycle_params)

    rv_mean = float(np.mean(train_rv))
    rv_std_raw = float(np.std(train_rv))
    rhk_mean = float(np.mean(train_rhk))
    rhk_std = float(np.std(train_rhk))

    if rv_std_raw == 0.0 or rhk_std == 0.0:
        raise ValueError("One of the training series has zero standard deviation after cycle removal.")

    train_rv_n = (train_rv - rv_mean) / rv_std_raw
    test_rv_n = (test_rv - rv_mean) / rv_std_raw
    train_rhk_n = (train_rhk - rhk_mean) / rhk_std
    test_rhk_n = (test_rhk - rhk_mean) / rhk_std

    train_rv_err_n = train_rv_err / rv_std_raw
    test_rv_err_n = test_rv_err / rv_std_raw
    train_rhk_err_n = train_rhk_err / rhk_std
    test_rhk_err_n = test_rhk_err / rhk_std

    t_train_full, y_train_full, yerr_train_full, series_index_train = cov.merge_series(
        [train_t, train_t],
        [train_rv_n, train_rhk_n],
        [train_rv_err_n, train_rhk_err_n],
    )

    t_test_full, y_test_full, yerr_test_full, series_index_test = cov.merge_series(
        [test_t, test_t],
        [test_rv_n, test_rhk_n],
        [test_rv_err_n, test_rhk_err_n],
    )

    # After normalisation, the relevant scale is close to unity.
    stds_train = [float(np.std(train_rv_n)), float(np.std(train_rhk_n))]
    stds_test = [float(np.std(test_rv_n)), float(np.std(test_rhk_n))]

    return (
        FoldData(t_train_full, y_train_full, yerr_train_full, series_index_train, rv_std_raw),
        FoldData(t_test_full, y_test_full, yerr_test_full, series_index_test, rv_std_raw),
        stds_train,
        stds_test,
    )


def guess_planet_period(args, train_fold: FoldData):
    if not args.fit_planet:
        return None
    if args.planet_p is not None:
        return args.planet_p

    periods, _ = mf.period_guess(
        train_fold.t_full,
        train_fold.y_full,
        train_fold.yerr_full,
        train_fold.series_index,
        PMIN=1.1,
        PMAX=310.0,
        MAX_FAP=1e-5,
        MAX_NPL=2,
        plot=False,
        output_name="periodogram_cv.png",
    )

    if len(periods) >= 2:
        return float(periods[1])
    return float(periods[0])


def optimise_on_fold(args, train_fold: FoldData, fit_planet: bool, planet_p0: float | None, stds):
    if fit_planet and planet_p0 is None:
        raise ValueError("planet_p0 is required when fit_planet=True")

    bounds_list = build_bounds_list_for_cv(args, stds, fit_planet)

    if fit_planet:
        A_inits = [args.planet_A_fit, args.planet_A_fit / 10.0, args.planet_A_fit / 100.0]
        B_inits = [args.planet_B_fit, args.planet_B_fit / 10.0, args.planet_B_fit / 100.0]
    else:
        A_inits = [None]
        B_inits = [None]

    best_train_loglike = -np.inf
    best_xbest = None
    best_C = None
    best_init = (None, None)

    for A0 in A_inits:
        for B0 in B_inits:
            C = build_covariance(train_fold.t_full, train_fold.yerr_full, train_fold.series_index, args, stds)
            xbest, C = mf.optimise_params(
                train_fold.t_full,
                train_fold.y_full,
                train_fold.series_index,
                C,
                bounds_list,
                delta_0=args.delta_0,
                delta_1=args.delta_1,
                planet_p=planet_p0 if planet_p0 is not None else 40.0,
                planet_A=A0 if A0 is not None else args.planet_A_fit,
                planet_B=B0 if B0 is not None else args.planet_B_fit,
                fit_planet=fit_planet,
                change_C=True,
            )

            train_loglike = -mf.negloglike_nocyc(
                xbest,
                train_fold.t_full,
                train_fold.y_full,
                train_fold.series_index,
                C,
                rv_std=train_fold.rv_std_raw,
                inject_planet=fit_planet,
            )[0]

            if train_loglike > best_train_loglike:
                best_train_loglike = float(train_loglike)
                best_xbest = xbest
                best_C = C
                best_init = (A0, B0)

    assert best_xbest is not None and best_C is not None
    return best_xbest, best_C, best_train_loglike, best_init


def evaluate_loglike(theta, fold: FoldData, args, fit_planet: bool, stds):
    C = build_covariance(fold.t_full, fold.yerr_full, fold.series_index, args, stds)
    loglike = -mf.negloglike_nocyc(
        theta,
        fold.t_full,
        fold.y_full,
        fold.series_index,
        C,
        rv_std=fold.rv_std_raw,
        inject_planet=fit_planet,
    )[0]
    return float(loglike)


def run_cv(args):
    raw = mf.load_and_norm_data(
        args.path,
        args.star_name,
        normalise=False,
        inject_planet=args.inject_planet,
        planet_params=(args.planet_period, args.planet_A, args.planet_B),
    )

    t_full, y_full, yerr_full, series_index = raw
    times = t_full[series_index[0]]
    rv = y_full[series_index[0]]
    rhk = y_full[series_index[1]]
    rv_err = yerr_full[series_index[0]]
    rhk_err = yerr_full[series_index[1]]

    obs_idx = np.arange(len(times))
    rng = np.random.default_rng(args.seed)
    if args.shuffle:
        obs_idx = rng.permutation(obs_idx)

    folds = np.array_split(obs_idx, args.n_folds)

    all_rows = []

    for cycle_fit in (False, True):
        mode_name = "cycle_fit_true" if cycle_fit else "cycle_fit_false"
        print(f"\n=== {mode_name} ===")

        fold_test_loglikes = []

        for fold_id, test_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(obs_idx, test_idx, assume_unique=False)

            train_fold, test_fold, stds_train, _ = prepare_fold_data(
                times, rv, rv_err, rhk, rhk_err, train_idx, test_idx, cycle_fit, args
            )

            # Determine planet period bounds on the training fold only.
            planet_p0 = guess_planet_period(args, train_fold)
            if args.fit_planet and args.planet_p is None:
                pmin = max(planet_p0 - 10.0, 2.0)
                pmax = min(planet_p0 + 10.0, 310.0)
            else:
                pmin = args.planet_p_min if args.planet_p_min is not None else 2.0
                pmax = args.planet_p_max if args.planet_p_max is not None else 310.0

            # Override the default planet bounds before optimisation.
            old_pmin, old_pmax = args.planet_p_min, args.planet_p_max
            args.planet_p_min, args.planet_p_max = pmin, pmax

            try:
                best_xbest, best_C, train_loglike, best_init = optimise_on_fold(
                    args,
                    train_fold,
                    args.fit_planet,
                    planet_p0,
                    stds_train,
                )
                test_loglike = evaluate_loglike(best_xbest, test_fold, args, args.fit_planet, stds_train)
            finally:
                args.planet_p_min, args.planet_p_max = old_pmin, old_pmax

            fold_test_loglikes.append(test_loglike)

            row = {
                "mode": mode_name,
                "fold": fold_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "train_loglike": train_loglike,
                "test_loglike": test_loglike,
                "best_A_init": best_init[0],
                "best_B_init": best_init[1],
                "planet_p0": planet_p0,
            }
            all_rows.append(row)

            print(
                f"fold {fold_id:02d}: train loglike={train_loglike:.6f}, "
                f"test loglike={test_loglike:.6f}, A0={best_init[0]}, B0={best_init[1]}"
            )

        print(f"{mode_name} sum test loglike = {np.sum(fold_test_loglikes):.6f}")

    return all_rows


def main():
    args = build_parser().parse_args()
    rows = run_cv(args)

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved per-fold results to {args.output_csv}")
        print(df.groupby("mode")["test_loglike"].sum())
    except Exception as exc:
        print(f"\nCould not save CSV: {exc}")
        print(rows)


if __name__ == "__main__":
    main()
