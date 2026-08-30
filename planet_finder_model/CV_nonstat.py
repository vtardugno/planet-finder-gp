import functions as mf
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from spleaf import cov, term
import emcee
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nonstationary"))
import functions_nonstat as mfn

class MultiSeriesKernel(term.MultiSeriesKernel):
  def _grad_param(self, grad_dU=None, grad_dV=None):
    if grad_dU is not None or grad_dV is not None:
      raise NotImplementedError()
    return super()._grad_param()

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the RV/RHK fit pipeline with command-line arguments."
    )

    # Data loading
    parser.add_argument("--path", default="Solar Data", help="Directory containing Analyse_summary.csv and Analyse_ccf.p")
    parser.add_argument("--star-name", default="Sun", help="Star name key used inside Analyse_ccf.p")
    parser.add_argument("--normalise", action=argparse.BooleanOptionalAction, default=False, help="Normalise RV and RHK")
    parser.add_argument("--inject-planet", action=argparse.BooleanOptionalAction, default=False, help="Inject a synthetic planet signal into RV")
    parser.add_argument("--planet-period", type=float, default=40.0, help="Injected planet period")
    parser.add_argument("--planet-A", type=float, default=0.001, help="Injected planet semi-amplitude")
    parser.add_argument("--planet-B", type=float, default=0.001, help="Injected planet phase")

    # Cycle fit (warm-start seeds for the joint cyc/nonstat optimisation)
    parser.add_argument("--cycle-b0", type=float, default=0.0, help="Initial shared linear coefficient")
    parser.add_argument("--cycle-P0", type=float, default=4000.0, help="Initial cycle period")
    parser.add_argument("--cycle-phi0", type=float, default=0.0, help="Initial cycle phase")
    parser.add_argument("--mu0", type=float, default=0.0, help="Initial covariance-modulation strength (nonstat only)")

    # Model / covariance settings
    parser.add_argument("--sig", type=float, default=1.0, help="MEPKernel sigma")
    parser.add_argument("--prot", type=float, default=27.0, help="Initial rotation period")
    parser.add_argument("--rho", type=float, default=20.0, help="Initial rho")
    parser.add_argument("--eta", type=float, default=0.25, help="Initial eta")
    parser.add_argument("--rvjit-frac", type=float, default=0.1, help="RV jitter as a fraction of the RV std")
    parser.add_argument("--rhkjit-frac", type=float, default=0.1, help="RHK jitter as a fraction of the RHK std")

    # Bounds
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
    parser.add_argument("--planet-A-min", type=float, default=0.000001)
    parser.add_argument("--planet-A-max", type=float, default=0.1)
    parser.add_argument("--planet-B-min", type=float, default=0.000001)
    parser.add_argument("--planet-B-max", type=float, default=0.1)
    parser.add_argument("--delta0-min", type=float, default=-0.5)
    parser.add_argument("--delta0-max", type=float, default=0.5)
    parser.add_argument("--delta1-min", type=float, default=-2.0)
    parser.add_argument("--delta1-max", type=float, default=2.0)

    # Bounds specific to the joint cyc / nonstat models (shared_core b, P, phi
    # and per-series cycle amplitude a0, a1 -- used by both branches)
    parser.add_argument("--mu-min", type=float, default=-5.0)
    parser.add_argument("--mu-max", type=float, default=5.0)
    parser.add_argument("--cycle-b-min", type=float, default=-0.5)
    parser.add_argument("--cycle-b-max", type=float, default=0.5)
    parser.add_argument("--Pcyc-min", type=float, default=0.5)
    parser.add_argument("--Pcyc-max", type=float, default=10000.0)
    parser.add_argument("--phi-min", type=float, default=-np.pi)
    parser.add_argument("--phi-max", type=float, default=np.pi)
    parser.add_argument("--a0-min", type=float, default=-0.1)
    parser.add_argument("--a0-max", type=float, default=0.1)
    parser.add_argument("--a1-min", type=float, default=-5)
    parser.add_argument("--a1-max", type=float, default=5)

    # Optimisation / MCMC
    parser.add_argument("--delta-0", type=float, default=-0.001, help="Initial delta_0")
    parser.add_argument("--delta-1", type=float, default=0.001, help="Initial delta_1")
    parser.add_argument("--planet-p", type=float, default=None, help="Initial planet period")
    parser.add_argument("--planet-A-fit", type=float, default=0.01, help="Initial planet K")
    parser.add_argument("--planet-B-fit", type=float, default=0.01, help="Initial planet phi")
    parser.add_argument("--fit-planet", action=argparse.BooleanOptionalAction, default=False, help="Include planet parameters in the optimisation")
    parser.add_argument("--change-C", action=argparse.BooleanOptionalAction, default=True, help="Write the optimised kernel parameters back into C")
    parser.add_argument("--output-csv", default="results/cv_nonstat_results.csv", help="Where to save per-fold CV results")

    return parser

def build_bounds_list(args, stds):
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
    ]

def build_bounds_list_cyc(args, stds):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]

    bounds_list = [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (args.prot_min, args.prot_max),
        (args.rho_min, args.rho_max),
        (args.eta_min, args.eta_max),
        (0.0, alpha_0_max),
        (-alpha_1_max, alpha_1_max),
        (-beta_0_max, beta_0_max),
        (args.cycle_b_min, args.cycle_b_max),
        (args.Pcyc_min, args.Pcyc_max),
        (args.phi_min, args.phi_max),
        (args.a0_min, args.a0_max),
        (args.a1_min, args.a1_max),
        (args.delta0_min, args.delta0_max),
        (args.delta1_min, args.delta1_max),
    ]

    if args.fit_planet:
        bounds_list += [
            (args.planet_p_min, args.planet_p_max),
            (args.planet_A_min, args.planet_A_max),
            (args.planet_B_min, args.planet_B_max),
        ]

    return bounds_list

def build_bounds_list_nonstat(args, stds):
    rvjit_max = args.rvjit_max_frac * stds[0]
    rhkjit_max = args.rhkjit_max_frac * stds[1]
    alpha_0_max = args.alpha0_max_frac * stds[0]
    alpha_1_max = args.alpha1_max_frac * stds[1]
    beta_0_max = args.beta0_max_frac * stds[0]

    bounds_list = [
        (0.0, rvjit_max),
        (0.0, rhkjit_max),
        (args.mu_min, args.mu_max),
        (args.cycle_b_min, args.cycle_b_max),
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
        (args.delta0_min, args.delta0_max),
        (args.delta1_min, args.delta1_max),
    ]

    if args.fit_planet:
        bounds_list += [
            (args.planet_p_min, args.planet_p_max),
            (args.planet_A_min, args.planet_A_max),
            (args.planet_B_min, args.planet_B_max),
        ]

    return bounds_list

def main():

    args = build_parser().parse_args()

    load_result = mf.load_and_norm_data(
                        args.path,
                        args.star_name,
                        normalise=args.normalise,
                        inject_planet=args.inject_planet,
                        planet_params=(args.planet_period, args.planet_A, args.planet_B),
                    )


    t_full, y_full, yerr_full, series_index = load_result
    rv_std = 1.0

    t = t_full[series_index[0]]
    y_rv = y_full[series_index[0]]
    yerr_rv = yerr_full[series_index[0]]
    y_rhk = y_full[series_index[1]]
    yerr_rhk = yerr_full[series_index[1]]

    N = len(t)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    fold_ids = np.empty(N, dtype=int)
    fold_ids[perm] = np.arange(N) % 12

    loglike_total_nocyc = 0
    loglike_total_cyc = 0
    loglike_total_nonstat = 0
    cv_rows = []


    for fold in range(12):

        test_mask = fold_ids == fold
        train_mask = ~test_mask

        test_t, test_rv, test_erv, test_rhk, test_erhk = t[test_mask], y_rv[test_mask], yerr_rv[test_mask], y_rhk[test_mask], yerr_rhk[test_mask]
        train_t, train_rv, train_erv, train_rhk, train_erhk = t[train_mask], y_rv[train_mask], yerr_rv[train_mask], y_rhk[train_mask], yerr_rhk[train_mask]

        t_test = [test_t,test_t]
        y_test = [test_rv, test_rhk]
        yerr_test = [test_erv, test_erhk]

        t_train = [train_t,train_t]
        y_train = [train_rv, train_rhk]
        yerr_train = [train_erv, train_erhk]

        t_full_train, y_full_train, yerr_full_train, series_index_train = cov.merge_series(t_train, y_train, yerr_train)
        t_full_test, y_full_test, yerr_full_test, series_index_test = cov.merge_series(t_test, y_test, yerr_test)

        stds = [np.std(y_full_train[series_index_train[0]]), np.std(y_full_train[series_index_train[1]])]
        bounds_list_nocyc = [(0.0, args.rvjit_max_frac * stds[0]),
        (0.0, args.rhkjit_max_frac * stds[1]),
        (args.prot_min, args.prot_max),
        (args.rho_min, 100.0),
        (args.eta_min, 3.0),
        (0.0, args.alpha0_max_frac * stds[0]),
        (-args.alpha1_max_frac * stds[1], args.alpha1_max_frac * stds[1]),
        (-args.beta0_max_frac * stds[0], args.beta0_max_frac * stds[0]),
        (args.delta0_min, args.delta0_max),
        (-5.0, 5.0),
        ]
        bounds_list_cyc = build_bounds_list_cyc(args, stds)
        bounds_list_nonstat = build_bounds_list_nonstat(args, stds)

        planet_p_fold = args.planet_p
        if args.fit_planet and planet_p_fold is None:
            periods, _ = mf.period_guess(t_full_train, y_full_train, yerr_full_train, series_index_train, PMIN=1.1, PMAX=310.0, MAX_FAP=1e-5, MAX_NPL=2, plot=False)
            planet_p_fold = float(periods[1]) if len(periods) >= 2 else float(periods[0])

        # NO CYCLE

        C = cov.Cov(
                    t_full_train,
                    err=term.Error(yerr_full_train),
                    rv_jit=term.InstrumentJitter(series_index_train[0], args.rvjit_frac * stds[0]),
                    rhk_jit=term.InstrumentJitter(series_index_train[1], args.rhkjit_frac * stds[1]),
                    rot = MultiSeriesKernel(term.MEPKernel(args.sig,args.prot,args.rho,args.eta), series_index_train,
                            np.array([stds[0], stds[1]]),
                            np.array([stds[0], 0.0])
                        ),
                    )

        xbest, C = mf.optimise_params(t_full_train, y_full_train, series_index_train, C, bounds_list_nocyc,
                                delta_0=args.delta_0,
                                delta_1=args.delta_1,
                                planet_p=planet_p_fold,
                                planet_A=args.planet_A_fit,
                                planet_B=args.planet_B_fit,
                                fit_planet=args.fit_planet,
                                change_C=args.change_C)

        C_test = cov.Cov(
                                t_full_test,
                                err=term.Error(yerr_full_test),
                                rv_jit=term.InstrumentJitter(series_index_test[0], xbest[0]),
                                rhk_jit=term.InstrumentJitter(series_index_test[1], xbest[1]),
                                rot = MultiSeriesKernel(term.MEPKernel(args.sig,xbest[2],xbest[3],xbest[4]), series_index_test,
                                        np.array([xbest[5], xbest[6]]),
                                        np.array([xbest[7], 0.0])
                                    ),
                                )

        test_loglike_nocyc = -1*mf.negloglike_nocyc(xbest, t_full_test, y_full_test, series_index_test, C_test, rv_std, inject_planet=args.fit_planet)[0]
        loglike_total_nocyc = loglike_total_nocyc + test_loglike_nocyc
        print(f"fold {fold:02d} [no_cycle]: test loglike={test_loglike_nocyc:.6f}")
        cv_rows.append({"fold": fold, "mode": "no_cycle", "test_loglike": test_loglike_nocyc, "planet_p": planet_p_fold})
        fit_plot_name_optim = f"results/optim_fit_plot_no_cycle_{fold}.png"
        mf.plot_fit(t_full_train, y_full_train, yerr_full_train, series_index_train, C, xbest, rv_std=rv_std,output_name=fit_plot_name_optim,inject_planet=args.fit_planet)

        # CYC (simultaneous GP + cycle mean-term optimisation)

        _, _, cycle_params = mf.fit_cycle(t_full_train, y_full_train, series_index_train, b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0, plot=False, print_results=False, return_fit=True)
        rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_params

        C = cov.Cov(
                    t_full_train,
                    err=term.Error(yerr_full_train),
                    rv_jit=term.InstrumentJitter(series_index_train[0], args.rvjit_frac * stds[0]),
                    rhk_jit=term.InstrumentJitter(series_index_train[1], args.rhkjit_frac * stds[1]),
                    rot = MultiSeriesKernel(term.MEPKernel(args.sig,args.prot,args.rho,args.eta), series_index_train,
                            np.array([stds[0], stds[1]]),
                            np.array([stds[0], 0.0])
                        ),
                    )

        xbest, C = mf.optimise_params_cyc(t_full_train, y_full_train, series_index_train, C, bounds_list_cyc,
                                b=b0,
                                P=P0,
                                phi=phi0,
                                a0=rv_amp0,
                                a1=rhk_amp0,
                                d0=rv_offset0,
                                d1=rhk_offset0,
                                planet_p=planet_p_fold,
                                planet_A=args.planet_A_fit,
                                planet_B=args.planet_B_fit,
                                fit_planet=args.fit_planet,
                                change_C=args.change_C)

        C_test = cov.Cov(
                                t_full_test,
                                err=term.Error(yerr_full_test),
                                rv_jit=term.InstrumentJitter(series_index_test[0], xbest[0]),
                                rhk_jit=term.InstrumentJitter(series_index_test[1], xbest[1]),
                                rot = MultiSeriesKernel(term.MEPKernel(args.sig,xbest[2],xbest[3],xbest[4]), series_index_test,
                                        np.array([xbest[5], xbest[6]]),
                                        np.array([xbest[7], 0.0])
                                    ),
                                )

        test_loglike_cyc = -1*mf.negloglike_cyc(xbest, t_full_test, y_full_test, series_index_test, C_test, rv_std, inject_planet=args.fit_planet)[0]
        loglike_total_cyc = loglike_total_cyc + test_loglike_cyc
        print(f"fold {fold:02d} [cyc]: test loglike={test_loglike_cyc:.6f}")
        cv_rows.append({"fold": fold, "mode": "cyc", "test_loglike": test_loglike_cyc, "planet_p": planet_p_fold})

        fit_plot_name_optim = f"results/optim_fit_plot_cyc_{fold}.png"
        mf.plot_fit_cyc(t_full_train, y_full_train, yerr_full_train, series_index_train, C, xbest, rv_std=rv_std,output_name=fit_plot_name_optim,inject_planet=args.fit_planet)

        # NONSTAT (cycle modulates the covariance envelope as well as the mean)

        C = cov.Cov(
                    t_full_train,
                    err=term.Error(yerr_full_train),
                    rv_jit=term.InstrumentJitter(series_index_train[0], args.rvjit_frac * stds[0]),
                    rhk_jit=term.InstrumentJitter(series_index_train[1], args.rhkjit_frac * stds[1]),
                    rot = term.SimpleProductKernel(
                            nonstat=mfn.NonStationaryKernel(mfn.alpha_cyc, mfn.alpha_cyc_grad, mu=args.mu0, b=b0, P=P0, phi=phi0),
                            qp=MultiSeriesKernel(term.MEPKernel(args.sig,args.prot,args.rho,args.eta), series_index_train,
                                    np.array([stds[0], stds[1]]),
                                    np.array([stds[0], 0.0])
                                ),
                        ),
                    )

        xbest, C = mfn.optimise_params_nonstat(t_full_train, y_full_train, series_index_train, C, bounds_list_nonstat,
                                a0=rv_amp0,
                                a1=rhk_amp0,
                                d0=rv_offset0,
                                d1=rhk_offset0,
                                planet_p=planet_p_fold,
                                planet_A=args.planet_A_fit,
                                planet_B=args.planet_B_fit,
                                fit_planet=args.fit_planet,
                                change_C=args.change_C)

        C_test = cov.Cov(
                                t_full_test,
                                err=term.Error(yerr_full_test),
                                rv_jit=term.InstrumentJitter(series_index_test[0], xbest[0]),
                                rhk_jit=term.InstrumentJitter(series_index_test[1], xbest[1]),
                                rot = term.SimpleProductKernel(
                                        nonstat=mfn.NonStationaryKernel(mfn.alpha_cyc, mfn.alpha_cyc_grad, mu=xbest[2], b=xbest[3], P=xbest[4], phi=xbest[5]),
                                        qp=MultiSeriesKernel(term.MEPKernel(args.sig,xbest[6],xbest[7],xbest[8]), series_index_test,
                                                np.array([xbest[9], xbest[10]]),
                                                np.array([xbest[11], 0.0])
                                            ),
                                    ),
                                )

        test_loglike_nonstat = -1*mfn.negloglike_nonstat(xbest, t_full_test, y_full_test, series_index_test, C_test, rv_std, inject_planet=args.fit_planet)[0]
        loglike_total_nonstat = loglike_total_nonstat + test_loglike_nonstat
        print(f"fold {fold:02d} [nonstat]: test loglike={test_loglike_nonstat:.6f}")
        cv_rows.append({"fold": fold, "mode": "nonstat", "test_loglike": test_loglike_nonstat, "planet_p": planet_p_fold})

        fit_plot_name_optim = f"results/optim_fit_plot_nonstat_{fold}.png"
        mfn.plot_fit_nonstat(t_full_train, y_full_train, yerr_full_train, series_index_train, C, xbest, rv_std=rv_std,output_name=fit_plot_name_optim,inject_planet=args.fit_planet)

    print(f"Total log-likelihood [no_cycle]: {loglike_total_nocyc}")
    print(f"Total log-likelihood [cyc]: {loglike_total_cyc}")
    print(f"Total log-likelihood [nonstat]: {loglike_total_nonstat}")

    try:
        import pandas as pd
        df = pd.DataFrame(cv_rows)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved per-fold CV results to {args.output_csv}")
    except Exception as exc:
        print(f"\nCould not save CV results CSV: {exc}")
        print(cv_rows)

    # Whole-dataset refit + AIC/BIC for all three models

    T = [t,t]
    Y = [y_rv, y_rhk]
    Yerr = [yerr_rv, yerr_rhk]

    t_full, y_full, yerr_full, series_index = cov.merge_series(T, Y, Yerr)

    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]

    n_train = len(y_full)

    # NO CYCLE

    bounds_list_nocyc = build_bounds_list(args, stds)
    C = cov.Cov(
                t_full,
                err=term.Error(yerr_full),
                rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
                rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
                rot = MultiSeriesKernel(term.MEPKernel(args.sig,args.prot,args.rho,args.eta), series_index,
                        np.array([stds[0], stds[1]]),
                        np.array([stds[0], 0.0])
                    ),
                )

    xbest, C = mf.optimise_params(t_full, y_full, series_index, C, bounds_list_nocyc,
                            delta_0=args.delta_0,
                            delta_1=args.delta_1,
                            planet_p=args.planet_p,
                            planet_A=args.planet_A_fit,
                            planet_B=args.planet_B_fit,
                            fit_planet=args.fit_planet,
                            change_C=args.change_C)

    print("xbest no_cycle: ", xbest)

    loglike_nocyc = -mf.negloglike_nocyc(
        xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]

    k_nocyc = len(xbest)
    aic_nocyc = 2 * k_nocyc - 2 * loglike_nocyc
    bic_nocyc = k_nocyc * np.log(n_train) - 2 * loglike_nocyc

    # CYC

    _, _, cycle_params = mf.fit_cycle(t_full, y_full, series_index, b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0, plot=False, print_results=False, return_fit=True)
    rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_params

    bounds_list_cyc = build_bounds_list_cyc(args, stds)
    C = cov.Cov(
                t_full,
                err=term.Error(yerr_full),
                rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
                rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
                rot = MultiSeriesKernel(term.MEPKernel(args.sig,args.prot,args.rho,args.eta), series_index,
                        np.array([stds[0], stds[1]]),
                        np.array([stds[0], 0.0])
                    ),
                )

    xbest, C = mf.optimise_params_cyc(t_full, y_full, series_index, C, bounds_list_cyc,
                            b=b0,
                            P=P0,
                            phi=phi0,
                            a0=rv_amp0,
                            a1=rhk_amp0,
                            d0=rv_offset0,
                            d1=rhk_offset0,
                            planet_p=args.planet_p,
                            planet_A=args.planet_A_fit,
                            planet_B=args.planet_B_fit,
                            fit_planet=args.fit_planet,
                            change_C=args.change_C)

    print("xbest cyc: ", xbest)
    loglike_cyc = -mf.negloglike_cyc(
        xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]

    k_cyc = len(xbest)
    aic_cyc = 2 * k_cyc - 2 * loglike_cyc
    bic_cyc = k_cyc * np.log(n_train) - 2 * loglike_cyc

    # NONSTAT

    bounds_list_nonstat = build_bounds_list_nonstat(args, stds)
    C = cov.Cov(
                t_full,
                err=term.Error(yerr_full),
                rv_jit=term.InstrumentJitter(series_index[0], args.rvjit_frac * stds[0]),
                rhk_jit=term.InstrumentJitter(series_index[1], args.rhkjit_frac * stds[1]),
                rot = term.SimpleProductKernel(
                        nonstat=mfn.NonStationaryKernel(mfn.alpha_cyc, mfn.alpha_cyc_grad, mu=args.mu0, b=b0, P=P0, phi=phi0),
                        qp=MultiSeriesKernel(term.MEPKernel(args.sig,args.prot,args.rho,args.eta), series_index,
                                np.array([stds[0], stds[1]]),
                                np.array([stds[0], 0.0])
                            ),
                    ),
                )

    xbest, C = mfn.optimise_params_nonstat(t_full, y_full, series_index, C, bounds_list_nonstat,
                            a0=rv_amp0,
                            a1=rhk_amp0,
                            d0=rv_offset0,
                            d1=rhk_offset0,
                            planet_p=args.planet_p,
                            planet_A=args.planet_A_fit,
                            planet_B=args.planet_B_fit,
                            fit_planet=args.fit_planet,
                            change_C=args.change_C)

    print("xbest nonstat: ", xbest)
    loglike_nonstat = -mfn.negloglike_nonstat(
        xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]

    k_nonstat = len(xbest)
    aic_nonstat = 2 * k_nonstat - 2 * loglike_nonstat
    bic_nonstat = k_nonstat * np.log(n_train) - 2 * loglike_nonstat

    print("BIC [no_cycle]: ", bic_nocyc)
    print("BIC [cyc]: ", bic_cyc)
    print("BIC [nonstat]: ", bic_nonstat)
    print("AIC [no_cycle]: ", aic_nocyc)
    print("AIC [cyc]: ", aic_cyc)
    print("AIC [nonstat]: ", aic_nonstat)

if __name__ == "__main__":
    main()
