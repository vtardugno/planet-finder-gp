import functions as mf
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from spleaf import cov, term
import emcee
import argparse

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

    # Cycle fit
    parser.add_argument("--cycle-fit", action=argparse.BooleanOptionalAction, default=True, help="Fit the long-term cycle and remove it before fitting the GP")
    parser.add_argument("--cycle-b0", type=float, default=0.0, help="Initial shared linear coefficient")
    parser.add_argument("--cycle-P0", type=float, default=4000.0, help="Initial cycle period")
    parser.add_argument("--cycle-phi0", type=float, default=0.0, help="Initial cycle phase")
    parser.add_argument("--cycle-print-results", action=argparse.BooleanOptionalAction, default=False, help="Print fitted shared cycle parameters")
    parser.add_argument("--cycle-plot", action=argparse.BooleanOptionalAction, default=True, help="Save the cycle fit plots")
    # parser.add_argument("--cycle-output-name", default="cycle_fit.png", help="Base filename for cycle fit plots")

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

    # Optimisation / MCMC
    parser.add_argument("--delta-0", type=float, default=-0.001, help="Initial delta_0")
    parser.add_argument("--delta-1", type=float, default=0.001, help="Initial delta_1")
    parser.add_argument("--planet-p", type=float, default=None, help="Initial planet period")
    parser.add_argument("--planet-A-fit", type=float, default=0.01, help="Initial planet K")
    parser.add_argument("--planet-B-fit", type=float, default=0.01, help="Initial planet phi")
    parser.add_argument("--fit-planet", action=argparse.BooleanOptionalAction, default=False, help="Include planet parameters in the optimisation")
    parser.add_argument("--change-C", action=argparse.BooleanOptionalAction, default=True, help="Write the optimised kernel parameters back into C")
    parser.add_argument("--output-csv", default="results/cv_results.csv", help="Where to save per-fold CV results")
    # Outputs
    # parser.add_argument("--fit-output", default="fit_plot.png", help="Output name for the fit plot")
    # parser.add_argument("--chains-output", default="chains_plot.png", help="Output name for the chains plot")
    # parser.add_argument("--corner-output", default="corner_plot.png", help="Output name for the corner plot")

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
    
    loglike_total_cyc = 0
    loglike_total_nocyc = 0
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
        bounds_list_cyc = build_bounds_list(args, stds)
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

        planet_p_fold = args.planet_p
        if args.fit_planet and planet_p_fold is None:
            periods, _ = mf.period_guess(t_full_train, y_full_train, yerr_full_train, series_index_train, PMIN=1.1, PMAX=310.0, MAX_FAP=1e-5, MAX_NPL=2, plot=False)
            planet_p_fold = float(periods[1]) if len(periods) >= 2 else float(periods[0])

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
        fit_plot_name_optim = f"results/optim_fit_plot_False_{fold}.png"
        mf.plot_fit(t_full_train, y_full_train, yerr_full_train, series_index_train, C, xbest, rv_std=rv_std,output_name=fit_plot_name_optim,inject_planet=args.fit_planet)

        # CYCLE PART

        cycle_out_name = f"results/cycle_fit_{fold}.png"
        rv_fit, rhk_fit, params = mf.fit_cycle(t_full_train, y_full_train, series_index_train, b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0, plot=args.cycle_plot, print_results=args.cycle_print_results, output_name=cycle_out_name, return_fit=True)
        y_full_train[series_index_train[0]] = y_full_train[series_index_train[0]] - rv_fit
        y_full_train[series_index_train[1]] = y_full_train[series_index_train[1]] - rhk_fit

        stds = [np.std(y_full_train[series_index_train[0]]), np.std(y_full_train[series_index_train[1]])]

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
        
        xbest, C = mf.optimise_params(t_full_train, y_full_train, series_index_train, C, bounds_list_cyc,
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


        a1, c1, a2, c2, b, P, phi = params
        y_full_test[series_index_test[0]] = y_full_test[series_index_test[0]] - mf.model_rv(t_full_test[series_index_test[0]], a1, c1, b, P, phi)
        y_full_test[series_index_test[1]] = y_full_test[series_index_test[1]] - mf.model_rhk(t_full_test[series_index_test[1]], a2, c2, b, P, phi)

        test_loglike_cyc = -1*mf.negloglike_nocyc(xbest, t_full_test, y_full_test, series_index_test, C_test, rv_std, inject_planet=args.fit_planet)[0]
        loglike_total_cyc = loglike_total_cyc + test_loglike_cyc
        print(f"fold {fold:02d} [cycle]: test loglike={test_loglike_cyc:.6f}")
        cv_rows.append({"fold": fold, "mode": "cycle", "test_loglike": test_loglike_cyc, "planet_p": planet_p_fold})

        fit_plot_name_optim = f"results/optim_fit_plot_True_{fold}.png"
        mf.plot_fit(t_full_train, y_full_train, yerr_full_train, series_index_train, C, xbest, rv_std=rv_std,output_name=fit_plot_name_optim,inject_planet=args.fit_planet)

        # np.save("results/xbest_" + args.output_name + ".npy", xbest_all)

    print(f"Total log-likelihood with cycle fit: {loglike_total_cyc}")
    print(f"Total log-likelihood without cycle fit: {loglike_total_nocyc}")

    try:
        import pandas as pd
        df = pd.DataFrame(cv_rows)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved per-fold CV results to {args.output_csv}")
    except Exception as exc:
        print(f"\nCould not save CV results CSV: {exc}")
        print(cv_rows)

    T = [t,t]
    Y = [y_rv, y_rhk]
    Yerr = [yerr_rv, yerr_rhk]

    t_full, y_full, yerr_full, series_index = cov.merge_series(T, Y, Yerr)

    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]

    bounds_list = build_bounds_list(args, stds)
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

    xbest, C = mf.optimise_params(t_full, y_full, series_index, C, bounds_list_cyc, 
                            delta_0=args.delta_0,
                            delta_1=args.delta_1,
                            planet_p=args.planet_p,
                            planet_A=args.planet_A_fit,
                            planet_B=args.planet_B_fit,
                            fit_planet=args.fit_planet,
                            change_C=args.change_C)

    print("xbest no cycle: ", xbest)

    loglike_nocyc = -mf.negloglike_nocyc(
    xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]

    n_train = len(y_full)  
    k_gp = len(xbest)

    aic_nocyc = 2 * k_gp - 2 * loglike_nocyc
    bic_nocyc = k_gp * np.log(n_train) - 2 * loglike_nocyc
    
    # CYCLE PART

    rv_fit, rhk_fit, params = mf.fit_cycle(t_full, y_full, series_index, b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0, plot=False, print_results=args.cycle_print_results, output_name=cycle_out_name, return_fit=True)
    y_full[series_index[0]] = y_full[series_index[0]] - rv_fit
    y_full[series_index[1]] = y_full[series_index[1]] - rhk_fit

    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]
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

    xbest, C = mf.optimise_params(t_full, y_full, series_index, C, bounds_list_cyc, 
                            delta_0=args.delta_0,
                            delta_1=args.delta_1,
                            planet_p=args.planet_p,
                            planet_A=args.planet_A_fit,
                            planet_B=args.planet_B_fit,
                            fit_planet=args.fit_planet,
                            change_C=args.change_C)


    print("xbest with cycle: ", xbest)
    loglike_cyc = -mf.negloglike_nocyc(
        xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]

    k_cycle = k_gp + 7   # a1, c1, a2, c2, b, P, phi
    aic_cyc = 2 * k_cycle - 2 * loglike_cyc
    bic_cyc = k_cycle * np.log(n_train) - 2 * loglike_cyc
        
    print("BIC with cycle fit: ", bic_cyc)
    print("BIC without cycle fit: ", bic_nocyc)
    print("AIC with cycle fit: ", aic_cyc)
    print("AIC without cycle fit: ", aic_nocyc)

if __name__ == "__main__":
    main()