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
    parser.add_argument("--inject-planet", action=argparse.BooleanOptionalAction, default=True, help="Inject a synthetic planet signal into RV")
    parser.add_argument("--planet-period", type=float, default=40.0, help="Injected planet period")
    parser.add_argument("--planet-A", type=float, default=0.001, help="Injected planet semi-amplitude")
    parser.add_argument("--planet-B", type=float, default=0.001, help="Injected planet phase")

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

    parser.add_argument("--planet-p-min", type=float, default=None)
    parser.add_argument("--planet-p-max", type=float, default=None)
    parser.add_argument("--planet-A-min", type=float, default=0.000001)
    parser.add_argument("--planet-A-max", type=float, default=0.1)
    parser.add_argument("--planet-B-min", type=float, default=0.000001)
    parser.add_argument("--planet-B-max", type=float, default=0.1)
   

    # Optimisation / MCMC
    parser.add_argument("--b", type=float, default=0.0, help="Initial shared linear coefficient")
    parser.add_argument("--Pcyc", type=float, default=4000.0, help="Initial cycle period")
    parser.add_argument("--phi", type=float, default=0.0, help="Initial cycle phase")
    parser.add_argument("--planet-p", type=float, default=None, help="Initial planet period")
    parser.add_argument("--planet-A-fit", type=float, default=0.01, help="Initial planet A")
    parser.add_argument("--planet-B-fit", type=float, default=0.01, help="Initial planet B")
    parser.add_argument("--fit-planet", action=argparse.BooleanOptionalAction, default=True, help="Include planet parameters in the optimisation")
    parser.add_argument("--change-C", action=argparse.BooleanOptionalAction, default=True, help="Write the optimised kernel parameters back into C")
    parser.add_argument("--run-mcmc", action=argparse.BooleanOptionalAction, default=False, help="Run MCMC after optimisation")
    parser.add_argument("--run-length", type=int, default=2000, help="Number of MCMC steps")
    # Outputs
    # parser.add_argument("--fit-output", default="fit_plot.png", help="Output name for the fit plot")
    # parser.add_argument("--chains-output", default="chains_plot.png", help="Output name for the chains plot")
    # parser.add_argument("--corner-output", default="corner_plot.png", help="Output name for the corner plot")
    parser.add_argument("--output-name", default=None, help="Base output name for plots")
    parser.add_argument("--corner-discard", type=int, default=1000, help="Discard this many samples before corner plotting")
    parser.add_argument("--map_thin", type=int, default=1, help="Thin the MCMC chain by this factor when calculating MAP parameters")
    parser.add_argument("--map_burn", type=int, default=1000, help="Discard this many samples before calculating MAP parameters")

    return parser

def build_bounds_list(args, stds):
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
        (args.b_min, args.b_max),
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


def main():
   
    args = build_parser().parse_args()

    if args.output_name is None:
        args.output_name = f"p{args.planet_period}_A{args.planet_A}_B{args.planet_B}"

    # if args.planet_K_fit is None:
    #     args.planet_K_fit = args.planet_K

    # if args.planet_phi_fit is None:
    #     args.planet_phi_fit = args.planet_phi

    # if args.planet_b_fit is None:
    #     args.planet_b_fit = args.planet_b * 5

    load_result = mf.load_and_norm_data(
                        args.path,
                        args.star_name,
                        normalise=args.normalise,
                        inject_planet=args.inject_planet,
                        planet_params=(args.planet_period, args.planet_A, args.planet_B),
                    )

    if args.inject_planet and args.normalise:
        t_full, y_full, yerr_full, series_index, rv_std = load_result
    else:
        t_full, y_full, yerr_full, series_index = load_result
        rv_std = 1.0

    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]

    bounds_list = build_bounds_list(args, stds)

    # Warm-start the cycle parameters from an unconstrained curve_fit so the joint
    # L-BFGS-B optimizer (which is prone to getting stuck for the highly non-convex
    # cycle period P) starts close to a good region. Everything is still refined
    # jointly with the GP/planet parameters below - nothing is frozen.
    _, _, cycle_params = mf.fit_cycle(t_full, y_full, series_index, b0=args.b, P0=args.Pcyc, phi0=args.phi, plot=False, print_results=False, return_fit=True)
    rv_offset0, rv_amp0, rhk_offset0, rhk_amp0, b0, P0, phi0 = cycle_params

    if args.fit_planet:
        A_inits = [args.planet_A_fit, args.planet_A_fit/10, args.planet_A_fit/100, args.planet_A_fit/1000]
        B_inits = [args.planet_B_fit, args.planet_B_fit/10, args.planet_B_fit/100, args.planet_B_fit/1000]
        best_loglike = -np.inf
        xbest_all = []
        best_p_init = 0

        if args.planet_p is None:
            periodogram_out_name = "results_cyc/periodogram_" + args.output_name + ".png"
            planet_guess, _ = mf.period_guess(t_full, y_full, yerr_full, series_index, PMIN = 1.1, PMAX = 400.0, MAX_FAP = 1e-5, MAX_NPL = 2, plot = True, output_name = periodogram_out_name)
        else:
            planet_guess = [args.planet_p]

        for p in planet_guess:

            if args.planet_p_min is None:
                args.planet_p_min = np.max([p - 10, 1.1])
            if args.planet_p_max is None:
                args.planet_p_max = np.min([p + 10, 400.0])

            bounds_list[-3] = (np.max([p - 10, 1.1]), np.min([p + 10, 400.0]))

            for A in A_inits:
                for B in B_inits:

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

                    xbest, C = mf.optimise_params_cyc(t_full, y_full, series_index, C, bounds_list,
                                                b=b0,
                                                P=P0,
                                                phi=phi0,
                                                a0=rv_amp0,
                                                a1=rhk_amp0,
                                                d0=rv_offset0,
                                                d1=rhk_offset0,
                                                planet_p=p,
                                                planet_A=A,
                                                planet_B=B,
                                                fit_planet=args.fit_planet,
                                                change_C=args.change_C)

                    loglike = -1*mf.negloglike_cyc(xbest, t_full, y_full, series_index,C, rv_std, inject_planet=args.fit_planet)[0]

                    if loglike > best_loglike:
                        best_loglike = loglike
                        xbest_all = xbest
                        best_p_init = p

        params, _ = mf.get_opt_params(C)
        C.set_param(xbest_all[:len(params)], params)

        fit_plot_name_optim = "results_cyc/optim_fit_plot_" + args.output_name + ".png"
        mf.plot_fit_cyc(t_full, y_full, yerr_full, series_index, C, xbest_all, rv_std=rv_std,output_name=fit_plot_name_optim,inject_planet=args.fit_planet)

        np.save("results_cyc/xbest_" + args.output_name + ".npy", xbest_all)
    else:
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

        xbest_all, C = mf.optimise_params_cyc(t_full, y_full, series_index, C, bounds_list,
                                                b=b0,
                                                P=P0,
                                                phi=phi0,
                                                a0=rv_amp0,
                                                a1=rhk_amp0,
                                                d0=rv_offset0,
                                                d1=rhk_offset0,
                                                fit_planet=args.fit_planet,
                                                change_C=args.change_C)

        loglike = -1*mf.negloglike_cyc(xbest_all, t_full, y_full, series_index,C, rv_std, inject_planet=args.fit_planet)[0]
        params, _ = mf.get_opt_params(C)
        C.set_param(xbest_all[:len(params)], params)

        fit_plot_name_optim = "results_cyc/optim_fit_plot_" + args.output_name + ".png"
        mf.plot_fit_cyc(t_full, y_full, yerr_full, series_index, C, xbest_all, rv_std=rv_std,output_name=fit_plot_name_optim,inject_planet=args.fit_planet)

        np.save("results_cyc/xbest_" + args.output_name + ".npy", xbest_all)

    if args.run_mcmc:

        if args.fit_planet:
            bounds_list[-3] = (np.max([best_p_init - 10, 1.1]), np.min([best_p_init + 10, 400.0]))

        sampler = mf.run_emcee_cyc(t_full, y_full, series_index, C, xbest_all, bounds_list, run_length=args.run_length, planet=args.fit_planet)

        corner_plot_name = "results_cyc/corner_plot_" + args.output_name + ".png"
        mf.plot_corner(sampler, C, discard=args.corner_discard,planet=args.fit_planet,output_name=corner_plot_name)

        map_params, _ = mf.get_MAP_params(sampler, args.map_burn, args.map_thin)
        np.save("results_cyc/map_params_" + args.output_name + ".npy", map_params)

if __name__ == "__main__":
    main()