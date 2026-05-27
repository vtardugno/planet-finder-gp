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
    parser.add_argument("--planet-a", type=float, default=0.001, help="Injected planet sine amplitude")
    parser.add_argument("--planet-b", type=float, default=0.001, help="Injected planet cosine amplitude")

    # Cycle fit
    parser.add_argument("--cycle-b0", type=float, default=0.0, help="Initial shared linear coefficient")
    parser.add_argument("--cycle-P0", type=float, default=4000.0, help="Initial cycle period")
    parser.add_argument("--cycle-phi0", type=float, default=0.0, help="Initial cycle phase")
    parser.add_argument("--cycle-print-results", action=argparse.BooleanOptionalAction, default=True, help="Print fitted shared cycle parameters")
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
    parser.add_argument("--planet-p-min", type=float, default=30.0)
    parser.add_argument("--planet-p-max", type=float, default=50.0)
    parser.add_argument("--planet-a-min", type=float, default=0.0001)
    parser.add_argument("--planet-a-max", type=float, default=0.01)
    parser.add_argument("--planet-b-min", type=float, default=0.0001)
    parser.add_argument("--planet-b-max", type=float, default=0.01)
    parser.add_argument("--delta0-min", type=float, default=-0.5)
    parser.add_argument("--delta0-max", type=float, default=0.5)
    parser.add_argument("--delta1-min", type=float, default=-2.0)
    parser.add_argument("--delta1-max", type=float, default=2.0)

    # Optimisation / MCMC
    parser.add_argument("--delta-0", type=float, default=-0.001, help="Initial delta_0")
    parser.add_argument("--delta-1", type=float, default=0.001, help="Initial delta_1")
    parser.add_argument("--planet-p", type=float, default=40.05, help="Initial planet period")
    parser.add_argument("--planet-a-fit", type=float, default=0.0005, help="Initial planet A")
    parser.add_argument("--planet-b-fit", type=float, default=0.0005, help="Initial planet B")
    parser.add_argument("--fit-planet", action=argparse.BooleanOptionalAction, default=True, help="Include planet parameters in the optimisation")
    parser.add_argument("--change-C", action=argparse.BooleanOptionalAction, default=True, help="Write the optimised kernel parameters back into C")
    parser.add_argument("--run-length", type=int, default=2000, help="Number of MCMC steps")
    # Outputs
    # parser.add_argument("--fit-output", default="fit_plot.png", help="Output name for the fit plot")
    # parser.add_argument("--chains-output", default="chains_plot.png", help="Output name for the chains plot")
    # parser.add_argument("--corner-output", default="corner_plot.png", help="Output name for the corner plot")
    parser.add_argument("--output-name", default="0", help="Base output name for plots")
    parser.add_argument("--corner-discard", type=int, default=1000, help="Discard this many samples before corner plotting")

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
        (args.planet_p_min, args.planet_p_max),
        (args.planet_a_min, args.planet_a_max),
        (args.planet_b_min, args.planet_b_max),
    ]


def main():
   
    args = build_parser().parse_args()

    load_result = mf.load_and_norm_data(
                        args.path,
                        args.star_name,
                        normalise=args.normalise,
                        inject_planet=args.inject_planet,
                        planet_params=(args.planet_period, args.planet_a, args.planet_b),
                    )

    if args.inject_planet and args.normalise:
        t_full, y_full, yerr_full, series_index, rv_std = load_result
    else:
        t_full, y_full, yerr_full, series_index = load_result
        rv_std = 1.0


    cycle_out_name = "cycle_fit_" + args.output_name + ".png"
    rv_fit, rhk_fit = mf.fit_cycle(t_full, y_full, series_index, b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0, plot=args.cycle_plot, print_results=args.cycle_print_results, output_name=cycle_out_name)

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

    bounds_list = build_bounds_list(args, stds)
    
    xbest = mf.optimise_params(t_full, y_full, series_index, C, bounds_list, 
                                delta_0=args.delta_0,
                                delta_1=args.delta_1,
                                planet_p=args.planet_p,
                                planet_a=args.planet_a_fit,
                                planet_b=args.planet_b_fit,
                                fit_planet=args.fit_planet,
                                change_C=args.change_C)

    fit_plot_name = "fit_plot_" + args.output_name + ".png"
    mf.plot_fit(t_full, y_full, yerr_full, series_index, C, xbest, rv_std=rv_std,output_name=fit_plot_name,inject_planet=args.fit_planet)
    sampler = mf.run_emcee(t_full, y_full, series_index, C, xbest, bounds_list, run_length=args.run_length, planet=args.fit_planet)
    chains_plot_name = "chains_plot_" + args.output_name + ".png"
    mf.plot_chains(sampler, C, planet=args.fit_planet, output_name=chains_plot_name)
    corner_plot_name = "corner_plot_" + args.output_name + ".png"
    mf.plot_corner(sampler, C, discard=args.corner_discard,planet=args.fit_planet,output_name=corner_plot_name)


if __name__ == "__main__":
    main()