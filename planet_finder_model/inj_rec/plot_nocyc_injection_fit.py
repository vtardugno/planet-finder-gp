"""Diagnostic for a single no_cycle injection in the high-K, short-period
regime where injection_recovery.py's sweep tends to miss.

Fits the no_cycle model (GP+planet and GP-only, same bounds/start as
injection_recovery.py's fit_nocyc) to one injected planet and plots the
GP+planet fit, the planet contribution on its own, and the GP-only fit for
comparison -- to see whether the GP is tracking (absorbing) the injected
oscillation rather than leaving it for the explicit planet term.

Defaults to a failing case pulled from inj_rec/injection_recovery_newtest.csv:
P=8.521443d, K=0.176979, phase=2.745362 (recovered P=7.98d, K=0.0106 there).
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spleaf import cov, term

import injection_recovery as ir
mf = ir.mf


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=float, default=8.521443)
    parser.add_argument("--k", type=float, default=0.176979)
    parser.add_argument("--phase", type=float, default=2.745362)
    parser.add_argument("--path", default="data/Solar_Data")
    parser.add_argument("--star-name", default="Sun")
    parser.add_argument("--output", default="inj_rec/nocyc_diagnostic_fit.png")
    parser.add_argument("--zoom-days", type=float, default=120.0,
                         help="Width of the zoomed-in time window used for the detail plot.")
    return parser


def build_C(t_full, yerr_full, series_index, stds, ir_args):
    return cov.Cov(
        t_full,
        err=term.Error(yerr_full),
        rv_jit=term.InstrumentJitter(series_index[0], ir_args.rvjit_frac * stds[0]),
        rhk_jit=term.InstrumentJitter(series_index[1], ir_args.rhkjit_frac * stds[1]),
        rot=ir.MultiSeriesKernel(term.MEPKernel(ir_args.sig, ir_args.prot, ir_args.rho, ir_args.eta), series_index,
                                  np.array([stds[0], stds[1]]), np.array([stds[0], 0.0])),
    )


def main():
    args = build_parser().parse_args()
    ir_args = ir.build_parser().parse_args([])

    A_inj = args.k * np.cos(args.phase)
    B_inj = args.k * np.sin(args.phase)

    t_full, y_full, yerr_full, series_index = mf.load_and_norm_data(
        args.path, args.star_name, normalise=False,
        inject_planet=True, planet_params=(args.period, A_inj, B_inj),
    )
    rv_std = 1.0

    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]
    amp_bound = ir_args.planet_amp_max if ir_args.planet_amp_max is not None else 1.5 * ir_args.k_max

    rv_offset0 = float(np.mean(y_full[series_index[0]]))
    rhk_offset0 = float(np.mean(y_full[series_index[1]]))

    planet_guess, _ = mf.period_guess(
        t_full, y_full, yerr_full, series_index,
        PMIN=1.1, PMAX=ir_args.period_max, MAX_FAP=1e-5, MAX_NPL=2, plot=False,
    )

    fit_args = (t_full, y_full, yerr_full, series_index, stds, ir_args,
                planet_guess, rv_offset0, rhk_offset0, amp_bound, rv_std)

    xbest1, loglike1, diag1 = ir.fit_nocyc(*fit_args, fit_planet=True)
    xbest0, loglike0, diag0 = ir.fit_nocyc(*fit_args, fit_planet=False)

    P_rec, A_rec, B_rec = xbest1[-3], xbest1[-2], xbest1[-1]
    K_rec = float(np.sqrt(A_rec ** 2 + B_rec ** 2))

    labels = ["rv_jit", "rhk_jit", "prot", "rho", "eta", "alpha0", "alpha1", "beta0", "delta0", "delta1"]
    print(f"injected:   P={args.period:.4f}  K={args.k:.5f}  phase={args.phase:.4f}")
    print(f"recovered:  P={P_rec:.4f}  K={K_rec:.5f}  A={A_rec:.5f}  B={B_rec:.5f}")
    print("GP+planet params:", {l: round(float(v), 4) for l, v in zip(labels, xbest1)})
    print(f"GP+planet:  loglike={loglike1:.3f}  at_bounds={[labels[i] if i < len(labels) else i for i in diag1['at_bounds']]}")
    print("GP-only params:   ", {l: round(float(v), 4) for l, v in zip(labels, xbest0)})
    print(f"GP-only:    loglike={loglike0:.3f}  at_bounds={[labels[i] if i < len(labels) else i for i in diag0['at_bounds']]}")

    C1 = build_C(t_full, yerr_full, series_index, stds, ir_args)
    params, _ = mf.get_opt_params(C1)
    C1.set_param(xbest1[:len(params)], params)

    C0 = build_C(t_full, yerr_full, series_index, stds, ir_args)
    C0.set_param(xbest0[:len(params)], params)

    tsmooth = np.linspace(np.min(t_full), np.max(t_full), 2000)

    C1.kernel['rot'].set_conditional_coef(series_id=0)
    y_model1 = y_full.copy()
    y_model1[series_index[0]] -= mf.planet_injection(t_full[series_index[0]], P_rec, A_rec, B_rec)
    y_model1[series_index[0]] -= xbest1[8]
    y_model1[series_index[1]] -= xbest1[9]
    mu1, _ = C1.conditional(y_model1, tsmooth, calc_cov='diag')

    C0.kernel['rot'].set_conditional_coef(series_id=0)
    y_model0 = y_full.copy()
    y_model0[series_index[0]] -= xbest0[8]
    y_model0[series_index[1]] -= xbest0[9]
    mu0, _ = C0.conditional(y_model0, tsmooth, calc_cov='diag')

    planet_true = mf.planet_injection(tsmooth, args.period, A_inj, B_inj)
    planet_rec = mf.planet_injection(tsmooth, P_rec, A_rec, B_rec)

    t_rv = t_full[series_index[0]]
    y_rv1 = y_full[series_index[0]] - xbest1[8]
    y_rv0 = y_full[series_index[0]] - xbest0[8]

    zoom_start = float(np.median(t_rv))
    zoom_end = zoom_start + args.zoom_days

    fig, axs = plt.subplots(3, 2, figsize=(20, 11))

    for col, (xlo, xhi, tag) in enumerate([(t_full.min(), t_full.max(), "full baseline"),
                                            (zoom_start, zoom_end, f"{args.zoom_days:.0f}-day zoom")]):
        ax = axs[0, col]
        ax.errorbar(t_rv, y_rv1, yerr_full[series_index[0]], fmt='.', color='k', alpha=0.5, label='data (offset-subtracted)')
        ax.plot(tsmooth, mu1 + planet_rec, 'g', label='GP+planet fit (total)')
        ax.plot(tsmooth, mu1, color='teal', label='GP fit (rot. term)')
        ax.plot(tsmooth, planet_rec, color='orange', label='found planet term')
        ax.set_ylabel("RV")
        ax.set_xlim(xlo, xhi)
        ax.set_title(f"GP+planet fit ({tag}): P_rec={P_rec:.2f}d K_rec={K_rec:.4f}  "
                     f"(injected P={args.period:.2f}d K={args.k:.4f})")
        ax.legend(fontsize=8)

        ax = axs[1, col]
        ax.plot(tsmooth, planet_rec, color='orange', label='recovered planet contribution')
        ax.plot(tsmooth, planet_true, 'r--', label='injected planet')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel("RV (planet term only)")
        ax.set_xlim(xlo, xhi)
        ax.legend(fontsize=8)

        ax = axs[2, col]
        ax.errorbar(t_rv, y_rv0, yerr_full[series_index[0]], fmt='.', color='k', alpha=0.5, label='data (offset-subtracted)')
        ax.plot(tsmooth, mu0, color='purple', label='GP-only fit (no planet term)')
        ax.plot(tsmooth, planet_true, 'r--', label='injected planet')
        ax.set_ylabel("RV")
        ax.set_xlabel("t [d]")
        ax.set_xlim(xlo, xhi)
        ax.set_title(f"GP-only fit ({tag}): rho={xbest0[3]:.2f} eta={xbest0[4]:.2f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
