import numpy as np
from scipy.linalg import cho_factor, cho_solve
from spleaf import cov
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CV_pipeline2 import build_parser, fit_nocyc, fit_cyc, fit_nonstat, fit_nonstat2, mf, mfn, mfn2

# MGIC (sections 15-16 of GPRV/MGIC tutorial/tutorial_basics_of_MGIC.ipynb) for
# the four pipeline-2 models, evaluated at the whole-dataset optimiser xbest.
# Every model is RV + RHK, so the RV likelihood is conditioned on RHK and the
# smoother is built from the RV latent GP covariance left after that conditioning.

MODES = ["no_cycle", "cyc", "nonstat", "nonstat2"]

def mean_model(mode, xbest, C, t_full, series_index, T, fit_planet):
    # Mirrors the y_model construction in each model's negloglike
    t0, t1 = t_full[series_index[0]], t_full[series_index[1]]
    mean = np.zeros_like(t_full)

    if mode == "no_cycle":
        mean[series_index[0]] = xbest[8]
        mean[series_index[1]] = xbest[9]
        if fit_planet:
            mean[series_index[0]] += mf.planet_injection(t0, xbest[10], xbest[11], xbest[12])

    elif mode == "cyc":
        b, P, phi = xbest[8], xbest[9], xbest[10]
        mean[series_index[0]] = xbest[11] * mf.shared_core(t0, b, P, phi) + xbest[13]
        mean[series_index[1]] = xbest[12] * mf.shared_core(t1, b, P, phi) + xbest[14]
        if fit_planet:
            mean[series_index[0]] += mf.planet_injection(t0, xbest[15], xbest[16], xbest[17])

    elif mode == "nonstat":
        n = len(mfn.get_opt_params_nonstat(C)[0])
        b, P, phi = C.get_param(['rot.nonstat_b', 'rot.nonstat_P', 'rot.nonstat_phi'])
        mean[series_index[0]] = xbest[n] * mfn.shared_core(t0, b, P, phi) + xbest[n+2]
        mean[series_index[1]] = xbest[n+1] * mfn.shared_core(t1, b, P, phi) + xbest[n+3]
        if fit_planet:
            mean[series_index[0]] += mfn.planet_injection(t0, xbest[n+4], xbest[n+5], xbest[n+6])

    elif mode == "nonstat2":
        n = len(mfn2.get_opt_params_nonstat(C)[0])
        b, P, phi, c = C.get_param(['rot.nonstat_b', 'rot.nonstat_P', 'rot.nonstat_phi', 'rot.nonstat_c'])
        mean[series_index[0]] = xbest[n] * mfn2.shared_core_pos(t0, b, P, phi, c, T) + xbest[n+2]
        mean[series_index[1]] = xbest[n+1] * mfn2.shared_core_pos(t1, b, P, phi, c, T) + xbest[n+3]
        if fit_planet:
            mean[series_index[0]] += mfn2.planet_injection(t0, xbest[n+4], xbest[n+5], xbest[n+6])

    return mean

def stable_loglike_from_cov(residual, covariance):
    """Evaluate a zero-mean multivariate Gaussian log-likelihood stably."""
    try:
        chol = cho_factor(covariance, lower=True, check_finite=False)
        Kinv_residual = cho_solve(chol, residual, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(chol[0])))
    except np.linalg.LinAlgError:
        return -np.inf

    return -0.5 * (residual @ Kinv_residual + logdet + len(residual) * np.log(2.0 * np.pi))

def mgic(C, xbest, y_full, yerr_full, series_index, mean):
    """Conditional RV likelihood (RV | RHK), smoother trace K_s and MGIC."""
    i0, i1 = series_index[0], series_index[1]

    # Dense joint covariance; the latent GP part is it minus the diagonal noise
    # (nominal errors + per-series jitter, which sit at xbest[0] / xbest[1])
    K = C.expand()
    noise = yerr_full**2
    noise[i0] += xbest[0]**2
    noise[i1] += xbest[1]**2
    G = K - np.diag(noise)

    r = y_full - mean
    joint_loglike = stable_loglike_from_cov(r, K)

    K11 = K[np.ix_(i0, i0)]
    K12 = K[np.ix_(i0, i1)]   # noise is diagonal, so this is also the latent cross-covariance
    K22 = K[np.ix_(i1, i1)]
    G11 = G[np.ix_(i0, i0)]

    chol22 = cho_factor(K22, lower=True, check_finite=False)

    # mu_RV|RHK and K_RV|RHK
    r_cond = r[i0] - K12 @ cho_solve(chol22, r[i1], check_finite=False)
    K22_inv_K21 = cho_solve(chol22, K12.T, check_finite=False)
    cond_cov = K11 - K12 @ K22_inv_K21

    # Latent RV GP covariance remaining after conditioning on RHK
    latent_cond_cov = G11 - K12 @ K22_inv_K21

    logL_cond = stable_loglike_from_cov(r_cond, cond_cov)

    # S = K_latent|RHK (K_RV|RHK)^{-1}
    cond_chol = cho_factor(cond_cov, lower=True, check_finite=False)
    S = cho_solve(cond_chol, latent_cond_cov.T, check_finite=False).T

    K_s = np.trace(S)
    K_p = len(xbest)

    return {
        "joint_loglike": joint_loglike,
        "logL_cond": logL_cond,
        "K_p": K_p,
        "K_s": K_s,
        "MGIC": -2.0 * logL_cond + 2.0 * (K_p + K_s),
    }

def main():

    args = build_parser().parse_args()

    t_full, y_full, yerr_full, series_index = mf.load_and_norm_data(
                        args.path,
                        args.star_name,
                        normalise=args.normalise,
                        inject_planet=args.inject_planet,
                        planet_params=(args.planet_period, args.planet_A, args.planet_B),
                    )
    rv_std = 1.0

    t = t_full[series_index[0]]
    T_ = [t, t]
    Y = [y_full[series_index[0]], y_full[series_index[1]]]
    Yerr = [yerr_full[series_index[0]], yerr_full[series_index[1]]]

    t_full, y_full, yerr_full, series_index = cov.merge_series(T_, Y, Yerr)

    stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]
    n_data = len(y_full)
    T = np.max(t_full)

    planet_p = args.planet_p
    if args.fit_planet and planet_p is None:
        periods, _ = mf.period_guess(t_full, y_full, yerr_full, series_index, PMIN=1.1, PMAX=310.0, MAX_FAP=1e-5, MAX_NPL=2, plot=False)
        planet_p = float(periods[1]) if len(periods) >= 2 else float(periods[0])

    _, _, cycle_params = mf.fit_cycle(t_full, y_full, series_index, b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0, plot=False, print_results=False, return_fit=True)

    results = {}

    for mode in MODES:
        if mode == "no_cycle":
            xbest, C = fit_nocyc(args, t_full, y_full, yerr_full, series_index, stds, planet_p, rv_std)
            loglike = -mf.negloglike_nocyc(xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]
        elif mode == "cyc":
            xbest, C = fit_cyc(args, t_full, y_full, yerr_full, series_index, stds, planet_p, cycle_params, rv_std)
            loglike = -mf.negloglike_cyc(xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]
        elif mode == "nonstat":
            xbest, C = fit_nonstat(args, t_full, y_full, yerr_full, series_index, stds, planet_p, cycle_params, rv_std)
            loglike = -mfn.negloglike_nonstat(xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]
        elif mode == "nonstat2":
            xbest, C = fit_nonstat2(args, t_full, y_full, yerr_full, series_index, stds, planet_p, cycle_params, rv_std)
            loglike = -mfn2.negloglike_nonstat(xbest, t_full, y_full, series_index, C, rv_std, inject_planet=args.fit_planet)[0]

        print(f"xbest {mode}: ", xbest)

        mean = mean_model(mode, xbest, C, t_full, series_index, T, args.fit_planet)
        r = mgic(C, xbest, y_full, yerr_full, series_index, mean)

        # The dense joint likelihood must reproduce spleaf's, otherwise the
        # mean vector / expanded covariance used for MGIC are inconsistent
        if not np.isclose(r["joint_loglike"], loglike, rtol=0, atol=1e-6 * max(1.0, abs(loglike))):
            print(f"WARNING [{mode}]: dense joint lnL {r['joint_loglike']:.6f} != spleaf lnL {loglike:.6f}")

        r["AIC"] = 2 * len(xbest) - 2 * loglike
        r["BIC"] = len(xbest) * np.log(n_data) - 2 * loglike
        results[mode] = r

    best_mgic = min(r["MGIC"] for r in results.values())

    print(f"\n{'Model':<10} {'lnL_RV|RHK':>12} {'K_p':>5} {'K_s':>10} {'MGIC':>12} {'Delta':>10} {'AIC':>12} {'BIC':>12}")
    print("-" * 90)
    for mode in MODES:
        r = results[mode]
        print(
            f"{mode:<10} {r['logL_cond']:12.3f} {r['K_p']:5d} "
            f"{r['K_s']:10.3f} {r['MGIC']:12.3f} {r['MGIC']-best_mgic:10.3f} "
            f"{r['AIC']:12.3f} {r['BIC']:12.3f}"
        )

if __name__ == "__main__":
    main()
