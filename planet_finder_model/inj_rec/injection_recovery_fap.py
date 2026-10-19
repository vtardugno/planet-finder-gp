"""Injection-recovery grid sweep where recovery also requires the planet signal
to be significant under the correlated-noise periodogram FAP of Delisle, Hara &
Segransan (2020, A&A 635, A83).

Same injections, grid, fits and CSV resumability as injection_recovery.py, but
each (combo, mode) is fitted twice: GP+planet and GP-only. A planet is
considered "recovered" only if BOTH:
  1. The GP+planet best-fit period/K land within tolerance of the injected
     truth (injection_recovery.py's criterion), AND
  2. FAP_max(z1(P_rec)) < --fap-threshold, where z1 is the periodogram of a
     sinusoid at the recovered period P_rec against the GP-only fit (base
     model H: its mean model subtracted, its covariance as the noise C).

Everything is done on the RVs conditioned on the activity indicator (as in
MGIC_rv), so n = N_rv and C^-1 is the conditional RV covariance's inverse:
  z1 = (n_H / 2) (chi2_H - chi2_K(P_rec)) / chi2_H                (Table 1)
  FAP = 1 - (1 - FAP_single) exp(-tau)                            (Eq. 6)
  FAP_single = (1 - 2 z1 / n_H)^(n_K / 2)
  tau = gamma_H W (1 - 2 z1 / n_H)^((n_K - 1) / 2) sqrt(z1)       (Table 1)
  W = nu_max T_eff / (2 pi), nu_max = 2 pi / --search-period-min  (Eq. 7)
with T_eff from Eq. 8. The base model's linear parameters are fixed at the
GP-only fit (not refitted with the sinusoid), so p = 0: n_H = n, n_K = n - 2.
"""

import argparse

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

import injection_recovery as ir
import injection_recovery_mgic as irm


CSV_FIELDS = ir.CSV_FIELDS + [
    "loglike_gp_only", "gp_only_at_bounds",
    "z1", "T_eff", "fap", "fap_threshold", "fap_pass",
]


def build_parser():
    parser = ir.build_parser()
    parser.description = ("Injection-recovery grid sweep: recovered = right P/K found AND "
                          "significant under the Delisle et al. (2020) correlated-noise FAP.")
    parser.set_defaults(output_csv="inj_rec/injection_recovery_fap.csv")
    parser.add_argument("--fap-threshold", type=float, default=0.01,
                         help="Maximum FAP of the periodogram at the recovered period for the "
                              "planet to count as detected.")
    return parser


def t_eff(t, Cinv, nu_max):
    """Effective time span (Eq. 8) of times t under noise precision Cinv."""
    t = t - np.mean(t)
    w = Cinv * np.sinc(nu_max * (t[:, None] - t[None, :]) / np.pi)  # np.sinc(x) = sin(pi x)/(pi x)
    s0 = np.sum(w)
    s_sigma = np.sum(w * (t[:, None] + t[None, :]))
    s_pi = t @ w @ t
    return np.sqrt(4 * np.pi) * np.sqrt(s_pi / s0 - (s_sigma / (2 * s0)) ** 2)


def fap_z1(z, n_H, n_K, W):
    """FAP_max of the z1 periodogram (Eq. 6, Table 1), with gamma_H included."""
    log_base = np.log1p(-2 * z / n_H)
    fap_single = np.exp(n_K / 2 * log_base)
    gamma_H = np.sqrt(2 / n_H) * np.exp(gammaln(n_H / 2) - gammaln((n_H - 1) / 2))
    tau = gamma_H * W * np.exp((n_K - 1) / 2 * log_base) * np.sqrt(z)
    return fap_single - (1 - fap_single) * np.expm1(-tau)


def fit_fap(mode, xbest0, diag0, P, prep, nu_max):
    """(z1, T_eff, FAP) of a sinusoid at period P against a GP-only fit_* result."""
    C, series_index = diag0["cov"], prep["series_index"]
    i_rv, i_act = series_index[0], series_index[1]
    resid = ir.residual(mode, xbest0, C, prep, fit_planet=False)

    # RV covariance and residuals conditioned on the activity indicator
    K = irm.expand_cov(C)
    K_ra = K[np.ix_(i_rv, i_act)]
    M = cho_solve(cho_factor(K[np.ix_(i_act, i_act)], lower=True), K_ra.T)
    L_cond = cho_factor(K[np.ix_(i_rv, i_rv)] - K_ra @ M, lower=True)
    r_cond = resid[i_rv] - M.T @ resid[i_act]
    Cinv = cho_solve(L_cond, np.eye(len(i_rv)))

    t = prep["t_full"][i_rv]
    X = np.column_stack([np.sin(2 * np.pi * t / P), np.cos(2 * np.pi * t / P)])
    b = X.T @ Cinv @ r_cond
    dchi2 = b @ np.linalg.solve(X.T @ Cinv @ X, b)
    chi2_H = r_cond @ Cinv @ r_cond

    n_H = len(i_rv)
    z = n_H / 2 * dchi2 / chi2_H
    T = t_eff(t, Cinv, nu_max)
    return z, T, fap_z1(z, n_H, n_H - 2, nu_max * T / (2 * np.pi))


def run_one_combo(args_dict, period_idx, k_idx, phase_idx, period, k, seed, modes_needed):
    args = argparse.Namespace(**args_dict)
    log_prefix = f"[combo p_idx={period_idx} k_idx={k_idx} phase_idx={phase_idx}]"
    prep = ir.prepare_injection(args, period, k, seed, modes_needed, log_prefix)
    nu_max = 2 * np.pi / args.search_period_min

    rows = []
    for mode in modes_needed:
        fit1, fit0, errors, guesses = ir.fit_mode(mode, prep, args, gp_only=True, log_prefix=log_prefix)
        xbest1, loglike1, diag1 = fit1 or (None, float("nan"), {"at_bounds": []})
        xbest0, loglike0, diag0 = fit0 or (None, float("nan"), {"at_bounds": []})

        row = ir.make_row(
            period_idx, k_idx, phase_idx, period, k, prep["phase"], prep["A_inj"], prep["B_inj"], mode,
            xbest1, loglike1, diag1, args.period_tolerance, args.k_tolerance,
            ";".join(errors) if errors else None,
            sigma_K=ir.planet_sigma_K(fit1, prep), k_sigma=args.k_sigma,
        )

        fap = {"z1": float("nan"), "T_eff": float("nan"), "fap": float("nan")}
        both_finite = (xbest1 is not None and np.isfinite(loglike1)
                       and xbest0 is not None and np.isfinite(loglike0))
        if both_finite:
            try:
                fap["z1"], fap["T_eff"], fap["fap"] = fit_fap(mode, xbest0, diag0, xbest1[-3], prep, nu_max)
            except Exception as exc:
                errors.append(f"fap_exception:{exc!r}")
                print(f"{log_prefix} mode={mode} FAP failed: {exc}", flush=True)

        fap_pass = bool(np.isfinite(fap["fap"]) and fap["fap"] < args.fap_threshold)

        flags = [f for f in row["diagnostic_flags"].split(",") if f] if row["diagnostic_flags"] else []
        if diag0.get("at_bounds"):
            flags.append("gp_only_at_bounds")
        flags += [e for e in errors if e.startswith("fap_")]

        row.update(fap)
        row.update({
            "loglike_gp_only": loglike0,
            "gp_only_at_bounds": ",".join(map(str, diag0.get("at_bounds", []))),
            "fap_threshold": args.fap_threshold,
            "fap_pass": fap_pass,
            "recovered": int(row["param_match_pass"] and fap_pass),
            "diagnostic_flags": ",".join(flags),
        })
        row.update(ir.period_search_columns(args, guesses))
        rows.append(row)

    return rows


def _worker(task):
    return run_one_combo(*task)


def main():
    ir.main_from_args(build_parser().parse_args(), csv_fields=CSV_FIELDS, worker_fn=_worker)


if __name__ == "__main__":
    main()
