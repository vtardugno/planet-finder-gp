"""Injection-recovery grid sweep where recovery also requires the planet model
to be preferred over a planet-less one.

Same injections, grid, fits and CSV resumability as injection_recovery.py, but
each (combo, mode) is fitted twice: GP+planet and GP-only. A planet is
considered "recovered" only if BOTH:
  1. The GP+planet best-fit period/K land within tolerance of the injected
     truth (injection_recovery.py's criterion), AND
  2. The GP+planet model is preferred under MGIC_rv (Barragan,
     arXiv:2606.04875): MGIC_rv(GP-only) - MGIC_rv(GP+planet) > --mgic-threshold.

MGIC_rv = -2 ln L'_rv + 2 (K_p + K_s), where L'_rv is the likelihood of the RVs
conditioned on the activity indicator (log R'HK) at the best-fit parameters,
K_p is the number of fitted parameters and K_s = tr(S_rv) is the effective
number of degrees of freedom of the GP smoother on the RVs.
"""

import argparse

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import injection_recovery as ir

mf, mfn, mfn2 = ir.mf, ir.mfn, ir.mfn2


CSV_FIELDS = ir.CSV_FIELDS + [
    "loglike_gp_only", "gp_only_at_bounds",
    "lnL_cond_planet", "lnL_cond_gp_only", "Ks_planet", "Ks_gp_only",
    "mgic_planet", "mgic_gp_only", "delta_mgic", "mgic_threshold", "mgic_pass",
]


def build_parser():
    parser = ir.build_parser()
    parser.description = ("Injection-recovery grid sweep: recovered = right P/K found AND "
                          "GP+planet preferred over GP-only under MGIC_rv.")
    parser.set_defaults(output_csv="inj_rec/injection_recovery_mgic.csv")
    parser.add_argument("--mgic-threshold", type=float, default=10.0,
                         help="Minimum MGIC_rv(GP-only) - MGIC_rv(GP+planet) for the planet model "
                              "to count as preferred (10 = 'strong' on the AIC scale).")
    return parser


def expand_cov(C):
    """Dense covariance of a spleaf Cov. Same algorithm as C.expand(), with the
    inner loop vectorised (C.expand() takes ~30 s for the ~4000-point Solar data)."""
    K = np.diag(C.A)
    for i in range(1, C.n):
        for j in range(i - C.b[i], i):
            K[i, j] = C.F[C.offsetrow[i] + j]
        # cumphi[m] = prod(phi[i-1-m:i]) multiplies U[i] V[j] for j = i-1-m
        cumphi = np.cumprod(C.phi[i - 1::-1], axis=0)
        K[i, i - 1::-1] += np.sum(cumphi * C.U[i] * C.V[i - 1::-1], axis=1)
    return np.tril(K) + np.tril(K, -1).T


def mgic_rv(C, resid, series_index, yerr_full, n_params):
    """MGIC_rv of a fitted model. C must already be set to the best-fit
    hyperparameters and resid is y minus the mean model (both series).
    Returns (MGIC_rv, ln L'_rv, K_s)."""
    i_rv, i_act = series_index[0], series_index[1]
    K = expand_cov(C)
    K_rr = K[np.ix_(i_rv, i_rv)]
    K_ra = K[np.ix_(i_rv, i_act)]
    K_aa = K[np.ix_(i_act, i_act)]

    # RV covariance and residuals conditioned on the activity indicator (Eqs. 6, 7, 9)
    M = cho_solve(cho_factor(K_aa, lower=True), K_ra.T)
    K_cond = K_rr - K_ra @ M
    r_cond = resid[i_rv] - M.T @ resid[i_act]

    L_cond = cho_factor(K_cond, lower=True)
    n_rv = len(i_rv)
    logdet = 2.0 * np.sum(np.log(np.diag(L_cond[0])))
    lnL_cond = -0.5 * (n_rv * np.log(2 * np.pi) + logdet + r_cond @ cho_solve(L_cond, r_cond))

    # Eq. 14 with k = K - R and noise-free cross-series blocks reduces to
    # S_rv = I - R_rv K_cond^-1, so tr(S_rv) = N_rv - sum_i R_ii (K_cond^-1)_ii
    R_rv = yerr_full[i_rv] ** 2 + C.get_param("rv_jit.sig") ** 2
    K_cond_inv_diag = np.diag(cho_solve(L_cond, np.eye(n_rv)))
    K_s = n_rv - np.sum(R_rv * K_cond_inv_diag)

    return -2.0 * lnL_cond + 2.0 * (n_params + K_s), lnL_cond, K_s


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


def run_one_combo(args_dict, period_idx, k_idx, phase_idx, period, k, seed, modes_needed):
    args = argparse.Namespace(**args_dict)
    log_prefix = f"[combo p_idx={period_idx} k_idx={k_idx} phase_idx={phase_idx}]"
    prep = ir.prepare_injection(args, period, k, seed, modes_needed, log_prefix)

    rows = []
    for mode in modes_needed:
        fit_fn, fit_args = ir.fit_fn_and_args(mode, prep, args)

        fits = {}
        errors = []
        for fit_planet, label in ((True, "gp_planet"), (False, "gp_only")):
            try:
                fits[fit_planet] = fit_fn(*fit_args, fit_planet=fit_planet)
            except Exception as exc:
                errors.append(f"{label}_exception:{exc!r}")
                print(f"{log_prefix} mode={mode} {label} fit failed: {exc}", flush=True)

        xbest1, loglike1, diag1 = fits.get(True, (None, float("nan"), {"at_bounds": []}))
        xbest0, loglike0, diag0 = fits.get(False, (None, float("nan"), {"at_bounds": []}))

        row = ir.make_row(
            period_idx, k_idx, phase_idx, period, k, prep["phase"], prep["A_inj"], prep["B_inj"], mode,
            xbest1, loglike1, diag1, args.period_tolerance, args.k_tolerance,
            ";".join(errors) if errors else None,
        )

        mgic = {"lnL_cond_planet": float("nan"), "lnL_cond_gp_only": float("nan"),
                "Ks_planet": float("nan"), "Ks_gp_only": float("nan"),
                "mgic_planet": float("nan"), "mgic_gp_only": float("nan"), "delta_mgic": float("nan")}
        both_finite = (xbest1 is not None and np.isfinite(loglike1)
                       and xbest0 is not None and np.isfinite(loglike0))
        if both_finite:
            try:
                for fit_planet, xbest, diag, tag in ((True, xbest1, diag1, "planet"),
                                                     (False, xbest0, diag0, "gp_only")):
                    resid = residual(mode, xbest, diag["cov"], prep, fit_planet)
                    value, lnL_cond, K_s = mgic_rv(diag["cov"], resid, prep["series_index"],
                                                   prep["yerr_full"], len(xbest))
                    mgic.update({f"mgic_{tag}": value, f"lnL_cond_{tag}": lnL_cond, f"Ks_{tag}": K_s})
                mgic["delta_mgic"] = mgic["mgic_gp_only"] - mgic["mgic_planet"]
            except Exception as exc:
                errors.append(f"mgic_exception:{exc!r}")
                print(f"{log_prefix} mode={mode} MGIC failed: {exc}", flush=True)

        mgic_pass = bool(np.isfinite(mgic["delta_mgic"]) and mgic["delta_mgic"] > args.mgic_threshold)

        flags = [f for f in row["diagnostic_flags"].split(",") if f] if row["diagnostic_flags"] else []
        if diag0.get("at_bounds"):
            flags.append("gp_only_at_bounds")
        flags += [e for e in errors if e.startswith("mgic_")]

        row.update(mgic)
        row.update({
            "loglike_gp_only": loglike0,
            "gp_only_at_bounds": ",".join(map(str, diag0.get("at_bounds", []))),
            "mgic_threshold": args.mgic_threshold,
            "mgic_pass": mgic_pass,
            "recovered": int(row["param_match_pass"] and mgic_pass),
            "diagnostic_flags": ",".join(flags),
        })
        rows.append(row)

    return rows


def _worker(task):
    return run_one_combo(*task)


def main():
    ir.main_from_args(build_parser().parse_args(), csv_fields=CSV_FIELDS, worker_fn=_worker)


if __name__ == "__main__":
    main()
