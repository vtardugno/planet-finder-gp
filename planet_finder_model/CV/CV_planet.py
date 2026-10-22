import numpy as np
from spleaf import cov
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "inj_rec"))
from CV import build_parser, fit_nocyc, fit_cyc, fit_nonstat2, build_C_qp, build_C_nonstat2, test_loglike_nonstat2, mf
# The planet fit is injection_recovery.py's with --period-search gp; imported
# after numpy, so its single-thread BLAS env defaults don't apply to this process
import injection_recovery as ir

# 12-fold CV (random split as in CV.py, contiguous-block split as in CV_block.py)
# of the no_cycle / cyc / nonstat2 models, each fitted with and without a planet,
# to check which is preferred on planet-less data. Scores are the test-fold
# log-likelihoods at the training-fold best fit, summed across folds.
# The planet-less fit is CV.py's; the planet fit is exactly injection_recovery.py's
# --period-search gp: P and A/B guesses from the GP period scan of the
# planet-less fit's residual, one guess (the top scan peak).

MODES = ["no_cycle", "cyc", "nonstat2"]
N_FOLDS = 12

def fold_ids_for(split, t):
    N = len(t)
    fold_ids = np.empty(N, dtype=int)
    if split == "random":
        rng = np.random.default_rng(42)
        fold_ids[rng.permutation(N)] = np.arange(N) % N_FOLDS
    elif split == "block":
        fold_ids[np.argsort(t)] = np.arange(N) * N_FOLDS // N
    return fold_ids

def fit_no_planet(mode, args, train, stds, cycle_params, rv_std):
    if mode == "no_cycle":
        return fit_nocyc(args, *train, stds, None, rv_std)
    if mode == "cyc":
        return fit_cyc(args, *train, stds, None, cycle_params, rv_std)
    if mode == "nonstat2":
        return fit_nonstat2(args, *train, stds, None, cycle_params, rv_std)

def loglike_fn(mode, args, xbest, T_train, rv_std):
    # loglike at xbest of any (t, y, yerr, series_index) set, as CV.py scores it
    if mode == "nonstat2":
        return lambda t_, y_, yerr_, si_: test_loglike_nonstat2(xbest, t_, y_, si_, build_C_nonstat2(args, xbest, t_, yerr_, si_, T_train), T_train, args.fit_planet)
    negloglike = mf.negloglike_nocyc if mode == "no_cycle" else mf.negloglike_cyc
    return lambda t_, y_, yerr_, si_: -negloglike(xbest, t_, y_, si_, build_C_qp(args, xbest, t_, yerr_, si_), rv_std, inject_planet=args.fit_planet)[0]

def main():

    parser = build_parser()
    parser.set_defaults(output_csv="CV/cv_planet_results.csv")
    args = parser.parse_args()

    # injection_recovery.py's defaults (period search range, number of guesses,
    # planet bounds), with the settings it shares with CV.py taken from args
    ir_args = ir.build_parser().parse_args([])
    for key, value in vars(args).items():
        if hasattr(ir_args, key):
            setattr(ir_args, key, value)
    ir_args.period_search = "gp"
    ir_args.n_period_guesses = 1
    amp_bound = ir_args.planet_amp_max if ir_args.planet_amp_max is not None else 1.5 * ir_args.k_max

    t_full, y_full, yerr_full, series_index = mf.load_and_norm_data(
                        args.path,
                        args.star_name,
                        normalise=args.normalise,
                        inject_planet=args.inject_planet,
                        planet_params=(args.planet_period, args.planet_A, args.planet_B),
                    )
    rv_std = 1.0

    t = t_full[series_index[0]]
    y_rv = y_full[series_index[0]]
    yerr_rv = yerr_full[series_index[0]]
    y_rhk = y_full[series_index[1]]
    yerr_rhk = yerr_full[series_index[1]]

    cv_rows = []

    for split in ["random", "block"]:

        fold_ids = fold_ids_for(split, t)

        for fold in range(N_FOLDS):

            test_mask = fold_ids == fold
            train_mask = ~test_mask

            train = cov.merge_series([t[train_mask], t[train_mask]], [y_rv[train_mask], y_rhk[train_mask]], [yerr_rv[train_mask], yerr_rhk[train_mask]])
            test = cov.merge_series([t[test_mask], t[test_mask]], [y_rv[test_mask], y_rhk[test_mask]], [yerr_rv[test_mask], yerr_rhk[test_mask]])
            t_full_train, y_full_train, yerr_full_train, series_index_train = train
            T_train = np.max(t_full_train)

            stds = [np.std(y_full_train[series_index_train[0]]), np.std(y_full_train[series_index_train[1]])]

            _, _, cycle_params = mf.fit_cycle(t_full_train, y_full_train, series_index_train, b0=args.cycle_b0, P0=args.cycle_P0, phi0=args.cycle_phi0, plot=False, print_results=False, return_fit=True)

            prep = {"t_full": t_full_train, "y_full": y_full_train, "yerr_full": yerr_full_train, "series_index": series_index_train,
                    "rv_std": rv_std, "stds": stds, "amp_bound": amp_bound, "cycle_params": cycle_params, "planet_guess": None}

            for mode in MODES:

                # NO PLANET
                args.fit_planet = False
                x0, C0 = fit_no_planet(mode, args, train, stds, cycle_params, rv_std)

                # PLANET, as in ir.fit_mode with --period-search gp (its GP-only
                # fit is the same as CV.py's planet-less fit above)
                guesses, amps = ir.gp_period_guesses(C0, ir.residual(mode, x0, C0, prep, fit_planet=False), prep, ir_args)
                fit_fn, fit_args = ir.fit_fn_and_args(mode, prep, ir_args, guesses)
                x1, _, diag1 = fit_fn(*fit_args, fit_planet=True, planet_AB=amps)

                results = [(False, x0, None, []), (True, x1, diag1["best_period_guess"], guesses)]

                for fit_planet, xbest, planet_p, guesses in results:
                    args.fit_planet = fit_planet
                    ll = loglike_fn(mode, args, xbest, T_train, rv_std)
                    test_loglike = ll(*test)
                    # log p(y_all | xbest) - log p(y_train | xbest), as in CV.py
                    cond_loglike = ll(t_full, y_full, yerr_full, series_index) - ll(*train)
                    label = f"{split} fold {fold:02d} [{mode}, planet={fit_planet}]"
                    print(f"{label}: test loglike={test_loglike:.6f}")
                    print(f"{label}: conditional test loglike={cond_loglike:.6f}")
                    cv_rows.append({"split": split, "fold": fold, "mode": mode, "planet": fit_planet,
                                    "test_loglike": test_loglike, "test_loglike_cond": cond_loglike,
                                    "planet_p": planet_p, "planet_P_fit": xbest[-3] if fit_planet else None,
                                    "period_guesses": ";".join(f"{g:.4f}" for g in guesses)})

    try:
        import pandas as pd
        df = pd.DataFrame(cv_rows)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved per-fold CV results to {args.output_csv}")
    except Exception as exc:
        print(f"\nCould not save CV results CSV: {exc}")
        print(cv_rows)

    # Summed across folds; Delta = planet - no planet (> 0 means the planet model is preferred)
    for split in ["random", "block"]:
        print(f"\n[{split} split]")
        print(f"{'Model':<10} {'score':<18} {'no planet':>14} {'planet':>14} {'Delta':>12}")
        print("-" * 72)
        for mode in MODES:
            for key in ["test_loglike", "test_loglike_cond"]:
                totals = {p: sum(r[key] for r in cv_rows if r["split"] == split and r["mode"] == mode and r["planet"] == p) for p in [False, True]}
                print(f"{mode:<10} {key:<18} {totals[False]:14.3f} {totals[True]:14.3f} {totals[True] - totals[False]:12.3f}")

if __name__ == "__main__":
    main()
