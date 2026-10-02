"""Per-model Delta MGIC_rv threshold from the real, uninjected Sun.

Fits GP+planet and GP-only to the Solar data with no planet injected, using
exactly the same warm starts, periodogram guesses and fits as the
injection-recovery sweep, and records Delta MGIC_rv = MGIC_rv(GP-only) -
MGIC_rv(GP+planet). That value is the model's threshold: an injected planet
only counts as preferred if it beats the Sun's own best spurious "planet".

The output JSON is read by injection_recovery_mgic.py --mgic-threshold-path
and rescore_mgic.py --threshold-path.
"""

import json
import os

import numpy as np

import injection_recovery as ir
import injection_recovery_mgic as irm


def build_parser():
    parser = irm.build_parser()
    parser.description = "Per-model Delta MGIC_rv threshold from the real, uninjected Sun."
    parser.set_defaults(models="cyc,nonstat,nonstat2,no_cycle")
    parser.add_argument("--threshold-path", default="results/mgic_calibration/mgic_threshold.json")
    return parser


def calibrate_mode(mode, prep, args):
    fit1, fit0, errors, _ = ir.fit_mode(mode, prep, args, gp_only=True)
    if errors:
        raise RuntimeError(f"{mode}: {errors}")
    xbest1, loglike1, diag1 = fit1
    xbest0, loglike0, diag0 = fit0
    mgic1, _, Ks1 = irm.fit_mgic(mode, xbest1, diag1, prep, fit_planet=True)
    mgic0, _, Ks0 = irm.fit_mgic(mode, xbest0, diag0, prep, fit_planet=False)
    delta_mgic = float(mgic0 - mgic1)
    return {
        "threshold": delta_mgic, "delta_mgic": delta_mgic,
        "P_rec": float(xbest1[-3]), "K_rec": float(np.hypot(xbest1[-2], xbest1[-1])),
        "loglike_planet": float(loglike1), "loglike_gp_only": float(loglike0),
        "mgic_planet": float(mgic1), "mgic_gp_only": float(mgic0),
        "Ks_planet": float(Ks1), "Ks_gp_only": float(Ks0),
        "period_search": args.period_search,
    }


def main():
    args = build_parser().parse_args()
    modes = [m.strip() for m in args.models.split(",")]

    # k=0: the "injected" planet is identically zero, i.e. the real Sun
    prep = ir.prepare_injection(args, 1.0, 0.0, args.seed, set(modes))

    results = {}
    for mode in modes:
        results[mode] = calibrate_mode(mode, prep, args)
        r = results[mode]
        print(f"{mode:9s} dMGIC={r['delta_mgic']:8.2f}  P_rec={r['P_rec']:8.3f} d  "
              f"K_rec={r['K_rec'] * 1000:.3f} m/s", flush=True)

    out_dir = os.path.dirname(args.threshold_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.threshold_path, "w") as f:
        json.dump({"null": "real_sun_uninjected", "modes": results}, f, indent=2)
    print(f"Saved {args.threshold_path}")


if __name__ == "__main__":
    main()
