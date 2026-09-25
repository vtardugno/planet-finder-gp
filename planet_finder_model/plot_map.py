"""
Plot a MAP fit from a saved map_params_*.npy file, for a run produced by
main_cycle_likelihood.py (the cycle is fit jointly in the likelihood, not
subtracted from the data beforehand).

Edit the CONFIG block below to match the run that produced the map_params
file (same --path/--star-name/--normalise/--inject-planet flags you used
with main_cycle_likelihood.py), then run: python plot_map.py
"""
import numpy as np
import functions as mf
from spleaf import cov, term

class MultiSeriesKernel(term.MultiSeriesKernel):
    def _grad_param(self, grad_dU=None, grad_dV=None):
        if grad_dU is not None or grad_dV is not None:
            raise NotImplementedError()
        return super()._grad_param()

# ---------------- CONFIG: edit to match the run ----------------
MAP_PARAMS_FILE = "results_cyc/map_params_cycle_final.npy"

DATA_PATH = "data/Solar_Data"
STAR_NAME = "Sun"
NORMALISE = False
INJECT_PLANET = False
PLANET_PERIOD, PLANET_A, PLANET_B = 40.0, 0.001, 0.001

SIG, PROT, RHO, ETA = 1.0, 27.0, 20.0, 0.25   # initial guesses, overwritten by map params
RVJIT_FRAC, RHKJIT_FRAC = 0.1, 0.1

OUTPUT_NAME = "results_cyc/MAP_fit_plot.png"
# -----------------------------------------------------------------

load_result = mf.load_and_norm_data(
    DATA_PATH, STAR_NAME,
    normalise=NORMALISE,
    inject_planet=INJECT_PLANET,
    planet_params=(PLANET_PERIOD, PLANET_A, PLANET_B),
)

if INJECT_PLANET and NORMALISE:
    t_full, y_full, yerr_full, series_index, rv_std = load_result
else:
    t_full, y_full, yerr_full, series_index = load_result
    rv_std = 1.0

stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]

C = cov.Cov(
    t_full,
    err=term.Error(yerr_full),
    rv_jit=term.InstrumentJitter(series_index[0], RVJIT_FRAC * stds[0]),
    rhk_jit=term.InstrumentJitter(series_index[1], RHKJIT_FRAC * stds[1]),
    rot=MultiSeriesKernel(
        term.MEPKernel(SIG, PROT, RHO, ETA), series_index,
        np.array([stds[0], stds[1]]),
        np.array([stds[0], 0.0]),
    ),
)

map_params = np.load(MAP_PARAMS_FILE)

params, _ = mf.get_opt_params(C)
C.set_param(map_params[:len(params)], params)

mf.plot_fit_cyc(
    t_full, y_full, yerr_full, series_index, C, map_params,
    rv_std=rv_std, output_name=OUTPUT_NAME, inject_planet=INJECT_PLANET,
)
