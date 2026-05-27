import functions as mf
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from spleaf import cov, term
import emcee
        

t_full, y_full, yerr_full, series_index = mf.load_and_norm_data("Solar Data", "Sun",normalise=False, inject_planet=True, planet_params=(40, 0.001, 0.004))

rv_fit, rhk_fit = mf.fit_cycle(t_full, y_full, series_index, plot=True, print_results=True)

y_full[series_index[0]] = y_full[series_index[0]] - rv_fit
y_full[series_index[1]] = y_full[series_index[1]] - rhk_fit

class MultiSeriesKernel(term.MultiSeriesKernel):
  def _grad_param(self, grad_dU=None, grad_dV=None):
    if grad_dU is not None or grad_dV is not None:
      raise NotImplementedError()
    return super()._grad_param()
  
stds = [np.std(y_full[series_index[0]]), np.std(y_full[series_index[1]])]
sig = 1.0

prot = 27.0
Q = 1.2

rvjit = 0.1*stds[0]
rhkjit = 0.1*stds[1]

rho = 20.0
eta = 0.25


C = cov.Cov(
  t_full,
  err=term.Error(yerr_full),
  rv_jit=term.InstrumentJitter(series_index[0], rvjit),
  rhk_jit=term.InstrumentJitter(series_index[1], rhkjit),
  rot = MultiSeriesKernel(term.MEPKernel(sig,prot,rho,eta), series_index, 
        np.array([stds[0], stds[1]]), 
        np.array([stds[0], 0.0])
    ),
  )


rvjit_max = 5*stds[0]
rhkjit_max = 5*stds[1]

prot_min = 20.0
prot_max = 32.0
Q_min = 0.9
Q_max = 3.0

alpha_0_max = 10*stds[0]
alpha_1_max = 5*stds[1]
beta_0_max  = 10*stds[0]

rho_min = 10.0
rho_max = 65.0
eta_min = 0.1
eta_max = 1.0

gamma_0_max = 5*stds[0]
gamma_1_max = 5*stds[1]

planet_p_min = 30.0
planet_p_max = 50.0
planet_A_min = 0.0001
planet_A_max = 0.01
planet_B_min = 0.0001
planet_B_max = 0.01

bounds_list = [
    (0.0, rvjit_max),                     # rv_jit.sig
    (0.0, rhkjit_max),                     # rhk_jit.sig
    
    (prot_min, prot_max),                 # rot.P0 (period)
    # (Q_min, Q_max),                      # rot.Q (quality factor)
    (rho_min, rho_max),                    # rot.rho (frequency)
    (eta_min, eta_max),                    # rot.eta (damping)
    (0, alpha_0_max),         # rot.alpha_0 (set lower bound to zero)
    (-alpha_1_max, alpha_1_max),         # rot.alpha_1
    (-beta_0_max, beta_0_max),         # rot.beta_0      # rot.beta_1
    # (0.01,0.99), 
    (-0.5, 0.5),         # delta_0 (offset for series 0)
    (-2,2),    
    (planet_p_min, planet_p_max),                         # P_planet
    (planet_A_min, planet_A_max),                          # A_planet
    (planet_B_min, planet_B_max),                          # B_planet

]

xbest = mf.optimise_params(t_full, y_full, series_index, C, bounds_list)
mf.plot_fit(t_full, y_full, yerr_full, series_index, C, xbest, inject_planet=True)
sampler = mf.run_emcee(t_full, y_full, series_index, C, xbest, bounds_list, run_length = 2000)
mf.plot_chains(sampler, C)
mf.plot_corner(sampler, C)