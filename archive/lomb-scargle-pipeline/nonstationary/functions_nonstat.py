import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, curve_fit
from spleaf import cov, term
import scipy.io as sp
import emcee
import pandas as pd
import corner
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import astropy

class NonStationaryKernel(term.Kernel):
  r"""
  Non-stationary rank 1 separable kernel.

  .. math:: k(t, t') = alpha(t) * alpha(t')

  Parameters
  ----------
  alpha_func : function
    Function alpha_func(t, **kwargs) computing the amplitudes
    from the times and hyper-parameters.
  alpha_grad : function or None
    Optional function alpha_grad(t, **kwargs) providing a dictionary
    of derivatives of alpha with respect to hyper-parameters.
    This is required to compute the gradient of the likelihood with
    respect to hyper-parameters.
  **kwargs:
    Hyper-parameters to be provided to alpha_func and alpha_grad.
  """

  def __init__(self, alpha_func, alpha_grad=None, **kwargs):
    super().__init__()
    self._alpha_func = alpha_func
    self._alpha_grad = alpha_grad
    self._value = kwargs
    self._param = list(self._value.keys())
    self._r = 1

  def _link(self, cov, offset):
    super()._link(cov, offset)

  def _compute(self):
    self._alpha = self._alpha_func(self._cov.t, **self._value)
    self._cov.U[:, self._offset] = self._alpha
    self._cov.V[:, self._offset] = self._alpha
    self._cov.A += self._alpha * self._alpha
    self._cov.phi[:, self._offset] = 1

  def _get_param(self, par):
    return self._value[par]

  def _set_param(self, *args, **kwargs):
    for karg, arg in enumerate(args):
      par = self._param[karg]
      if par in kwargs:
        raise Exception(f'NonStatKernel._set_param: parameter {par} multiply defined.')
      kwargs[par] = arg
    self._value.update(kwargs)

  def _grad_param(self, grad_dU=None, grad_dV=None):

    if grad_dU is not None or grad_dV is not None:
      raise NotImplementedError()


    grad_alpha = (
      2 * self._alpha * self._cov._grad_A
      + self._cov._grad_U[:, self._offset]
      + self._cov._grad_V[:, self._offset]
    )

    # if grad_dU is not None:
    #   grad_alpha += grad_dU[:, self._offset]
    # if grad_dV is not None:
    #   grad_alpha += grad_dV[:, self._offset]

    return {
      key: grad_alpha @ dalpha_dpk
      for key, dalpha_dpk in self._alpha_grad(self._cov.t, **self._value).items()
    }

  def _compute_t2(
    self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right, phi2left, phi2right
  ):
    alpha2 = self._alpha_func(t2, **self._value)
    U2[:, self._offset] = alpha2
    V2[:, self._offset] = alpha2
    phi2[:, self._offset] = 1
    phi2left[:, self._offset] = 1
    phi2right[:, self._offset] = 1



def planet_injection(t, P, A, B):
    return A * np.sin(2 * np.pi * (t) / (P+0.000001)) + B * np.cos(2 * np.pi * (t) / (P+0.000001))


def load_and_norm_data(path,star_name,normalise=True,inject_planet=True, planet_params = (0.0, 0.0, 0.0)):

    data_rhk = pd.read_csv(f"{path}/Analyse_summary.csv")
    data_rhk = data_rhk[data_rhk["flag"] != 1]

    tt = np.array(data_rhk["jdb"])
    tt = tt - np.min(tt)
    # tt = tt - np.min(tt)

    rhk = np.array(data_rhk["MHK_cleaned"])
    rhk_err = np.array(data_rhk["MHK_cleaned_std"])

    data_rv = pd.read_pickle(f"{path}/Analyse_ccf.p")[f'CCF_kitcat_mask_{star_name}']['matching_instrument']['table']
    data_rv = data_rv.loc[data_rv.index.isin(data_rhk.index)]

    rv = np.array(data_rv['rv'])
    erv = np.sqrt(np.array(data_rv['rv_std'])**2 + (np.ones_like(rv) * 0.0005)**2)

    # Planet injection
    if inject_planet==True:
        rv_planet = planet_injection(tt, P=planet_params[0], A=planet_params[1], B=planet_params[2])
        rv += rv_planet



    T = [tt,tt]

    if normalise==True:
        rv_mean = np.mean(rv)
        rv_std  = np.std(rv)
        rv_norm = (rv - rv_mean) / rv_std

        rhk_mean = np.mean(rhk)
        rhk_std  = np.std(rhk)
        rhk_norm = (rhk - rhk_mean) / rhk_std

        # # ERROR normalization using SAME scaling as parent data
        erv_norm     = erv / rv_std
        rhk_err_norm = rhk_err / rhk_std

        Y = [rv_norm,rhk_norm]
        Yerr = [erv_norm,rhk_err_norm]

    else:
        Y = [rv,rhk]
        Yerr = [erv,rhk_err]

    t_full, y_full, yerr_full, series_index = cov.merge_series(T,Y,Yerr)

    if inject_planet == True:
        if normalise == True:
            return t_full, y_full, yerr_full, series_index, rv_std
        else:
            return t_full, y_full, yerr_full, series_index
    else:

        return t_full, y_full, yerr_full, series_index


def shared_core(x, b, P, phi):
    return b*x + np.sin(2*np.pi*x/P + phi)

def shared_core_grads(x, b, P, phi):
    arg = 2 * np.pi * x / P + phi

    dcore_db = x
    dcore_dP = -(2 * np.pi * x / (P**2 + 1e-12)) * np.cos(arg)
    dcore_dphi = np.cos(arg)

    return dcore_db, dcore_dP, dcore_dphi

def shared_core_and_grads(x, b, P, phi):
    arg = 2 * np.pi * x / P + phi
    core = b * x + np.sin(arg)

    dcore_db = x
    dcore_dP = -(2 * np.pi * x / (P**2 + 1e-12)) * np.cos(arg)
    dcore_dphi = np.cos(arg)

    return core, dcore_db, dcore_dP, dcore_dphi

def alpha_cyc(t, mu, b, P, phi):
    # clipped to avoid exp() overflow when the optimiser/MCMC explores large
    # mu together with the b*t drift term over a long time baseline
    return np.exp(np.clip(mu * shared_core(t, b, P, phi), -50.0, 50.0))

def alpha_cyc_grad(t, mu, b, P, phi):
    fc = shared_core(t, b, P, phi)
    dcore_db, dcore_dP, dcore_dphi = shared_core_grads(t, b, P, phi)
    exponent = mu * fc
    clipped_exponent = np.clip(exponent, -50.0, 50.0)
    a = np.exp(clipped_exponent)
    active = (exponent == clipped_exponent).astype(float)

    return {
        'mu': a * fc * active,
        'b': a * mu * dcore_db * active,
        'P': a * mu * dcore_dP * active,
        'phi': a * mu * dcore_dphi * active,
    }

# RV model
def model_rv(x, a1, c1, b, P, phi):
    return a1 + c1 * shared_core(x, b, P, phi)

# RHK model
def model_rhk(x, a2, c2, b, P, phi):
    return a2 + c2 * shared_core(x, b, P, phi)

# Joint model for curve_fit
def joint_model(x_concat,
                a1, c1,
                a2, c2,
                b, P, phi):

    n = len(x_concat) // 2

    x1 = x_concat[:n]
    x2 = x_concat[n:]

    y1 = model_rv(x1, a1, c1, b, P, phi)
    y2 = model_rhk(x2, a2, c2, b, P, phi)

    return np.concatenate([y1, y2])

def fit_cycle(t_full, y_full, series_index, b0=0, P0=4000, phi0=0, print_results=False, plot=False, output_name='cycle_fit.png', return_fit=False):
    x = t_full[series_index[0]]  # same for both series, but just take from one
    x_concat = np.concatenate([t_full[series_index[0]], t_full[series_index[1]]])
    y_rv = y_full[series_index[0]]
    y_rhk = y_full[series_index[1]]
    y_concat = np.concatenate([y_rv, y_rhk])

    a1_0 = np.mean(y_rv)
    c1_0 = np.std(y_rv)

    a2_0 = np.mean(y_rhk)
    c2_0 = np.std(y_rhk)

    p0 = [
        a1_0, c1_0,
        a2_0, c2_0,
        b0, P0, phi0
    ]


    params, covar = curve_fit(
        joint_model,
        x_concat,
        y_concat,
        p0=p0,
        maxfev=20000
    )

    (
        a1, c1,
        a2, c2,
        b, P, phi
    ) = params

    rv_fit = model_rv(x, a1, c1, b, P, phi)
    rhk_fit = model_rhk(x, a2, c2, b, P, phi)

    if print_results == True:
        print("Shared parameters:")
        print(f"b    = {b}")
        print(f"P    = {P}")
        print(f"phi  = {phi}")

    if plot == True:

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# RV panel
        axs[0].plot(x, y_rv, '.', alpha=0.4, markersize=2, label='RV')
        axs[0].plot(x, rv_fit, linewidth=3, label='RV fit')
        axs[0].set_ylabel("RV")
        axs[0].legend()

# RHK panel
        axs[1].plot(x, y_rhk, '.', alpha=0.4, markersize=2, label='RHK')
        axs[1].plot(x, rhk_fit, linewidth=3, label='RHK fit')
        axs[1].set_ylabel("RHK")
        axs[1].set_xlabel("Time")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(output_name)

        # plt.figure(figsize=(12,6))

        # plt.plot(x, y_rv, '.', alpha=0.4, markersize=2, label='RV')
        # plt.plot(x, rv_fit, linewidth=3, label='RV fit')
        # plt.legend()
        # # plt.show()
        # plt.savefig(f'rv_{output_name}')


        # plt.figure(figsize=(12,6))
        # plt.plot(x, y_rhk, '.', alpha=0.4, markersize=2, label='RHK')
        # plt.plot(x, rhk_fit, linewidth=3, label='RHK fit')

        # plt.legend()
        # # plt.show()
        # plt.savefig(f'rhk_{output_name}')

    if return_fit == True:
        return rv_fit, rhk_fit, params

    return rv_fit, rhk_fit


def get_opt_params(C):
    params_inds = [k for k, key in enumerate(C.param) if key != 'rot.sig' and key != "rot.beta_1"]
    params = [C.param[k] for k in params_inds]
    return params, params_inds

def get_opt_params_nonstat(C):
    params_inds = [k for k, key in enumerate(C.param) if key != 'rot.qp_sig' and key != "rot.qp_beta_1"]
    params = [C.param[k] for k in params_inds]
    return params, params_inds

def negloglike_nocyc(theta, t_full, y_full, series_index,C, rv_std = 1.0, inject_planet=False):

  params, params_inds = get_opt_params(C)

  C.set_param(theta[:len(params)], params)

  y_model = y_full.copy()
  if inject_planet == True:
    y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], theta[10], theta[11], theta[12]) )/rv_std

  y_model[series_index[0]] -=  theta[8]
  y_model[series_index[1]] -=  theta[9]

  # gradient§
  nll = -C.loglike(y_model)

  lg = C.loglike_grad()
  dL_dy = np.asarray(lg[0]).reshape(-1)      # shape (N_total,)
  dL_dparams = np.asarray(lg[1]).reshape(-1) # shape (12,)

  base_grad = - dL_dparams[params_inds]

  # gradients wrt delta (additive offsets)
  grad_delta_0 = np.sum(dL_dy[series_index[0]])
  grad_delta_1 = np.sum(dL_dy[series_index[1]])


  if inject_planet == True:
    argument = 2 * np.pi * t_full[series_index[0]] / (theta[10]+0.000001)
    grad_planet_p = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(-theta[11] * 2 * np.pi * t_full[series_index[0]] / ((theta[10]+0.000001)**2) * np.cos(argument)+theta[12]*2*np.pi*t_full[series_index[0]]/((theta[10]+0.000001)**2)*np.sin(argument)))
    grad_planet_a = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(np.sin(argument)))
    grad_planet_b = np.sum(dL_dy[series_index[0]] * (1 / rv_std) * (np.cos(argument)))
    nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_delta_0, grad_delta_1, grad_planet_p, grad_planet_a, grad_planet_b])])
  else:
    nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_delta_0, grad_delta_1])])
  # nll_grad = -C.loglike_grad()[1][fitted]

  return (nll, nll_grad)



def negloglike_cyc(theta, t_full, y_full, series_index,C, rv_std = 1.0, inject_planet=False):

  params, params_inds = get_opt_params(C)

  C.set_param(theta[:len(params)], params)

  y_model = y_full.copy()
  if inject_planet == True:
    y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], theta[15], theta[16], theta[17]) )/rv_std

  core0, dcore0_db, dcore0_dP, dcore0_dphi = shared_core_and_grads(t_full[series_index[0]], theta[8], theta[9], theta[10])
  core1, dcore1_db, dcore1_dP, dcore1_dphi = shared_core_and_grads(t_full[series_index[1]], theta[8], theta[9], theta[10])
  a0, a1 = theta[11], theta[12]
  d0, d1 = theta[13], theta[14]
  y_model[series_index[0]] -=  a0*core0 + d0
  y_model[series_index[1]] -=  a1*core1 + d1

  # gradient§
  nll = -C.loglike(y_model)

  lg = C.loglike_grad()
  dL_dy = np.asarray(lg[0]).reshape(-1)      # shape (N_total,)
  dL_dparams = np.asarray(lg[1]).reshape(-1) # shape (12,)

  base_grad = - dL_dparams[params_inds]

  # gradients wrt delta (additive offsets)
  grad_delta_0 = np.sum(dL_dy[series_index[0]])
  grad_delta_1 = np.sum(dL_dy[series_index[1]])

  grad_amp_0 = np.sum(dL_dy[series_index[0]] * core0)
  grad_amp_1 = np.sum(dL_dy[series_index[1]] * core1)

  grad_b = (
        np.sum(dL_dy[series_index[0]] * a0 * dcore0_db) +
        np.sum(dL_dy[series_index[1]] * a1 * dcore1_db))

  grad_P = (
        np.sum(dL_dy[series_index[0]] * a0 * dcore0_dP) +
        np.sum(dL_dy[series_index[1]] * a1 * dcore1_dP))

  grad_phi = (
        np.sum(dL_dy[series_index[0]] * a0 * dcore0_dphi) +
        np.sum(dL_dy[series_index[1]] * a1 * dcore1_dphi))


  if inject_planet == True:
    argument = 2 * np.pi * t_full[series_index[0]] / (theta[15]+0.000001)
    grad_planet_p = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(-theta[16] * 2 * np.pi * t_full[series_index[0]] / ((theta[15]+0.000001)**2) * np.cos(argument)+theta[17]*2*np.pi*t_full[series_index[0]]/((theta[15]+0.000001)**2)*np.sin(argument)))
    grad_planet_a = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(np.sin(argument)))
    grad_planet_b = np.sum(dL_dy[series_index[0]] * (1 / rv_std) * (np.cos(argument)))
    nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_b, grad_P, grad_phi,grad_amp_0, grad_amp_1,grad_delta_0, grad_delta_1,grad_planet_p, grad_planet_a, grad_planet_b])])
  else:
    nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_b, grad_P, grad_phi,grad_amp_0, grad_amp_1,grad_delta_0, grad_delta_1])])
  # nll_grad = -C.loglike_grad()[1][fitted]

  return (nll, nll_grad)


def negloglike_nonstat(theta, t_full, y_full, series_index,C, rv_std = 1.0, inject_planet=False):
  """
  Like negloglike_cyc, but b/P/phi are spleaf kernel hyper-parameters
  (living inside the NonStationaryKernel envelope that modulates the
  rotation kernel's amplitude) instead of theta-only mean parameters, so
  the same magnetic-cycle b/P/phi drive both the covariance envelope and
  the cycle mean term. Their gradient therefore needs the mean-path
  contribution added on top of what C.loglike_grad() already returns via
  the kernel path (alpha_cyc_grad).
  """

  params, params_inds = get_opt_params_nonstat(C)
  n = len(params)

  C.set_param(theta[:n], params)

  b, P, phi = C.get_param(['rot.nonstat_b', 'rot.nonstat_P', 'rot.nonstat_phi'])

  y_model = y_full.copy()
  if inject_planet == True:
    y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], theta[n+4], theta[n+5], theta[n+6]) )/rv_std

  core0, dcore0_db, dcore0_dP, dcore0_dphi = shared_core_and_grads(t_full[series_index[0]], b, P, phi)
  core1, dcore1_db, dcore1_dP, dcore1_dphi = shared_core_and_grads(t_full[series_index[1]], b, P, phi)
  a0, a1 = theta[n], theta[n+1]
  d0, d1 = theta[n+2], theta[n+3]
  y_model[series_index[0]] -=  a0*core0 + d0
  y_model[series_index[1]] -=  a1*core1 + d1

  # gradient§
  nll = -C.loglike(y_model)

  lg = C.loglike_grad()
  dL_dy = np.asarray(lg[0]).reshape(-1)      # shape (N_total,)
  dL_dparams = np.asarray(lg[1]).reshape(-1) # shape (n,)

  base_grad = - dL_dparams[params_inds]

  idx_b = params.index('rot.nonstat_b')
  idx_P = params.index('rot.nonstat_P')
  idx_phi = params.index('rot.nonstat_phi')

  # b, P, phi also enter the mean term directly, on top of the covariance
  # path already captured in base_grad via alpha_cyc_grad
  base_grad[idx_b] += (
        np.sum(dL_dy[series_index[0]] * a0 * dcore0_db) +
        np.sum(dL_dy[series_index[1]] * a1 * dcore1_db))

  base_grad[idx_P] += (
        np.sum(dL_dy[series_index[0]] * a0 * dcore0_dP) +
        np.sum(dL_dy[series_index[1]] * a1 * dcore1_dP))

  base_grad[idx_phi] += (
        np.sum(dL_dy[series_index[0]] * a0 * dcore0_dphi) +
        np.sum(dL_dy[series_index[1]] * a1 * dcore1_dphi))

  grad_a0 = np.sum(dL_dy[series_index[0]] * core0)
  grad_a1 = np.sum(dL_dy[series_index[1]] * core1)
  grad_d0 = np.sum(dL_dy[series_index[0]])
  grad_d1 = np.sum(dL_dy[series_index[1]])

  if inject_planet == True:
    argument = 2 * np.pi * t_full[series_index[0]] / (theta[n+4]+0.000001)
    grad_planet_p = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(-theta[n+5] * 2 * np.pi * t_full[series_index[0]] / ((theta[n+4]+0.000001)**2) * np.cos(argument)+theta[n+6]*2*np.pi*t_full[series_index[0]]/((theta[n+4]+0.000001)**2)*np.sin(argument)))
    grad_planet_a = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(np.sin(argument)))
    grad_planet_b = np.sum(dL_dy[series_index[0]] * (1 / rv_std) * (np.cos(argument)))
    nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_a0, grad_a1, grad_d0, grad_d1, grad_planet_p, grad_planet_a, grad_planet_b])])
  else:
    nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_a0, grad_a1, grad_d0, grad_d1])])

  return (nll, nll_grad)


def optimise_params(t_full, y_full, series_index, C, bounds_list, delta_0 = -0.001, delta_1 = 0.001, planet_p = 40.05, planet_A = 0.0005, planet_B = 0.0, fit_planet = True, change_C = True):

    params, _ = get_opt_params(C)

    x0 = C.get_param(params)

    if fit_planet == True:
        x0 = np.append(x0,[delta_0, delta_1, planet_p, planet_A, planet_B])
        result = fmin_l_bfgs_b(negloglike_nocyc, x0, args=(t_full, y_full, series_index, C, 1.0, True), bounds=bounds_list)
        xbest = result[0]
    else:
        x0 = np.append(x0,[delta_0, delta_1])
        result = fmin_l_bfgs_b(negloglike_nocyc, x0, args=(t_full, y_full, series_index, C, 1.0, False), bounds=bounds_list)
        xbest = result[0]

    if change_C == True:
        C.set_param(xbest[:len(params)], params)

    return xbest, C


def optimise_params_cyc(t_full, y_full, series_index, C, bounds_list, b = 0.0, P = 4000, phi = 0.0, a0 = None, a1 = None, d0 = None, d1 = None, planet_p = 40.05, planet_A = 0.0005, planet_B = 0.0, fit_planet = True, change_C = True):

    if a0 is None:
        a0 = np.std(y_full[series_index[0]])
    if a1 is None:
        a1 = np.std(y_full[series_index[1]])
    if d0 is None:
        d0 = np.mean(y_full[series_index[0]])
    if d1 is None:
        d1 = np.mean(y_full[series_index[1]])

    params, _ = get_opt_params(C)

    x0 = C.get_param(params)

    if fit_planet == True:
        x0 = np.append(x0,[b, P, phi, a0, a1, d0, d1, planet_p, planet_A, planet_B])
        result = fmin_l_bfgs_b(negloglike_cyc, x0, args=(t_full, y_full, series_index, C, 1.0, True), bounds=bounds_list)
        xbest = result[0]
    else:
        x0 = np.append(x0,[b, P, phi, a0, a1, d0, d1])
        result = fmin_l_bfgs_b(negloglike_cyc, x0, args=(t_full, y_full, series_index, C, 1.0, False), bounds=bounds_list)
        xbest = result[0]

    if change_C == True:
        C.set_param(xbest[:len(params)], params)

    return xbest, C


def optimise_params_nonstat(t_full, y_full, series_index, C, bounds_list, a0 = None, a1 = None, d0 = None, d1 = None, planet_p = 40.05, planet_A = 0.0005, planet_B = 0.0, fit_planet = True, change_C = True):

    if a0 is None:
        a0 = np.std(y_full[series_index[0]])
    if a1 is None:
        a1 = np.std(y_full[series_index[1]])
    if d0 is None:
        d0 = np.mean(y_full[series_index[0]])
    if d1 is None:
        d1 = np.mean(y_full[series_index[1]])

    params, _ = get_opt_params_nonstat(C)

    x0 = C.get_param(params)

    if fit_planet == True:
        x0 = np.append(x0,[a0, a1, d0, d1, planet_p, planet_A, planet_B])
        result = fmin_l_bfgs_b(negloglike_nonstat, x0, args=(t_full, y_full, series_index, C, 1.0, True), bounds=bounds_list)
        xbest = result[0]
    else:
        x0 = np.append(x0,[a0, a1, d0, d1])
        result = fmin_l_bfgs_b(negloglike_nonstat, x0, args=(t_full, y_full, series_index, C, 1.0, False), bounds=bounds_list)
        xbest = result[0]

    if change_C == True:
        C.set_param(xbest[:len(params)], params)

    return xbest, C


def log_prior(theta, bounds_list, planet = True):
    if planet == True:
        rv_jit, rhk_jit, rot_P0, rot_rho, rot_eta, rot_alpha_0, rot_alpha_1, rot_beta_0, delta_0, delta_1, planet_period, planet_A, planet_B = theta
        if bounds_list[0][0] < rv_jit < bounds_list[0][1] and bounds_list[1][0] < rhk_jit < bounds_list[1][1] and bounds_list[2][0] < rot_P0 < bounds_list[2][1] and bounds_list[3][0] < rot_rho < bounds_list[3][1] and bounds_list[4][0] < rot_eta < bounds_list[4][1] and bounds_list[5][0] < rot_alpha_0 < bounds_list[5][1] and bounds_list[6][0] < rot_alpha_1 < bounds_list[6][1] and bounds_list[7][0] < rot_beta_0 < bounds_list[7][1] and bounds_list[8][0] < delta_0 < bounds_list[8][1] and bounds_list[9][0] < delta_1 < bounds_list[9][1] and bounds_list[10][0] < planet_period < bounds_list[10][1] and bounds_list[11][0] < planet_A < bounds_list[11][1] and bounds_list[12][0] < planet_B < bounds_list[12][1]:
            return -np.log(planet_A) - np.log(planet_B)
        # planet_phi = planet_phi % (2*np.pi)
        return -np.inf

    else:
        rv_jit, rhk_jit, rot_P0, rot_rho, rot_eta, rot_alpha_0, rot_alpha_1, rot_beta_0, delta_0, delta_1 = theta
        if bounds_list[0][0] < rv_jit < bounds_list[0][1] and bounds_list[1][0] < rhk_jit < bounds_list[1][1] and bounds_list[2][0] < rot_P0 < bounds_list[2][1] and bounds_list[3][0] < rot_rho < bounds_list[3][1] and bounds_list[4][0] < rot_eta < bounds_list[4][1] and bounds_list[5][0] < rot_alpha_0 < bounds_list[5][1] and bounds_list[6][0] < rot_alpha_1 < bounds_list[6][1] and bounds_list[7][0] < rot_beta_0 < bounds_list[7][1] and bounds_list[8][0] < delta_0 < bounds_list[8][1] and bounds_list[9][0] < delta_1 < bounds_list[9][1]:
            return 0
        return -np.inf


def log_prior_cyc(theta, bounds_list, planet = True):
    if planet == True:
        rv_jit, rhk_jit, rot_P0, rot_rho, rot_eta, rot_alpha_0, rot_alpha_1, rot_beta_0, b, P, phi, a0, a1, d0, d1, planet_period, planet_A, planet_B = theta
        if bounds_list[0][0] < rv_jit < bounds_list[0][1] and bounds_list[1][0] < rhk_jit < bounds_list[1][1] and bounds_list[2][0] < rot_P0 < bounds_list[2][1] and bounds_list[3][0] < rot_rho < bounds_list[3][1] and bounds_list[4][0] < rot_eta < bounds_list[4][1] and bounds_list[5][0] < rot_alpha_0 < bounds_list[5][1] and bounds_list[6][0] < rot_alpha_1 < bounds_list[6][1] and bounds_list[7][0] < rot_beta_0 < bounds_list[7][1] and bounds_list[8][0] < b < bounds_list[8][1] and bounds_list[9][0] < P < bounds_list[9][1] and bounds_list[10][0] < phi < bounds_list[10][1] and bounds_list[11][0] < a0 < bounds_list[11][1] and bounds_list[12][0] < a1 < bounds_list[12][1] and bounds_list[13][0] < d0 < bounds_list[13][1] and bounds_list[14][0] < d1 < bounds_list[14][1] and bounds_list[15][0] < planet_period < bounds_list[15][1] and bounds_list[16][0] < planet_A < bounds_list[16][1] and bounds_list[17][0] < planet_B < bounds_list[17][1]:
            return -np.log(planet_A) - np.log(planet_B)
        # planet_phi = planet_phi % (2*np.pi)
        return -np.inf

    else:
        rv_jit, rhk_jit, rot_P0, rot_rho, rot_eta, rot_alpha_0, rot_alpha_1, rot_beta_0, b, P, phi, a0, a1, d0, d1 = theta
        if bounds_list[0][0] < rv_jit < bounds_list[0][1] and bounds_list[1][0] < rhk_jit < bounds_list[1][1] and bounds_list[2][0] < rot_P0 < bounds_list[2][1] and bounds_list[3][0] < rot_rho < bounds_list[3][1] and bounds_list[4][0] < rot_eta < bounds_list[4][1] and bounds_list[5][0] < rot_alpha_0 < bounds_list[5][1] and bounds_list[6][0] < rot_alpha_1 < bounds_list[6][1] and bounds_list[7][0] < rot_beta_0 < bounds_list[7][1] and bounds_list[8][0] < b < bounds_list[8][1] and bounds_list[9][0] < P < bounds_list[9][1] and bounds_list[10][0] < phi < bounds_list[10][1] and bounds_list[11][0] < a0 < bounds_list[11][1] and bounds_list[12][0] < a1 < bounds_list[12][1] and bounds_list[13][0] < d0 < bounds_list[13][1] and bounds_list[14][0] < d1 < bounds_list[14][1]:
            return 0
        return -np.inf


def log_prior_nonstat(theta, bounds_list, planet = True):
    if planet == True:
        rv_jit, rhk_jit, mu, b, P, phi, rot_P0, rot_rho, rot_eta, rot_alpha_0, rot_alpha_1, rot_beta_0, a0, a1, d0, d1, planet_period, planet_A, planet_B = theta
        if bounds_list[0][0] < rv_jit < bounds_list[0][1] and bounds_list[1][0] < rhk_jit < bounds_list[1][1] and bounds_list[2][0] < mu < bounds_list[2][1] and bounds_list[3][0] < b < bounds_list[3][1] and bounds_list[4][0] < P < bounds_list[4][1] and bounds_list[5][0] < phi < bounds_list[5][1] and bounds_list[6][0] < rot_P0 < bounds_list[6][1] and bounds_list[7][0] < rot_rho < bounds_list[7][1] and bounds_list[8][0] < rot_eta < bounds_list[8][1] and bounds_list[9][0] < rot_alpha_0 < bounds_list[9][1] and bounds_list[10][0] < rot_alpha_1 < bounds_list[10][1] and bounds_list[11][0] < rot_beta_0 < bounds_list[11][1] and bounds_list[12][0] < a0 < bounds_list[12][1] and bounds_list[13][0] < a1 < bounds_list[13][1] and bounds_list[14][0] < d0 < bounds_list[14][1] and bounds_list[15][0] < d1 < bounds_list[15][1] and bounds_list[16][0] < planet_period < bounds_list[16][1] and bounds_list[17][0] < planet_A < bounds_list[17][1] and bounds_list[18][0] < planet_B < bounds_list[18][1]:
            return -np.log(planet_A) - np.log(planet_B)
        return -np.inf

    else:
        rv_jit, rhk_jit, mu, b, P, phi, rot_P0, rot_rho, rot_eta, rot_alpha_0, rot_alpha_1, rot_beta_0, a0, a1, d0, d1 = theta
        if bounds_list[0][0] < rv_jit < bounds_list[0][1] and bounds_list[1][0] < rhk_jit < bounds_list[1][1] and bounds_list[2][0] < mu < bounds_list[2][1] and bounds_list[3][0] < b < bounds_list[3][1] and bounds_list[4][0] < P < bounds_list[4][1] and bounds_list[5][0] < phi < bounds_list[5][1] and bounds_list[6][0] < rot_P0 < bounds_list[6][1] and bounds_list[7][0] < rot_rho < bounds_list[7][1] and bounds_list[8][0] < rot_eta < bounds_list[8][1] and bounds_list[9][0] < rot_alpha_0 < bounds_list[9][1] and bounds_list[10][0] < rot_alpha_1 < bounds_list[10][1] and bounds_list[11][0] < rot_beta_0 < bounds_list[11][1] and bounds_list[12][0] < a0 < bounds_list[12][1] and bounds_list[13][0] < a1 < bounds_list[13][1] and bounds_list[14][0] < d0 < bounds_list[14][1] and bounds_list[15][0] < d1 < bounds_list[15][1]:
            return 0
        return -np.inf


def log_probability(theta, t_full, y_full, series_index, C, bounds_list, planet = True):
    # params, params_inds = get_opt_params(C)
    lp = log_prior(theta, bounds_list, planet)
    if not np.isfinite(lp):
        return -np.inf
    if planet == True:
        return lp + -1*negloglike_nocyc(theta, t_full, y_full, series_index, C, 1.0, True)[0]
    else:
        return lp + -1*negloglike_nocyc(theta, t_full, y_full, series_index, C, 1.0, False)[0]


def log_probability_cyc(theta, t_full, y_full, series_index, C, bounds_list, planet = True):
    # params, params_inds = get_opt_params(C)
    lp = log_prior_cyc(theta, bounds_list, planet)
    if not np.isfinite(lp):
        return -np.inf
    if planet == True:
        return lp + -1*negloglike_cyc(theta, t_full, y_full, series_index, C, 1.0, True)[0]
    else:
        return lp + -1*negloglike_cyc(theta, t_full, y_full, series_index, C, 1.0, False)[0]


def log_probability_nonstat(theta, t_full, y_full, series_index, C, bounds_list, planet = True):
    lp = log_prior_nonstat(theta, bounds_list, planet)
    if not np.isfinite(lp):
        return -np.inf
    if planet == True:
        return lp + -1*negloglike_nonstat(theta, t_full, y_full, series_index, C, 1.0, True)[0]
    else:
        return lp + -1*negloglike_nonstat(theta, t_full, y_full, series_index, C, 1.0, False)[0]

def run_emcee(t_full, y_full, series_index, C, x0, bounds_list, run_length = 3000, planet = True):

    print(log_prior(x0, bounds_list, planet))

    ndim = len(x0)
    pos = x0 + 1e-5 * np.random.randn(ndim*3, ndim)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(t_full, y_full, series_index, C, bounds_list, planet)
    )
    sampler.run_mcmc(pos, run_length, progress=True);
    return sampler


def run_emcee_cyc(t_full, y_full, series_index, C, x0, bounds_list, run_length = 3000, planet = True):

    print(log_prior_cyc(x0, bounds_list, planet))

    ndim = len(x0)
    pos = x0 + 1e-5 * np.random.randn(ndim*3, ndim)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_cyc, args=(t_full, y_full, series_index, C, bounds_list, planet)
    )
    sampler.run_mcmc(pos, run_length, progress=True);
    return sampler


def run_emcee_nonstat(t_full, y_full, series_index, C, x0, bounds_list, run_length = 3000, planet = True):

    print(log_prior_nonstat(x0, bounds_list, planet))

    ndim = len(x0)
    pos = x0 + 1e-5 * np.random.randn(ndim*3, ndim)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_nonstat, args=(t_full, y_full, series_index, C, bounds_list, planet)
    )
    sampler.run_mcmc(pos, run_length, progress=True);
    return sampler


def plot_fit(t_full, y_full, yerr_full, series_index, C, xbest, rv_std = 1.0, output_name = 'fit_plot.png', return_residuals=True, inject_planet=True):
    print("xbest:", xbest)
    tsmooth = np.linspace(np.min(t_full), np.max(t_full), 1000)
    _, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    mus = []
    vars = []

    for k in range(2):
        C.kernel['rot'].set_conditional_coef(series_id=k)

        y_model = y_full.copy()

        if inject_planet == True:
            y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], xbest[10], xbest[11], xbest[12]))/rv_std

        y_model[series_index[0]] -= xbest[8]
        y_model[series_index[1]] -= xbest[9]


        mu, var = C.conditional(y_model, tsmooth, calc_cov='diag')
        mu_res, _ = C.conditional(y_model, t_full[series_index[k]], calc_cov='diag')

        mus.append(mu)
        vars.append(var)

        ax = axs[k]
        if k ==0 :
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
            if inject_planet == True:
                ax.plot(t_full[series_index[k]], planet_injection(t_full[series_index[k]], xbest[10], xbest[11], xbest[12])/rv_std, 'r', label='injected planet')
        if k == 1:
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
        ax.fill_between(tsmooth,
            mu - np.sqrt(var),
            mu + np.sqrt(var),
            color='g',
            alpha=0.5)
        ax.plot(tsmooth, mu, 'g', label='predict.')
        if k == 0:
            ax.set_ylabel("RV")
        if k == 1:
            ax.set_ylabel('RHK')

        if return_residuals==True:
            if k == 0:
                res_rv = y_model[series_index[k]] - mu_res
            else:
                res_rhk = y_model[series_index[k]] - mu_res

    ax.set_xlabel('$t$')
    axs[0].legend()
    plt.savefig(output_name)
    if return_residuals==True:
        return tsmooth, mus, res_rv, res_rhk
    else:
        return tsmooth, mus


def plot_fit_cyc(t_full, y_full, yerr_full, series_index, C, xbest, rv_std = 1.0, output_name = 'fit_plot.png', return_residuals=True, inject_planet=True):

    tsmooth = np.linspace(np.min(t_full), np.max(t_full), 1000)
    _, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    mus = []
    vars = []

    for k in range(2):
        C.kernel['rot'].set_conditional_coef(series_id=k)

        y_model = y_full.copy()
        print("xbest:", xbest)
        if inject_planet == True:
            y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], xbest[15], xbest[16], xbest[17]))/rv_std

        core0, _,_,_ = shared_core_and_grads(t_full[series_index[0]], xbest[8], xbest[9], xbest[10])
        core1, _,_,_ = shared_core_and_grads(t_full[series_index[1]], xbest[8], xbest[9], xbest[10])
        a0, a1 = xbest[11], xbest[12]
        d0, d1 = xbest[13], xbest[14]
        y_model[series_index[0]] -=  a0*core0 + d0
        y_model[series_index[1]] -=  a1*core1 + d1

        mu, var = C.conditional(y_model, tsmooth, calc_cov='diag')
        mu_res, _ = C.conditional(y_model, t_full[series_index[k]], calc_cov='diag')

        mus.append(mu)
        vars.append(var)

        ax = axs[k]
        if k ==0 :
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
            if inject_planet == True:
                ax.plot(t_full[series_index[k]], planet_injection(t_full[series_index[k]], xbest[15], xbest[16], xbest[17])/rv_std, 'r', label='injected planet')
        if k == 1:
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
        ax.fill_between(tsmooth,
            mu - np.sqrt(var),
            mu + np.sqrt(var),
            color='g',
            alpha=0.5)
        ax.plot(tsmooth, mu, 'g', label='predict.')
        if k == 0:
            ax.set_ylabel("RV")
        if k == 1:
            ax.set_ylabel('RHK')

        if return_residuals==True:
            if k == 0:
                res_rv = y_model[series_index[k]] - mu_res
            else:
                res_rhk = y_model[series_index[k]] - mu_res

    ax.set_xlabel('$t$')
    axs[0].legend()
    plt.savefig(output_name)
    if return_residuals==True:
        return tsmooth, mus, res_rv, res_rhk
    else:
        return tsmooth, mus


def plot_fit_nonstat(t_full, y_full, yerr_full, series_index, C, xbest, rv_std = 1.0, output_name = 'fit_plot.png', return_residuals=True, inject_planet=True):

    params, _ = get_opt_params_nonstat(C)
    n = len(params)
    idx_b = params.index('rot.nonstat_b')
    idx_P = params.index('rot.nonstat_P')
    idx_phi = params.index('rot.nonstat_phi')

    tsmooth = np.linspace(np.min(t_full), np.max(t_full), 1000)
    _, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    mus = []
    vars = []

    for k in range(2):
        C.kernel['rot']._kernel2.set_conditional_coef(series_id=k)

        y_model = y_full.copy()
        print("xbest:", xbest)
        if inject_planet == True:
            y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], xbest[n+4], xbest[n+5], xbest[n+6]))/rv_std

        b, P, phi = xbest[idx_b], xbest[idx_P], xbest[idx_phi]
        core0, _,_,_ = shared_core_and_grads(t_full[series_index[0]], b, P, phi)
        core1, _,_,_ = shared_core_and_grads(t_full[series_index[1]], b, P, phi)
        a0, a1 = xbest[n], xbest[n+1]
        d0, d1 = xbest[n+2], xbest[n+3]
        y_model[series_index[0]] -=  a0*core0 + d0
        y_model[series_index[1]] -=  a1*core1 + d1

        mu, var = C.conditional(y_model, tsmooth, calc_cov='diag')
        mu_res, _ = C.conditional(y_model, t_full[series_index[k]], calc_cov='diag')

        mus.append(mu)
        vars.append(var)

        ax = axs[k]
        if k ==0 :
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
            if inject_planet == True:
                ax.plot(t_full[series_index[k]], planet_injection(t_full[series_index[k]], xbest[n+4], xbest[n+5], xbest[n+6])/rv_std, 'r', label='injected planet')
        if k == 1:
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
        ax.fill_between(tsmooth,
            mu - np.sqrt(var),
            mu + np.sqrt(var),
            color='g',
            alpha=0.5)
        ax.plot(tsmooth, mu, 'g', label='predict.')
        if k == 0:
            ax.set_ylabel("RV")
        if k == 1:
            ax.set_ylabel('RHK')

        if return_residuals==True:
            if k == 0:
                res_rv = y_model[series_index[k]] - mu_res
            else:
                res_rhk = y_model[series_index[k]] - mu_res

    ax.set_xlabel('$t$')
    axs[0].legend()
    plt.savefig(output_name)
    if return_residuals==True:
        return tsmooth, mus, res_rv, res_rhk
    else:
        return tsmooth, mus


def plot_kernel_draw(t_full, C, xbest, sig=1.0, n_points=2000, output_name='kernel_draw.png'):
    """
    Unconditional sample draw from the optimised rotation kernel alone
    (SimpleProductKernel(NonStationaryKernel, MultiSeriesKernel(MEPKernel)),
    negligible nugget instead of the fitted jitter/error), so the cycle's
    amplitude modulation is visible without being masked by measurement noise.
    """
    params, _ = get_opt_params_nonstat(C)
    idx_mu = params.index('rot.nonstat_mu')
    idx_b = params.index('rot.nonstat_b')
    idx_P = params.index('rot.nonstat_P')
    idx_phi = params.index('rot.nonstat_phi')
    idx_prot = params.index('rot.qp_P')
    idx_rho = params.index('rot.qp_rho')
    idx_eta = params.index('rot.qp_eta')
    idx_alpha0 = params.index('rot.qp_alpha_0')
    idx_alpha1 = params.index('rot.qp_alpha_1')
    idx_beta0 = params.index('rot.qp_beta_0')

    mu, b, P, phi = xbest[idx_mu], xbest[idx_b], xbest[idx_P], xbest[idx_phi]
    prot, rho, eta = xbest[idx_prot], xbest[idx_rho], xbest[idx_eta]
    alpha_0, alpha_1, beta_0 = xbest[idx_alpha0], xbest[idx_alpha1], xbest[idx_beta0]

    tsmooth = np.linspace(np.min(t_full), np.max(t_full), n_points)
    t_grid, yerr_grid, series_index_grid = cov.merge_series(
        [tsmooth, tsmooth], [np.full(n_points, 1e-6), np.full(n_points, 1e-6)]
    )

    C_draw = cov.Cov(
        t_grid,
        err=term.Error(yerr_grid),
        rot=term.SimpleProductKernel(
            nonstat=NonStationaryKernel(alpha_cyc, alpha_cyc_grad, mu=mu, b=b, P=P, phi=phi),
            qp=term.MultiSeriesKernel(term.MEPKernel(sig, prot, rho, eta), series_index_grid,
                                       np.array([alpha_0, alpha_1]), np.array([beta_0, 0.0])),
        ),
    )
    y_draw = C_draw.sample()

    _, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
    axs[0].plot(tsmooth, y_draw[series_index_grid[0]], color='C0')
    axs[0].set_ylabel('RV draw')
    axs[1].plot(tsmooth, y_draw[series_index_grid[1]], color='C1')
    axs[1].set_ylabel('RHK draw')
    axs[1].set_xlabel('$t$')
    plt.tight_layout()
    plt.savefig(output_name)


def plot_chains(sampler, C, planet = True, output_name = 'chains_plot.png'):
    params , _ = get_opt_params(C)
    if planet == True:
        ig, axes = plt.subplots(len(params) + 6, figsize=(10, 12), sharex=True)
        labels = params + ['delta_0', 'delta_1', 'planet_p', 'planet_A', 'planet_B']
    else:
        ig, axes = plt.subplots(len(params) + 3, figsize=(10, 12), sharex=True)
        labels = params + ['delta_0', 'delta_1']

    samples = sampler.get_chain()

    for i in range(len(axes)-1):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");

    logp = sampler.get_log_prob(flat=False)
    axes[-1].plot(logp)
    plt.savefig(output_name)


def plot_corner(sampler, C, discard = 1000, planet = True, output_name = 'corner_plot.png'):

    flat_samples = sampler.get_chain(discard=discard, thin=1, flat=True)
    params, _ = get_opt_params(C)
    if planet == True:
        labels = params + ['delta_0', 'delta_1', 'planet_p', 'planet_A', 'planet_B']
    else:
        labels = params + ['delta_0', 'delta_1']

    # true_vals = [0.0, 0.0, prot, Q, stds[0], stds[1], stds[0], 8000, np.pi, 0.28, gamma_0, gamma_1, delta_0, delta_1, 100.0, 0.5, 0.5]

    fig = corner.corner(
        flat_samples, labels=labels, show_titles=True
    );
    plt.savefig(output_name)


def plot_corner_nonstat(sampler, C, discard = 1000, planet = True, output_name = 'corner_plot.png'):

    flat_samples = sampler.get_chain(discard=discard, thin=1, flat=True)
    params, _ = get_opt_params_nonstat(C)
    if planet == True:
        labels = params + ['a0', 'a1', 'delta_0', 'delta_1', 'planet_p', 'planet_A', 'planet_B']
    else:
        labels = params + ['a0', 'a1', 'delta_0', 'delta_1']

    fig = corner.corner(
        flat_samples, labels=labels, show_titles=True
    );
    plt.savefig(output_name)


def get_MAP_params(sampler, burn=1500, thin=1):

    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)   # shape (n_samples, ndim)
    flat_logp    = sampler.get_log_prob(discard=burn, thin=thin, flat=True) # shape (n_samples,)

    finite_mask = np.isfinite(flat_logp)
    flat_samples = flat_samples[finite_mask]
    flat_logp    = flat_logp[finite_mask]

    map_idx = np.argmax(flat_logp)
    map_sample = flat_samples[map_idx]
    map_logp = flat_logp[map_idx]

    print("MAP log-posterior:", map_logp)
    print("MAP sample (theta):", map_sample)
    return map_sample, map_logp

def period_guess(t_full, y_full, yerr_full, series_index, PMIN = 1.1, PMAX = 5000.0, MAX_FAP = 1e-5, MAX_NPL = 6, plot = True, output_name = 'periodogram.png'):
    x = t_full[series_index[0]]
    y = y_full[series_index[0]]
    y_sig = yerr_full[series_index[0]]
    z = y_full[series_index[1]]

    znorm = z / np.std(z) # now has unit variance
    X = [znorm]
    xreg = np.linspace(x.min(), x.max(), 5000)
    zreg = interp1d(x, znorm)(xreg)
    Xreg = [zreg]

    Xm = np.array(X).T
    reg = LinearRegression().fit(Xm, y, sample_weight = 1./y_sig**2)
    y_fit = reg.predict(Xm)
    Xregm = np.array(Xreg).T
    y_fit_reg = reg.predict(Xregm)
    resid = np.copy(y)

    if plot == True:
        fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 12))
        ax[1].plot(x, y, 'k.', ms=4)
        ax[1].plot(xreg, y_fit_reg, 'C0-', lw = 0.5)

        off = 0
        ax[2].plot(x, resid - off, f'C0.', ms=2)
        off += 1.1 * (resid.max() - resid.min())

    npl = 0
    i = 0

    pers_notr = []
    faps_notr = []
    while True:
        LS = LombScargle(x, resid, dy = y_sig)
        freq, power = LS.autopower(
            minimum_frequency = 1.0 / PMAX,
            maximum_frequency = 1.0 / PMIN,
            samples_per_peak = 10, nyquist_factor = 10.0)
        imax = np.argmax(power)
        p = 1/freq[imax]
        fap = LS.false_alarm_probability(power[imax])
        pers_notr.append(p)
        faps_notr.append(fap)
        pm = power.max()
        if plot == True:
            ax[0].semilogx(1/freq, power/pm - 1.5*i, f'C{i+1}-', lw = 0.5)
        print(f"initial period guess: {p}, FAP: {fap}")
        if fap >= MAX_FAP:
            break
        if plot == True:
            ax[0].plot(p, 1 - 1.5*i, f'C{i+1}o',ms=5)
            ax[0].text(p, 1 - 1.5*i, f' {p:.3f}',color=f'k')

        X.append(np.sin(2 * np.pi * x / p))
        X.append(np.cos(2 * np.pi * x / p))
        Xm = np.array(X).T
        reg = LinearRegression().fit(Xm, y, sample_weight = 1./y_sig**2)
        y_fit = reg.predict(Xm)
        resid = y - y_fit

        Xreg.append(np.sin(2 * np.pi * xreg / p))
        Xreg.append(np.cos(2 * np.pi * xreg / p))
        Xregm = np.array(Xreg).T
        y_fit_reg = reg.predict(Xregm)

        if plot == True:
            ax[1].plot(xreg, y_fit_reg, f'C{i+1}-',lw=0.5)
            ax[2].plot(x, resid - off, f'C{i+1}.', ms=2)
            off += 1.1 * (resid.max() - resid.min())

        theta = reg.coef_
        sin_coef = theta[-2]
        cos_coef = theta[-1]
        amp = np.sqrt(cos_coef**2 + sin_coef**2)
        phase = np.arctan2(cos_coef, sin_coef)
        ttr = p * (0.5 - phase / (2 * np.pi))

        i += 1
        npl += 1
        print(npl, len(theta), len(X))
        if npl == MAX_NPL:
            print("Maximum number of planets reached")
            break
    if plot == True:
        ax[0].set_xlim(PMIN,PMAX)
        ax[0].set_xlabel('period (days)')
        ax[0].set_ylabel('LS power')
        ax[1].set_xlim(xreg.min(),xreg.max())
        ax[1].set_ylabel('RV (m/s)')
        ax[2].set_xlim(xreg.min(),xreg.max())
        ax[2].set_xlabel('time (days)')
        ax[2].set_ylabel('residuals (m/s)')
        plt.tight_layout()
        plt.savefig(output_name)

    return pers_notr, faps_notr


# def alpha(t, pcyc, phi, k):

#     x = (2.0 * np.pi / pcyc) * (t) + phi

#     fc = np.sin((x) + k * np.sin(x))  # in [-1, 1]

#     return fc



# def alpha_grad(t, pcyc, phi, k):

#     omega = 2.0 * np.pi / pcyc
#     x = omega * (t) + phi

#     inner = x + k * np.sin(x)
#     c_inner = np.cos(inner)
#     d_inner_dx = 1.0 + k * np.cos(x)
#     d_alpha_dinner =  c_inner

#     grads = {}

#     # derivative wrt phi

#     grads['phi'] = d_alpha_dinner * d_inner_dx

#     # derivative wrt pcyc

#     domega_dp = -2.0 * np.pi / (pcyc ** 2)
#     dx_dp = domega_dp * (t)
#     grads['pcyc'] = d_alpha_dinner * d_inner_dx * dx_dp

#     # derivative wrt k

#     d_inner_dk = np.sin(x)
#     grads['k'] = d_alpha_dinner * d_inner_dk

#     return grads


# def negloglike(x, y, C, params, params_inds, t_full, series_index, rv_std = 1.0, inject_planet=False):

#   C.set_param(x[:len(params)], params)
#   fc = alpha(t_full[series_index[0]], x[7],x[8],x[9])
#   # fc_unnorm = (fc - x[10]) / ((1.0 - x[10]) / 2.0) - 1
#   fc_grads = alpha_grad(t_full[series_index[0]] ,x[7],x[8],x[9])
#   y_model = y.copy()
#   y_model[series_index[0]] -= (x[10]*fc + x[12])
#   y_model[series_index[1]] -= (x[11]*fc + x[13])

#   if inject_planet == True:
#       y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], x[14], x[15], x[16]) )/rv_std

#   # gradient§
#   nll = -C.loglike(y_model)

#   lg = C.loglike_grad()
#   dL_dy = np.asarray(lg[0]).reshape(-1)      # shape (N_total,)
#   dL_dparams = np.asarray(lg[1]).reshape(-1) # shape (12,)

#   base_grad = - dL_dparams[params_inds]

#   # gradients wrt gamma (scale of fc)
#   grad_gamma_0 = np.sum(dL_dy[series_index[0]] * fc)
#   grad_gamma_1 = np.sum(dL_dy[series_index[1]] * fc)
#   # gradients wrt delta (additive offsets)
#   grad_delta_0 = np.sum(dL_dy[series_index[0]])
#   grad_delta_1 = np.sum(dL_dy[series_index[1]])

#   grad_pcyc = np.sum(dL_dy[series_index[0]] * x[10] * fc_grads['pcyc']) + np.sum(dL_dy[series_index[1]] * x[11] * fc_grads['pcyc'])
#   grad_phi = np.sum(dL_dy[series_index[0]] * x[10] * fc_grads['phi']) + np.sum(dL_dy[series_index[1]] * x[11] * fc_grads['phi'])
#   grad_k_alpha = np.sum(dL_dy[series_index[0]] * x[10] * fc_grads['k']) + np.sum(dL_dy[series_index[1]] * x[11] * fc_grads['k'])
#   # grad_c = np.sum(dL_dy[series_index[0]] * x[-4] * fc_grads['c'][series_index[0]]) + np.sum(dL_dy[series_index[1]] * x[-3] * fc_grads['c'][series_index[1]])
#   if inject_planet == True:
#     argument = 2 * np.pi * t_full[series_index[0]] / (x[14]+0.000001)
#     grad_planet_p = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(-x[15] * 2 * np.pi * t_full[series_index[0]] / ((x[14]+0.000001)**2) * np.cos(argument)+x[16]*2*np.pi*t_full[series_index[0]]/((x[14]+0.000001)**2)*np.sin(argument)))
#     grad_planet_a = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(np.sin(argument)))
#     grad_planet_b = np.sum(dL_dy[series_index[0]] * (1 / rv_std) * (np.cos(argument)))
#     nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_pcyc, grad_phi, grad_k_alpha, grad_gamma_0, grad_gamma_1, grad_delta_0, grad_delta_1, grad_planet_p, grad_planet_a, grad_planet_b])])
#   else:
#     nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_pcyc, grad_phi, grad_k_alpha, grad_gamma_0, grad_gamma_1, grad_delta_0, grad_delta_1])])
#   # nll_grad = -C.loglike_grad()[1][fitted]

#   return (nll, nll_grad)

# def plot_fit(t_full, y_full, yerr_full, C, xbest, series_index, rv_std = 1.0, output_name = 'fit_plot.png', return_residuals=True, inject_planet=False):

#     tsmooth = np.linspace(np.min(t_full), np.max(t_full), 1000)
#     _, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
#     # params = ['rv_jit.sig', 'rhk_jit.sig', 'rot.P0', 'rot.Q', 'rot.alpha_0', 'rot.alpha_1', 'rot.beta_0']
#     # C.set_param(xbest[:len(params)], params)
#     for k in range(2):
#     # Predict time series k

#         C.kernel['rot'].set_conditional_coef(series_id=k)
#         # C.kernel['rot'].set_conditional_coef(series_id=k)

#         fc = alpha(t_full[series_index[0]], xbest[7],xbest[8],xbest[9])
#         # fc_unnorm = (fc - xbest[10]) / ((1.0 - xbest[10]) / 2.0) - 1
#         y_model = y_full.copy()

#         y_model[series_index[0]] -= (xbest[10]*fc + xbest[12])
#         y_model[series_index[1]] -= (xbest[11]*fc + xbest[13])

#         if inject_planet == True:
#             y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], xbest[14], xbest[15], xbest[16]))/rv_std

#         mu, var = C.conditional(y_model, tsmooth, calc_cov='diag')
#         mu_res, _ = C.conditional(y_model, t_full[series_index[k]], calc_cov='diag')

#         ax = axs[k]
#         if k ==0 :
#             ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
#             if inject_planet == True:
#                 ax.plot(t_full[series_index[k]], planet_injection(t_full[series_index[k]], xbest[14], xbest[15], xbest[16])/rv_std, 'r', label='injected planet')
#         if k == 1:
#             ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
#         ax.fill_between(tsmooth,
#             mu - np.sqrt(var),
#             mu + np.sqrt(var),
#             color='g',
#             alpha=0.5)
#         ax.plot(tsmooth, mu, 'g', label='predict.')
#         ax.set_ylabel(f'$y_{k}$')

#         if return_residuals==True:
#             if k == 0:
#                 res_rv = y_model[series_index[k]] - mu_res
#             else:
#                 res_rhk = y_model[series_index[k]] - mu_res

#     ax.set_xlabel('$t$')
#     axs[0].legend()
#     plt.savefig(output_name)
#     if return_residuals==True:
#         return tsmooth, mu,res_rv, res_rhk
#     else:
#         return tsmooth, mu





