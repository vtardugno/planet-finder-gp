import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, curve_fit
from spleaf import cov, term
import scipy.io as sp
import emcee
import pandas as pd


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


def alpha(t, pcyc, phi, k):

    x = (2.0 * np.pi / pcyc) * (t) + phi

    fc = np.sin((x) + k * np.sin(x))  # in [-1, 1]

    return fc



def alpha_grad(t, pcyc, phi, k):
    
    omega = 2.0 * np.pi / pcyc
    x = omega * (t) + phi

    inner = x + k * np.sin(x)
    c_inner = np.cos(inner)
    d_inner_dx = 1.0 + k * np.cos(x)
    d_alpha_dinner =  c_inner

    grads = {}

    # derivative wrt phi

    grads['phi'] = d_alpha_dinner * d_inner_dx

    # derivative wrt pcyc 
  
    domega_dp = -2.0 * np.pi / (pcyc ** 2)
    dx_dp = domega_dp * (t)
    grads['pcyc'] = d_alpha_dinner * d_inner_dx * dx_dp

    # derivative wrt k 

    d_inner_dk = np.sin(x)
    grads['k'] = d_alpha_dinner * d_inner_dk

    return grads


def negloglike(x, y, C, params, params_inds, t_full, series_index, rv_std = 1.0, inject_planet=False):
  
  C.set_param(x[:len(params)], params)
  fc = alpha(t_full[series_index[0]], x[7],x[8],x[9])
  # fc_unnorm = (fc - x[10]) / ((1.0 - x[10]) / 2.0) - 1
  fc_grads = alpha_grad(t_full[series_index[0]] ,x[7],x[8],x[9])
  y_model = y.copy()
  y_model[series_index[0]] -= (x[10]*fc + x[12])
  y_model[series_index[1]] -= (x[11]*fc + x[13])

  if inject_planet == True:
      y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], x[14], x[15], x[16]) )/rv_std

  # gradient§
  nll = -C.loglike(y_model)

  lg = C.loglike_grad()
  dL_dy = np.asarray(lg[0]).reshape(-1)      # shape (N_total,) 
  dL_dparams = np.asarray(lg[1]).reshape(-1) # shape (12,)      

  base_grad = - dL_dparams[params_inds]

  # gradients wrt gamma (scale of fc)
  grad_gamma_0 = np.sum(dL_dy[series_index[0]] * fc)
  grad_gamma_1 = np.sum(dL_dy[series_index[1]] * fc)
  # gradients wrt delta (additive offsets)
  grad_delta_0 = np.sum(dL_dy[series_index[0]])
  grad_delta_1 = np.sum(dL_dy[series_index[1]])

  grad_pcyc = np.sum(dL_dy[series_index[0]] * x[10] * fc_grads['pcyc']) + np.sum(dL_dy[series_index[1]] * x[11] * fc_grads['pcyc'])
  grad_phi = np.sum(dL_dy[series_index[0]] * x[10] * fc_grads['phi']) + np.sum(dL_dy[series_index[1]] * x[11] * fc_grads['phi'])
  grad_k_alpha = np.sum(dL_dy[series_index[0]] * x[10] * fc_grads['k']) + np.sum(dL_dy[series_index[1]] * x[11] * fc_grads['k'])
  # grad_c = np.sum(dL_dy[series_index[0]] * x[-4] * fc_grads['c'][series_index[0]]) + np.sum(dL_dy[series_index[1]] * x[-3] * fc_grads['c'][series_index[1]])
  if inject_planet == True:
    argument = 2 * np.pi * t_full[series_index[0]] / (x[14]+0.000001)
    grad_planet_p = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(-x[15] * 2 * np.pi * t_full[series_index[0]] / ((x[14]+0.000001)**2) * np.cos(argument)+x[16]*2*np.pi*t_full[series_index[0]]/((x[14]+0.000001)**2)*np.sin(argument))) 
    grad_planet_a = np.sum(dL_dy[series_index[0]] * (1/rv_std)*(np.sin(argument)))
    grad_planet_b = np.sum(dL_dy[series_index[0]] * (1 / rv_std) * (np.cos(argument)))
    nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_pcyc, grad_phi, grad_k_alpha, grad_gamma_0, grad_gamma_1, grad_delta_0, grad_delta_1, grad_planet_p, grad_planet_a, grad_planet_b])])
  else:
    nll_grad = np.concatenate([np.asarray(base_grad).ravel(), np.array([grad_pcyc, grad_phi, grad_k_alpha, grad_gamma_0, grad_gamma_1, grad_delta_0, grad_delta_1])])
  # nll_grad = -C.loglike_grad()[1][fitted]

  return (nll, nll_grad)

def plot_fit(t_full, y_full, yerr_full, C, xbest, series_index, rv_std = 1.0, output_name = 'fit_plot.png', return_residuals=True, inject_planet=False):

    tsmooth = np.linspace(np.min(t_full), np.max(t_full), 1000)
    _, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    # params = ['rv_jit.sig', 'rhk_jit.sig', 'rot.P0', 'rot.Q', 'rot.alpha_0', 'rot.alpha_1', 'rot.beta_0']
    # C.set_param(xbest[:len(params)], params)
    for k in range(2):
    # Predict time series k

        C.kernel['rot'].set_conditional_coef(series_id=k)
        # C.kernel['rot'].set_conditional_coef(series_id=k)

        fc = alpha(t_full[series_index[0]], xbest[7],xbest[8],xbest[9])
        # fc_unnorm = (fc - xbest[10]) / ((1.0 - xbest[10]) / 2.0) - 1
        y_model = y_full.copy()
        
        y_model[series_index[0]] -= (xbest[10]*fc + xbest[12])
        y_model[series_index[1]] -= (xbest[11]*fc + xbest[13])
        
        if inject_planet == True:
            y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], xbest[14], xbest[15], xbest[16]))/rv_std

        mu, var = C.conditional(y_model, tsmooth, calc_cov='diag')
        mu_res, _ = C.conditional(y_model, t_full[series_index[k]], calc_cov='diag')
  
        ax = axs[k]
        if k ==0 :
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
            if inject_planet == True:
                ax.plot(t_full[series_index[k]], planet_injection(t_full[series_index[k]], xbest[14], xbest[15], xbest[16])/rv_std, 'r', label='injected planet')
        if k == 1:
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='meas.')
        ax.fill_between(tsmooth,
            mu - np.sqrt(var),
            mu + np.sqrt(var),
            color='g',
            alpha=0.5)
        ax.plot(tsmooth, mu, 'g', label='predict.')
        ax.set_ylabel(f'$y_{k}$')

        if return_residuals==True:
            if k == 0:
                res_rv = y_model[series_index[k]] - mu_res
            else:
                res_rhk = y_model[series_index[k]] - mu_res

    ax.set_xlabel('$t$')
    axs[0].legend()
    plt.savefig(output_name)
    if return_residuals==True:
        return tsmooth, mu,res_rv, res_rhk
    else:
        return tsmooth, mu
    



    