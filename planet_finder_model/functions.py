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

def fit_cycle(t_full, y_full, series_index, b0=0, P0=4000, phi0=0, print_results=False, plot=False, output_name='cycle_fit.png', return_fit = False):
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
        print("\nRV parameters:")
        print(f"a1   = {a1}")
        print(f"c1   = {c1}")
        print("\nRHK parameters:")
        print(f"a2   = {a2}")
        print(f"c2   = {c2}")

    if plot == True:

        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# RV panel
        axs[0].plot(x, y_rv, '.', alpha=0.7, color='k',markersize=2, label='RV')
        axs[0].plot(x, rv_fit, linewidth=3, color = 'g',label='RV fit')
        axs[0].set_ylabel("RV",fontsize=14)
        axs[0].legend(fontsize=14)
        axs[0].tick_params(axis='both', labelsize=12)

# RHK panel
        axs[1].plot(x, y_rhk, '.', alpha=0.7, color='k', markersize=2, label='RHK')
        axs[1].plot(x, rhk_fit, linewidth=3, color='g', label='RHK fit')
        axs[1].set_ylabel("RHK",fontsize=14)
        axs[1].set_xlabel("Time",fontsize=14)
        axs[1].legend(fontsize=14)
        axs[1].tick_params(axis='both', labelsize=12)

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
    else:
        return rv_fit, rhk_fit


def get_opt_params(C):
    params_inds = [k for k, key in enumerate(C.param) if key != 'rot.sig' and key != "rot.beta_1"]  
    params = [C.param[k] for k in params_inds]
    return params, params_inds

def negloglike_cyc(theta, t_full, y_full, series_index,C, rv_std = 1.0, inject_planet=False):
#   ADD GRADIENTS FOR CYCLE PARAMETERS
  params, params_inds = get_opt_params(C)

  C.set_param(theta[:len(params)], params)

  y_model = y_full.copy()
  if inject_planet == True:
    y_model[series_index[0]] -= (planet_injection(t_full[series_index[0]], theta[15], theta[16], theta[17]) )/rv_std

  y_model[series_index[0]] -=  model_rv(t_full[series_index[0]], theta[8], theta[10], theta[12], theta[13], theta[14])
  y_model[series_index[1]] -=  model_rhk(t_full[series_index[1]], theta[9], theta[11], theta[12], theta[13], theta[14])

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


def optimise_params(t_full, y_full, series_index, C, bounds_list, delta_0 = -0.001, delta_1 = 0.001, planet_p = 40.05, planet_A = 0.0005, planet_B = 0.0005, fit_planet = True, change_C = True):

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



def log_probability(theta, t_full, y_full, series_index, C, bounds_list, planet = True):
    # params, params_inds = get_opt_params(C)
    lp = log_prior(theta, bounds_list, planet)
    if not np.isfinite(lp):
        return -np.inf
    if planet == True:
        return lp + -1*negloglike_nocyc(theta, t_full, y_full, series_index, C, 1.0, True)[0]
    else:
        return lp + -1*negloglike_nocyc(theta, t_full, y_full, series_index, C, 1.0, False)[0]


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


def plot_fit(t_full, y_full, yerr_full, series_index, C, xbest, rv_std = 1.0, output_name = 'fit_plot.png', return_residuals=True, inject_planet=True):

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
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='Data')
            if inject_planet == True:
                ax.plot(t_full[series_index[k]], planet_injection(t_full[series_index[k]], xbest[10], xbest[11], xbest[12])/rv_std, 'r', label='injected planet')
        if k == 1:
            ax.errorbar(t_full[series_index[k]], y_model[series_index[k]], yerr_full[series_index[k]], fmt='.', color='k', label='Data')
        ax.fill_between(tsmooth,
            mu - np.sqrt(var),
            mu + np.sqrt(var),
            color='g',
            alpha=0.5)
        ax.plot(tsmooth, mu, 'g', label='GP fit')
        if k == 0:
            ax.set_ylabel("RV",fontsize=14)
        if k == 1:
            ax.set_ylabel('RHK',fontsize=14)   

        if return_residuals==True:
            if k == 0:
                res_rv = y_model[series_index[k]] - mu_res
            else:
                res_rhk = y_model[series_index[k]] - mu_res

    ax.set_xlabel('Time',fontsize=14)
    axs[0].legend(fontsize=14)
    plt.savefig(output_name)
    if return_residuals==True:
        return tsmooth, mus, res_rv, res_rhk
    else:
        return tsmooth, mus

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
        flat_samples, labels=labels, show_titles=True, label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 12}
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
    



    