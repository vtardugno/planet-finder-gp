import matplotlib.pyplot as plt
import numpy as np
from nskernel import NonStationaryKernel
from scipy.optimize import fmin_l_bfgs_b
from spleaf import cov, term

np.random.seed(0)

# Generate calendar + measurements errorbars
nt = 200
tmax = 100
t = np.sort(np.random.uniform(0, tmax, nt))
y1err = np.random.uniform(0.1, 0.2, nt)
y2err = np.random.uniform(0.1, 0.2, nt)

full_t, full_yerr, series_index = cov.merge_series([t, t], [y1err, y2err])

P = 25.7
rho = 50.2
eta = 0.5
a_qp = np.array([0.5, 0.25])
b_qp = np.array([0, -2.2])


# Initialize a non-stationary GP
# We assume the kernel to be the product
# of a purely non-stationary separable part
# and a stationary semi-separable part
# k(t, t') = alpha(t) alpha(t') K(|t-t'|)
def alpha(t, a, b, c):
  return a * t**2 + b * t + c


def alpha_grad(t, a, b, c):
  return {'a': t**2, 'b': t, 'c': np.ones_like(t)}


class MultiSeriesKernel(term.MultiSeriesKernel):
  def _grad_param(self, grad_dU=None, grad_dV=None):
    if grad_dU is not None or grad_dV is not None:
      raise NotImplementedError()
    return super()._grad_param()


cov = cov.Cov(
  full_t,
  err=term.Error(full_yerr),
  gp=term.SimpleProductKernel(
    nonstat=NonStationaryKernel(alpha, alpha_grad, a=1e-4, b=0, c=0),
    qp=MultiSeriesKernel(term.ESPKernel(1, P, rho, eta), series_index, a_qp, b_qp),
  ),
)
print('Full list of kernel hyper-parameters:', cov.param)

# Generate a random sample from the covariance matrix (fake measurements)
full_y = cov.sample()
y1 = full_y[series_index[0]]
y2 = full_y[series_index[1]]

# Plot measurements
_, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
axs[0].errorbar(t, y1, y1err, fmt='.', color='k')
axs[0].set_ylabel('$y_1$')
axs[1].errorbar(t, y2, y2err, fmt='.', color='k')
axs[1].set_ylabel('$y_2$')
axs[1].set_xlabel('$t$')

# We now fit the hyper-parameters using the fmin_l_bfgs_b function from scipy.optimize.
# Define the function to minimize
# We exclude gp.qp_sig, gp.qp_alpha_0, gp.qp_beta_0 from the list of fitted parameters
# as it will be degenerated with a, b, c from the non-stationary part:
fitted = [
  k
  for k, key in enumerate(cov.param)
  if key not in ['gp.qp_sig', 'gp.qp_alpha_0', 'gp.qp_beta_0']
]
params = [cov.param[k] for k in fitted]
x0 = cov.get_param(params) * 0.95
print('Parameters to be fitted:', params)


def negloglike(x, y, cov):
  cov.set_param(x, params)
  nll = -cov.loglike(y)
  # gradient
  nll_grad = -cov.loglike_grad()[1][fitted]
  return (nll, nll_grad)


# Fit
result = fmin_l_bfgs_b(negloglike, x0, args=(full_y, cov))
print(result)
xbest = result[0]

# Compute GP prediction
cov.set_param(xbest, params)

tsmooth = np.linspace(-0.1 * tmax, 1.025 * tmax, 2000)

cov.kernel['gp']._kernel2.set_conditional_coef(alpha=1, beta=0)
mugp, vargp = mu1, var1 = cov.conditional(full_y, tsmooth, calc_cov='diag')
cov.kernel['gp']._kernel2.set_conditional_coef(alpha=0, beta=1)
mudgp, vardgp = cov.conditional(full_y, tsmooth, calc_cov='diag')

cov.kernel['gp']._kernel2.set_conditional_coef(series_id=0)
mu1, var1 = cov.conditional(full_y, tsmooth, calc_cov='diag')
cov.kernel['gp']._kernel2.set_conditional_coef(series_id=1)
mu2, var2 = cov.conditional(full_y, tsmooth, calc_cov='diag')

# Plot a(t) * f(t) and a(t) * f'(t)
_, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 5))

axs[0].fill_between(
  tsmooth, mugp - np.sqrt(vargp), mugp + np.sqrt(vargp), color='g', alpha=0.5
)
axs[0].plot(tsmooth, mugp, 'g', label='predict.')
axs[0].set_ylabel('$a(t) f(t)$')
axs[1].fill_between(
  tsmooth, mudgp - np.sqrt(vardgp), mudgp + np.sqrt(vardgp), color='r', alpha=0.5
)
axs[1].plot(tsmooth, mudgp, 'r', label='predict.')
axs[1].set_ylabel("$a(t) f'(t)$")
axs[1].set_xlabel('$t$')


# Plot GP predictions for both timeseries
_, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
axs[0].errorbar(t, y1, y1err, fmt='.', color='k')
axs[0].fill_between(
  tsmooth, mu1 - np.sqrt(var1), mu1 + np.sqrt(var1), color='g', alpha=0.5
)
axs[0].plot(tsmooth, mu1, 'g', label='predict.')
axs[0].set_ylabel('$y_1$')
axs[1].errorbar(t, y2, y2err, fmt='.', color='k')
axs[1].fill_between(
  tsmooth, mu2 - np.sqrt(var2), mu2 + np.sqrt(var2), color='g', alpha=0.5
)
axs[1].plot(tsmooth, mu2, 'g', label='predict.')
axs[1].set_ylabel('$y_2$')
axs[1].set_xlabel('$t$')
plt.show()
