import matplotlib.pyplot as plt
import numpy as np
from nskernel import NonStationaryKernel
from scipy.optimize import fmin_l_bfgs_b
from spleaf import cov, term

# from me import *

np.random.seed(0)

# Generate calendar + measurements errorbars
nt = 200
tmax = 100
t = np.sort(np.random.uniform(0, tmax, nt))
yerr = np.random.uniform(0.1, 0.2, nt)

# Initialize a non-stationary GP
# We assume the kernel to be the product
# of a purely non-stationary separable part
# and a stationary semi-separable part
# k(t, t') = alpha(t) alpha(t') K(|t-t'|)
# In the following example
# alpha is an order 2 polynomial
# and K is a Matérn 1/2 kernel.


def alpha(t, a, b, c):
  return a * t**2 + b * t + c


def alpha_grad(t, a, b, c):
  return {'a': t**2, 'b': t, 'c': np.ones_like(t)}


cov = cov.Cov(
  t,
  err=term.Error(yerr),
  gp=term.SimpleProductKernel(
    nonstat=NonStationaryKernel(alpha, alpha_grad, a=1e-4, b=0, c=0),
    mat12=term.Matern12Kernel(1, 5),
  ),
)
print('List of kernel hyper-parameters:', cov.param)

# Generate a random sample from the covariance matrix (fake measurements)
y = cov.sample()

# Plot measurements
plt.figure()
plt.errorbar(t, y, yerr, fmt='.', color='k', label='meas.')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()


# We now fit the hyper-parameters using the fmin_l_bfgs_b function from scipy.optimize.
# Define the function to minimize
# We exclude gp.mat12_sig from the list of fitted parameters as it will be degenerated with a, b, c from the non-stationary part:
fitted = [k for k, key in enumerate(cov.param) if key != 'gp.mat12_sig']
params = [cov.param[k] for k in fitted]
x0 = cov.get_param(params)


def negloglike(x, y, cov):
  cov.set_param(x, params)
  nll = -cov.loglike(y)
  # gradient
  nll_grad = -cov.loglike_grad()[1][fitted]
  return (nll, nll_grad)


# Fit
# xbest, _, _ = fmin_l_bfgs_b(negloglike, cov.get_param(), args=(y, cov))
result = fmin_l_bfgs_b(negloglike, cov.get_param(), args=(y, cov))
print(result)
xbest = result[0]

# Compute GP prediction
cov.set_param(xbest)
tsmooth = np.linspace(-0.1 * tmax, 1.025 * tmax, 2000)
mu, var = cov.conditional(y, tsmooth, calc_cov='diag')

# Plot
plt.figure()
plt.errorbar(t, y, yerr, fmt='.', color='k', label='meas.')
plt.fill_between(tsmooth, mu - np.sqrt(var), mu + np.sqrt(var), color='g', alpha=0.5)
plt.plot(tsmooth, mu, 'g', label='predict.')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()
