from spleaf import term


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
