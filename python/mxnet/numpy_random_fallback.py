import numpy as np
from . import operator
from . import numpy as _mx_np  # pylint: disable=reimported
from .util import np_array, use_np
from .ndarray.numpy import _internal as _nd_npi
from .symbol.numpy import _internal as _sym_npi
from .numpy_op_fallback import register


@use_np
class multivariate_normal(operator.CustomOp):
  def __init__(self, loc, scale, shape):
    super(multivariate_normal, self).__init__()
    self._loc = loc
    self._scale = scale
    self._shape = None
  
  def forward(self, in_train, req, in_data, out_data, aux):
    loc = self.in_data[0]
    L = _mx_np.linalg.cholesky(self.in_data[1])
    noise = _mx_np.random.normal(size=out_data[0].shape)
    samples = loc + _mx_np.dot(L, noise)

  def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    raise NotImplementedError('Operator Unravel_index does not support gradient computation')