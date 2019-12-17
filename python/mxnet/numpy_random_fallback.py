import numpy as np
import ast
from . import operator
from . import numpy as _mx_np  # pylint: disable=reimported
from .util import np_array, use_np
from .ndarray.numpy import _internal as _nd_npi
from .symbol.numpy import _internal as _sym_npi
from .numpy_op_fallback import register


@use_np
class multivariate_normal(operator.CustomOp):
    def __init__(self, shape=None):
      super(multivariate_normal, self).__init__()
      self._shape = shape

    def forward(self, in_train, req, in_data, out_data, aux):
      loc = in_data[0]
      cov = in_data[1]
      L = _mx_np.linalg.cholesky(cov)
      noise = _mx_np.random.normal(size=out_data[0].shape)
      samples = loc + _mx_np.dot(L, noise)
      self.assign(out_data[0], req[0], samples)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
      raise NotImplementedError(
          'Operator Unravel_index does not support gradient computation')


@register('mvn_fallback')
class MvnProp(operator.CustomOpProp):
    """Fallback resize operator properties."""

    def __init__(self, shape=None):
      super(MvnProp, self).__init__(need_top_grad=True)
      self._shape = ast.literal_eval(
          shape) if shape is not None else None

    def list_arguments(self):
      return ['mean', 'variance']

    def infer_shape(self, in_shape):
      loc_shape = in_shape[0]
      cov_shape = in_shape[1]
      if self._shape is None:
        out_shape = np.broadcast(np.empty(loc_shape), np.empty(cov_shape)).shape
      else:
        out_shape = (self._shape,) if np.isscalar(
            self._shape) else self._shape

      return (in_shape[0], in_shape[1]), (out_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
      return multivariate_normal(self._shape)
