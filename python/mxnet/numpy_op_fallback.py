# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Fallback-to-NumPy operator implementation."""

from distutils.version import StrictVersion
import functools
import ast
import numpy as np
from . import operator
from . import numpy as _mx_np  # pylint: disable=reimported
from .util import np_array, use_np
from .numpy.utils import _STR_2_DTYPE_
from .ndarray.numpy import _internal as _nd_npi
from .symbol.numpy import _internal as _sym_npi


def register(op_name, imperative=True, symbolic=True):
    """Register operators that fallback to NumPy in modules
    ``mxnet.ndarray.numpy._internal`` and ``mxnet.symbol.numpy._internal``."""
    def _save_op(mod):
        if hasattr(mod, op_name):
            raise ValueError('Duplicate name {} found in module {}'.format(op_name, str(mod)))
        op = functools.partial(mod.Custom, op_type=op_name)
        setattr(mod, op_name, op)

    def _register_helper(prop_cls):
        with np_array():
            prop_cls = operator.register(op_name)(prop_cls)
        if imperative:
            _save_op(_nd_npi)
        if symbolic:
            _save_op(_sym_npi)
        return prop_cls

    return _register_helper


@use_np  # enforce np shape and array semantics for all the methods in this class
class EmptyLike(operator.CustomOp):
    """Fallback to NumPy empty_like operator."""
    def __init__(self, dtype, order, subok, shape):
        super(EmptyLike, self).__init__()
        self._dtype = dtype
        self._order = order
        self._subok = subok
        self._shape = shape

    def forward(self, is_train, req, in_data, out_data, aux):
        np_version = np.version.version
        if StrictVersion(np_version) >= StrictVersion('1.6.0'):
            out = np.empty_like(in_data[0].asnumpy(), dtype=self._dtype, order=self._order,
                                subok=self._subok)
        else:
            out = np.empty_like(in_data[0].asnumpy())
        self.assign(out_data[0], req[0], _mx_np.array(out, device=in_data[0].device))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError('Operator empty_like does not support gradient computation')


@register('empty_like_fallback')
class EmptyLikeProp(operator.CustomOpProp):
    """Fallback empty_like operator properties."""
    def __init__(self, dtype, order, subok, shape):
        super(EmptyLikeProp, self).__init__(need_top_grad=True)
        self._dtype = None if dtype == 'None' else dtype
        self._order = order
        self._subok = ast.literal_eval(subok)
        self._shape = ast.literal_eval(shape)

    def list_arguments(self):
        return ['prototype']

    def infer_shape(self, in_shape):
        return (in_shape[0],), (in_shape[0],), ()

    def infer_type(self, in_type):
        if self._dtype is None:
            return (in_type[0],), (in_type[0],), ()
        else:
            return (in_type[0],), (_STR_2_DTYPE_[self._dtype],), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return EmptyLike(self._dtype, self._order, self._subok, self._shape)


@use_np  # enforce np shape and array semantics for all the methods in this class
class Resize(operator.CustomOp):
    """Fallback to NumPy resize operator."""
    def __init__(self, new_shape):
        super(Resize, self).__init__()
        self._new_shape = new_shape

    def forward(self, is_train, req, in_data, out_data, aux):
        out = np.resize(in_data[0].asnumpy(), self._new_shape)
        self.assign(out_data[0], req[0], _mx_np.array(out, dtype=out.dtype, device=out_data[0].device))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError('Operator resize does not support gradient computation')


@register('resize_fallback')
class ResizeProp(operator.CustomOpProp):
    """Fallback resize operator properties."""
    def __init__(self, new_shape):
        super(ResizeProp, self).__init__(need_top_grad=True)
        self._new_shape = ast.literal_eval(new_shape)

    def list_arguments(self):
        return ['a']

    def infer_shape(self, in_shape):
        out_shape = (self._new_shape,) if np.isscalar(self._new_shape) else self._new_shape
        return (in_shape[0],), (out_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Resize(self._new_shape)


@use_np
class Unravel_index(operator.CustomOp):
    """Fallback to NumPy Unravel_index operator."""
    def __init__(self, shape):
        super(Unravel_index, self).__init__()
        self._shape = shape

    def forward(self, is_train, req, in_data, out_data, aux):
        out = np.unravel_index(in_data[0].asnumpy(), self._shape)
        self.assign(out_data[0], req[0], _mx_np.array(out, dtype=out[0].dtype, device=out_data[0].device))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError('Operator Unravel_index does not support gradient computation')


@register('unravel_index_fallback')
class Unravel_indexProp(operator.CustomOpProp):
    """Fallback unravel_index operator properties."""
    def __init__(self, shape):
        super(Unravel_indexProp, self).__init__(need_top_grad=True)
        self._shape = ast.literal_eval(shape)

    def list_arguments(self):
        return ['indices']

    def infer_shape(self, in_shape):
        dim_list = (1,) if np.isscalar(self._shape) else (len(self._shape),)
        out_shape = dim_list + tuple(in_shape[0])
        return (in_shape[0],), (out_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Unravel_index(self._shape)


@use_np
class MultivariateNormal(operator.CustomOp):
    """Fallback to the front-end implementation of random.multivariate_normal."""
    def __init__(self, size=None):
        super(MultivariateNormal, self).__init__()
        self._size = size

    def forward(self, is_train, req, in_data, out_data, aux):
        loc = in_data[0]
        cov = in_data[1]
        if cov.dtype == np.float16:
            scale = _mx_np.linalg.cholesky(cov.astype(np.float32)).astype(np.float16)
        else:
            scale = _mx_np.linalg.cholesky(cov)
        #set context
        noise = _mx_np.random.normal(size=out_data[0].shape, dtype=loc.dtype, device=loc.device)
        out = loc + _mx_np.einsum('...jk,...k->...j', scale, noise)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError('Operator random.multivariate_normal'
                                  ' does not support gradient computation')


@register('mvn_fallback')
class MultivariateNormalProp(operator.CustomOpProp):
    """Fallback np.random.multivariate_normal operator properties."""

    def __init__(self, size=None):
        super(MultivariateNormalProp, self).__init__(need_top_grad=True)
        self._size = ast.literal_eval(
            size) if size is not None else None

    def list_arguments(self):
        return ['mean', 'cov']

    def infer_shape(self, in_shape):
        loc_shape = in_shape[0]
        cov_shape = in_shape[1]
        if len(loc_shape) < 1:
            raise ValueError("mean must be at least 1 dimensional")
        if len(cov_shape) < 2:
            raise ValueError("cov must be at least 2 dimensional")
        if cov_shape[-1] != cov_shape[-2]:
            raise ValueError("the last two dimentions of the parameter cov have to be the same,"
                             " whereas the shape of cov is {}".format(cov_shape))
        if cov_shape[-1] != loc_shape[-1]:
            raise ValueError("mean and cov must have same length."
                             "The shape of mean is {} but the shape of cov is {}"
                             .format(loc_shape[-1:], cov_shape[-2:]))
        # handle shape mismatch here
        out_shape = np.broadcast(np.empty(loc_shape), np.empty(cov_shape[:-1])).shape
        if self._size is not None:
            self._size = (self._size,) if np.isscalar(
                self._size) else self._size
            out_shape = self._size + out_shape

        return in_shape, (out_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return MultivariateNormal(self._size)
