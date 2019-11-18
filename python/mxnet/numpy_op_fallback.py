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

from __future__ import absolute_import
import functools
import ast
import numpy as np
from . import operator
from . import numpy as _mx_np  # pylint: disable=reimported
from .util import np_array, use_np
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
class Resize(operator.CustomOp):
    """Fallback to NumPy resize operator."""
    def __init__(self, new_shape):
        super(Resize, self).__init__()
        self._new_shape = new_shape

    def forward(self, is_train, req, in_data, out_data, aux):
        out = np.resize(in_data[0].asnumpy(), self._new_shape)
        self.assign(out_data[0], req[0], _mx_np.array(out, dtype=out.dtype, ctx=out_data[0].ctx))

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
