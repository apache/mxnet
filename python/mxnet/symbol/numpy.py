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

"""numpy namespace for operators used in Gluon APIs dispatched by F=symbol module."""

from __future__ import absolute_import
import ctypes
import numpy as _np
from ..base import _sanity_check_params, use_np_compat, check_call, _LIB, SymbolHandle
from ..context import current_context
from . import _internal
from .symbol import Symbol
from . import op as _op
from ._internal import _set_np_symbol_class

__all__ = ['zeros']


class _NumpySymbol(Symbol):
    def asNDArray(self):
        """Convert _NumpySymbol to mxnet.symbol.Symbol to use its fluent methods."""
        hdl = SymbolHandle()
        check_call(_LIB.MXShallowCopySymbol(self.handle, ctypes.byref(hdl)))
        return Symbol(handle=hdl)

    @use_np_compat
    def sin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sin`.

        The arguments are the same as for :py:func:`sin`, with
        this array as data.
        """
        raise NotImplementedError('mxnet.numpy.ndarray.sin is not implemented. Please '
                                  'convert the mxnet.numpy.ndarray to mxnet.ndarray.NDArray '
                                  'and call the sin function as follows: '
                                  'self.asNDArray().sin(*args, **kwargs).')

    @use_np_compat
    def sum(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sum`.

        The arguments are the same as for :py:func:`sum`, with
        this array as data.
        """
        return _op.sum(self, *args, **kwargs)


@use_np_compat
def zeros(shape, dtype=_np.float64, **kwargs):
    """Return a new array of given shape and type, filled with zeros.
    This function does not support the parameter `order` as in NumPy package.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`).
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : Symbol
        Array of zeros with the given shape, dtype, and ctx.
    """
    _sanity_check_params('zeros', ['order'], kwargs)
    ctx = kwargs.get('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    dtype = _np.float64 if dtype is None else dtype
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, **kwargs)


_set_np_symbol_class(_NumpySymbol)
