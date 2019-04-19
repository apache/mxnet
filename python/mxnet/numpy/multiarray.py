#!/usr/bin/env python

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


from __future__ import absolute_import
import numpy as _np
import ctypes
from ..ndarray import NDArray, _new_alloc_handle
from ..ndarray._internal import _set_np_ndarray_class
from . import _op
from ..base import use_np_compat, check_call, _LIB, NDArrayHandle, _sanity_check_params
from ..context import current_context
from ..ndarray import numpy as _mx_nd_np


__all__ = ['ndarray', 'empty', 'array', 'zeros']


def _np_ndarray_cls(handle, writable=True):
    return ndarray(handle, writable=writable)


_set_np_ndarray_class(_np_ndarray_cls)


class ndarray(NDArray):
    def asNDArray(self):
        """Convert mxnet.numpy.ndarray to mxnet.ndarray.NDArray to use its fluent methods."""
        hdl = NDArrayHandle()
        check_call(_LIB.MXShallowCopyNDArray(self.handle, ctypes.byref(hdl)))
        return NDArray(handle=hdl, writable=self.writable)

    @use_np_compat
    def __repr__(self):
        """Returns a string representation of the array."""
        return '%s\n<%s shape=%s ctx=%s>' % (str(self.asnumpy()), self.__class__.__name__,
                                             self.shape, self.context)

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
def empty(shape, dtype=None, **kwargs):
    """Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array, e.g, `numpy.int8`. Default is
        `numpy.float64`.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape, dtype, and order.
    """
    _sanity_check_params('emtpy', ['order'], **kwargs)
    ctx = kwargs.get('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    if dtype is None:
        dtype = _np.float64
    if isinstance(shape, int):
        shape = (shape,)
    return ndarray(handle=_new_alloc_handle(shape, ctx, False, dtype))


@use_np_compat
def array(object, dtype=None, **kwargs):
    """
    Create an array.

    Parameters
    ----------
    object : array_like or `mxnet.ndarray.NDArray` or `mxnet.numpy.ndarray`
        An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence. This argument can only be used to 'upcast' the array.  For
        downcasting, use the .astype(t) method.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape, dtype and ctx.
    """
    _sanity_check_params('array', ['copy', 'order', 'subok', 'ndim'], **kwargs)
    ctx = kwargs.get('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    if not isinstance(object, (ndarray, NDArray, _np.ndarray)):
        try:
            object = _np.array(object, dtype=dtype)
        except:
            raise TypeError('source array must be an array like object')
    if dtype is None:
        dtype = object.dtype
    ret = empty(object.shape, dtype=dtype, ctx=ctx)
    ret[:] = object
    return ret


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
    out : NDArray
        Array of zeros with the given shape, dtype, and ctx.
    """
    return _mx_nd_np.zeros(shape, dtype, **kwargs)
