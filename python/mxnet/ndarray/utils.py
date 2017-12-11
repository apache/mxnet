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

# coding: utf-8
"""Utility functions for NDArray and BaseSparseNDArray."""
import ctypes

from ..base import _LIB, check_call, py_str, c_str, string_types, mx_uint, NDArrayHandle
from ..base import c_array, c_handle_array, c_str_array
from .ndarray import NDArray
from .ndarray import array as _array
from .ndarray import empty as _empty_ndarray
from .ndarray import zeros as _zeros_ndarray
from .sparse import zeros as _zeros_sparse_ndarray
from .sparse import empty as _empty_sparse_ndarray
from .sparse import array as _sparse_array
from .sparse import _ndarray_cls
try:
    import scipy.sparse as spsp
except ImportError:
    spsp = None

__all__ = ['zeros', 'empty', 'array', 'load', 'save']


def zeros(shape, ctx=None, dtype=None, stype=None, **kwargs):
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array
    ctx : Context, optional
        An optional device context (default is the current default context)
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`)
    stype: string, optional
        The storage type of the empty array, such as 'row_sparse', 'csr', etc.

    Returns
    -------
    NDArray, CSRNDArray or RowSparseNDArray
        A created array
    Examples
    --------
    >>> mx.nd.zeros((1,2), mx.cpu(), stype='csr')
    <CSRNDArray 1x2 @cpu(0)>
    >>> mx.nd.zeros((1,2), mx.cpu(), 'float16', stype='row_sparse').asnumpy()
    array([[ 0.,  0.]], dtype=float16)
    """

    if stype is None or stype == 'default':
        return _zeros_ndarray(shape, ctx, dtype, **kwargs)
    else:
        return _zeros_sparse_ndarray(stype, shape, ctx, dtype, **kwargs)


def empty(shape, ctx=None, dtype=None, stype=None):
    """Returns a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    ctx : Context, optional
        An optional device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`).
    stype : str, optional
        An optional storage type (default is `default`).

    Returns
    -------
    NDArray, CSRNDArray or RowSparseNDArray
        A created array.

    Examples
    --------
    >>> mx.nd.empty(1)
    <NDArray 1 @cpu(0)>
    >>> mx.nd.empty((1,2), mx.gpu(0))
    <NDArray 1x2 @gpu(0)>
    >>> mx.nd.empty((1,2), mx.gpu(0), 'float16')
    <NDArray 1x2 @gpu(0)>
    >>> mx.nd.empty((1,2), stype='csr')
    <CSRNDArray 1x2 @cpu(0)>
    """
    if stype is None or stype == 'default':
        return _empty_ndarray(shape, ctx, dtype)
    else:
        return _empty_sparse_ndarray(stype, shape, ctx, dtype)


def array(source_array, ctx=None, dtype=None):
    """Creates an array from any object exposing the array interface.

    Parameters
    ----------
    source_array : array_like
        An object exposing the array interface, an object whose `__array__`
        method returns an array, or any (nested) sequence.
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``source_array.dtype``
        if `source_array` is an `NDArray`, `float32` otherwise.

    Returns
    -------
    NDArray, RowSparseNDArray or CSRNDArray
        An array with the same contents as the `source_array`.

    Examples
    --------
    >>> import numpy as np
    >>> mx.nd.array([1, 2, 3])
    <NDArray 3 @cpu(0)>
    >>> mx.nd.array([[1, 2], [3, 4]])
    <NDArray 2x2 @cpu(0)>
    >>> mx.nd.array(np.zeros((3, 2)))
    <NDArray 3x2 @cpu(0)>
    >>> mx.nd.array(np.zeros((3, 2)), mx.gpu(0))
    <NDArray 3x2 @gpu(0)>
    >>> mx.nd.array(mx.nd.zeros((3, 2), stype='row_sparse'))
    <RowSparseNDArray 3x2 @cpu(0)>
    """
    if spsp is not None and isinstance(source_array, spsp.csr.csr_matrix):
        return _sparse_array(source_array, ctx=ctx, dtype=dtype)
    elif isinstance(source_array, NDArray) and source_array.stype != 'default':
        return _sparse_array(source_array, ctx=ctx, dtype=dtype)
    else:
        return _array(source_array, ctx=ctx, dtype=dtype)


def load(fname):
    """Loads an array from file.

    See more details in ``save``.

    Parameters
    ----------
    fname : str
        The filename.

    Returns
    -------
    list of NDArray, RowSparseNDArray or CSRNDArray, or \
    dict of str to NDArray, RowSparseNDArray or CSRNDArray
        Loaded data.
    """
    if not isinstance(fname, string_types):
        raise TypeError('fname required to be a string')
    out_size = mx_uint()
    out_name_size = mx_uint()
    handles = ctypes.POINTER(NDArrayHandle)()
    names = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXNDArrayLoad(c_str(fname),
                                  ctypes.byref(out_size),
                                  ctypes.byref(handles),
                                  ctypes.byref(out_name_size),
                                  ctypes.byref(names)))
    if out_name_size.value == 0:
        return [_ndarray_cls(NDArrayHandle(handles[i])) for i in range(out_size.value)]
    else:
        assert out_name_size.value == out_size.value
        return dict(
            (py_str(names[i]), _ndarray_cls(NDArrayHandle(handles[i])))
            for i in range(out_size.value))


def save(fname, data):
    """Saves a list of arrays or a dict of str->array to file.

    Examples of filenames:

    - ``/path/to/file``
    - ``s3://my-bucket/path/to/file`` (if compiled with AWS S3 supports)
    - ``hdfs://path/to/file`` (if compiled with HDFS supports)

    Parameters
    ----------
    fname : str
        The filename.
    data : NDArray, RowSparseNDArray or CSRNDArray, \
           or list of NDArray, RowSparseNDArray or CSRNDArray, \
           or dict of str to NDArray, RowSparseNDArray or CSRNDArray
        The data to save.

    Examples
    --------
    >>> x = mx.nd.zeros((2,3))
    >>> y = mx.nd.ones((1,4))
    >>> mx.nd.save('my_list', [x,y])
    >>> mx.nd.save('my_dict', {'x':x, 'y':y})
    >>> mx.nd.load('my_list')
    [<NDArray 2x3 @cpu(0)>, <NDArray 1x4 @cpu(0)>]
    >>> mx.nd.load('my_dict')
    {'y': <NDArray 1x4 @cpu(0)>, 'x': <NDArray 2x3 @cpu(0)>}
    """
    if isinstance(data, NDArray):
        data = [data]
        handles = c_array(NDArrayHandle, [])
    if isinstance(data, dict):
        str_keys = data.keys()
        nd_vals = data.values()
        if any(not isinstance(k, string_types) for k in str_keys) or \
           any(not isinstance(v, NDArray) for v in nd_vals):
            raise TypeError('save only accept dict str->NDArray or list of NDArray')
        keys = c_str_array(str_keys)
        handles = c_handle_array(nd_vals)
    elif isinstance(data, list):
        if any(not isinstance(v, NDArray) for v in data):
            raise TypeError('save only accept dict str->NDArray or list of NDArray')
        keys = None
        handles = c_handle_array(data)
    else:
        raise ValueError("data needs to either be a NDArray, dict of str, NDArray pairs "
                         "or a list of NDarrays.")
    check_call(_LIB.MXNDArraySave(c_str(fname),
                                  mx_uint(len(handles)),
                                  handles,
                                  keys))
