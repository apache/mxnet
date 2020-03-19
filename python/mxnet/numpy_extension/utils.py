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

"""Util functions for the numpy module."""



import ctypes
from .. util import is_np_array, is_np_shape
from .. base import _LIB, check_call, string_types, c_str_array
from .. base import c_handle_array, c_str, mx_uint, NDArrayHandle, py_str
from ..numpy import ndarray

__all__ = ['save', 'load']


def save(file, arr):
    """Saves a list of `ndarray`s or a dict of `str`->`ndarray` to file.

    Examples of filenames:

    - ``/path/to/file``
    - ``s3://my-bucket/path/to/file`` (if compiled with AWS S3 supports)
    - ``hdfs://path/to/file`` (if compiled with HDFS supports)

    Parameters
    ----------
    file : str
        Filename to which the data is saved.
    arr : `ndarray` or list of `ndarray`s or dict of `str` to `ndarray`
        The data to be saved.

    Notes
    -----
    This function can only be called within numpy semantics, i.e., `npx.is_np_shape()`
    and `npx.is_np_array()` must both return true.
    """
    if not (is_np_shape() and is_np_array()):
        raise ValueError('Cannot save `mxnet.numpy.ndarray` in legacy mode. Please activate'
                         ' numpy semantics by calling `npx.set_np()` in the global scope'
                         ' before calling this function.')
    if isinstance(arr, ndarray):
        arr = [arr]
    if isinstance(arr, dict):
        str_keys = arr.keys()
        nd_vals = arr.values()
        if any(not isinstance(k, string_types) for k in str_keys) or \
                any(not isinstance(v, ndarray) for v in nd_vals):
            raise TypeError('Only accepts dict str->ndarray or list of ndarrays')
        keys = c_str_array(str_keys)
        handles = c_handle_array(nd_vals)
    elif isinstance(arr, list):
        if any(not isinstance(v, ndarray) for v in arr):
            raise TypeError('Only accepts dict str->ndarray or list of ndarrays')
        keys = None
        handles = c_handle_array(arr)
    else:
        raise ValueError("data needs to either be a ndarray, dict of (str, ndarray) pairs "
                         "or a list of ndarrays.")
    check_call(_LIB.MXNDArraySave(c_str(file),
                                  mx_uint(len(handles)),
                                  handles,
                                  keys))


def load(file):
    """Loads an array from file.

    See more details in ``save``.

    Parameters
    ----------
    file : str
        The filename.

    Returns
    -------
    result : list of ndarrays or dict of str -> ndarray
        Data stored in the file.

    Notes
    -----
    This function can only be called within numpy semantics, i.e., `npx.is_np_shape()`
    and `npx.is_np_array()` must both return true.
    """
    if not (is_np_shape() and is_np_array()):
        raise ValueError('Cannot load `mxnet.numpy.ndarray` in legacy mode. Please activate'
                         ' numpy semantics by calling `npx.set_np()` in the global scope'
                         ' before calling this function.')
    if not isinstance(file, string_types):
        raise TypeError('file required to be a string')
    out_size = mx_uint()
    out_name_size = mx_uint()
    handles = ctypes.POINTER(NDArrayHandle)()
    names = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXNDArrayLoad(c_str(file),
                                  ctypes.byref(out_size),
                                  ctypes.byref(handles),
                                  ctypes.byref(out_name_size),
                                  ctypes.byref(names)))
    if out_name_size.value == 0:
        return [ndarray(NDArrayHandle(handles[i])) for i in range(out_size.value)]
    else:
        assert out_name_size.value == out_size.value
        return dict(
            (py_str(names[i]), ndarray(NDArrayHandle(handles[i])))
            for i in range(out_size.value))
