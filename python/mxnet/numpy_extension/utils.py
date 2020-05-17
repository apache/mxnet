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
from .. base import _LIB, check_call, string_types, c_str_array, DLPackHandle
from .. base import c_handle_array, c_str, mx_uint, NDArrayHandle, py_str
from ..numpy import ndarray

__all__ = ['save', 'load', 'to_dlpack_for_read', 'to_dlpack_for_write', 'from_dlpack']

PyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_c_str_dltensor = c_str('dltensor')
_c_str_used_dltensor = c_str('used_dltensor')

def _dlpack_deleter(pycapsule):
    pycapsule = ctypes.c_void_p(pycapsule)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
        ptr = ctypes.c_void_p(
            ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor))
        check_call(_LIB.MXNDArrayCallDLPackDeleter(ptr))

_c_dlpack_deleter = PyCapsuleDestructor(_dlpack_deleter)

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


def from_dlpack(dlpack):
    """Returns a np.ndarray backed by a dlpack tensor.

    Parameters
    ----------
    dlpack: PyCapsule (the pointer of DLManagedTensor)
        input data

    Returns
    -------
    np.ndarray
        an ndarray backed by a dlpack tensor

    Examples
    --------
    >>> x = mx.np.ones((2,3))
    >>> y = mx.npx.to_dlpack_for_read(x)
    >>> type(y)
    <class 'PyCapsule'>
    >>> z = mx.npx.from_dlpack(y)
    >>> type(z)
    <class 'mxnet.numpy.ndarray'>
    >>> z
    array([[1., 1., 1.],
           [1., 1., 1.]])

    >>> w = mx.npx.to_dlpack_for_write(x)
    >>> type(w)
    <class 'PyCapsule'>
    >>> u = mx.npx.from_dlpack(w)
    >>> u += 1
    >>> x
    array([[2., 2., 2.],
           [2., 2., 2.]])
    """
    handle = NDArrayHandle()
    dlpack = ctypes.py_object(dlpack)
    assert ctypes.pythonapi.PyCapsule_IsValid(dlpack, _c_str_dltensor), ValueError(
        'Invalid DLPack Tensor. DLTensor capsules can be consumed only once.')
    dlpack_handle = ctypes.c_void_p(ctypes.pythonapi.PyCapsule_GetPointer(dlpack, _c_str_dltensor))
    check_call(_LIB.MXNDArrayFromDLPackEx(dlpack_handle, False, ctypes.byref(handle)))
    # Rename PyCapsule (DLPack)
    ctypes.pythonapi.PyCapsule_SetName(dlpack, _c_str_used_dltensor)
    # delete the deleter of the old dlpack
    ctypes.pythonapi.PyCapsule_SetDestructor(dlpack, None)
    return ndarray(handle=handle)

def to_dlpack_for_read(data):
    """Returns a reference view of np.ndarray that represents as DLManagedTensor until
       all previous write operations on the current array are finished.

    Parameters
    ----------
    data: np.ndarray
        input data.

    Returns
    -------
    PyCapsule (the pointer of DLManagedTensor)
        a reference view of ndarray that represents as DLManagedTensor.

    Examples
    --------
    >>> x = mx.np.ones((2,3))
    >>> y = mx.npx.to_dlpack_for_read(x)
    >>> type(y)
    <class 'PyCapsule'>
    >>> z = mx.npx.from_dlpack(y)
    >>> z
    array([[1., 1., 1.],
           [1., 1., 1.]])
    """
    data.wait_to_read()
    dlpack = DLPackHandle()
    check_call(_LIB.MXNDArrayToDLPack(data.handle, ctypes.byref(dlpack)))
    return ctypes.pythonapi.PyCapsule_New(dlpack, _c_str_dltensor, _c_dlpack_deleter)

def to_dlpack_for_write(data):
    """Returns a reference view of ndarray that represents as DLManagedTensor until
       all previous read/write operations on the current array are finished.

    Parameters
    ----------
    data: np.ndarray
        input data.

    Returns
    -------
    PyCapsule (the pointer of DLManagedTensor)
        a reference view of np.ndarray that represents as DLManagedTensor.

    Examples
    --------
    >>> x = mx.np.ones((2,3))
    >>> w = mx.npx.to_dlpack_for_write(x)
    >>> type(w)
    <class 'PyCapsule'>
    >>> u = mx.npx.from_dlpack(w)
    >>> u += 1
    >>> x
    array([[2., 2., 2.],
           [2., 2., 2.]])
    """
    check_call(_LIB.MXNDArrayWaitToWrite(data.handle))
    dlpack = DLPackHandle()
    check_call(_LIB.MXNDArrayToDLPack(data.handle, ctypes.byref(dlpack)))
    return ctypes.pythonapi.PyCapsule_New(dlpack, _c_str_dltensor, _c_dlpack_deleter)
