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
from ..util import is_np_array, is_np_shape
from ..base import _LIB, check_call, string_types, c_str_array
from ..base import c_handle_array, c_str, mx_uint, NDArrayHandle, py_str
from ..dlpack import ndarray_to_dlpack_for_read, ndarray_to_dlpack_for_write
from ..dlpack import ndarray_from_dlpack, ndarray_from_numpy
from ..numpy import ndarray, array
from ..ndarray import NDArray

__all__ = ['save', 'savez', 'load', 'to_dlpack_for_read', 'to_dlpack_for_write',
           'from_dlpack', 'from_numpy']

def save(file, arr):
    """Save an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : str
        File or filename to which the data is saved.  If file is a file-object,
        then the filename is unchanged.
    arr : ndarray
        Array data to be saved. Sparse formats are not supported. Please use
        savez function to save sparse arrays.

    See Also
    --------
    savez : Save several arrays into a ``.npz`` archive

    Notes
    -----
    For a description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.
    """
    if not isinstance(arr, NDArray):
        raise ValueError("data needs to either be a MXNet ndarray")
    arr = [arr]
    keys = None
    handles = c_handle_array(arr)
    check_call(_LIB.MXNDArraySave(c_str(file), mx_uint(len(handles)), handles, keys))


def savez(file, *args, **kwds):
    """Save several arrays into a single file in uncompressed ``.npz`` format.

    If arguments are passed in with no keywords, the corresponding variable
    names, in the ``.npz`` file, are 'arr_0', 'arr_1', etc. If keyword
    arguments are given, the corresponding variable names, in the ``.npz``
    file will match the keyword names.

    Parameters
    ----------
    file : str
        Either the filename (string) or an open file (file-like object)
        where the data will be saved.
    args : Arguments, optional
        Arrays to save to the file. Since it is not possible for Python to
        know the names of the arrays outside `savez`, the arrays will be saved
        with names "arr_0", "arr_1", and so on. These arguments can be any
        expression.
    kwds : Keyword arguments, optional
        Arrays to save to the file. Arrays will be saved in the file with the
        keyword names.

    Returns
    -------
    None

    See Also
    --------
    save : Save a single array to a binary file in NumPy format.

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  The archive is not compressed and each file
    in the archive contains one variable in ``.npy`` format. For a
    description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.

    When opening the saved ``.npz`` file with `load` a dictionary object
    mapping file-names to the arrays themselves.

    When saving dictionaries, the dictionary keys become filenames
    inside the ZIP archive. Therefore, keys should be valid filenames.
    E.g., avoid keys that begin with ``/`` or contain ``.``.
    """

    if len(args):
        for i, arg in enumerate(args):
            name = 'arr_{}'.format(str(i))
            assert name not in kwds, 'Naming conflict between arg {} and kwargs.'.format(str(i))
            kwds[name] = arg

    str_keys = kwds.keys()
    nd_vals = kwds.values()
    if any(not isinstance(k, string_types) for k in str_keys) or \
            any(not isinstance(v, NDArray) for v in nd_vals):
        raise TypeError('Only accepts dict str->ndarray or list of ndarrays')

    keys = c_str_array(str_keys)
    handles = c_handle_array(nd_vals)
    check_call(_LIB.MXNDArraySave(c_str(file), mx_uint(len(handles)), handles, keys))


def load(file):
    """Load arrays from ``.npy``, ``.npz`` or legacy MXNet file format.

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
        if out_size.value != 1:
            return [ndarray(NDArrayHandle(handles[i])) for i in range(out_size.value)]
        return ndarray(NDArrayHandle(handles[0]))
    else:
        assert out_name_size.value == out_size.value
        return dict(
            (py_str(names[i]), ndarray(NDArrayHandle(handles[i])))
            for i in range(out_size.value))

from_dlpack = ndarray_from_dlpack(ndarray)
from_dlpack_doc = """Returns a np.ndarray backed by a dlpack tensor.

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
from_dlpack.__doc__ = from_dlpack_doc


from_numpy = ndarray_from_numpy(ndarray, array)
from_numpy_doc = """Returns an MXNet's np.ndarray backed by numpy's ndarray.
    When `zero_copy` is set to be true,
    this API consumes numpy's ndarray and produces MXNet's np.ndarray
    without having to copy the content. In this case, we disallow
    users to modify the given numpy ndarray, and it is suggested
    not to read the numpy ndarray as well for internal correctness.

    Parameters
    ----------
    ndarray: np.ndarray
        input data
    zero_copy: bool
        Whether we use DLPack's zero-copy conversion to convert to MXNet's
        np.ndarray.
        This is only available for c-contiguous arrays, i.e. array.flags[C_CONTIGUOUS] == True.

    Returns
    -------
    np.ndarray
        a np.ndarray backed by a dlpack tensor
    """
from_numpy.__doc__ = from_numpy_doc

to_dlpack_for_read = ndarray_to_dlpack_for_read()
to_dlpack_for_read_doc = """Returns a reference view of np.ndarray that represents
as DLManagedTensor until all previous write operations on the current array are finished.

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
to_dlpack_for_read.__doc__ = to_dlpack_for_read_doc

to_dlpack_for_write = ndarray_to_dlpack_for_write()
to_dlpack_for_write_doc = """Returns a reference view of ndarray that represents
as DLManagedTensor until all previous read/write operations on the current array are finished.

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
to_dlpack_for_write.__doc__ = to_dlpack_for_write_doc
