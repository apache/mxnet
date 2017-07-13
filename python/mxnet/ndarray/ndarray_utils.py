# coding: utf-8
"""Utility functions for NDArray and SparseNDArray."""
import ctypes

from ..base import _LIB, check_call, py_str, c_str, string_types, mx_uint, NDArrayHandle, c_array
from .ndarray import NDArray, _zeros_ndarray
from .sparse_ndarray import _ndarray_cls, _zeros_sparse_ndarray


def zeros(shape, ctx=None, dtype=None, stype=None, aux_types=None, **kwargs):
    if stype is None:
        return _zeros_ndarray(shape, ctx, dtype, **kwargs)
    else:
        return _zeros_sparse_ndarray(stype, shape, ctx, dtype, aux_types, **kwargs)


def load(fname):
    """Loads an array from file.

    See more details in ``save``.

    Parameters
    ----------
    fname : str
        The filename.

    Returns
    -------
    list of NDArray or dict of str to NDArray
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
    data : list of ``NDArray` or dict of str to ``NDArray``
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
    handles = []
    if isinstance(data, dict):
        keys = []
        for key, val in data.items():
            if not isinstance(key, string_types):
                raise TypeError('save only accept dict str->NDArray or list of NDArray')
            if not isinstance(val, NDArray):
                raise TypeError('save only accept dict str->NDArray or list of NDArray')
            keys.append(c_str(key))
            handles.append(val.handle)
        keys = c_array(ctypes.c_char_p, keys)
    else:
        for val in data:
            if not isinstance(val, NDArray):
                raise TypeError('save only accept dict str->NDArray or list of NDArray')
            handles.append(val.handle)
        keys = None
    check_call(_LIB.MXNDArraySave(c_str(fname),
                                  mx_uint(len(handles)),
                                  c_array(NDArrayHandle, handles),
                                  keys))
