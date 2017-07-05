# coding: utf-8
"""Utility functions for NDArray and SparseNDArray."""
import ctypes
import sys as _sys

from mxnet import Context
from mxnet.base import mx_real_t, _LIB, check_call, py_str, c_str, string_types, mx_uint,\
    NDArrayHandle, c_array
from mxnet.ndarray import NDArray
from mxnet.sparse_ndarray import _STORAGE_AUX_TYPES, _new_alloc_handle, _ndarray_cls
from . import _ndarray_internal as _internal


def _zeros_ndarray(shape, ctx=None, dtype=None, **kwargs):
    """Returns a new array filled with all zeros, with the given shape and type.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    ctx : Context, optional
        An optional device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`).
    out : NDArray, optional
        The output NDArray (default is `None`).

    Returns
    -------
    NDArray
        A created array

    Examples
    --------
    >>> mx.nd.zeros(1).asnumpy()
    array([ 0.], dtype=float32)
    >>> mx.nd.zeros((1,2), mx.gpu(0))
    <NDArray 1x2 @gpu(0)>
    >>> mx.nd.zeros((1,2), mx.gpu(0), 'float16').asnumpy()
    array([[ 0.,  0.]], dtype=float16)
    """
    # pylint: disable= unused-argument
    if ctx is None:
        ctx = Context.default_ctx
    dtype = mx_real_t if dtype is None else dtype
    # pylint: disable= no-member, protected-access
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, **kwargs)
    # pylint: enable= no-member, protected-access


def _zeros_sparse_ndarray(storage_type, shape, ctx=None, dtype=None, aux_types=None, **kwargs):
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array
    storage_type: string
        The storage type of the empty array, such as 'row_sparse', 'csr', etc
    ctx : Context, optional
        An optional device context (default is the current default context)
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`)
    aux_types: list of numpy.dtype, optional
        An optional type for the aux data for SparseNDArray (default values depends
        on the storage type)

    Returns
    -------
    SparseNDArray
        A created array
    Examples
    --------
    >>> mx.sparse_nd.zeros('csr', (1,2), mx.gpu(0))
    <SparseNDArray 1x2 @gpu(0)>
    >>> mx.sparse_nd.zeros('row_sparse', (1,2), mx.gpu(0), 'float16').asnumpy()
    array([[ 0.,  0.]], dtype=float16)
    """
    if storage_type == 'default':
        return _zeros_ndarray(shape, ctx=ctx, dtype=dtype, **kwargs)
    if ctx is None:
        ctx = Context.default_ctx
    dtype = mx_real_t if dtype is None else dtype
    if aux_types is None:
        if storage_type == 'row_sparse' or storage_type == 'csr':
            aux_types = _STORAGE_AUX_TYPES[storage_type]
        else:
            raise Exception("unknown storage type")
    assert(len(aux_types) == len(_STORAGE_AUX_TYPES[storage_type]))
    out = _ndarray_cls(_new_alloc_handle(storage_type, shape, ctx, True, dtype, aux_types))
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out, **kwargs)


def zeros(shape, ctx=None, dtype=None, storage_type=None, aux_types=None, **kwargs):
    if storage_type is None:
        return _zeros_ndarray(shape, ctx, dtype, **kwargs)
    else:
        return _zeros_sparse_ndarray(storage_type, shape, ctx, dtype, aux_types, **kwargs)


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


def _init_ndarray_module_frontend(function, root_namespace, module_name):
    """Register front end functions defined in this file to mxnet.ndarray module.
    The functions registered were originally defined in mxnet.ndarray. They were
    moved here because they need to know SparseNDArray class, while it's not allowed
    in ndarray.py since that would result in circular import."""
    module_obj = _sys.modules["%s.%s" % (root_namespace, module_name)]
    setattr(module_obj, function.__name__, function)


# register the following front end functions in mx.nd
_init_ndarray_module_frontend(zeros, "mxnet", "ndarray")
_init_ndarray_module_frontend(load, "mxnet", "ndarray")
_init_ndarray_module_frontend(save, "mxnet", "ndarray")
