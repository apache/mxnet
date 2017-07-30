# coding: utf-8
"""SparseNDArray API of mxnet."""
from __future__ import absolute_import
from __future__ import division
try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

import ctypes
import warnings

import os as _os
import sys as _sys

# import operator
import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call
from ..context import Context
from . import _internal
from .ndarray import _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import NDArray, _storage_type, _zeros_ndarray, _array
from . import cast_storage
from . import slice as nd_slice

# Use different verison of SymbolBase
# When possible, use cython to speedup part of computation.
# pylint: disable=unused-import
try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from .._ctypes.ndarray import NDArrayBase, _set_ndarray_class
    elif _sys.version_info >= (3, 0):
        from .._cy3.ndarray import NDArrayBase, _set_ndarray_class
    else:
        from .._cy2.ndarray import NDArrayBase, _set_ndarray_class
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from .._ctypes.ndarray import NDArrayBase, _set_ndarray_class

# pylint: enable=unused-import
_STORAGE_AUX_TYPES = {
    'row_sparse': [np.int64],
    'csr': [np.int64, np.int64]
}


def _new_alloc_handle(stype, shape, ctx, delay_alloc, dtype, aux_types, aux_shapes=None):
    """Return a new handle with specified storage type, shape, dtype and context.

    Empty handle is only used to hold results

    Returns
    -------
    handle
        A new empty ndarray handle
    """
    hdl = NDArrayHandle()
    aux_type_ids = [int(_DTYPE_NP_TO_MX[np.dtype(aux_t).type]) for aux_t in aux_types]
    aux_shapes = [(0,) for aux_t in aux_types] if aux_shapes is None else aux_shapes
    aux_shape_lens = [len(aux_shape) for aux_shape in aux_shapes]
    aux_shapes = sum(aux_shapes, ())
    num_aux = mx_uint(len(aux_types))
    check_call(_LIB.MXNDArrayCreateSparseEx(
        ctypes.c_int(int(_STORAGE_TYPE_STR_TO_ID[stype])),
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        num_aux,
        c_array(ctypes.c_int, aux_type_ids),
        c_array(mx_uint, aux_shape_lens),
        c_array(mx_uint, aux_shapes),
        ctypes.byref(hdl)))
    return hdl

class BaseSparseNDArray(NDArray):
    """The base class of an NDArray stored in a sparse storage format.

    See CSRNDArray and RowSparseNDArray for more details.
    """

    def __iadd__(self, other):
        raise NotImplementedError()

    def __isub__(self, other):
        raise NotImplementedError()

    def __imul__(self, other):
        raise NotImplementedError()

    def __idiv__(self, other):
        raise NotImplementedError()

    def __itruediv__(self, other):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value. Only slice [:] is supported.

        Parameters
        ----------
        key : slice
            The indexing key.
        value : NDArray or numpy.ndarray
            The value to set.

        Examples
        --------
        >>> src = mx.nd.row_sparse([[1, 0, 2], [4, 5, 6]], [0, 2], (3,3))
        >>> src.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  0.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> # assign BaseSparseNDArray with same storage type
        >>> x = mx.nd.zeros('row_sparse', (3,3))
        >>> x[:] = src
        >>> x.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  0.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> # assign NDArray to BaseSparseNDArray
        >>> x[:] = mx.nd.ones((3,3))
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        """
        if not self.writable:
            raise ValueError('Failed to assign to a readonly NDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('Assignment with slicing not supported in BaseSparseNDArray.')
            if isinstance(value, NDArray):
                # avoid copying to itself
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                raise Exception("Assigning numeric types to BaseSparseNDArray not supported yet.")
            elif isinstance(value, (np.ndarray, np.generic)):
                # TODO(haibin) Implement _sync_copyfrom for sparse ndarray to avoid an extra copy
                warnings.warn('Assigning non-NDArray object to BaseSparseNDArray is not efficient',
                              RuntimeWarning)
                tmp = _array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert(isinstance(key, (int, tuple)))
            raise Exception('BaseSparseNDArray only supports [:] for assignment')

    def __getitem__(self, key):
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of this array.

        Parameters
        ----------
        key : int or slice
            Indexing key.

        Examples
        --------
        >>> x = mx.nd.zeros((2, 3), stype='row_sparse')
        >>> x[:] = mx.nd.arange(0,6).reshape((2,3))
        >>> x.asnumpy()
        array([[ 0.,  1.,  2.],
               [ 3.,  4.,  5.]], dtype=float32)
        >>> x[1:2].asnumpy()
        array([[ 3.,  4.,  5.]], dtype=float32)
        """
        stype = self.stype
        if isinstance(key, int):
            raise Exception("__getitem__ with int key is not implemented yet")
        if isinstance(key, py_slice):
            if key.step is not None:
                raise ValueError('NDArray only supports continuous slicing on axis 0')
            if key.start is not None or key.stop is not None:
                assert(stype == 'csr'), "__getitem__ with slice is only implemented for CSRNDArray"
                begin = key.start if key.start else 0
                end = key.stop if key.stop else self.shape[0]
                return nd_slice(self, begin=begin, end=end)
            else:
                return self
        if isinstance(key, tuple):
            raise ValueError('Multi-dimension indexing is not supported')

    def _sync_copyfrom(self, source_array):
        raise Exception('Not implemented for SparseND yet!')

    def _at(self, idx):
        raise NotSupportedForSparseNDArray(self._at, '[idx]', idx)

    def _slice(self, start, stop):
        raise NotSupportedForSparseNDArray(self._slice, None, start, stop)

    def reshape(self, shape):
        raise NotSupportedForSparseNDArray(self.reshape, None, shape)

    def _aux_type(self, i):
        """Data-type of the array’s ith aux data.

        Returns
        -------
        numpy.dtype
            This BaseSparseNDArray's aux data type.
        """
        aux_type = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetAuxType(self.handle, i, ctypes.byref(aux_type)))
        return _DTYPE_MX_TO_NP[aux_type.value]

    @property
    def data(self):
        """Get a deep copy NDArray of the data array associated with the BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
        """
        self.wait_to_read()
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetDataNDArray(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl)

    @property
    def indices(self):
        """Get a deep copy NDArray of the indices array associated with the BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
        """
        raise NotImplementedError()

    @property
    def _num_aux(self):
        """The number of aux data used to help store the sparse ndarray.
        """
        return len(_STORAGE_AUX_TYPES[self.stype])

    @property
    def _aux_types(self):
        """The data types of the aux data for the BaseSparseNDArray.
        """
        aux_types = []
        num_aux = self._num_aux
        for i in range(num_aux):
            aux_types.append(self._aux_type(i))
        return aux_types

    def asnumpy(self):
        """Return a dense ``numpy.ndarray`` object with value copied from this array
        """
        return self.todense().asnumpy()

    def astype(self, dtype):
        """Returns a copy of the array after casting to a specified type.
        Parameters
        ----------
        dtype : numpy.dtype or str
            The type of the returned array.
        Examples
        --------
        >>> x = mx.nd.zeros('row_sparse', (2,3), dtype='float32')
        >>> y = x.astype('int32')
        >>> y.dtype
        <type 'numpy.int32'>
        """
        res = _zeros_sparse_ndarray(shape=self.shape, ctx=self.context,
                                    dtype=dtype, stype=self.stype)
        self.copyto(res)
        return res

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``NDArray`` object, then ``other.shape`` and
        ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``NDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : NDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray
            The copied array. If ``other`` is an ``NDArray``, then the return value
            and ``other`` will point to the same ``NDArray``.
        """
        if isinstance(other, NDArray):
            if other.handle is self.handle:
                warnings.warn('You are attempting to copy an array to itself', RuntimeWarning)
                return
            return _internal._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = _ndarray_cls(_new_alloc_handle(self.stype, self.shape, other,
                                                  True, self.dtype, self._aux_types))
            return _internal._copyto(self, out=hret)
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def todense(self):
        return todense(self)

    def _aux_data(self, i):
        """ Get a deep copy NDArray of the i-th aux data array associated with the
        BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
        """
        self.wait_to_read()
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetAuxNDArray(self.handle, i, ctypes.byref(hdl)))
        return NDArray(hdl)


# pylint: disable=abstract-method
class CSRNDArray(BaseSparseNDArray):
    """A CSRNDArray represents a NDArray as three separate arrays: `data`,
    `indptr` and `indices`. It uses the standard CSR representation where the column indices for
    row i are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored
    in values[indptr[i]:indptr[i+1]].

    Example
    -------
    >>> a = mx.nd.array([[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 0, 3]])
    >>> a = a._to_csr()
    >>> a.indices.asnumpy()
    array([1, 0, 2])
    >>> a.indptr.asnumpy()
    array([0, 1, 2, 2, 3])
    >>> a.data.asnumpy()
    array([ 1.,  2.,  3.], dtype=float32)
    """

    def __reduce__(self):
        return CSRNDArray, (None,), super(CSRNDArray, self).__getstate__()

    def __iadd__(self, other):
        (self + other).copyto(self)
        return self

    def __isub__(self, other):
        (self - other).copyto(self)
        return self

    def __imul__(self, other):
        (self * other).copyto(self)
        return self

    def __idiv__(self, other):
        (self / other).copyto(self)
        return self

    def __itruediv__(self, other):
        (self / other).copyto(self)
        return self

    @property
    def indices(self):
        """The indices array of the CSRNDArray.
        This generates a deep copy of the column indices of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's indices array.
        """
        return self._aux_data(1)

    @property
    def indptr(self):
        """The indptr array of the CSRNDArray with `csr` storage type.
        This generates a deep copy of the `indptr` of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's indptr array.
        """
        return self._aux_data(0)

# pylint: disable=abstract-method
class RowSparseNDArray(BaseSparseNDArray):
    """A RowSparseNDArray is typically used to represent a subset of a larger
    NDArray  with `default` of shape [LARGE0, D1, .. , DN] where LARGE0 >> D0. The values
    in indices are the indices in the first dimension of the slices that have been extracted from
    the larger NDArray. The indices are expected to be sorted in ascending order.

    The corresponding NDArray ``dense`` with `default` storage represented by a ``rsp``
    RowSparseNDArray

    ``dense[rsp.indices[i], :, :, :, ...] = rsp.values[i, :, :, :, ...]``

    RowSparseNDArray is used principally in the definition of gradients for operations
    that have sparse gradients (e.g. dot with sparse inputs).

    Examples
    --------
    >>> import mxnet as mx
    >>> dense = mx.nd.array([[1,2],[0,0],[3,0],[0,0]])
    >>> rsp = dense._to_rsp()
    >>> rsp.indices.asnumpy()
    array([0, 2], dtype=int32)
    >>> rsp.data.asnumpy()
    array([[ 1.,  2.],
           [ 3.,  0.]], dtype=float32)
    """
    def __reduce__(self):
        return RowSparseNDArray, (None,), super(RowSparseNDArray, self).__getstate__()

    def __iadd__(self, other):
        (self + other).copyto(self)
        return self

    def __isub__(self, other):
        (self - other).copyto(self)
        return self

    def __imul__(self, other):
        (self * other).copyto(self)
        return self

    def __idiv__(self, other):
        (self / other).copyto(self)
        return self

    def __itruediv__(self, other):
        (self / other).copyto(self)
        return self

    @property
    def indices(self):
        """The indices array of the RowSparseNDArray with `row_sparse` storage type.
        This generates a deep copy of the row indices of the current row-sparse matrix.

        Returns
        -------
        NDArray
            This RowSparseNDArray's indices array.
        """
        return self._aux_data(0)


def _prepare_src_array(src, dtype, default_dtype):
    if isinstance(src, NDArray):
        dtype = src.dtype if dtype is None else dtype
    else:
        dtype = default_dtype if dtype is None else dtype
        if not isinstance(src, np.ndarray):
            try:
                src = np.array(src, dtype=dtype)
            except:
                raise TypeError('values must be array like object')
    return src, dtype


def csr(data, indptr, indices, shape, ctx=None, dtype=None, indptr_type=None, indices_type=None):
    """Creates a 2D array with compressed sparse row format.

    Parameters
    ----------
    data: array_like
        An object exposing the array interface, with shape [nnz], where D0 is the number of
        non-zero entries.
    indptr: array_like
        An object exposing the array interface, with shape [D0 + 1]. The first element in indptr
        should always be zero.
    indices: array_like
        An object exposing the array interface, with shape [nnz].
    ctx: Context, optional
        Device context (default is the current default context).
    dtype: str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``values.dtype``
        if `values` is an `NDArray`, `float32` otherwise.
    indptr_type: str or numpy.dtype, optional
        The data type of the indices array. The default dtype is ``indptr.dtype``
        if `indptr` is an `NDArray`, `int32` otherwise.
    indices_type: str or numpy.dtype, optional
        The data type of the indices array. The default dtype is ``indices.dtype``
        if `indicies` is an `NDArray`, `int32` otherwise.

    Returns
    -------
    CSRNDArray
        A `CSRNDArray` with the `csr` storage representation.

    Example
    -------
    >>> import mxnet as mx
    >>> a = mx.nd.csr([1, 2, 3], [0, 1, 2, 2, 3], [1, 0, 2], (4, 3))
    >>> a.asnumpy()
    array([[ 0.,  1.,  0.],
           [ 2.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  3.]], dtype=float32)
    """
    storage_type = 'csr'
    # context
    if ctx is None:
        ctx = Context.default_ctx
    # prepare src array and types
    data, dtype = _prepare_src_array(data, dtype, mx_real_t)
    indptr, indptr_type = _prepare_src_array(indptr, indptr_type,
                                             _STORAGE_AUX_TYPES[storage_type][0])
    indices, indices_type = _prepare_src_array(indices, indices_type,
                                               _STORAGE_AUX_TYPES[storage_type][1])
    # verify types
    assert('int64' in str(indptr_type)), "expected int64 for indptr"
    assert('int64' in str(indices_type)), "expected int64 for indices"
    # verify shapes
    aux_shapes = [indptr.shape, indices.shape]
    assert(data.ndim == 1)
    assert(indptr.ndim == 1)
    assert(indices.ndim == 1)
    assert(len(shape) == 2)
    result = CSRNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype,
                                          [indptr_type, indices_type], aux_shapes))
    # TODO(junwu): Convert data, indptr, and indices to mxnet NDArrays
    # if they are not for now. In the future, we should provide a c-api
    # to accept np.ndarray types to copy from to result.data and aux_data
    if not isinstance(data, NDArray):
        data = _array(data, ctx, dtype)
    if not isinstance(indptr, NDArray):
        indptr = _array(indptr, ctx, indptr_type)
    if not isinstance(indices, NDArray):
        indices = _array(indices, ctx, indices_type)
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, data.handle, ctypes.c_int(-1)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indptr.handle, ctypes.c_int(0)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indices.handle, ctypes.c_int(1)))
    return result


def row_sparse(data, indices, shape, ctx=None, dtype=None, indices_type=None):
    """Creates a row sparse array with a set of tensor slices at given indices.

    Parameters
    ----------
    data: array_like
        An object exposing the array interface, with shape [D0, D1, .. Dn], where D0 is
        the number of rows with non-zeros entries.
    indices: array_like
        An object exposing the array interface, with shape [D0].
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``data.dtype``
        if `data` is an `NDArray`, `float32` otherwise.
    indices_type: str or numpy.dtype, optional
        The data type of the indices array. The default dtype is ``indices.dtype``
        if `indicies` is an `NDArray`, `int32` otherwise.

    Returns
    -------
    RowSparseNDArray
        An `RowSparseNDArray` with the `row_sparse` storage representation.

    Example
    -------
    >>> a = mx.nd.row_sparse([[1, 2], [3, 4]], [1, 4], (6, 2))
    >>> a.asnumpy()
    array([[ 0.,  0.],
           [ 1.,  2.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 3.,  4.],
           [ 0.,  0.]], dtype=float32)
    """
    storage_type = 'row_sparse'
    # context
    if ctx is None:
        ctx = Context.default_ctx
    # prepare src array and types
    data, dtype = _prepare_src_array(data, dtype, mx_real_t)
    indices, indices_type = _prepare_src_array(indices, indices_type,
                                               _STORAGE_AUX_TYPES[storage_type][0])
    # verify types
    assert('int64' in str(indices_type)), "expected int64 for indices"
    # verify shapes
    assert(data.ndim == len(shape))
    assert(indices.ndim == 1)
    result = RowSparseNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype,
                                                [indices_type], [indices.shape]))

    # TODO(junwu): Convert data, indptr, and indices to mxnet NDArrays
    # if they are not for now. In the future, we should provide a c-api
    # to accept np.ndarray types to copy from to result.data and aux_data
    if not isinstance(data, NDArray):
        data = _array(data, ctx, dtype)
    if not isinstance(indices, NDArray):
        indices = _array(indices, ctx, indices_type)
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, data.handle, ctypes.c_int(-1)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indices.handle, ctypes.c_int(0)))
    return result


def todense(source):
    """ Return a dense array representation of a BaseSparseNDArray.

    Returns
    -------
    NDArray
        A copy of the array with `default` storage stype
    """
    return cast_storage(source, stype='default')


def _ndarray_cls(handle, writable=True, stype=None):
    if stype is None:
        stype = _storage_type(handle)
    if stype == 'default':
        return NDArray(handle, writable=writable)
    elif stype == 'csr':
        return CSRNDArray(handle, writable=writable)
    elif stype == 'row_sparse':
        return RowSparseNDArray(handle, writable=writable)
    else:
        raise Exception("unknown storage type")


_set_ndarray_class(_ndarray_cls)


def _zeros_sparse_ndarray(stype, shape, ctx=None, dtype=None, aux_types=None, **kwargs):
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    stype: string
        The storage type of the empty array, such as 'row_sparse', 'csr', etc
    shape : int or tuple of int
        The shape of the empty array
    ctx : Context, optional
        An optional device context (default is the current default context)
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`)
    aux_types: list of numpy.dtype, optional
        An optional type for the aux data for BaseSparseNDArray (default values depends
        on the storage type)

    Returns
    -------
    RowSparseNDArray or CSRNDArray
        A created array
    Examples
    --------
    >>> mx.nd.zeros((1,2), mx.cpu(), stype='csr')
    <CSRNDArray 1x2 @cpu(0)>
    >>> mx.nd.zeros((1,2), mx.cpu(), 'float16', stype='row_sparse').asnumpy()
    array([[ 0.,  0.]], dtype=float16)
    """
    if stype == 'default':
        return _zeros_ndarray(shape, ctx=ctx, dtype=dtype, **kwargs)
    if ctx is None:
        ctx = Context.default_ctx
    dtype = mx_real_t if dtype is None else dtype
    if aux_types is None:
        if stype == 'row_sparse' or stype == 'csr':
            aux_types = _STORAGE_AUX_TYPES[stype]
        else:
            raise Exception("unknown storage type")
    assert(len(aux_types) == len(_STORAGE_AUX_TYPES[stype]))
    out = _ndarray_cls(_new_alloc_handle(stype, shape, ctx, True, dtype, aux_types))
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out, **kwargs)

def _empty_sparse_ndarray(stype, shape, ctx=None, dtype=None, aux_types=None):
    """Returns a new array of given shape and type, without initializing entries.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    assert(stype is not None)
    if stype == 'csr' or stype == 'row_sparse':
        return _zeros_sparse_ndarray(stype, shape, ctx=ctx, dtype=dtype, aux_types=aux_types)
    else:
        raise Exception("unknown stype : " + str(stype))

def _sparse_array(source_array, ctx=None, dtype=None, aux_types=None):
    """Creates a sparse array from any object exposing the array interface.
    """
    if isinstance(source_array, NDArray):
        assert(source_array.stype != 'default'), \
               "Please use `cast_storage` to create BaseSparseNDArray from an NDArray"
        dtype = source_array.dtype if dtype is None else dtype
        aux_types = source_array._aux_types if aux_types is None else aux_types
    else:
        # TODO(haibin/anisub) support creation from scipy object when `_sync_copy_from` is ready
        raise NotImplementedError('creating BaseSparseNDArray from ' \
                                  ' a non-NDArray object is not implemented.')
    arr = _empty_sparse_ndarray(source_array.stype, source_array.shape, ctx, dtype, aux_types)
    arr[:] = source_array
    return arr
