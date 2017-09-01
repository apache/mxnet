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
"""Sparse NDArray API of MXNet."""

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

__all__ = ["_ndarray_cls", "csr_matrix", "row_sparse_array",
           "BaseSparseNDArray", "CSRNDArray", "RowSparseNDArray"]

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
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR
from .ndarray import NDArray, _storage_type
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from . import op

# When possible, use cython to speedup part of computation.
# pylint: disable=unused-import
try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from .._ctypes.ndarray import _set_ndarray_class
    elif _sys.version_info >= (3, 0):
        from .._cy3.ndarray import _set_ndarray_class
    else:
        from .._cy2.ndarray import _set_ndarray_class
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from .._ctypes.ndarray import _set_ndarray_class
# pylint: enable=unused-import

try:
    import scipy.sparse as spsp
except ImportError:
    spsp = None

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

    def __repr__(self):
        """Returns a string representation of the sparse array."""
        shape_info = 'x'.join(['%d' % x for x in self.shape])
        # The data content is not displayed since the array usually has big shape
        return '\n<%s %s @%s>' % (self.__class__.__name__,
                                  shape_info, self.context)

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

    def _sync_copyfrom(self, source_array):
        raise NotImplementedError()

    def _at(self, idx):
        raise NotSupportedForSparseNDArray(self._at, '[idx]', idx)

    def _slice(self, start, stop):
        raise NotSupportedForSparseNDArray(self._slice, None, start, stop)

    def reshape(self, shape):
        raise NotSupportedForSparseNDArray(self.reshape, None, shape)

    def _aux_type(self, i):
        """Data-type of the array's ith aux data.

        Returns
        -------
        numpy.dtype
            This BaseSparseNDArray's aux data type.
        """
        aux_type = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetAuxType(self.handle, i, ctypes.byref(aux_type)))
        return _DTYPE_MX_TO_NP[aux_type.value]

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
        return self.tostype('default').asnumpy()

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
        res = zeros(shape=self.shape, ctx=self.context,
                    dtype=dtype, stype=self.stype)
        self.copyto(res)
        return res

    def copyto(self, other):
        """Copies the value of this array to another array.

        Parameters
        ----------
        other : NDArray or CSRNDArray or RowSparseNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or CSRNDArray or RowSparseNDArray
            The copied array.
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

    def _data(self):
        """A deep copy NDArray of the data array associated with the BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
        """
        self.wait_to_read()
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetDataNDArray(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl)


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
    """A sparse representation of 2D NDArray in the standard CSR format.

    A CSRNDArray represents an NDArray as three separate arrays: `data`,
    `indptr` and `indices`. It uses the standard CSR representation where the column indices for
    row i are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored
    in values[indptr[i]:indptr[i+1]].

    The column indices for a given row are expected to be sorted in ascending order.
    Duplicate column entries for the same row are not allowed.

    Example
    -------
    >>> a = mx.nd.array([[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 0, 3]])
    >>> a = a.tostype('csr')
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

    def __getitem__(self, key):
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of this array.

        Parameters
        ----------
        key : slice
            Indexing key.

        Examples
        --------
        >>> indptr = np.array([0, 2, 3, 6])
        >>> indices = np.array([0, 2, 2, 0, 1, 2])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> a = mx.nd.sparse.csr_matrix(data, indptr, indices, (3, 3))
        >>> a.asnumpy()
        array([[1, 0, 2],
               [0, 0, 3],
               [4, 5, 6]])
        >>> a[1:2].asnumpy()
        array([[0, 0, 3]], dtype=float32)
        """
        if isinstance(key, int):
            raise ValueError("__getitem__ with int key is not implemented for CSRNDArray")
        if isinstance(key, py_slice):
            if key.step is not None:
                raise ValueError('CSRNDArray only supports continuous slicing on axis 0')
            if key.start is not None or key.stop is not None:
                begin = key.start if key.start else 0
                end = key.stop if key.stop else self.shape[0]
                return op.slice(self, begin=begin, end=end)
            else:
                return self
        if isinstance(key, tuple):
            raise ValueError('Multi-dimension indexing is not supported')

    def __setitem__(self, key, value):
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value. Only slice key [:] is supported.

        Parameters
        ----------
        key : slice
            The indexing key.
        value : NDArray or CSRNDArray or numpy.ndarray
            The value to set.

        Examples
        --------
        >>> src = mx.nd.zeros((3,3), stype='csr')
        >>> src.asnumpy()
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
        >>> # assign CSRNDArray with same storage type
        >>> x = mx.nd.ones('row_sparse', (3,3)).tostype('csr')
        >>> x[:] = src
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> # assign NDArray to CSRNDArray
        >>> x[:] = mx.nd.ones((3,3)) * 2
        >>> x.asnumpy()
        array([[ 2.,  2.,  2.],
               [ 2.,  2.,  2.],
               [ 2.,  2.,  2.]], dtype=float32)
        """
        if not self.writable:
            raise ValueError('Failed to assign to a readonly CSRNDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('Assignment with slice for CSRNDArray is not ' \
                                 'implmented yet.')
            if isinstance(value, NDArray):
                # avoid copying to itself
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                raise ValueError("Assigning numeric types to CSRNDArray is " \
                                 "not implemented yet.")
            elif isinstance(value, (np.ndarray, np.generic)):
                # TODO(haibin/anisub) check scipy.sparse and use _sync_copy_from to
                # avoid the temporary copy
                warnings.warn('Assigning non-NDArray object to CSRNDArray is not efficient',
                              RuntimeWarning)
                tmp = _array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert(isinstance(key, (int, tuple)))
            raise Exception('CSRNDArray only supports [:] for assignment')

    @property
    def indices(self):
        """A deep copy NDArray of the indices array of the CSRNDArray.
        This generates a deep copy of the column indices of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's indices array.
        """
        return self._aux_data(1)

    @property
    def indptr(self):
        """A deep copy NDArray of the indptr array of the CSRNDArray.
        This generates a deep copy of the `indptr` of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's indptr array.
        """
        return self._aux_data(0)

    @property
    def data(self):
        """A deep copy NDArray of the data array of the CSRNDArray.
        This generates a deep copy of the `data` of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's data array.
        """
        return self._data()

    @indices.setter
    def indices(self, indices):
        raise NotImplementedError()

    @indptr.setter
    def indptr(self, indptr):
        raise NotImplementedError()

    @data.setter
    def data(self, data):
        raise NotImplementedError()


    def tostype(self, stype):
        """Return a copy of the array with chosen storage type.

        Returns
        -------
        NDArray or CSRNDArray
            A copy of the array with the chosen storage stype
        """
        if stype == 'row_sparse':
            raise ValueError("cast_storage from csr to row_sparse is not supported")
        return op.cast_storage(self, stype=stype)

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``NDArray`` or ``CSRNDArray`` object, then ``other.shape`` and
        ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``CSRNDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : NDArray or CSRNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or CSRNDArray
            The copied array. If ``other`` is an ``NDArray`` or ``CSRNDArray``, then the return
            value and ``other`` will point to the same ``NDArray`` or ``CSRNDArray``.
        """
        if isinstance(other, Context):
            return super(CSRNDArray, self).copyto(other)
        elif isinstance(other, NDArray):
            stype = other.stype
            if stype == 'default' or stype == 'csr':
                return super(CSRNDArray, self).copyto(other)
            else:
                raise TypeError('copyto does not support destination NDArray stype ' + str(stype))
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

# pylint: disable=abstract-method
class RowSparseNDArray(BaseSparseNDArray):
    """A sparse representation of a set of NDArray row slices at given indices.

    A RowSparseNDArray represents a multidimensional NDArray using two separate arrays: `data` and
    `indices`.

    - data: an NDArray of any dtype with shape [D0, D1, ..., Dn].
    - indices: a 1-D int64 NDArray with shape [D0].

    The `indices` stores the indices of the row slices with non-zeros,
    while the values are stored in `data`. The corresponding NDArray ``dense``
    represented by RowSparseNDArray ``rsp`` has

    ``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

        >>> dense.asnumpy()
        array([[ 1.,  2., 3.],
               [ 0.,  0., 0.],
               [ 4.,  0., 5.],
               [ 0.,  0., 0.],
               [ 0.,  0., 0.]], dtype=float32)
        >>> rsp = dense.tostype('row_sparse')
        >>> rsp.indices.asnumpy()
        array([0, 2], dtype=int64)
        >>> rsp.data.asnumpy()
        array([[ 1.,  2., 3.],
               [ 4.,  0., 5.]], dtype=float32)

    A RowSparseNDArray is typically used to represent non-zero row-slices of a large NDArray
    of shape [LARGE0, D1, .. , Dn] where LARGE0 >> D0 and most row slices are zeros.

    The indices are expected to be sorted in ascending order.

    RowSparseNDArray is used principally in the definition of gradients for operations
    that have sparse gradients (e.g. sparse dot and sparse embedding).
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

    def __getitem__(self, key):
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of this array.

        Parameters
        ----------
        key : slice
            Indexing key.

        Examples
        --------
        >>> x = mx.nd.zeros((2, 3), stype='row_sparse')
        >>> x[:].asnumpy()
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
        """
        if isinstance(key, int):
            raise Exception("__getitem__ with int key is not implemented for RowSparseNDArray yet")
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise Exception('RowSparseNDArray only supports [:] for __getitem__')
            else:
                return self
        if isinstance(key, tuple):
            raise ValueError('Multi-dimension indexing is not supported')

    def __setitem__(self, key, value):
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value. Only slice key [:] is supported.

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
        >>> # assign RowSparseNDArray with same storage type
        >>> x = mx.nd.zeros('row_sparse', (3,3))
        >>> x[:] = src
        >>> x.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  0.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> # assign NDArray to RowSparseNDArray
        >>> x[:] = mx.nd.ones((3,3))
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        """
        if not self.writable:
            raise ValueError('Failed to assign to a readonly RowSparseNDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('Assignment with slice for RowSparseNDArray ' \
                                 'is not implmented yet.')
            if isinstance(value, NDArray):
                # avoid copying to itself
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                raise ValueError("Assigning numeric types to RowSparseNDArray " \
                                 "is not implemented yet.")
            elif isinstance(value, (np.ndarray, np.generic)):
                warnings.warn('Assigning non-NDArray object to RowSparseNDArray is not efficient',
                              RuntimeWarning)
                tmp = _array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert(isinstance(key, (int, tuple)))
            raise TypeError('RowSparseNDArray only supports [:] for assignment')

    @property
    def indices(self):
        """A deep copy NDArray of the indices array of the RowSparseNDArray.
        This generates a deep copy of the row indices of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This RowSparseNDArray's indices array.
        """
        return self._aux_data(0)

    @property
    def data(self):
        """A deep copy NDArray of the data array of the RowSparseNDArray.
        This generates a deep copy of the `data` of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This RowSparseNDArray's data array.
        """
        return self._data()

    @indices.setter
    def indices(self, indices):
        raise NotImplementedError()

    @data.setter
    def data(self, data):
        raise NotImplementedError()

    def tostype(self, stype):
        """Return a copy of the array with chosen storage type.

        Returns
        -------
        NDArray or RowSparseNDArray
            A copy of the array with the chosen storage stype
        """
        if stype == 'csr':
            raise ValueError("cast_storage from row_sparse to csr is not supported")
        return op.cast_storage(self, stype=stype)

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``NDArray`` or ``RowSparseNDArray`` object, then ``other.shape``
        and ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``RowSparseNDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : NDArray or RowSparseNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or RowSparseNDArray
            The copied array. If ``other`` is an ``NDArray`` or ``RowSparseNDArray``, then the
            return value and ``other`` will point to the same ``NDArray`` or ``RowSparseNDArray``.
        """
        if isinstance(other, Context):
            return super(RowSparseNDArray, self).copyto(other)
        elif isinstance(other, NDArray):
            stype = other.stype
            if stype == 'default' or stype == 'row_sparse':
                return super(RowSparseNDArray, self).copyto(other)
            else:
                raise TypeError('copyto does not support destination NDArray stype ' + str(stype))
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))


def _prepare_src_array(src, dtype, default_dtype):
    """Prepare `src` and its dtype so that they can be used to construct NDArray.
    `src` is converted to a `np.ndarray` if it's neither an `NDArray` nor an `np.ndarray`.
    """
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


def csr_matrix(data, indptr, indices, shape, ctx=None, dtype=None, indptr_type=None,
               indices_type=None):
    """Creates a 2D array with compressed sparse row(CSR) format.

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
        if `indptr` is an `NDArray`, `int64` otherwise.
    indices_type: str or numpy.dtype, optional
        The data type of the indices array. The default dtype is ``indices.dtype``
        if `indicies` is an `NDArray`, `int64` otherwise.

    Returns
    -------
    CSRNDArray
        A `CSRNDArray` with the `csr` storage representation.

    Example
    -------
    >>> import mxnet as mx
    >>> a = mx.nd.sparse.csr_matrix([1, 2, 3], [0, 1, 2, 2, 3], [1, 0, 2], (4, 3))
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


def row_sparse_array(data, indices, shape, ctx=None, dtype=None, indices_type=None):
    """Creates a multidimensional row sparse array with a set of tensor slices at given indices.

    Parameters
    ----------
    data: array_like
        An object exposing the array interface, with shape [D0, D1, .. DK], where D0 is
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
        if `indicies` is an `NDArray`, `int64` otherwise.

    Returns
    -------
    RowSparseNDArray
        An `RowSparseNDArray` with the `row_sparse` storage representation.

    Example
    -------
    >>> a = mx.nd.sparse.row_sparse_array([[1, 2], [3, 4]], [1, 4], (6, 2))
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


def _ndarray_cls(handle, writable=True, stype=_STORAGE_TYPE_UNDEFINED):
    if stype == _STORAGE_TYPE_UNDEFINED:
        stype = _storage_type(handle)
    if stype == _STORAGE_TYPE_DEFAULT:
        return NDArray(handle, writable=writable)
    elif stype == _STORAGE_TYPE_CSR:
        return CSRNDArray(handle, writable=writable)
    elif stype == _STORAGE_TYPE_ROW_SPARSE:
        return RowSparseNDArray(handle, writable=writable)
    else:
        raise Exception("unknown storage type")


_set_ndarray_class(_ndarray_cls)


def zeros(stype, shape, ctx=None, dtype=None, aux_types=None, **kwargs):
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
        An optional list of types of the aux data for RowSparseNDArray or CSRNDArray
        (default values depends on the storage type)

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


def empty(stype, shape, ctx=None, dtype=None, aux_types=None):
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
        return zeros(stype, shape, ctx=ctx, dtype=dtype, aux_types=aux_types)
    else:
        raise Exception("unknown stype : " + str(stype))


def array(source_array, ctx=None, dtype=None, aux_types=None):
    """Creates a sparse array from any object exposing the array interface.

    Parameters
    ----------
    source_array : RowSparseNDArray, CSRNDArray or scipy.sparse.csr.csr_matrix
        The source sparse array
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``source_array.dtype``
        if `source_array` is an `NDArray`, `float32` otherwise.
    aux_types: list of numpy.dtype, optional
        An optional list of types of the aux data for RowSparseNDArray or CSRNDArray.
        The default value for CSRNDArray is [`int64`, `int64`] for `indptr` and `indices`.
        The default value for RowSparseNDArray is [`int64`] for `indices`.

    Returns
    -------
    RowSparseNDArray or CSRNDArray
        An array with the same contents as the `source_array`.

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> csr = sp.csr_matrix((2, 100))
    >>> mx.nd.sparse.array(csr)
    <CSRNDArray 2x100 @cpu(0)>
    >>> mx.nd.sparse.array(mx.nd.zeros((3, 2), stype='csr'))
    <CSRNDArray 3x2 @cpu(0)>
    >>> mx.nd.sparse.array(mx.nd.zeros((3, 2), stype='row_sparse'))
    <RowSparseNDArray 3x2 @cpu(0)>
    """
    if isinstance(source_array, NDArray):
        assert(source_array.stype != 'default'), \
               "Please use `cast_storage` to create RowSparseNDArray or CSRNDArray from an NDArray"
        dtype = source_array.dtype if dtype is None else dtype
        aux_types = source_array._aux_types if aux_types is None else aux_types
        arr = empty(source_array.stype, source_array.shape, ctx, dtype, aux_types)
        arr[:] = source_array
        return arr
    if spsp is not None and isinstance(source_array, spsp.csr.csr_matrix):
        # TODO(haibin) implement `_sync_copy_from` with scipy csr object to reduce a copy
        indptr_type = None
        indices_type = None
        if aux_types is not None:
            assert(len(aux_types) == 2), "Expected types for both indices and indptr"
            indptr_type = aux_types[0]
            indices_type = aux_types[1]
        # preprocess scipy csr to canonical form
        csr = source_array.sorted_indices()
        csr.sum_duplicates()
        arr = csr_matrix(csr.data, csr.indptr, csr.indices, csr.shape, dtype=dtype,
                         indptr_type=indptr_type, indices_type=indices_type)
        return arr
    elif isinstance(source_array, (np.ndarray, np.generic)):
        raise ValueError("Please use mx.nd.array to create an NDArray with source_array of type ",
                         type(source_array))
    else:
        raise ValueError("Unexpected source_array type: ", type(source_array))
