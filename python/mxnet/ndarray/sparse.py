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
# pylint: disable=wildcard-import, unused-wildcard-import, too-many-lines
"""Sparse NDArray API of MXNet."""

from __future__ import absolute_import
from __future__ import division
try:
    from __builtin__ import slice as py_slice
    from __builtin__ import sum as py_sum
except ImportError:
    from builtins import slice as py_slice
    from builtins import sum as py_sum

import ctypes
import warnings
import operator
from array import array as native_array

__all__ = ["_ndarray_cls", "csr_matrix", "row_sparse_array",
           "BaseSparseNDArray", "CSRNDArray", "RowSparseNDArray",
           "add", "subtract", "multiply", "divide"]

import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array_buf, mx_real_t, integer_types
from ..base import mx_uint, NDArrayHandle, check_call
from ..context import Context
from . import _internal
from . import op
try:
    from .gen_sparse import * # pylint: disable=redefined-builtin
except ImportError:
    pass
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from .ndarray import _ufunc_helper


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
    for aux_t in aux_types:
        if np.dtype(aux_t) != np.dtype("int64"):
            raise NotImplementedError("only int64 is supported for aux types")
    aux_type_ids = [int(_DTYPE_NP_TO_MX[np.dtype(aux_t).type]) for aux_t in aux_types]
    aux_shapes = [(0,) for aux_t in aux_types] if aux_shapes is None else aux_shapes
    aux_shape_lens = [len(aux_shape) for aux_shape in aux_shapes]
    aux_shapes = py_sum(aux_shapes, ())
    num_aux = mx_uint(len(aux_types))
    check_call(_LIB.MXNDArrayCreateSparseEx(
        ctypes.c_int(int(_STORAGE_TYPE_STR_TO_ID[stype])),
        c_array_buf(mx_uint, native_array('I', shape)),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        num_aux,
        c_array_buf(ctypes.c_int, native_array('i', aux_type_ids)),
        c_array_buf(mx_uint, native_array('I', aux_shape_lens)),
        c_array_buf(mx_uint, native_array('I', aux_shapes)),
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

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return subtract(self, other)

    def __mul__(self, other):
        return multiply(self, other)

    def __div__(self, other):
        return divide(self, other)

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

    def reshape(self, *shape, **kwargs):
        raise NotSupportedForSparseNDArray(self.reshape, None, shape)

    @property
    def size(self):
        # the `size` for a sparse ndarray is ambiguous, hence disabled.
        raise NotImplementedError()

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

    def astype(self, dtype, copy=True):
        """Returns a copy of the array after casting to a specified type.
        Parameters
        ----------
        dtype : numpy.dtype or str
            The type of the returned array.
        copy : bool
            Default `True`. By default, astype always returns a newly
            allocated ndarray on the same context. If this is set to
            `False`, and the dtype requested is the same as the ndarray's
            dtype, the ndarray is returned instead of a copy.
        Examples
        --------
        >>> x = mx.nd.sparse.zeros('row_sparse', (2,3), dtype='float32')
        >>> y = x.astype('int32')
        >>> y.dtype
        <type 'numpy.int32'>
        """
        if not copy and np.dtype(dtype) == self.dtype:
            return self

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
        # pylint: disable= no-member, protected-access
        if isinstance(other, NDArray):
            if other.handle is self.handle:
                warnings.warn('You are attempting to copy an array to itself', RuntimeWarning)
                return False
            return _internal._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = _ndarray_cls(_new_alloc_handle(self.stype, self.shape, other,
                                                  True, self.dtype, self._aux_types))
            return _internal._copyto(self, out=hret)
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))
        # pylint: enable= no-member, protected-access

    def check_format(self, full_check=True):
        """Check whether the NDArray format is valid.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, rigorous check, O(N) operations. Otherwise
            basic check, O(1) operations (default True).
        """
        check_call(_LIB.MXNDArraySyncCheckFormat(self.handle, ctypes.c_bool(full_check)))

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
    """A sparse representation of 2D NDArray in the Compressed Sparse Row format.

    A CSRNDArray represents an NDArray as three separate arrays: `data`,
    `indptr` and `indices`. It uses the CSR representation where the column indices for
    row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their corresponding values are stored
    in ``data[indptr[i]:indptr[i+1]]``.

    The column indices for a given row are expected to be sorted in ascending order.
    Duplicate column entries for the same row are not allowed.

    Example
    -------
    >>> a = mx.nd.array([[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 0, 3]])
    >>> a = a.tostype('csr')
    >>> a.data.asnumpy()
    array([ 1.,  2.,  3.], dtype=float32)
    >>> a.indices.asnumpy()
    array([1, 0, 2])
    >>> a.indptr.asnumpy()
    array([0, 1, 2, 2, 3])

    See Also
    --------
    csr_matrix: Several ways to construct a CSRNDArray
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

        Returns a newly created NDArray based on the indexing key.

        Parameters
        ----------
        key : int or slice
            Indexing key.

        Examples
        --------
        >>> indptr = np.array([0, 2, 3, 6])
        >>> indices = np.array([0, 2, 2, 0, 1, 2])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> a = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        >>> a.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> a[1:2].asnumpy()
        array([[ 0.,  0.,  3.]], dtype=float32)
        >>> a[1].asnumpy()
        array([[ 0.,  0.,  3.]], dtype=float32)
        >>> a[-1].asnumpy()
        array([[ 4.,  5.,  6.]], dtype=float32)
        """
        # pylint: disable= no-member, protected-access
        if isinstance(key, int):
            if key == -1:
                begin = self.shape[0] - 1
            else:
                begin = key
            return op.slice(self, begin=begin, end=begin+1)
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
        raise ValueError('Undefined behaviour for {}'.format(key))
        # pylint: enable= no-member, protected-access

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
        >>> src = mx.nd.sparse.zeros('csr', (3,3))
        >>> src.asnumpy()
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
        >>> # assign CSRNDArray with same storage type
        >>> x = mx.nd.ones((3,3)).tostype('csr')
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
        # pylint: disable= no-member, protected-access
        if stype == 'row_sparse':
            raise ValueError("cast_storage from csr to row_sparse is not supported")
        return op.cast_storage(self, stype=stype)
        # pylint: enable= no-member, protected-access

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

    def asscipy(self):
        """Returns a ``scipy.sparse.csr.csr_matrix`` object with value copied from this array

        Examples
        --------
        >>> x = mx.nd.sparse.zeros('csr', (2,3))
        >>> y = x.asscipy()
        >>> type(y)
        <type 'scipy.sparse.csr.csr_matrix'>
        >>> y
        <2x3 sparse matrix of type '<type 'numpy.float32'>'
        with 0 stored elements in Compressed Sparse Row format>
        """
        data = self.data.asnumpy()
        indices = self.indices.asnumpy()
        indptr = self.indptr.asnumpy()
        if not spsp:
            raise ImportError("scipy is not available. \
                               Please check if the scipy python bindings are installed.")
        return spsp.csr_matrix((data, indices, indptr), shape=self.shape, dtype=self.dtype)

# pylint: disable=abstract-method
class RowSparseNDArray(BaseSparseNDArray):
    """A sparse representation of a set of NDArray row slices at given indices.

    A RowSparseNDArray represents a multidimensional NDArray using two separate arrays: `data` and
    `indices`. The number of dimensions has to be at least 2.

    - data: an NDArray of any dtype with shape [D0, D1, ..., Dn].
    - indices: a 1-D int64 NDArray with shape [D0] with values sorted in ascending order.

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

    A RowSparseNDArray is typically used to represent non-zero row slices of a large NDArray
    of shape [LARGE0, D1, .. , Dn] where LARGE0 >> D0 and most row slices are zeros.

    RowSparseNDArray is used principally in the definition of gradients for operations
    that have sparse gradients (e.g. sparse dot and sparse embedding).

    See Also
    --------
    row_sparse_array: Several ways to construct a RowSparseNDArray
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
        >>> x = mx.nd.sparse.zeros('row_sparse', (2, 3))
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
        raise ValueError('Undefined behaviour for {}'.format(key))

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
        >>> x = mx.nd.sparse.zeros('row_sparse', (3,3))
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
        # pylint: disable= no-member, protected-access
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
                _internal._set_value(float(value), out=self)
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
        # pylint: enable= no-member, protected-access

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
        # pylint: disable= no-member, protected-access
        if stype == 'csr':
            raise ValueError("cast_storage from row_sparse to csr is not supported")
        return op.cast_storage(self, stype=stype)
        # pylint: enable= no-member, protected-access

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

    def retain(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`retain`.

        The arguments are the same as for :py:func:`retain`, with
        this array as data.
        """
        return retain(self, *args, **kwargs)

def _prepare_src_array(source_array, dtype):
    """Prepare `source_array` so that it can be used to construct NDArray.
    `source_array` is converted to a `np.ndarray` if it's neither an `NDArray` \
    nor an `np.ndarray`.
    """
    if not isinstance(source_array, NDArray) and not isinstance(source_array, np.ndarray):
        try:
            source_array = np.array(source_array, dtype=dtype)
        except:
            raise TypeError('values must be array like object')
    return source_array

def _prepare_default_dtype(src_array, dtype):
    """Prepare the value of dtype if `dtype` is None. If `src_array` is an NDArray, numpy.ndarray
    or scipy.sparse.csr.csr_matrix, return src_array.dtype. float32 is returned otherwise."""
    if dtype is None:
        if isinstance(src_array, (NDArray, np.ndarray)):
            dtype = src_array.dtype
        elif spsp and isinstance(src_array, spsp.csr.csr_matrix):
            dtype = src_array.dtype
        else:
            dtype = mx_real_t
    return dtype

def _check_shape(s1, s2):
    """check s1 == s2 if both are not None"""
    if s1 and s2 and s1 != s2:
        raise ValueError("Shape mismatch detected. " + str(s1) + " v.s. " + str(s2))

def csr_matrix(arg1, shape=None, ctx=None, dtype=None):
    """Creates a `CSRNDArray`, an 2D array with compressed sparse row (CSR) format.

    The CSRNDArray can be instantiated in several ways:

    - csr_matrix(D):
        to construct a CSRNDArray with a dense 2D array ``D``
            -  **D** (*array_like*) - An object exposing the array interface, an object whose \
            `__array__` method returns an array, or any (nested) sequence.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``D.dtype`` if ``D`` is an NDArray or numpy.ndarray, \
            float32 otherwise.

    - csr_matrix(S)
        to construct a CSRNDArray with a sparse 2D array ``S``
            -  **S** (*CSRNDArray or scipy.sparse.csr.csr_matrix*) - A sparse matrix.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``S.dtype``.

    - csr_matrix((M, N))
        to construct an empty CSRNDArray with shape ``(M, N)``
            -  **M** (*int*) - Number of rows in the matrix
            -  **N** (*int*) - Number of columns in the matrix
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is float32.

    - csr_matrix((data, indices, indptr))
        to construct a CSRNDArray based on the definition of compressed sparse row format \
        using three separate arrays, \
        where the column indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]`` \
        and their corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``. \
        The column indices for a given row are expected to be **sorted in ascending order.** \
        Duplicate column entries for the same row are not allowed.
            - **data** (*array_like*) - An object exposing the array interface, which \
            holds all the non-zero entries of the matrix in row-major order.
            - **indices** (*array_like*) - An object exposing the array interface, which \
            stores the column index for each non-zero element in ``data``.
            - **indptr** (*array_like*) - An object exposing the array interface, which \
            stores the offset into ``data`` of the first non-zero element number of each \
            row of the matrix.
            - **shape** (*tuple of int, optional*) - The shape of the array. The default \
            shape is inferred from the indices and indptr arrays.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``data.dtype`` if ``data`` is an NDArray or numpy.ndarray, \
            float32 otherwise.

    - csr_matrix((data, (row, col)))
        to construct a CSRNDArray based on the COOrdinate format \
        using three seperate arrays, \
        where ``row[i]`` is the row index of the element, \
        ``col[i]`` is the column index of the element \
        and ``data[i]`` is the data corresponding to the element. All the missing \
        elements in the input are taken to be zeroes.
            - **data** (*array_like*) - An object exposing the array interface, which \
            holds all the non-zero entries of the matrix in COO format.
            - **row** (*array_like*) - An object exposing the array interface, which \
            stores the row index for each non zero element in ``data``.
            - **col** (*array_like*) - An object exposing the array interface, which \
            stores the col index for each non zero element in ``data``.
            - **shape** (*tuple of int, optional*) - The shape of the array. The default \
            shape is inferred from the ``row`` and ``col`` arrays.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is float32.

    Parameters
    ----------
    arg1: tuple of int, tuple of array_like, array_like, CSRNDArray, scipy.sparse.csr_matrix, \
    scipy.sparse.coo_matrix, tuple of int or tuple of array_like
        The argument to help instantiate the csr matrix. See above for further details.
    shape : tuple of int, optional
        The shape of the csr matrix.
    ctx: Context, optional
        Device context (default is the current default context).
    dtype: str or numpy.dtype, optional
        The data type of the output array.

    Returns
    -------
    CSRNDArray
        A `CSRNDArray` with the `csr` storage representation.

    Example
    -------
    >>> a = mx.nd.sparse.csr_matrix(([1, 2, 3], [1, 0, 2], [0, 1, 2, 2, 3]), shape=(4, 3))
    >>> a.asnumpy()
    array([[ 0.,  1.,  0.],
           [ 2.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  3.]], dtype=float32)

    See Also
    --------
    CSRNDArray : MXNet NDArray in compressed sparse row format.
    """
    # construct a csr matrix from (M, N) or (data, indices, indptr)
    if isinstance(arg1, tuple):
        arg_len = len(arg1)
        if arg_len == 2:
            # construct a sparse csr matrix from
            # scipy coo matrix if input format is coo
            if isinstance(arg1[1], tuple) and len(arg1[1]) == 2:
                data, (row, col) = arg1
                if isinstance(data, NDArray):
                    data = data.asnumpy()
                if isinstance(row, NDArray):
                    row = row.asnumpy()
                if isinstance(col, NDArray):
                    col = col.asnumpy()
                coo = spsp.coo_matrix((data, (row, col)), shape=shape)
                _check_shape(coo.shape, shape)
                csr = coo.tocsr()
                return array(csr, ctx=ctx, dtype=dtype)
            else:
                # empty matrix with shape
                _check_shape(arg1, shape)
                return empty('csr', arg1, ctx=ctx, dtype=dtype)
        elif arg_len == 3:
            # data, indices, indptr
            return _csr_matrix_from_definition(arg1[0], arg1[1], arg1[2], shape=shape,
                                               ctx=ctx, dtype=dtype)
        else:
            raise ValueError("Unexpected length of input tuple: " + str(arg_len))
    else:
        # construct a csr matrix from a sparse / dense one
        if isinstance(arg1, CSRNDArray) or (spsp and isinstance(arg1, spsp.csr.csr_matrix)):
            # construct a csr matrix from scipy or CSRNDArray
            _check_shape(arg1.shape, shape)
            return array(arg1, ctx=ctx, dtype=dtype)
        elif isinstance(arg1, RowSparseNDArray):
            raise ValueError("Unexpected input type: RowSparseNDArray")
        else:
            # construct a csr matrix from a dense one
            # prepare default ctx and dtype since mx.nd.array doesn't use default values
            # based on source_array
            dtype = _prepare_default_dtype(arg1, dtype)
            # create dns array with provided dtype. ctx is not passed since copy across
            # ctx requires dtype to be the same
            dns = _array(arg1, dtype=dtype)
            if ctx is not None and dns.context != ctx:
                dns = dns.as_in_context(ctx)
            _check_shape(dns.shape, shape)
            return dns.tostype('csr')

def _csr_matrix_from_definition(data, indices, indptr, shape=None, ctx=None,
                                dtype=None, indices_type=None, indptr_type=None):
    """Create a `CSRNDArray` based on data, indices and indptr"""
    # pylint: disable= no-member, protected-access
    storage_type = 'csr'
    # context
    ctx = Context.default_ctx if ctx is None else ctx
    # types
    dtype = _prepare_default_dtype(data, dtype)
    indptr_type = _STORAGE_AUX_TYPES[storage_type][0] if indptr_type is None else indptr_type
    indices_type = _STORAGE_AUX_TYPES[storage_type][1] if indices_type is None else indices_type
    # prepare src array and types
    data = _prepare_src_array(data, dtype)
    indptr = _prepare_src_array(indptr, indptr_type)
    indices = _prepare_src_array(indices, indices_type)

    # TODO(junwu): Convert data, indptr, and indices to mxnet NDArrays
    # if they are not for now. In the future, we should provide a c-api
    # to accept np.ndarray types to copy from to result.data and aux_data
    if not isinstance(data, NDArray):
        data = _array(data, ctx, dtype)
    if not isinstance(indptr, NDArray):
        indptr = _array(indptr, ctx, indptr_type)
    if not isinstance(indices, NDArray):
        indices = _array(indices, ctx, indices_type)
    if shape is None:
        if indices.shape[0] == 0:
            raise ValueError('invalid shape')
        shape = (len(indptr) - 1, op.max(indices).asscalar() + 1)
    # verify shapes
    aux_shapes = [indptr.shape, indices.shape]
    if data.ndim != 1 or indptr.ndim != 1 or indices.ndim != 1 or \
        indptr.shape[0] == 0 or len(shape) != 2:
        raise ValueError('invalid shape')
    result = CSRNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype,
                                          [indptr_type, indices_type], aux_shapes))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, data.handle, ctypes.c_int(-1)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indptr.handle, ctypes.c_int(0)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indices.handle, ctypes.c_int(1)))
    return result
    # pylint: enable= no-member, protected-access

def row_sparse_array(arg1, shape=None, ctx=None, dtype=None):
    """Creates a `RowSparseNDArray`, a multidimensional row sparse array with a set of \
    tensor slices at given indices.

    The RowSparseNDArray can be instantiated in several ways:

    - row_sparse_array(D):
        to construct a RowSparseNDArray with a dense ndarray ``D``
            -  **D** (*array_like*) - An object exposing the array interface, an object whose \
            `__array__` method returns an array, or any (nested) sequence.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``D.dtype`` if ``D`` is an NDArray or numpy.ndarray, \
            float32 otherwise.

    - row_sparse_array(S)
        to construct a RowSparseNDArray with a sparse ndarray ``S``
            -  **S** (*RowSparseNDArray*) - A sparse ndarray.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``S.dtype``.

    - row_sparse_array((D0, D1 .. Dn))
        to construct an empty RowSparseNDArray with shape ``(D0, D1, ... Dn)``
            -  **D0, D1 .. Dn** (*int*) - The shape of the ndarray
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is float32.

    - row_sparse_array((data, indices))
        to construct a RowSparseNDArray based on the definition of row sparse format \
        using two separate arrays, \
        where the `indices` stores the indices of the row slices with non-zeros,
        while the values are stored in `data`. The corresponding NDArray ``dense``
        represented by RowSparseNDArray ``rsp`` has \
        ``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``
        The row indices for are expected to be **sorted in ascending order.** \
            - **data** (*array_like*) - An object exposing the array interface, which \
            holds all the non-zero row slices of the array.
            - **indices** (*array_like*) - An object exposing the array interface, which \
            stores the row index for each row slice with non-zero elements.
            - **shape** (*tuple of int, optional*) - The shape of the array. The default \
            shape is inferred from the indices and indptr arrays.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is float32.

    Parameters
    ----------
    arg1: NDArray, numpy.ndarray, RowSparseNDArray, tuple of int or tuple of array_like
        The argument to help instantiate the row sparse ndarray. See above for further details.
    shape : tuple of int, optional
        The shape of the row sparse ndarray.
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array.

    Returns
    -------
    RowSparseNDArray
        An `RowSparseNDArray` with the `row_sparse` storage representation.

    Example
    -------
    >>> a = mx.nd.sparse.row_sparse_array(([[1, 2], [3, 4]], [1, 4]), shape=(6, 2))
    >>> a.asnumpy()
    array([[ 0.,  0.],
           [ 1.,  2.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 3.,  4.],
           [ 0.,  0.]], dtype=float32)

    See Also
    --------
    RowSparseNDArray : MXNet NDArray in row sparse format.
    """
    # construct a row sparse array from (D0, D1 ..) or (data, indices)
    if isinstance(arg1, tuple):
        arg_len = len(arg1)
        if arg_len < 2:
            raise ValueError("Unexpected length of input tuple: " + str(arg_len))
        elif arg_len > 2:
            # empty ndarray with shape
            _check_shape(arg1, shape)
            return empty('row_sparse', arg1, ctx=ctx, dtype=dtype)
        else:
            # len(arg1) = 2, is either shape or (data, indices)
            if isinstance(arg1[0], integer_types) and isinstance(arg1[1], integer_types):
                # empty ndarray with shape
                _check_shape(arg1, shape)
                return empty('row_sparse', arg1, ctx=ctx, dtype=dtype)
            else:
                # data, indices, indptr
                return _row_sparse_ndarray_from_definition(arg1[0], arg1[1], shape=shape,
                                                           ctx=ctx, dtype=dtype)
    else:
        # construct a row sparse ndarray from a dense / sparse array
        if isinstance(arg1, RowSparseNDArray):
            # construct a row sparse ndarray from RowSparseNDArray
            _check_shape(arg1.shape, shape)
            return array(arg1, ctx=ctx, dtype=dtype)
        elif isinstance(arg1, CSRNDArray):
            raise ValueError("Unexpected input type: CSRNDArray")
        else:
            # construct a csr matrix from a dense one
            # prepare default dtype since mx.nd.array doesn't use default values
            # based on source_array
            dtype = _prepare_default_dtype(arg1, dtype)
            # create dns array with provided dtype. ctx is not passed since copy across
            # ctx requires dtype to be the same
            dns = _array(arg1, dtype=dtype)
            if ctx is not None and dns.context != ctx:
                dns = dns.as_in_context(ctx)
            _check_shape(dns.shape, shape)
            return dns.tostype('row_sparse')

def _row_sparse_ndarray_from_definition(data, indices, shape=None, ctx=None,
                                        dtype=None, indices_type=None):
    """Create a `RowSparseNDArray` based on data and indices"""
    storage_type = 'row_sparse'
    # context
    ctx = Context.default_ctx if ctx is None else ctx
    # types
    dtype = _prepare_default_dtype(data, dtype)
    indices_type = _STORAGE_AUX_TYPES[storage_type][0] if indices_type is None else indices_type
    # prepare src array and types
    data = _prepare_src_array(data, dtype)
    indices = _prepare_src_array(indices, indices_type)

    # TODO(junwu): Convert data, indptr, and indices to mxnet NDArrays
    # if they are not for now. In the future, we should provide a c-api
    # to accept np.ndarray types to copy from to result.data and aux_data
    if not isinstance(data, NDArray):
        data = _array(data, ctx, dtype)
    if not isinstance(indices, NDArray):
        indices = _array(indices, ctx, indices_type)
    if shape is None:
        num_indices = indices.shape[0]
        if num_indices == 0:
            raise ValueError('invalid shape')
        dim0 = indices[num_indices - 1].asscalar() + 1
        shape = (dim0, ) + data.shape[1:]
    # verify shapes
    if data.ndim != len(shape) or indices.ndim != 1 or np.prod(shape[1:]) == 0:
        raise ValueError("invalid shape")
    result = RowSparseNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype,
                                                [indices_type], [indices.shape]))
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
        raise Exception("unknown storage type: %s"%stype)


_set_ndarray_class(_ndarray_cls)


def add(lhs, rhs):
    """Returns element-wise sum of the input arrays with broadcasting.

    Equivalent to ``lhs + rhs``, ``mx.nd.broadcast_add(lhs, rhs)`` and
    ``mx.nd.broadcast_plus(lhs, rhs)`` when shapes of lhs and rhs do not
    match. If lhs.shape == rhs.shape, this is equivalent to
    ``mx.nd.elemwise_add(lhs, rhs)``

    .. note::

        If the corresponding dimensions of two arrays have the same size or one of them has size 1,
        then the arrays are broadcastable to a common shape.abs

    Parameters
    ----------
    lhs : scalar or array
        First array to be added.
    rhs : scalar or array
         Second array to be added.
        If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        The element-wise sum of the input arrays.

    Examples
    --------
    >>> a = mx.nd.ones((2,3)).tostype('csr')
    >>> b = mx.nd.ones((2,3)).tostype('csr')
    >>> a.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> b.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (a+b).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> c = mx.nd.ones((2,3)).tostype('row_sparse')
    >>> d = mx.nd.ones((2,3)).tostype('row_sparse')
    >>> c.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> d.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (c+d).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray) and lhs.shape == rhs.shape:
        return _ufunc_helper(
            lhs,
            rhs,
            op.elemwise_add,
            operator.add,
            _internal._plus_scalar,
            None)

    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_add,
        operator.add,
        _internal._plus_scalar,
        None)
    # pylint: enable= no-member, protected-access


def subtract(lhs, rhs):
    """Returns element-wise difference of the input arrays with broadcasting.

    Equivalent to ``lhs - rhs``, ``mx.nd.broadcast_sub(lhs, rhs)`` and
    ``mx.nd.broadcast_minus(lhs, rhs)`` when shapes of lhs and rhs do not
    match. If lhs.shape == rhs.shape, this is equivalent to
    ``mx.nd.elemwise_sub(lhs, rhs)``

    .. note::

        If the corresponding dimensions of two arrays have the same size or one of them has size 1,
        then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be subtracted.
    rhs : scalar or array
         Second array to be subtracted.
        If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.__spec__

    Returns
    -------
    NDArray
        The element-wise difference of the input arrays.

    Examples
    --------
    >>> a = mx.nd.ones((2,3)).tostype('csr')
    >>> b = mx.nd.ones((2,3)).tostype('csr')
    >>> a.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> b.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (a-b).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> c = mx.nd.ones((2,3)).tostype('row_sparse')
    >>> d = mx.nd.ones((2,3)).tostype('row_sparse')
    >>> c.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> d.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (c-d).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray) and lhs.shape == rhs.shape:
        return _ufunc_helper(
            lhs,
            rhs,
            op.elemwise_sub,
            operator.sub,
            _internal._minus_scalar,
            None)

    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_sub,
        operator.sub,
        _internal._minus_scalar,
        None)
    # pylint: enable= no-member, protected-access


def multiply(lhs, rhs):
    """Returns element-wise product of the input arrays with broadcasting.

        Equivalent to ``lhs * rhs`` and ``mx.nd.broadcast_mul(lhs, rhs)``
        when shapes of lhs and rhs do not match. If lhs.shape == rhs.shape,
        this is equivalent to ``mx.nd.elemwise_mul(lhs, rhs)``

    .. note::

        If the corresponding dimensions of two arrays have the same size or one of them has size 1,
        then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be multiplied.
    rhs : scalar or array
         Second array to be multiplied.
        If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        The element-wise multiplication of the input arrays.

    Examples
    --------
    >>> x = mx.nd.ones((2,3)).tostype('csr')
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(3)
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([ 0.,  1.,  2.], dtype=float32)
    >>> (x*2).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> (x*y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.sparse.multiply(x, y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (x*z).asnumpy()
    array([[ 0.,  1.,  2.],
           [ 0.,  1.,  2.]], dtype=float32)
    >>> mx.nd.sparse.multiply(x, z).asnumpy()
    array([[ 0.,  1.,  2.],
           [ 0.,  1.,  2.]], dtype=float32)
    >>> z = z.reshape((1, 3))
    >>> z.asnumpy()
    array([[ 0.,  1.,  2.]], dtype=float32)
    >>> (x*z).asnumpy()
    array([[ 0.,  1.,  2.],
           [ 0.,  1.,  2.]], dtype=float32)
    >>> mx.nd.sparse.multiply(x, z).asnumpy()
    array([[ 0.,  1.,  2.],
           [ 0.,  1.,  2.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray) and lhs.shape == rhs.shape:
        return _ufunc_helper(
            lhs,
            rhs,
            op.elemwise_mul,
            operator.mul,
            _internal._mul_scalar,
            None)

    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_mul,
        operator.mul,
        _internal._mul_scalar,
        None)
    # pylint: enable= no-member, protected-access


def divide(lhs, rhs):
    """Returns element-wise division of the input arrays with broadcasting.

    Equivalent to ``lhs / rhs`` and ``mx.nd.broadcast_div(lhs, rhs)``
    when shapes of lhs and rhs do not match. If lhs.shape == rhs.shape,
    this is equivalent to ``mx.nd.elemwise_div(lhs, rhs)``

    .. note::

        If the corresponding dimensions of two arrays have the same size or one of them has size 1,
        then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array in division.
    rhs : scalar or array
         Second array in division.
        The arrays to be divided. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        The element-wise division of the input arrays.

    Examples
    --------
    >>> x = (mx.nd.ones((2,3))*6).tostype('csr')
    >>> y = mx.nd.arange(2).reshape((2,1)) + 1
    >>> z = mx.nd.arange(3) + 1
    >>> x.asnumpy()
    array([[ 6.,  6.,  6.],
           [ 6.,  6.,  6.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 1.],
           [ 2.]], dtype=float32)
    >>> z.asnumpy()
    array([ 1.,  2.,  3.], dtype=float32)
    >>> x/2
    <NDArray 2x3 @cpu(0)>
    >>> (x/3).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> (x/y).asnumpy()
    array([[ 6.,  6.,  6.],
           [ 3.,  3.,  3.]], dtype=float32)
    >>> mx.nd.sparse.divide(x,y).asnumpy()
    array([[ 6.,  6.,  6.],
           [ 3.,  3.,  3.]], dtype=float32)
    >>> (x/z).asnumpy()
    array([[ 6.,  3.,  2.],
           [ 6.,  3.,  2.]], dtype=float32)
    >>> mx.nd.sprase.divide(x,z).asnumpy()
    array([[ 6.,  3.,  2.],
           [ 6.,  3.,  2.]], dtype=float32)
    >>> z = z.reshape((1,3))
    >>> z.asnumpy()
    array([[ 1.,  2.,  3.]], dtype=float32)
    >>> (x/z).asnumpy()
    array([[ 6.,  3.,  2.],
           [ 6.,  3.,  2.]], dtype=float32)
    >>> mx.nd.sparse.divide(x,z).asnumpy()
    array([[ 6.,  3.,  2.],
           [ 6.,  3.,  2.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray) and lhs.shape == rhs.shape:
        return _ufunc_helper(
            lhs,
            rhs,
            op.elemwise_div,
            operator.truediv,
            _internal._div_scalar,
            None)

    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_div,
        operator.truediv,
        _internal._div_scalar,
        None)
    # pylint: enable= no-member, protected-access


def zeros(stype, shape, ctx=None, dtype=None, **kwargs):
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

    Returns
    -------
    RowSparseNDArray or CSRNDArray
        A created array
    Examples
    --------
    >>> mx.nd.sparse.zeros('csr', (1,2))
    <CSRNDArray 1x2 @cpu(0)>
    >>> mx.nd.sparse.zeros('row_sparse', (1,2), ctx=mx.cpu(), dtype='float16').asnumpy()
    array([[ 0.,  0.]], dtype=float16)
    """
    # pylint: disable= no-member, protected-access
    if stype == 'default':
        return _zeros_ndarray(shape, ctx=ctx, dtype=dtype, **kwargs)
    if ctx is None:
        ctx = Context.default_ctx
    dtype = mx_real_t if dtype is None else dtype
    if stype == 'row_sparse' or stype == 'csr':
        aux_types = _STORAGE_AUX_TYPES[stype]
    else:
        raise ValueError("unknown storage type" + stype)
    out = _ndarray_cls(_new_alloc_handle(stype, shape, ctx, True, dtype, aux_types))
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out, **kwargs)
    # pylint: enable= no-member, protected-access


def empty(stype, shape, ctx=None, dtype=None):
    """Returns a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    stype: string
        The storage type of the empty array, such as 'row_sparse', 'csr', etc
    shape : int or tuple of int
        The shape of the empty array.
    ctx : Context, optional
        An optional device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`).

    Returns
    -------
    CSRNDArray or RowSparseNDArray
        A created array.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    assert(stype is not None)
    if stype == 'csr' or stype == 'row_sparse':
        return zeros(stype, shape, ctx=ctx, dtype=dtype)
    else:
        raise Exception("unknown stype : " + str(stype))


def array(source_array, ctx=None, dtype=None):
    """Creates a sparse array from any object exposing the array interface.

    Parameters
    ----------
    source_array : RowSparseNDArray, CSRNDArray or scipy.sparse.csr.csr_matrix
        The source sparse array
    ctx : Context, optional
        The default context is ``source_array.context`` if ``source_array`` is an NDArray. \
        The current default context otherwise.
    dtype : str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``source_array.dtype``
        if `source_array` is an `NDArray`, `numpy.ndarray` or `scipy.sparse.csr.csr_matrix`, \
        `float32` otherwise.

    Returns
    -------
    RowSparseNDArray or CSRNDArray
        An array with the same contents as the `source_array`.

    Examples
    --------
    >>> import scipy.sparse as spsp
    >>> csr = spsp.csr_matrix((2, 100))
    >>> mx.nd.sparse.array(csr)
    <CSRNDArray 2x100 @cpu(0)>
    >>> mx.nd.sparse.array(mx.nd.sparse.zeros('csr', (3, 2)))
    <CSRNDArray 3x2 @cpu(0)>
    >>> mx.nd.sparse.array(mx.nd.sparse.zeros('row_sparse', (3, 2)))
    <RowSparseNDArray 3x2 @cpu(0)>
    """
    ctx = Context.default_ctx if ctx is None else ctx
    if isinstance(source_array, NDArray):
        assert(source_array.stype != 'default'), \
               "Please use `tostype` to create RowSparseNDArray or CSRNDArray from an NDArray"
        # prepare dtype and ctx based on source_array, if not provided
        dtype = _prepare_default_dtype(source_array, dtype)
        # if both dtype and ctx are different from source_array, we cannot copy directly
        if source_array.dtype != dtype and source_array.context != ctx:
            arr = empty(source_array.stype, source_array.shape, dtype=dtype)
            arr[:] = source_array
            arr = arr.as_in_context(ctx)
        else:
            arr = empty(source_array.stype, source_array.shape, dtype=dtype, ctx=ctx)
            arr[:] = source_array
        return arr
    elif spsp and isinstance(source_array, spsp.csr.csr_matrix):
        # TODO(haibin) implement `_sync_copy_from` with scipy csr object to reduce a copy
        # preprocess scipy csr to canonical form
        csr = source_array.sorted_indices()
        csr.sum_duplicates()
        dtype = _prepare_default_dtype(source_array, dtype)
        return csr_matrix((csr.data, csr.indices, csr.indptr), shape=csr.shape, \
                          dtype=dtype, ctx=ctx)
    elif isinstance(source_array, (np.ndarray, np.generic)):
        raise ValueError("Please use mx.nd.array to create an NDArray with source_array of type ",
                         type(source_array))
    else:
        raise ValueError("Unexpected source_array type: ", type(source_array))
