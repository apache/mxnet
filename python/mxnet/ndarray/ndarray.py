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
# pylint: disable=too-many-lines, protected-access
# pylint: disable=import-error, no-name-in-module, undefined-variable
"""NDArray API of MXNet."""
from __future__ import absolute_import
from __future__ import division

try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call
from ..base import ctypes2buffer
from ..context import Context
from . import _internal
from . import op
from ._internal import NDArrayBase

__all__ = ["NDArray", "concatenate", "_DTYPE_NP_TO_MX", "_DTYPE_MX_TO_NP", "_GRAD_REQ_MAP",
           "ones", "add", "arange", "divide", "equal", "full", "greater", "greater_equal",
           "imdecode", "lesser", "lesser_equal", "maximum", "minimum", "moveaxis", "modulo",
           "multiply", "not_equal", "onehot_encode", "power", "subtract", "true_divide",
           "waitall", "_new_empty_handle"]

_STORAGE_TYPE_UNDEFINED = -1
_STORAGE_TYPE_DEFAULT = 0
_STORAGE_TYPE_ROW_SPARSE = 1
_STORAGE_TYPE_CSR = 2

# pylint: disable= no-member
_DTYPE_NP_TO_MX = {
    None: -1,
    np.float32: 0,
    np.float64: 1,
    np.float16: 2,
    np.uint8: 3,
    np.int32: 4,
    np.int8: 5,
    np.int64: 6,
}

_DTYPE_MX_TO_NP = {
    -1: None,
    0: np.float32,
    1: np.float64,
    2: np.float16,
    3: np.uint8,
    4: np.int32,
    5: np.int8,
    6: np.int64,
}

_STORAGE_TYPE_STR_TO_ID = {
    'undefined': _STORAGE_TYPE_UNDEFINED,
    'default': _STORAGE_TYPE_DEFAULT,
    'row_sparse': _STORAGE_TYPE_ROW_SPARSE,
    'csr': _STORAGE_TYPE_CSR,
}

_STORAGE_TYPE_ID_TO_STR = {
    _STORAGE_TYPE_UNDEFINED: 'undefined',
    _STORAGE_TYPE_DEFAULT: 'default',
    _STORAGE_TYPE_ROW_SPARSE: 'row_sparse',
    _STORAGE_TYPE_CSR: 'csr',
}

_GRAD_REQ_MAP = {
    'null': 0,
    'write': 1,
    'add': 3
}
# pylint: enable= no-member


def _new_empty_handle():
    """Returns a new empty handle.

    Empty handle can be used to hold a result.

    Returns
    -------
    handle
        A new empty `NDArray` handle.
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateNone(ctypes.byref(hdl)))
    return hdl


def _new_alloc_handle(shape, ctx, delay_alloc, dtype=mx_real_t):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results.

    Returns
    -------
    handle
        A new empty `NDArray` handle.
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateEx(
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        ctypes.byref(hdl)))
    return hdl


def waitall():
    """Wait for all async operations to finish in MXNet.

    This function is used for benchmarking only.
    """
    check_call(_LIB.MXNDArrayWaitAll())


def _storage_type(handle):
    storage_type = ctypes.c_int(0)
    check_call(_LIB.MXNDArrayGetStorageType(handle, ctypes.byref(storage_type)))
    return storage_type.value


class NDArray(NDArrayBase):
    """An array object representing a multidimensional, homogeneous array of
fixed-size items.

    """
    __slots__ = []
    # make numpy functions return NDArray instead of numpy object array
    __array_priority__ = 1000.0
    # pylint: disable= no-member, undefined-variable

    def __repr__(self):
        """Returns a string representation of the array."""
        shape_info = 'x'.join(['%d' % x for x in self.shape])
        return '\n%s\n<%s %s @%s>' % (str(self.asnumpy()),
                                      self.__class__.__name__,
                                      shape_info, self.context)

    def __reduce__(self):
        return NDArray, (None,), self.__getstate__()

    def __add__(self, other):
        """x.__add__(y) <=> x+y <=> mx.nd.add(x, y) """
        return add(self, other)

    def __iadd__(self, other):
        """x.__iadd__(y) <=> x+=y """
        if not self.writable:
            raise ValueError('trying to add to a readonly NDArray')
        if isinstance(other, NDArray):
            return op.broadcast_add(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._plus_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """x.__sub__(y) <=> x-y <=> mx.nd.subtract(x, y) """
        return subtract(self, other)

    def __isub__(self, other):
        """x.__isub__(y) <=> x-=y """
        if not self.writable:
            raise ValueError('trying to subtract from a readonly NDArray')
        if isinstance(other, NDArray):
            return op.broadcast_sub(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._minus_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rsub__(self, other):
        """x.__rsub__(y) <=> y-x <=> mx.nd.subtract(y, x) """
        return subtract(other, self)

    def __mul__(self, other):
        """x.__mul__(y) <=> x*y <=> mx.nd.multiply(x, y) """
        return multiply(self, other)

    def __neg__(self):
        """x.__neg__(y) <=> -x """
        return _internal._mul_scalar(self, -1.0)

    def __imul__(self, other):
        """x.__imul__(y) <=> x*=y """
        if not self.writable:
            raise ValueError('trying to multiply to a readonly NDArray')
        if isinstance(other, NDArray):
            return op.broadcast_mul(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._mul_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """x.__div__(y) <=> x/y <=> mx.nd.divide(x, y) """
        return divide(self, other)

    def __rdiv__(self, other):
        """x.__rdiv__(y) <=> y/x <=> mx.nd.divide(y, x) """
        return divide(other, self)

    def __idiv__(self, other):
        """x.__rdiv__(y) <=> x/=y """
        if not self.writable:
            raise ValueError('trying to divide from a readonly NDArray')
        if isinstance(other, NDArray):
            return op.broadcast_div(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._div_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __mod__(self, other):
        """x.__mod__(y) <=> x%y <=> mx.nd.modulo(x, y) """
        return modulo(self, other)

    def __rmod__(self, other):
        """x.__rmod__(y) <=> y%x <=> mx.nd.modulo(y, x) """
        return modulo(other, self)

    def __imod__(self, other):
        """x.__rmod__(y) <=> x%=y """
        if not self.writable:
            raise ValueError('trying to take modulo from a readonly NDArray')
        if isinstance(other, NDArray):
            return op.broadcast_mod(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._mod_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __pow__(self, other):
        """x.__pow__(y) <=> x**y <=> mx.nd.power(x,y) """
        return power(self, other)

    def __rpow__(self, other):
        """x.__pow__(y) <=> y**x <=> mx.nd.power(y,x) """
        return power(other, self)

    def __eq__(self, other):
        """x.__eq__(y) <=> x==y <=> mx.nd.equal(x, y) """
        return equal(self, other)

    def __ne__(self, other):
        """x.__ne__(y) <=> x!=y <=> mx.nd.not_equal(x, y) """
        return not_equal(self, other)

    def __gt__(self, other):
        """x.__gt__(y) <=> x>y <=> mx.nd.greater(x, y) """
        return greater(self, other)

    def __ge__(self, other):
        """x.__ge__(y) <=> x>=y <=> mx.nd.greater_equal(x, y) """
        return greater_equal(self, other)

    def __lt__(self, other):
        """x.__lt__(y) <=> x<y <=> mx.nd.lesser(x, y) """
        return lesser(self, other)

    def __le__(self, other):
        """x.__le__(y) <=> x<=y <=> mx.nd.less_equal(x, y) """
        return lesser_equal(self, other)

    def __bool__(self):
        num_elements = reduce(operator.mul, self.shape, 1)
        if num_elements == 0:
            return False
        elif num_elements == 1:
            return bool(self.asscalar())
        else:
            raise ValueError("The truth value of an NDArray with multiple elements " \
                             "is ambiguous.")

    __nonzero__ = __bool__

    def __len__(self):
        """Number of element along the first axis."""
        return self.shape[0]

    def __getstate__(self):
        handle = self.handle
        this = {'handle' : None}
        if handle is not None:
            length = ctypes.c_size_t()
            cptr = ctypes.POINTER(ctypes.c_char)()
            check_call(_LIB.MXNDArraySaveRawBytes(self.handle,
                                                  ctypes.byref(length),
                                                  ctypes.byref(cptr)))
            this['handle'] = ctypes2buffer(cptr, length.value)
        return this

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot
        handle = state['handle']
        if handle is not None:
            buf = handle
            handle = NDArrayHandle()
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            length = ctypes.c_size_t(len(buf))
            check_call(_LIB.MXNDArrayLoadFromRawBytes(ptr, length, ctypes.byref(handle)))
            self.handle = handle
        else:
            self.handle = None

    def __setitem__(self, key, value):
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value.

        Parameters
        ----------
        key : int, slice or tuple
            The indexing key.
        value : scalar, NDArray or numpy.ndarray
            The value to set.

        Examples
        --------
        >>> x = mx.nd.zeros((2,3))
        >>> x[:] = 1
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> x[:,1:2] = 2
        >>> x.asnumpy()
        array([[ 1.,  2.,  1.],
               [ 1.,  2.,  1.]], dtype=float32)
        >>> x[1:2,1:] = 3
        >>> x.asnumpy()
        array([[ 1.,  2.,  1.],
               [ 1.,  3.,  3.]], dtype=float32)
        >>> x[1:,0:2] = mx.nd.zeros((1,2))
        >>> x.asnumpy()
        array([[ 1.,  2.,  1.],
               [ 0.,  0.,  3.]], dtype=float32)
        >>> x[1,2] = 4
        >>> x.asnumpy()
        array([[ 1.,  2.,  1.],
               [ 0.,  0.,  4.]], dtype=float32)
        """
        # pylint: disable=too-many-branches
        if not self.writable:
            raise ValueError('Cannot assign to readonly NDArray')
        if isinstance(key, integer_types):
            sliced_arr = self._at(key)
            sliced_arr[:] = value
            return
        elif isinstance(key, py_slice):
            if key.step is not None:
                raise ValueError('NDArray only supports slicing with step size 1')
            if key.start is not None or key.stop is not None:
                sliced_arr = self._slice(key.start, key.stop)
                sliced_arr[:] = value
                return
            if isinstance(value, NDArray):
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                _internal._set_value(float(value), out=self)
            elif isinstance(value, (np.ndarray, np.generic)):
                self._sync_copyfrom(value)
            else:
                raise TypeError(
                    'NDArray does not support assignment with %s of type %s'%(
                        str(value), str(type(value))))
        elif isinstance(key, tuple):
            # multi-dimension indexing
            my_shape = self.shape
            assert len(key) <= len(my_shape), \
                "Indexing dimensions exceed array dimensions, %d vs %d"%(
                    len(key), len(my_shape))
            begin = [0 for _ in my_shape]
            end = [x for x in my_shape]
            expand = []
            for i, slice_i in enumerate(key):
                if isinstance(slice_i, integer_types):
                    assert slice_i < my_shape[i]
                    begin[i] = slice_i
                    end[i] = slice_i + 1
                    expand.append(i)
                elif isinstance(slice_i, py_slice):
                    # only support continuous slicing
                    assert slice_i.step is None, \
                        "NDArray only supports slicing with step size 1."
                    begin[i] = slice_i.start or 0
                    end[i] = slice_i.stop or my_shape[i]
                    assert begin[i] < end[i]
                    assert end[i] <= my_shape[i]
                else:
                    raise ValueError(
                        "NDArray does not support slicing with key %s of type %s."%(
                            str(slice_i), str(type(slice_i))))

            if isinstance(value, NDArray):
                value = value.as_in_context(self.context)
                self._slice_assign(value, begin, end, expand)
            elif isinstance(value, numeric_types):
                _internal._crop_assign_scalar(self, out=self,
                                              begin=begin, end=end,
                                              scalar=value)
            elif isinstance(value, (np.ndarray, np.generic)):
                value = array(value, ctx=self.context, dtype=self.dtype)
                self._slice_assign(value, begin, end, expand)
            else:
                raise TypeError(
                    'NDArray does not support assignment with %s of type %s'%(
                        str(value), str(type(value))))
        else:
            raise ValueError(
                "NDArray does not support slicing with key %s of type %s."%(
                    str(key), str(type(key))))
        # pylint: enable=too-many-branches

    def _slice_assign(self, value, begin, end, expand):
        vshape = list(value.shape)
        if expand and len(vshape) != len(begin):
            if len(expand) + len(vshape) != len(begin):
                sshape = [e - b for e, b in zip(end, begin)]
                for i in reversed(expand):
                    sshape.pop(i)
                raise ValueError(
                    "Cannot assign NDArray with shape %s to NDArray slice with " \
                    "shape %s"%(str(vshape), str(sshape)))
            for i in expand:
                vshape.insert(i, 1)
            value = value.reshape(vshape)
        _internal._crop_assign(self, value, out=self,
                               begin=begin, end=end)

    def __getitem__(self, key):
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of this array.

        Parameters
        ----------
        key : int or slice, or array like
            Indexing key.

        Examples
        --------
        >>> x = mx.nd.arange(0,6).reshape((2,3))
        >>> x.asnumpy()
        array([[ 0.,  1.,  2.],
               [ 3.,  4.,  5.]], dtype=float32)
        >>> x[1].asnumpy()
        array([ 3.,  4.,  5.], dtype=float32)
        >>> y = x[0:1]
        >>> y[:] = 2
        >>> x.asnumpy()
        array([[ 2.,  2.,  2.],
               [ 3.,  4.,  5.]], dtype=float32)
        """
        # multi-dimensional slicing is not supported yet
        if isinstance(key, integer_types):
            if key > self.shape[0] - 1:
                raise IndexError(
                    'index {} is out of bounds for axis 0 with size {}'.format(
                        key, self.shape[0]))
            return self._at(key)
        elif isinstance(key, py_slice):
            if key.step is not None:
                raise ValueError("NDArray only supports slicing with step size 1.")
            if key.start is not None or key.stop is not None:
                return self._slice(key.start, key.stop)
            return self
        elif isinstance(key, tuple):
            shape = self.shape
            assert len(key) > 0, "Cannot slice with empty indices"
            assert len(shape) >= len(key), \
                "Slicing dimensions exceeds array dimensions, %d vs %d"%(
                    len(key), len(shape))
            if isinstance(key[0], (NDArray, np.ndarray, list, tuple)):
                indices = []
                dtype = 'int32'
                shape = None
                for idx_i in key:
                    if not isinstance(idx_i, NDArray):
                        assert isinstance(idx_i, (NDArray, np.ndarray, list, tuple)), \
                            "Combining basic and advanced indexing is not supported " \
                            "yet. Indices must be all NDArray or all slice, not a " \
                            "mix of both."
                        idx_i = array(idx_i, ctx=self.context, dtype=dtype)
                    else:
                        dtype = idx_i.dtype
                    if shape is None:
                        shape = idx_i.shape
                    else:
                        assert shape == idx_i.shape, \
                            "All index arrays must have the same shape: %s vs %s. " \
                            "Broadcasting is not supported yet."%(shape, idx_i.shape)
                    indices.append(idx_i)
                indices = op.stack(*indices)
                return op.gather_nd(self, indices)
            else:
                oshape = []
                begin = []
                end = []
                i = -1
                for i, slice_i in enumerate(key):
                    if isinstance(slice_i, integer_types):
                        begin.append(slice_i)
                        end.append(slice_i+1)
                    elif isinstance(slice_i, py_slice):
                        if slice_i.step is not None:
                            raise ValueError("NDArray only supports slicing with step size 1.")
                        begin.append(0 if slice_i.start is None else slice_i.start)
                        end.append(shape[i] if slice_i.stop is None else slice_i.stop)
                        oshape.append(end[i] - begin[i])
                    elif isinstance(slice_i, (NDArray, np.ndarray, list, tuple)):
                        raise ValueError(
                            "Combining basic and advanced indexing is not supported " \
                            "yet. Indices must be all NDArray or all slice, not a " \
                            "mix of both.")
                    else:
                        raise ValueError(
                            "NDArray does not support slicing with key %s of type %s."%(
                                str(slice_i), str(type(slice_i))))
                oshape.extend(shape[i+1:])
                if len(oshape) == 0:
                    oshape.append(1)
                return op.slice(self, begin, end).reshape(oshape)
        else:
            raise ValueError(
                "NDArray does not support slicing with key %s of type %s."%(
                    str(key), str(type(key))))

    def _sync_copyfrom(self, source_array):
        """Performs a synchronized copy from the `source_array` to the current array.
        This is called through ``x[:] = source_array``, where the `source_array`
        is a `numpy.ndarray` or array-like object.
        This function blocks until all the pending read/write operations with respect
        to the current `NDArray` are finished and carry out the copy operation to the
        current NDArray.

        Parameters
        ----------
        source_array : array_like
            The data source we would like to copy from.

        Example
        -------
        >>> a = mx.nd.array([1, 2])
        >>> a.asnumpy()
        array([ 1.,  2.], dtype=float32)
        >>> a[:] = np.array([3, 4])
        >> a.asnumpy()
        array([ 3.,  4.], dtype=float32)
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must consist of array-like data,' +
                                'type %s is not supported' % str(type(array)))
        source_array = np.ascontiguousarray(source_array, dtype=self.dtype)
        if source_array.shape != self.shape:
            raise ValueError('Shape inconsistent: expected %s vs got %s'%(
                str(self.shape), str(source_array.shape)))
        check_call(_LIB.MXNDArraySyncCopyFromCPU(
            self.handle,
            source_array.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(source_array.size)))

    def _slice(self, start, stop):
        """Returns a sliced NDArray that shares memory with the current one.
        This is called through ``x[start:stop]``.

        Parameters
        ----------
        start : int
            Starting inclusive index of slice in the first dim.
        stop : int
            Finishing exclusive index of slice in the first dim.

        Returns
        -------
            `NDArray` sharing the memory with the current one sliced from
            start to stop in the first dim.

        Examples:
        >>> a = mx.nd.array([[1,2], [3, 4], [5, 6], [7, 8]])
        >>> a[1:2].asnumpy()
        array([[ 3.,  4.]], dtype=float32)
        >>> a[1:1].asnumpy()
        array([], shape=(0, 2), dtype=float32)
        """
        handle = NDArrayHandle()
        if start is None:
            start = mx_uint(0)
        elif start < 0:
            length = self.shape[0]
            start += length
            assert start >= 0, "Slicing start %d exceeds limit of %d"%(start-length, length)
            start = mx_uint(start)
        else:
            start = mx_uint(start)
        if stop is None:
            stop = mx_uint(self.shape[0])
        elif stop < 0:
            length = self.shape[0]
            stop += length
            assert stop >= 0, "Slicing end %d exceeds limit of %d"%(stop-length, length)
            stop = mx_uint(stop)
        else:
            stop = mx_uint(stop)
        check_call(_LIB.MXNDArraySlice(
            self.handle, start, stop, ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    def _at(self, idx):
        """Returns a view of the array sliced at `idx` in the first dim.
        This is called through ``x[idx]``.

        Parameters
        ----------
        idx : int
            index for slicing the `NDArray` in the first dim.

        Returns
        -------
        NDArray
            `NDArray` sharing the memory with the current one sliced at `idx` in the first dim.

        Examples
        --------
        >>> a = mx.nd.array([[1,2], [3, 4]])
        >>> a[1].asnumpy()
        array([ 3.,  4.], dtype=float32)
        >>> b = mx.nd.array([1, 2, 3, 4])
        >>> b[0].asnumpy()
        array([ 1.], dtype=float32)
        """
        handle = NDArrayHandle()
        idx = mx_uint(idx)
        check_call(_LIB.MXNDArrayAt(
            self.handle, idx, ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    def reshape(self, shape):
        """Returns a **view** of this array with a new shape without altering any data.

        Parameters
        ----------
        shape : tuple of int
            The new shape should not change the array size, namely
            ``np.prod(new_shape)`` should be equal to ``np.prod(self.shape)``.

            One dimension can be -1. In this case, the value is inferred
            from the length of the array and remaining dimensions.

            0 Dimensions in shape will be copied from original shape, i.e.
            if x.shape == (3, 4, 5), x.reshape((0, 20)).shape will be (3, 20).


        Returns
        -------
        NDArray
            An array with desired shape that shares data with this array.

        Examples
        --------
        >>> x = mx.nd.arange(0,6).reshape((2,3))
        >>> x.asnumpy()
        array([[ 0.,  1.,  2.],
               [ 3.,  4.,  5.]], dtype=float32)
        >>> y = x.reshape((3,2))
        >>> y.asnumpy()
        array([[ 0.,  1.],
               [ 2.,  3.],
               [ 4.,  5.]], dtype=float32)
        >>> y = x.reshape((3,-1))
        >>> y.asnumpy()
        array([[ 0.,  1.],
               [ 2.,  3.],
               [ 4.,  5.]], dtype=float32)
        >>> y[:] = -1
        >>> x.asnumpy()
        array([[-1., -1., -1.],
               [-1., -1., -1.]], dtype=float32)
        """
        handle = NDArrayHandle()

        # Actual reshape
        check_call(_LIB.MXNDArrayReshape(self.handle,
                                         len(shape),
                                         c_array(ctypes.c_int, shape),
                                         ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    def reshape_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reshape_like`.

        The arguments are the same as for :py:func:`reshape_like`, with
        this array as data.
        """
        return op.reshape_like(self, *args, **kwargs)

    def zeros_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`zeros_like`.

        The arguments are the same as for :py:func:`zeros_like`, with
        this array as data.
        """
        return op.zeros_like(self, *args, **kwargs)

    def ones_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ones_like`.

        The arguments are the same as for :py:func:`ones_like`, with
        this array as data.
        """
        return op.ones_like(self, *args, **kwargs)

    def broadcast_axes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`broadcast_axes`.

        The arguments are the same as for :py:func:`broadcast_axes`, with
        this array as data.
        """
        return op.broadcast_axes(self, *args, **kwargs)

    def repeat(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`repeat`.

        The arguments are the same as for :py:func:`repeat`, with
        this array as data.
        """
        return op.repeat(self, *args, **kwargs)

    def pad(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pad`.

        The arguments are the same as for :py:func:`pad`, with
        this array as data.
        """
        return op.pad(self, *args, **kwargs)

    def swapaxes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`swapaxes`.

        The arguments are the same as for :py:func:`swapaxes`, with
        this array as data.
        """
        return op.swapaxes(self, *args, **kwargs)

    def split(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split`.

        The arguments are the same as for :py:func:`split`, with
        this array as data.
        """
        return op.split(self, *args, **kwargs)

    def slice(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice`.

        The arguments are the same as for :py:func:`slice`, with
        this array as data.
        """
        return op.slice(self, *args, **kwargs)

    def slice_axis(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_axis`.

        The arguments are the same as for :py:func:`slice_axis`, with
        this array as data.
        """
        return op.slice_axis(self, *args, **kwargs)

    def take(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`take`.

        The arguments are the same as for :py:func:`take`, with
        this array as data.
        """
        return op.take(self, *args, **kwargs)

    def one_hot(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`one_hot`.

        The arguments are the same as for :py:func:`one_hot`, with
        this array as data.
        """
        return op.one_hot(self, *args, **kwargs)

    def pick(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pick`.

        The arguments are the same as for :py:func:`pick`, with
        this array as data.
        """
        return op.pick(self, *args, **kwargs)

    def sort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sort`.

        The arguments are the same as for :py:func:`sort`, with
        this array as data.
        """
        return op.sort(self, *args, **kwargs)

    def topk(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`topk`.

        The arguments are the same as for :py:func:`topk`, with
        this array as data.
        """
        return op.topk(self, *args, **kwargs)

    def argsort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argsort`.

        The arguments are the same as for :py:func:`argsort`, with
        this array as data.
        """
        return op.argsort(self, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax`.

        The arguments are the same as for :py:func:`argmax`, with
        this array as data.
        """
        return op.argmax(self, *args, **kwargs)

    def argmax_channel(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax_channel`.

        The arguments are the same as for :py:func:`argmax_channel`, with
        this array as data.
        """
        return op.argmax_channel(self, *args, **kwargs)

    def argmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmin`.

        The arguments are the same as for :py:func:`argmin`, with
        this array as data.
        """
        return op.argmin(self, *args, **kwargs)

    def clip(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`clip`.

        The arguments are the same as for :py:func:`clip`, with
        this array as data.
        """
        return op.clip(self, *args, **kwargs)

    def abs(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`abs`.

        The arguments are the same as for :py:func:`abs`, with
        this array as data.
        """
        return op.abs(self, *args, **kwargs)

    def sign(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sign`.

        The arguments are the same as for :py:func:`sign`, with
        this array as data.
        """
        return op.sign(self, *args, **kwargs)

    def flatten(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`flatten`.

        The arguments are the same as for :py:func:`flatten`, with
        this array as data.
        """
        return op.flatten(self, *args, **kwargs)

    def expand_dims(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expand_dims`.

        The arguments are the same as for :py:func:`expand_dims`, with
        this array as data.
        """
        return op.expand_dims(self, *args, **kwargs)

    def tile(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tile`.

        The arguments are the same as for :py:func:`tile`, with
        this array as data.
        """
        return op.tile(self, *args, **kwargs)

    def transpose(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`transpose`.

        The arguments are the same as for :py:func:`transpose`, with
        this array as data.
        """
        return op.transpose(self, *args, **kwargs)

    def flip(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`flip`.

        The arguments are the same as for :py:func:`flip`, with
        this array as data.
        """
        return op.flip(self, *args, **kwargs)

    def sum(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sum`.

        The arguments are the same as for :py:func:`sum`, with
        this array as data.
        """
        return op.sum(self, *args, **kwargs)

    def nansum(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nansum`.

        The arguments are the same as for :py:func:`nansum`, with
        this array as data.
        """
        return op.nansum(self, *args, **kwargs)

    def prod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`prod`.

        The arguments are the same as for :py:func:`prod`, with
        this array as data.
        """
        return op.prod(self, *args, **kwargs)

    def nanprod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nanprod`.

        The arguments are the same as for :py:func:`nanprod`, with
        this array as data.
        """
        return op.nanprod(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`mean`.

        The arguments are the same as for :py:func:`mean`, with
        this array as data.
        """
        return op.mean(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`max`.

        The arguments are the same as for :py:func:`max`, with
        this array as data.
        """
        return op.max(self, *args, **kwargs)

    def min(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`min`.

        The arguments are the same as for :py:func:`min`, with
        this array as data.
        """
        return op.min(self, *args, **kwargs)

    def norm(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`norm`.

        The arguments are the same as for :py:func:`norm`, with
        this array as data.
        """
        return op.norm(self, *args, **kwargs)

    def round(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`round`.

        The arguments are the same as for :py:func:`round`, with
        this array as data.
        """
        return op.round(self, *args, **kwargs)

    def rint(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rint`.

        The arguments are the same as for :py:func:`rint`, with
        this array as data.
        """
        return op.rint(self, *args, **kwargs)

    def fix(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`fix`.

        The arguments are the same as for :py:func:`fix`, with
        this array as data.
        """
        return op.fix(self, *args, **kwargs)

    def floor(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`floor`.

        The arguments are the same as for :py:func:`floor`, with
        this array as data.
        """
        return op.floor(self, *args, **kwargs)

    def ceil(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ceil`.

        The arguments are the same as for :py:func:`ceil`, with
        this array as data.
        """
        return op.ceil(self, *args, **kwargs)

    def trunc(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`trunc`.

        The arguments are the same as for :py:func:`trunc`, with
        this array as data.
        """
        return op.trunc(self, *args, **kwargs)

    def sin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sin`.

        The arguments are the same as for :py:func:`sin`, with
        this array as data.
        """
        return op.sin(self, *args, **kwargs)

    def cos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cos`.

        The arguments are the same as for :py:func:`cos`, with
        this array as data.
        """
        return op.cos(self, *args, **kwargs)

    def tan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tan`.

        The arguments are the same as for :py:func:`tan`, with
        this array as data.
        """
        return op.tan(self, *args, **kwargs)

    def arcsin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsin`.

        The arguments are the same as for :py:func:`arcsin`, with
        this array as data.
        """
        return op.arcsin(self, *args, **kwargs)

    def arccos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccos`.

        The arguments are the same as for :py:func:`arccos`, with
        this array as data.
        """
        return op.arccos(self, *args, **kwargs)

    def arctan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctan`.

        The arguments are the same as for :py:func:`arctan`, with
        this array as data.
        """
        return op.arctan(self, *args, **kwargs)

    def degrees(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`degrees`.

        The arguments are the same as for :py:func:`degrees`, with
        this array as data.
        """
        return op.degrees(self, *args, **kwargs)

    def radians(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`radians`.

        The arguments are the same as for :py:func:`radians`, with
        this array as data.
        """
        return op.radians(self, *args, **kwargs)

    def sinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sinh`.

        The arguments are the same as for :py:func:`sinh`, with
        this array as data.
        """
        return op.sinh(self, *args, **kwargs)

    def cosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cosh`.

        The arguments are the same as for :py:func:`cosh`, with
        this array as data.
        """
        return op.cosh(self, *args, **kwargs)

    def tanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tanh`.

        The arguments are the same as for :py:func:`tanh`, with
        this array as data.
        """
        return op.tanh(self, *args, **kwargs)

    def arcsinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsinh`.

        The arguments are the same as for :py:func:`arcsinh`, with
        this array as data.
        """
        return op.arcsinh(self, *args, **kwargs)

    def arccosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccosh`.

        The arguments are the same as for :py:func:`arccosh`, with
        this array as data.
        """
        return op.arccosh(self, *args, **kwargs)

    def arctanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctanh`.

        The arguments are the same as for :py:func:`arctanh`, with
        this array as data.
        """
        return op.arctanh(self, *args, **kwargs)

    def exp(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`exp`.

        The arguments are the same as for :py:func:`exp`, with
        this array as data.
        """
        return op.exp(self, *args, **kwargs)

    def expm1(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expm1`.

        The arguments are the same as for :py:func:`expm1`, with
        this array as data.
        """
        return op.expm1(self, *args, **kwargs)

    def log(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log`.

        The arguments are the same as for :py:func:`log`, with
        this array as data.
        """
        return op.log(self, *args, **kwargs)

    def log10(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log10`.

        The arguments are the same as for :py:func:`log10`, with
        this array as data.
        """
        return op.log10(self, *args, **kwargs)

    def log2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log2`.

        The arguments are the same as for :py:func:`log2`, with
        this array as data.
        """
        return op.log2(self, *args, **kwargs)

    def log1p(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log1p`.

        The arguments are the same as for :py:func:`log1p`, with
        this array as data.
        """
        return op.log1p(self, *args, **kwargs)

    def sqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sqrt`.

        The arguments are the same as for :py:func:`sqrt`, with
        this array as data.
        """
        return op.sqrt(self, *args, **kwargs)

    def rsqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rsqrt`.

        The arguments are the same as for :py:func:`rsqrt`, with
        this array as data.
        """
        return op.rsqrt(self, *args, **kwargs)

    def cbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cbrt`.

        The arguments are the same as for :py:func:`cbrt`, with
        this array as data.
        """
        return op.cbrt(self, *args, **kwargs)

    def rcbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rcbrt`.

        The arguments are the same as for :py:func:`rcbrt`, with
        this array as data.
        """
        return op.rcbrt(self, *args, **kwargs)

    def square(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`square`.

        The arguments are the same as for :py:func:`square`, with
        this array as data.
        """
        return op.square(self, *args, **kwargs)

    def reciprocal(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reciprocal`.

        The arguments are the same as for :py:func:`reciprocal`, with
        this array as data.
        """
        return op.reciprocal(self, *args, **kwargs)

    def relu(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`relu`.

        The arguments are the same as for :py:func:`relu`, with
        this array as data.
        """
        return op.relu(self, *args, **kwargs)

    def sigmoid(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sigmoid`.

        The arguments are the same as for :py:func:`sigmoid`, with
        this array as data.
        """
        return op.sigmoid(self, *args, **kwargs)

    def softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmax`.

        The arguments are the same as for :py:func:`softmax`, with
        this array as data.
        """
        return op.softmax(self, *args, **kwargs)

    def log_softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log_softmax`.

        The arguments are the same as for :py:func:`log_softmax`, with
        this array as data.
        """
        return op.log_softmax(self, *args, **kwargs)

    # pylint: disable= undefined-variable
    def broadcast_to(self, shape):
        """Broadcasts the input array to a new shape.

        Broadcasting is only allowed on axes with size 1. The new shape cannot change
        the number of dimensions.
        For example, you could broadcast from shape (2, 1) to (2, 3), but not from
        shape (2, 3) to (2, 3, 3).

        Parameters
        ----------
        shape : tuple of int
            The shape of the desired array.

        Returns
        -------
        NDArray
            A NDArray with the desired shape that is not sharing data with this
            array, even if the new shape is the same as ``self.shape``.

        Examples
        --------
        >>> x = mx.nd.arange(0,3).reshape((1,3,1))
        >>> x.asnumpy()
        array([[[ 0.],
                [ 1.],
                [ 2.]]], dtype=float32)
        >>> y = x.broadcast_to((2,3,3))
        >>> y.asnumpy()
        array([[[ 0.,  0.,  0.],
                [ 1.,  1.,  1.],
                [ 2.,  2.,  2.]],
        <BLANKLINE>
               [[ 0.,  0.,  0.],
                [ 1.,  1.,  1.],
                [ 2.,  2.,  2.]]], dtype=float32)
        """
        cur_shape = self.shape
        err_str = 'operands could not be broadcast together with remapped shapes' \
                  '[original->remapped]: {} and requested shape {}'.format(cur_shape, shape)
        if len(shape) < len(cur_shape):
            raise ValueError(err_str)
        cur_shape = (1,) * (len(shape) - len(cur_shape)) + cur_shape
        cur_shape_arr = np.array(cur_shape)
        broadcasting_axes = np.nonzero(cur_shape_arr != np.array(shape))
        if (cur_shape_arr[broadcasting_axes] != 1).any():
            raise ValueError(err_str)
        if cur_shape != self.shape:
            return op.broadcast_to(self.reshape(cur_shape), shape=shape)
        else:
            return op.broadcast_to(self, shape=tuple(shape))
    # pylint: enable= undefined-variable

    def wait_to_read(self):
        """Waits until all previous write operations on the current array are finished.

        This method guarantees that all previous write operations that pushed
        into the backend engine for execution are actually finished.

        Examples
        --------
        >>> import time
        >>> tic = time.time()
        >>> a = mx.nd.ones((1000,1000))
        >>> b = mx.nd.dot(a, a)
        >>> print(time.time() - tic) # doctest: +SKIP
        0.003854036331176758
        >>> b.wait_to_read()
        >>> print(time.time() - tic) # doctest: +SKIP
        0.0893700122833252
        """
        check_call(_LIB.MXNDArrayWaitToRead(self.handle))

    @property
    def ndim(self):
        """Returns the number of dimensions of this array

        Examples
        --------
        >>> x = mx.nd.array([1, 2, 3, 4])
        >>> x.ndim
        1
        >>> x = mx.nd.array([[1, 2], [3, 4]])
        >>> x.ndim
        2
        """
        return len(self.shape)

    @property
    def shape(self):
        """Tuple of array dimensions.

        Examples
        --------
        >>> x = mx.nd.array([1, 2, 3, 4])
        >>> x.shape
        (4L,)
        >>> y = mx.nd.zeros((2, 3, 4))
        >>> y.shape
        (2L, 3L, 4L)
        """
        ndim = mx_uint()
        pdata = ctypes.POINTER(mx_uint)()
        check_call(_LIB.MXNDArrayGetShape(
            self.handle, ctypes.byref(ndim), ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])


    @property
    def size(self):
        """Number of elements in the array.

        Equivalent to the product of the array's dimensions.

        Examples
        --------
        >>> import numpy as np
        >>> x = mx.nd.zeros((3, 5, 2))
        >>> x.size
        30
        >>> np.prod(x.shape)
        30
        """
        size = 1
        for i in self.shape:
            size *= i
        return size

    @property
    def context(self):
        """Device context of the array.

        Examples
        --------
        >>> x = mx.nd.array([1, 2, 3, 4])
        >>> x.context
        cpu(0)
        >>> type(x.context)
        <class 'mxnet.context.Context'>
        >>> y = mx.nd.zeros((2,3), mx.gpu(0))
        >>> y.context
        gpu(0)
        """
        dev_typeid = ctypes.c_int()
        dev_id = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetContext(
            self.handle, ctypes.byref(dev_typeid), ctypes.byref(dev_id)))
        return Context(Context.devtype2str[dev_typeid.value], dev_id.value)

    @property
    def dtype(self):
        """Data-type of the array's elements.

        Returns
        -------
        numpy.dtype
            This NDArray's data type.

        Examples
        --------
        >>> x = mx.nd.zeros((2,3))
        >>> x.dtype
        <type 'numpy.float32'>
        >>> y = mx.nd.zeros((2,3), dtype='int32')
        >>> y.dtype
        <type 'numpy.int32'>
        """
        mx_dtype = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetDType(
            self.handle, ctypes.byref(mx_dtype)))
        return _DTYPE_MX_TO_NP[mx_dtype.value]

    @property
    def stype(self):
        """Storage-type of the array.
        """
        return _STORAGE_TYPE_ID_TO_STR[_storage_type(self.handle)]

    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        """Returns a copy of the array with axes transposed.

        Equivalent to ``mx.nd.transpose(self)`` except that
        self is returned if ``self.ndim < 2``.

        Unlike ``numpy.ndarray.T``, this function returns a copy
        rather than a view of the array unless ``self.ndim < 2``.

        Examples
        --------
        >>> x = mx.nd.arange(0,6).reshape((2,3))
        >>> x.asnumpy()
        array([[ 0.,  1.,  2.],
               [ 3.,  4.,  5.]], dtype=float32)
        >>> x.T.asnumpy()
        array([[ 0.,  3.],
               [ 1.,  4.],
               [ 2.,  5.]], dtype=float32)

        """
        if len(self.shape) < 2:
            return self
        return op.transpose(self)
    # pylint: enable= invalid-name, undefined-variable

    @property
    def _fresh_grad(self):
        """Whether this array's corresponding gradient array
        (registered via `autograd.mark_variables`) has been
        updated by `autograd.backward` since last reset.

        `_fresh_grad` need to be manually set to False
        after consuming gradient (usually after updating this
        array).
        """
        out = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetGradState(self.handle, ctypes.byref(out)))
        return out.value

    @_fresh_grad.setter
    def _fresh_grad(self, state):
        check_call(_LIB.MXNDArraySetGradState(self.handle, ctypes.c_int(state)))

    def asnumpy(self):
        """Returns a ``numpy.ndarray`` object with value copied from this array.

        Examples
        --------
        >>> x = mx.nd.ones((2,3))
        >>> y = x.asnumpy()
        >>> type(y)
        <type 'numpy.ndarray'>
        >>> y
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> z = mx.nd.ones((2,3), dtype='int32')
        >>> z.asnumpy()
        array([[1, 1, 1],
               [1, 1, 1]], dtype=int32)
        """
        data = np.empty(self.shape, dtype=self.dtype)
        check_call(_LIB.MXNDArraySyncCopyToCPU(
            self.handle,
            data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(data.size)))
        return data

    def asscalar(self):
        """Returns a scalar whose value is copied from this array.

        This function is equivalent to ``self.asnumpy()[0]``. This NDArray must have shape (1,).

        Examples
        --------
        >>> x = mx.nd.ones((1,), dtype='int32')
        >>> x.asscalar()
        1
        >>> type(x.asscalar())
        <type 'numpy.int32'>
        """
        if self.shape != (1,):
            raise ValueError("The current array is not a scalar")
        return self.asnumpy()[0]

    def astype(self, dtype):
        """Returns a copy of the array after casting to a specified type.

        Parameters
        ----------
        dtype : numpy.dtype or str
            The type of the returned array.

        Returns
        -------
        NDArray, CSRNDArray or RowSparseNDArray
            The copied array after casting to the specified type.

        Examples
        --------
        >>> x = mx.nd.zeros((2,3), dtype='float32')
        >>> y = x.astype('int32')
        >>> y.dtype
        <type 'numpy.int32'>
        """
        res = empty(self.shape, ctx=self.context, dtype=dtype)
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
        NDArray, CSRNDArray or RowSparseNDArray
            The copied array. If ``other`` is an ``NDArray``, then the return value
            and ``other`` will point to the same ``NDArray``.

        Examples
        --------
        >>> x = mx.nd.ones((2,3))
        >>> y = mx.nd.zeros((2,3), mx.gpu(0))
        >>> z = x.copyto(y)
        >>> z is y
        True
        >>> y.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> y.copyto(mx.gpu(0))
        <NDArray 2x3 @gpu(0)>

        """
        if isinstance(other, NDArray):
            if other.handle is self.handle:
                warnings.warn('You are attempting to copy an array to itself', RuntimeWarning)
                return
            return _internal._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = NDArray(_new_alloc_handle(self.shape, other, True, self.dtype))
            return _internal._copyto(self, out=hret)
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def copy(self):
        """Makes a copy of this ``NDArray``, keeping the same context.

        Returns
        -------
        NDArray, CSRNDArray or RowSparseNDArray
            The copied array

        Examples
        --------
        >>> x = mx.nd.ones((2,3))
        >>> y = x.copy()
        >>> y.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        """
        return self.copyto(self.context)

    def as_in_context(self, context):
        """Returns an array on the target device with the same value as this array.

        If the target context is the same as ``self.context``, then ``self`` is
        returned.  Otherwise, a copy is made.

        Parameters
        ----------
        context : Context
            The target context.

        Returns
        -------
        NDArray, CSRNDArray or RowSparseNDArray
            The target array.


        Examples
        --------
        >>> x = mx.nd.ones((2,3))
        >>> y = x.as_in_context(mx.cpu())
        >>> y is x
        True
        >>> z = x.as_in_context(mx.gpu(0))
        >>> z is x
        False
        """
        if self.context == context:
            return self
        return self.copyto(context)

    def attach_grad(self, grad_req='write', stype=None):
        """Attach a gradient buffer to this NDArray, so that `backward`
        can compute gradient with respect to it.

        Parameters
        ----------
        grad_req : {'write', 'add', 'null'}
            How gradient will be accumulated.
            - 'write': gradient will be overwritten on every backward.
            - 'add': gradient will be added to existing value on every backward.
            - 'null': do not compute gradient for this NDArray.
        stype : str, optional
            The storage type of the gradient array. Defaults to the same stype of this NDArray.
        """
        from . import zeros as _zeros
        if stype is not None:
            grad = _zeros(self.shape, stype=stype)
        else:
            grad = op.zeros_like(self)  # pylint: disable=undefined-variable
        grad_req = _GRAD_REQ_MAP[grad_req]
        check_call(_LIB.MXAutogradMarkVariables(
            1, ctypes.pointer(self.handle),
            ctypes.pointer(mx_uint(grad_req)),
            ctypes.pointer(grad.handle)))

    @property
    def grad(self):
        """Returns gradient buffer attached to this NDArray."""
        from . import _ndarray_cls
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetGrad(self.handle, ctypes.byref(hdl)))
        if hdl.value is None:
            return None
        return _ndarray_cls(hdl)

    def detach(self):
        """Returns a new NDArray, detached from the current graph."""
        from . import _ndarray_cls
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayDetach(self.handle, ctypes.byref(hdl)))
        return _ndarray_cls(hdl)

    def backward(self, out_grad=None, retain_graph=False, train_mode=True):
        """Compute the gradients of this NDArray w.r.t variables.

        Parameters
        ----------
        out_grad : NDArray, optional
            Gradient with respect to head.
        retain_graph : bool, optional
            Whether to retain the computaion graph for another backward
            pass on the same graph. By default the computaion history
            is cleared.
        train_mode : bool, optional
            Whether to compute gradient for training or inference.
        """
        if out_grad is None:
            ograd_handles = [NDArrayHandle(0)]
        else:
            ograd_handles = [out_grad.handle]

        check_call(_LIB.MXAutogradBackwardEx(
            1, c_array(NDArrayHandle, [self.handle]),
            c_array(NDArrayHandle, ograd_handles),
            0,
            ctypes.c_void_p(0),
            ctypes.c_int(retain_graph),
            ctypes.c_int(0),
            ctypes.c_int(train_mode),
            ctypes.c_void_p(0),
            ctypes.c_void_p(0)))

    def tostype(self, stype):
        """Return a copy of the array with chosen storage type.

        See Also
        ----------
        :meth:`mxnet.ndarray.cast_storage`.

        Returns
        -------
        NDArray, CSRNDArray or RowSparseNDArray
            A copy of the array with the chosen storage stype
        """
        return op.cast_storage(self, stype=stype)


def onehot_encode(indices, out):
    """One-hot encoding indices into matrix out.

    .. note:: `onehot_encode` is deprecated. Use `one_hot` instead.

    """
    # pylint: disable= no-member, protected-access
    return _internal._onehot_encode(indices, out, out=out)
    # pylint: enable= no-member, protected-access


def ones(shape, ctx=None, dtype=None, **kwargs):
    """Returns a new array filled with all ones, with the given shape and type.

    Parameters
    ----------
    shape : int or tuple of int or list of int
        The shape of the empty array.
    ctx : Context, optional
        An optional device context.
        Defaults to the current default context (``mxnet.Context.default_ctx``).
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`).
    out : NDArray, optional
        The output NDArray (default is `None`).

    Returns
    -------
    NDArray
        A new array of the specified shape filled with all ones.

    Examples
    --------
    >>> mx.nd.ones(1).asnumpy()
    array([ 1.], dtype=float32)
    >>> mx.nd.ones((1,2), mx.gpu(0))
    <NDArray 1x2 @gpu(0)>
    >>> mx.nd.ones((1,2), dtype='float16').asnumpy()
    array([[ 1.,  1.]], dtype=float16)
    """
    # pylint: disable= unused-argument
    if ctx is None:
        ctx = Context.default_ctx
    dtype = mx_real_t if dtype is None else dtype
    # pylint: disable= no-member, protected-access
    return _internal._ones(shape=shape, ctx=ctx, dtype=dtype, **kwargs)
    # pylint: enable= no-member, protected-access


def full(shape, val, ctx=None, dtype=mx_real_t, out=None):
    """Returns a new array of given shape and type, filled with the given value `val`.

    Parameters
    --------
    shape : int or tuple of int
        The shape of the new array.
    val : scalar
        Fill value.
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : `str` or `numpy.dtype`, optional
        The data type of the returned `NDArray`. The default datatype is `float32`.
    out : NDArray, optional
        The output NDArray (default is `None`).

    Returns
    -------
    NDArray
        `NDArray` filled with `val`, with the given shape, ctx, and dtype.

    Examples
    --------
    >>> mx.nd.full(1, 2.0).asnumpy()
    array([ 2.], dtype=float32)
    >>> mx.nd.full((1, 2), 2.0, mx.gpu(0))
    <NDArray 1x2 @gpu(0)>
    >>> mx.nd.full((1, 2), 2.0, dtype='float16').asnumpy()
    array([[ 2.,  2.]], dtype=float16)
    """
    out = empty(shape, ctx, dtype) if out is None else out
    out[:] = val
    return out


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
    NDArray
        An `NDArray` with the same contents as the `source_array`.
    """
    if isinstance(source_array, NDArray):
        dtype = source_array.dtype if dtype is None else dtype
    else:
        dtype = mx_real_t if dtype is None else dtype
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=dtype)
            except:
                raise TypeError('source_array must be array like object')
    arr = empty(source_array.shape, ctx, dtype)
    arr[:] = source_array
    return arr


def moveaxis(tensor, source, destination):
    """Moves the `source` axis into the `destination` position
    while leaving the other axes in their original order

    Parameters
    ----------
    tensor : mx.nd.array
        The array which axes should be reordered
    source : int
        Original position of the axes to move.
    destination : int
        Destination position for each of the original axes.

    Returns
    -------
    result : mx.nd.array
        Array with moved axes.

    Examples
    --------
    >>> X = mx.nd.array([[1, 2, 3], [4, 5, 6]])
    >>> mx.nd.moveaxis(X, 0, 1).shape
    (3L, 2L)
    """
    axes = list(range(tensor.ndim))
    try:
        axes.pop(source)
    except IndexError:
        raise ValueError('Source should verify 0 <= source < tensor.ndim'
                         'Got %d' % source)
    try:
        axes.insert(destination, source)
    except IndexError:
        raise ValueError('Destination should verify 0 <= destination < tensor.ndim'
                         'Got %d' % destination)
    return op.transpose(tensor, axes)


# pylint: disable= no-member, protected-access, too-many-arguments, redefined-outer-name
def arange(start, stop=None, step=1.0, repeat=1, ctx=None, dtype=mx_real_t):
    """Returns evenly spaced values within a given interval.

    Values are generated within the half-open interval [`start`, `stop`). In other
    words, the interval includes `start` but excludes `stop`. The function is
    similar to the built-in Python function `range` and to `numpy.arange`,
    but returns an `NDArray`.

    Parameters
    ----------
    start : number, optional
        Start of interval. The default start value is 0.
    stop : number
        End of interval.
    step : number, optional
        Spacing between values. The default step size is 1.
    repeat : int, optional
        Number of times to repeat each element. The default repeat count is 1.
    ctx : Context, optional
        Device context. Default context is the current default context.
    dtype : str or numpy.dtype, optional
        The data type of the `NDArray`. The default datatype is `np.float32`.

    Returns
    -------
    NDArray
        `NDArray` of evenly spaced values in the specified range.

    Examples
    --------
    >>> mx.nd.arange(3).asnumpy()
    array([ 0.,  1.,  2.], dtype=float32)
    >>> mx.nd.arange(2, 6).asnumpy()
    array([ 2.,  3.,  4.,  5.], dtype=float32)
    >>> mx.nd.arange(2, 6, step=2).asnumpy()
    array([ 2.,  4.], dtype=float32)
    >>> mx.nd.arange(2, 6, step=1.5, repeat=2).asnumpy()
    array([ 2. ,  2. ,  3.5,  3.5,  5. ,  5. ], dtype=float32)
    >>> mx.nd.arange(2, 6, step=2, repeat=3, dtype='int32').asnumpy()
    array([2, 2, 2, 4, 4, 4], dtype=int32)
    """
    if ctx is None:
        ctx = Context.default_ctx
    return _internal._arange(start=start, stop=stop, step=step, repeat=repeat,
                             dtype=dtype, ctx=str(ctx))
# pylint: enable= no-member, protected-access, too-many-arguments


#pylint: disable= too-many-arguments, no-member, protected-access
def _ufunc_helper(lhs, rhs, fn_array, fn_scalar, lfn_scalar, rfn_scalar=None):
    """ Helper function for element-wise operation.
    The function will perform numpy-like broadcasting if needed and call different functions.

    Parameters
    --------
    lhs : NDArray or numeric value
        Left-hand side operand.

    rhs : NDArray or numeric value
        Right-hand operand,

    fn_array : function
        Function to be called if both lhs and rhs are of ``NDArray`` type.

    fn_scalar : function
        Function to be called if both lhs and rhs are numeric values.

    lfn_scalar : function
        Function to be called if lhs is ``NDArray`` while rhs is numeric value

    rfn_scalar : function
        Function to be called if lhs is numeric value while rhs is ``NDArray``;
        if none is provided, then the function is commutative, so rfn_scalar is equal to lfn_scalar

    Returns
    --------
    NDArray
        result array
    """
    if isinstance(lhs, numeric_types):
        if isinstance(rhs, numeric_types):
            return fn_scalar(lhs, rhs)
        else:
            if rfn_scalar is None:
                # commutative function
                return lfn_scalar(rhs, float(lhs))
            else:
                return rfn_scalar(rhs, float(lhs))
    elif isinstance(rhs, numeric_types):
        return lfn_scalar(lhs, float(rhs))
    elif isinstance(rhs, NDArray):
        return fn_array(lhs, rhs)
    else:
        raise TypeError('type %s not supported' % str(type(rhs)))
#pylint: enable= too-many-arguments, no-member, protected-access


def add(lhs, rhs):
    """Returns element-wise sum of the input arrays with broadcasting.

    Equivalent to ``lhs + rhs``, ``mx.nd.broadcast_add(lhs, rhs)`` and
    ``mx.nd.broadcast_plus(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

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
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x+2).asnumpy()
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]], dtype=float32)
    >>> (x+y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> mx.nd.add(x,y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> (z + y).asnumpy()
    array([[ 0.,  1.],
           [ 1.,  2.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
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
    ``mx.nd.broadcast_minus(lhs, rhs)``.

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
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        The element-wise difference of the input arrays.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x-2).asnumpy()
    array([[-1., -1., -1.],
           [-1., -1., -1.]], dtype=float32)
    >>> (x-y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> mx.nd.subtract(x,y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> (z-y).asnumpy()
    array([[ 0.,  1.],
           [-1.,  0.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_sub,
        operator.sub,
        _internal._minus_scalar,
        _internal._rminus_scalar)
    # pylint: enable= no-member, protected-access


def multiply(lhs, rhs):
    """Returns element-wise product of the input arrays with broadcasting.

    Equivalent to ``lhs * rhs`` and ``mx.nd.broadcast_mul(lhs, rhs)``.

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
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x*2).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> (x*y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.multiply(x, y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (z*y).asnumpy()
    array([[ 0.,  0.],
           [ 0.,  1.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
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

    Equivalent to ``lhs / rhs`` and ``mx.nd.broadcast_div(lhs, rhs)``.

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
    >>> x = mx.nd.ones((2,3))*6
    >>> y = mx.nd.ones((2,1))*2
    >>> x.asnumpy()
    array([[ 6.,  6.,  6.],
           [ 6.,  6.,  6.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 2.],
           [ 2.]], dtype=float32)
    >>> x/2
    <NDArray 2x3 @cpu(0)>
    >>> (x/3).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> (x/y).asnumpy()
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]], dtype=float32)
    >>> mx.nd.divide(x,y).asnumpy()
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_div,
        operator.truediv,
        _internal._div_scalar,
        _internal._rdiv_scalar)
    # pylint: enable= no-member, protected-access


def modulo(lhs, rhs):
    """Returns element-wise modulo of the input arrays with broadcasting.

    Equivalent to ``lhs % rhs`` and ``mx.nd.broadcast_mod(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array in modulo.
    rhs : scalar or array
         Second array in modulo.
        The arrays to be taken modulo. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        The element-wise modulo of the input arrays.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))*6
    >>> y = mx.nd.ones((2,1))*4
    >>> x.asnumpy()
    array([[ 6.,  6.,  6.],
           [ 6.,  6.,  6.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 4.],
           [ 4.]], dtype=float32)
    >>> x%5
    <NDArray 2x3 @cpu(0)>
    >>> (x%5).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (x%y).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> mx.nd.modulo(x,y).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_mod,
        operator.mod,
        _internal._mod_scalar,
        _internal._rmod_scalar)
    # pylint: enable= no-member, protected-access


def power(base, exp):
    """Returns result of first array elements raised to powers from second array, element-wise
    with broadcasting.

    Equivalent to ``base ** exp`` and ``mx.nd.broadcast_power(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    base : scalar or NDArray
         The base array
    exp : scalar or NDArray
         The exponent array. If ``base.shape != exp.shape``, they must be
        broadcastable to a common shape.

    Returns
    --------
    NDArray
        The bases in x raised to the exponents in y.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))*2
    >>> y = mx.nd.arange(1,3).reshape((2,1))
    >>> z = mx.nd.arange(1,3).reshape((2,1))
    >>> x.asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 1.],
           [ 2.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 1.],
           [ 2.]], dtype=float32)
    >>> (x**2).asnumpy()
    array([[ 4.,  4.,  4.],
           [ 4.,  4.,  4.]], dtype=float32)
    >>> (x**y).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 4.,  4.,  4.]], dtype=float32)
    >>> mx.nd.power(x,y).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 4.,  4.,  4.]], dtype=float32)
    >>> (z**y).asnumpy()
    array([[ 1.],
           [ 4.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        base,
        exp,
        op.broadcast_power,
        operator.pow,
        _internal._power_scalar,
        _internal._rpower_scalar)
    # pylint: enable= no-member, protected-access


def maximum(lhs, rhs):
    """Returns element-wise maximum of the input arrays with broadcasting.

    Equivalent to ``mx.nd.broadcast_maximum(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be compared.
    rhs : scalar or array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        The element-wise maximum of the input arrays.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> mx.nd.maximum(x, 2).asnumpy()
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)
    >>> mx.nd.maximum(x, y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.maximum(y, z).asnumpy()
    array([[ 0.,  1.],
           [ 1.,  1.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_maximum,
        lambda x, y: x if x > y else y,
        _internal._maximum_scalar,
        None)
    # pylint: enable= no-member, protected-access


def minimum(lhs, rhs):
    """Returns element-wise minimum of the input arrays with broadcasting.

    Equivalent to ``mx.nd.broadcast_minimum(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be compared.
    rhs : scalar or array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        The element-wise minimum of the input arrays.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> mx.nd.minimum(x, 2).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.minimum(x, y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.minimum(z, y).asnumpy()
    array([[ 0.,  0.],
           [ 0.,  1.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_minimum,
        lambda x, y: x if x < y else y,
        _internal._minimum_scalar,
        None)
    # pylint: enable= no-member, protected-access


def equal(lhs, rhs):
    """Returns the result of element-wise **equal to** (==) comparison operation with
    broadcasting.

    For each element in input arrays, return 1(true) if corresponding elements are same,
    otherwise return 0(false).

    Equivalent to ``lhs == rhs`` and ``mx.nd.broadcast_equal(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be compared.
    rhs : scalar or array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        Output array of boolean values.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x == 1).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (x == y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.equal(x,y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (z == y).asnumpy()
    array([[ 1.,  0.],
           [ 0.,  1.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_equal,
        lambda x, y: 1 if x == y else 0,
        _internal._equal_scalar,
        None)
    # pylint: enable= no-member, protected-access


def not_equal(lhs, rhs):
    """Returns the result of element-wise **not equal to** (!=) comparison operation
    with broadcasting.

    For each element in input arrays, return 1(true) if corresponding elements are different,
    otherwise return 0(false).

    Equivalent to ``lhs != rhs`` and ``mx.nd.broadcast_not_equal(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be compared.
    rhs : scalar or array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        Output array of boolean values.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (z == y).asnumpy()
    array([[ 1.,  0.],
           [ 0.,  1.]], dtype=float32)
    >>> (x != 1).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> (x != y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> mx.nd.not_equal(x, y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> (z != y).asnumpy()
    array([[ 0.,  1.],
           [ 1.,  0.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_not_equal,
        lambda x, y: 1 if x != y else 0,
        _internal._not_equal_scalar,
        None)
    # pylint: enable= no-member, protected-access


def greater(lhs, rhs):
    """Returns the result of element-wise **greater than** (>) comparison operation
    with broadcasting.

    For each element in input arrays, return 1(true) if lhs elements are greater than rhs,
    otherwise return 0(false).

    Equivalent to ``lhs > rhs`` and ``mx.nd.broadcast_greater(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be compared.
    rhs : scalar or array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        Output array of boolean values.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x > 1).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> (x > y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> mx.nd.greater(x, y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> (z > y).asnumpy()
    array([[ 0.,  1.],
           [ 0.,  0.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_greater,
        lambda x, y: 1 if x > y else 0,
        _internal._greater_scalar,
        _internal._lesser_scalar)
    # pylint: enable= no-member, protected-access


def greater_equal(lhs, rhs):
    """Returns the result of element-wise **greater than or equal to** (>=) comparison
    operation with broadcasting.

    For each element in input arrays, return 1(true) if lhs elements are greater than equal to rhs,
    otherwise return 0(false).

    Equivalent to ``lhs >= rhs`` and ``mx.nd.broadcast_greater_equal(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be compared.
    rhs : scalar or array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        Output array of boolean values.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x >= 1).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (x >= y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.greater_equal(x, y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (z >= y).asnumpy()
    array([[ 1.,  1.],
           [ 0.,  1.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_greater_equal,
        lambda x, y: 1 if x >= y else 0,
        _internal._greater_equal_scalar,
        _internal._lesser_equal_scalar)
    # pylint: enable= no-member, protected-access


def lesser(lhs, rhs):
    """Returns the result of element-wise **lesser than** (<) comparison operation
    with broadcasting.

    For each element in input arrays, return 1(true) if lhs elements are less than rhs,
    otherwise return 0(false).

    Equivalent to ``lhs < rhs`` and ``mx.nd.broadcast_lesser(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be compared.
    rhs : scalar or array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        Output array of boolean values.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x < 1).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> (x < y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> mx.nd.lesser(x, y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> (z < y).asnumpy()
    array([[ 0.,  0.],
           [ 1.,  0.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_lesser,
        lambda x, y: 1 if x < y else 0,
        _internal._lesser_scalar,
        _internal._greater_scalar)
    # pylint: enable= no-member, protected-access


def lesser_equal(lhs, rhs):
    """Returns the result of element-wise **lesser than or equal to** (<=) comparison
    operation with broadcasting.

    For each element in input arrays, return 1(true) if lhs elements are
    lesser than equal to rhs, otherwise return 0(false).

    Equivalent to ``lhs <= rhs`` and ``mx.nd.broadcast_lesser_equal(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or array
        First array to be compared.
    rhs : scalar or array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        Output array of boolean values.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x <= 1).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (x <= y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.lesser_equal(x, y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (z <= y).asnumpy()
    array([[ 1.,  0.],
           [ 1.,  1.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_lesser_equal,
        lambda x, y: 1 if x <= y else 0,
        _internal._lesser_equal_scalar,
        _internal._greater_equal_scalar)
    # pylint: enable= no-member, protected-access


def true_divide(lhs, rhs):

    """This function is similar to :meth:`divide`.
    """
    return divide(lhs, rhs)


def concatenate(arrays, axis=0, always_copy=True):
    """DEPRECATED, use ``concat`` instead

    Parameters
    ----------
    arrays : list of `NDArray`
        Arrays to be concatenate. They must have identical shape except
        the first dimension. They also must have the same data type.
    axis : int
        The axis along which to concatenate.
    always_copy : bool
        Default `True`. When not `True`, if the arrays only contain one
        `NDArray`, that element will be returned directly, avoid copying.

    Returns
    -------
    NDArray
        An `NDArray` that lives on the same context as `arrays[0].context`.
    """
    assert isinstance(arrays, list)
    assert len(arrays) > 0
    assert isinstance(arrays[0], NDArray)

    if not always_copy and len(arrays) == 1:
        return arrays[0]

    shape_axis = arrays[0].shape[axis]
    shape_rest1 = arrays[0].shape[0:axis]
    shape_rest2 = arrays[0].shape[axis+1:]
    dtype = arrays[0].dtype
    for arr in arrays[1:]:
        shape_axis += arr.shape[axis]
        assert shape_rest1 == arr.shape[0:axis]
        assert shape_rest2 == arr.shape[axis+1:]
        assert dtype == arr.dtype
    ret_shape = shape_rest1 + (shape_axis,) + shape_rest2
    ret = empty(ret_shape, ctx=arrays[0].context, dtype=dtype)

    idx = 0
    begin = [0 for _ in ret_shape]
    end = list(ret_shape)
    for arr in arrays:
        if axis == 0:
            ret[idx:idx+arr.shape[0]] = arr
        else:
            begin[axis] = idx
            end[axis] = idx+arr.shape[axis]
            # pylint: disable=no-member,protected-access
            _internal._crop_assign(ret, arr, out=ret,
                                   begin=tuple(begin),
                                   end=tuple(end))
            # pylint: enable=no-member,protected-access
        idx += arr.shape[axis]

    return ret


# pylint: disable=redefined-outer-name
def imdecode(str_img, clip_rect=(0, 0, 0, 0), out=None, index=0, channels=3, mean=None):
    """DEPRECATED, use mx.img instead

    Parameters
    ----------
    str_img : str
        Binary image data
    clip_rect : iterable of 4 int
        Clip decoded image to rectangle (x0, y0, x1, y1).
    out : NDArray
        Output buffer. Can be 3 dimensional (c, h, w) or 4 dimensional (n, c, h, w).
    index : int
        Output decoded image to i-th slice of 4 dimensional buffer.
    channels : int
        Number of channels to output. Decode to grey scale when channels = 1.
    mean : NDArray
        Subtract mean from decode image before outputing.
    """
    # pylint: disable= no-member, protected-access, too-many-arguments
    if mean is None:
        mean = NDArray(_new_empty_handle())
    if out is None:
        return _internal._imdecode(mean, index,
                                   clip_rect[0],
                                   clip_rect[1],
                                   clip_rect[2],
                                   clip_rect[3],
                                   channels,
                                   len(str_img),
                                   str_img=str_img)
    else:
        return _internal._imdecode(mean, index,
                                   clip_rect[0],
                                   clip_rect[1],
                                   clip_rect[2],
                                   clip_rect[3],
                                   channels,
                                   len(str_img),
                                   str_img=str_img,
                                   out=out)


def zeros(shape, ctx=None, dtype=None, **kwargs):
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


def empty(shape, ctx=None, dtype=None):
    """Returns a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    ctx : Context, optional
        An optional device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`).

    Returns
    -------
    NDArray
        A created array.

    """
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    return NDArray(handle=_new_alloc_handle(shape, ctx, False, dtype))
