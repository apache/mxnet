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


try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, mx_int, mx_int64
from ..base import ctypes2buffer
from ..dlpack import ndarray_to_dlpack_for_read, ndarray_to_dlpack_for_write
from ..dlpack import ndarray_from_dlpack, ndarray_from_numpy
from ..runtime import Features
from ..device import Device, current_device
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase

__all__ = ["NDArray", "concatenate", "dtype_np_to_mx", "dtype_mx_to_np", "_GRAD_REQ_MAP",
           "ones", "add", "arange", "linspace", "eye", "divide", "equal", "full", "greater",
           "greater_equal", "imdecode", "lesser", "lesser_equal", "logical_and", "logical_or",
           "logical_xor", "maximum", "minimum", "moveaxis", "modulo", "multiply", "not_equal",
           "onehot_encode", "power", "subtract", "true_divide", "waitall", "_new_empty_handle",
           "histogram", "split_v2", "to_dlpack_for_read", "to_dlpack_for_write", "from_dlpack",
           "from_numpy", "zeros", "indexing_key_expand_implicit_axes", "get_indexing_dispatch_code",
           "get_oshape_of_gather_nd_op", "bfloat16", "get_dtype_type", "is_mx_dtype",
           "get_dtype_name"]

_STORAGE_TYPE_UNDEFINED = -1
_STORAGE_TYPE_DEFAULT = 0
_STORAGE_TYPE_ROW_SPARSE = 1
_STORAGE_TYPE_CSR = 2
_SIGNED_INT32_UPPER_LIMIT = (2**31 - 1)

bfloat16 = np.dtype([('bfloat16', np.uint16)])

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
    np.bool_: 7,
    np.int16: 8,
    np.uint16 : 9,
    np.uint32 : 10,
    np.uint64 : 11,
    bfloat16: 12,
}

def _register_platform_dependent_mx_dtype():
    """Register platform dependent types to the fixed size counterparts."""
    kind_map = {'i': 'int', 'u': 'uint', 'f': 'float'}
    for np_type in [
            np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc, np.int_,
            np.uint, np.longlong, np.ulonglong, np.half, np.float16, np.single,
            np.double, np.longdouble]:
        dtype = np.dtype(np_type)
        kind, size = dtype.kind, dtype.itemsize
        bits = size * 8
        fixed_dtype = getattr(np, kind_map[kind]+str(bits))
        if fixed_dtype in _DTYPE_NP_TO_MX:
            _DTYPE_NP_TO_MX[np_type] = _DTYPE_NP_TO_MX[fixed_dtype]
_register_platform_dependent_mx_dtype()

_DTYPE_MX_TO_NP = {
    -1: None,
    0: np.float32,
    1: np.float64,
    2: np.float16,
    3: np.uint8,
    4: np.int32,
    5: np.int8,
    6: np.int64,
    7: np.bool_,
    8: np.int16,
    9: np.uint16,
    10: np.uint32,
    11: np.uint64,
    12: bfloat16,
}

def get_dtype_type(dtype):
    if (isinstance(dtype, str) and dtype in bfloat16.names) or np.dtype(dtype) == bfloat16:
        return bfloat16
    return np.dtype(dtype).type

def is_mx_dtype(dtype):
    return get_dtype_type(dtype) in _DTYPE_NP_TO_MX

def get_dtype_name(dtype):
    dtype = np.dtype(get_dtype_type(dtype))
    return bfloat16.names[0] if dtype == bfloat16 else dtype.name

def dtype_np_to_mx(dtype):
    if not is_mx_dtype(dtype):
        raise TypeError('dtype must be one of: ' + str(_DTYPE_NP_TO_MX))
    dtype_type = get_dtype_type(dtype)
    return _DTYPE_NP_TO_MX[dtype_type]

def dtype_mx_to_np(dtype_idx):
    return _DTYPE_MX_TO_NP[dtype_idx]


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

# Return code for dispatching indexing function call
_NDARRAY_UNSUPPORTED_INDEXING = -1
_NDARRAY_BASIC_INDEXING = 0
_NDARRAY_ADVANCED_INDEXING = 1
_NDARRAY_EMPTY_TUPLE_INDEXING = 2

# Return code for 0-d boolean array handler
_NDARRAY_NO_ZERO_DIM_BOOL_ARRAY = -1
_NDARRAY_ZERO_DIM_BOOL_ARRAY_FALSE = 0
_NDARRAY_ZERO_DIM_BOOL_ARRAY_TRUE = 1

# Caching whether MXNet was built with INT64 support or not
_INT64_TENSOR_SIZE_ENABLED = None

def _int64_enabled():
    global _INT64_TENSOR_SIZE_ENABLED
    if _INT64_TENSOR_SIZE_ENABLED is None:
        _INT64_TENSOR_SIZE_ENABLED = Features().is_enabled('INT64_TENSOR_SIZE')
    return _INT64_TENSOR_SIZE_ENABLED

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
    if _int64_enabled():
        check_call(_LIB.MXNDArrayCreate64(
            c_array_buf(mx_int64, native_array('q', shape)),
            ctypes.c_int(len(shape)),
            ctypes.c_int(ctx.device_typeid),
            ctypes.c_int(ctx.device_id),
            ctypes.c_int(int(delay_alloc)),
            ctypes.c_int(int(dtype_np_to_mx(dtype))),
            ctypes.byref(hdl)))
    else:
        # When shape is larger than unit32 then there is an overflow error at python end itself.
        # It needs to be caught here since the call doesn't even reach backend.
        size = 1
        for idx in shape:
            size = size * idx
        if size > _SIGNED_INT32_UPPER_LIMIT:
            raise Exception("[_new_alloc_handle] Size of tensor you are trying to allocate is " +
                            "larger than 2^31 elements. Please build with flag " +
                            "USE_INT64_TENSOR_SIZE=1")
        check_call(_LIB.MXNDArrayCreate(
            c_array_buf(mx_uint, native_array('I', shape)),
            mx_uint(len(shape)),
            ctypes.c_int(ctx.device_typeid),
            ctypes.c_int(ctx.device_id),
            ctypes.c_int(int(delay_alloc)),
            ctypes.c_int(int(dtype_np_to_mx(dtype))),
            ctypes.byref(hdl)))
    return hdl


def _new_from_shared_mem(shared_pid, shared_id, shape, dtype):
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateFromSharedMem(
        ctypes.c_int(shared_pid),
        ctypes.c_int(shared_id),
        c_array(mx_int, shape),
        mx_int(len(shape)),
        ctypes.c_int(int(dtype_np_to_mx(dtype))),
        ctypes.byref(hdl)))
    return hdl


def waitall():
    """Wait for all async operations to finish in MXNet.

    This function is used for benchmarking only.

    .. note::

       If your mxnet code throws an exception, then waitall can cause performance impact.
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
    # Extension type code for TVM function.
    # See C++ side of definition(kTVMNDArrayTypeCode) at include/mxmet/tensor_blob.h
    _tvm_tcode = 19
    # pylint: disable= no-member, undefined-variable

    def as_np_ndarray(self):
        """Convert mxnet.ndarray.NDArray to mxnet.numpy.ndarray."""
        storage_type = self.stype
        if storage_type != 'default':
            raise ValueError('cannot convert ndarray of stype {} to numpy ndarray'
                             .format(str(type(storage_type))))
        from ..numpy import ndarray
        hdl = NDArrayHandle()
        check_call(_LIB.MXShallowCopyNDArray(self.handle, ctypes.byref(hdl)))
        return ndarray(handle=hdl, writable=self.writable)

    def as_nd_ndarray(self):
        """A convenience function for creating a classic ndarray from the current
        ndarray with zero copy. For this class, it just returns itself since it is
        already a classic ndarray."""
        return self

    @property
    def _tvm_handle(self):
        return self.handle.value

    def __repr__(self):
        """Returns a string representation of the array."""
        if self._alive:
            shape_info = 'x'.join([f'{x}' for x in self.shape])
            return f'\n{str(self.asnumpy())}\n<{self.__class__.__name__} {shape_info} @{self.ctx}>'
        else:
            return '<FREED {}>'.format(self.__class__.__name__)

    def __reduce__(self):
        return NDArray, (None,), self.__getstate__()

    def _to_shared_mem(self):
        shared_pid = ctypes.c_int()
        shared_id = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetSharedMemHandle(
            self.handle, ctypes.byref(shared_pid), ctypes.byref(shared_id)))
        return shared_pid.value, shared_id.value, self.shape, self.dtype

    def __abs__(self):
        """x.__abs__() <=> abs(x) <=> x.abs() <=> mx.nd.abs(x, y)"""
        return self.abs()

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
            raise TypeError(f'type {str(type(other))} not supported')

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
            raise TypeError(f'type {str(type(other))} not supported')

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
            raise TypeError(f'type {str(type(other))} not supported')

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
            raise TypeError(f'type {str(type(other))} not supported')

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
            raise TypeError(f'type {str(type(other))} not supported')

    def __pow__(self, other):
        """x.__pow__(y) <=> x**y <=> mx.nd.power(x,y) """
        return power(self, other)

    def __rpow__(self, other):
        """x.__pow__(y) <=> y**x <=> mx.nd.power(y,x) """
        return power(other, self)

    def __eq__(self, other):
        """x.__eq__(y) <=> x==y <=> mx.nd.equal(x, y) """
        return equal(self, other)

    def __hash__(self):
        """Default hash function."""
        return id(self)//16

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

    def __str__(self):
        """Returns a readable string representation of the array."""
        if self.dtype == bfloat16:
            return super(NDArray, self.astype(float)).__str__()
        else:
            return super(NDArray, self).__str__()

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

        Sets ``self[key]`` to ``value``.

        This functions supports advanced indexing as defined in `the NumPy
        advanced indexing documentation
        <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing>`_,
        with the restriction that boolean array indexing is not supported.

        Parameters
        ----------
        key : int, mxnet.ndarray.slice, list, np.ndarray, NDArray, or tuple of all previous types
            The indexing key.
        value : scalar or array-like object that can be broadcast to the shape of self[key]
            The value to set.

        Examples
        --------
        >>> x = mx.nd.zeros((2, 3))
        >>> x[:] = 1
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> x[:, 1:2] = 2
        >>> x.asnumpy()
        array([[ 1.,  2.,  1.],
               [ 1.,  2.,  1.]], dtype=float32)
        >>> x[1:2, 1:] = 3
        >>> x.asnumpy()
        array([[ 1.,  2.,  1.],
               [ 1.,  3.,  3.]], dtype=float32)
        >>> x[1:, 0:2] = mx.nd.zeros((1, 2))
        >>> x.asnumpy()
        array([[ 1.,  2.,  1.],
               [ 0.,  0.,  3.]], dtype=float32)
        >>> x[1, 2] = 4
        >>> x.asnumpy()
        array([[ 1.,  2.,  1.],
               [ 0.,  0.,  4.]], dtype=float32)
        >>> x[[0], [1, 2]] = 5
        >>> x.asnumpy()
        array([[ 1.,  5.,  5.],
               [ 0.,  0.,  4.]], dtype=float32)
        >>> x[::-1, 0:2:2] = [6]
        >>> x.asnumpy()
        array([[ 6.,  5.,  5.],
               [ 6.,  0.,  4.]], dtype=float32)
        """
        if self.ndim == 0:
            if not isinstance(key, (tuple, py_slice)):
                raise IndexError('scalar tensor can only accept `()` and `:` as index')
            if isinstance(key, tuple) and len(key) != 0:
                raise IndexError('scalar tensor can only accept `()` and `:` as index')
            if isinstance(value, numeric_types):
                self._full(value)
            elif isinstance(value, NDArray) and value.size == 1:
                if value.shape != self.shape:
                    value = value.reshape(self.shape)
                value.copyto(self)
            elif isinstance(value, (np.ndarray, np.generic)) and value.size == 1:
                if isinstance(value, np.generic) or value.shape != self.shape:
                    value = value.reshape(self.shape)
                self._sync_copyfrom(value)
            else:
                raise ValueError('setting an array element with a sequence.')

        elif self.size == 0:
            return

        else:
            key, _ = indexing_key_expand_implicit_axes(key, self.shape)
            slc_key = tuple(idx for idx in key if idx is not None)

            if len(slc_key) < self.ndim:
                raise RuntimeError(
                    'too few indices after normalization: expected `ndim` ({}) '
                    'but got {}. This is a bug, please report it!'
                    ''.format(self.ndim, len(slc_key))
                )
            if len(slc_key) > self.ndim:
                raise IndexError(
                    'too many indices ({}) for array with {} dimensions'
                    ''.format(len(slc_key), self.ndim)
                )

            indexing_dispatch_code = get_indexing_dispatch_code(slc_key)
            if indexing_dispatch_code == _NDARRAY_BASIC_INDEXING:
                self._set_nd_basic_indexing(key, value)
            elif indexing_dispatch_code == _NDARRAY_ADVANCED_INDEXING:
                self._set_nd_advanced_indexing(key, value)
            else:
                raise ValueError(
                    'Indexing NDArray with index {} of type {} is not supported'
                    ''.format(key, type(key))
                )

    def __getitem__(self, key):  # pylint: disable=too-many-return-statements
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of this array if the elements fetched are contiguous in memory;
        otherwise, returns a newly created NDArray.
        This functions supports advanced indexing defined in the following reference with
        some restrictions.

        For basic indexing, i.e., if ``key`` consists only of integers,
        ``slice``, ``Ellipsis`` (``...``) and ``None``, a mutable view is
        returned that shares memory with this array if the accessed portion is
        contiguous in memory.
        Otherwise, a newly created ``NDArray`` is returned.

        This functions supports advanced indexing as defined in `the NumPy
        advanced indexing documentation
        <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing>`_,
        with the restriction that boolean array indexing is not supported.

        Parameters
        ----------
        key : int, mxnet.ndarray.slice, list, np.ndarray, NDArray, or tuple of all previous types
            Indexing key.

        Examples
        --------
        The default is to give explicit indices for all axes:

        >>> x = mx.nd.arange(0, 6).reshape((2, 3))
        >>> x.asnumpy()
        array([[ 0.,  1.,  2.],
               [ 3.,  4.,  5.]], dtype=float32)
        >>> x[0, :].asnumpy()
        array([0., 1., 2.], dtype=float32)
        >>> x[0, :2].asnumpy()
        array([0., 1.], dtype=float32)
        >>> x[:, :-1].asnumpy()
        array([[0., 1.],
               [3., 4.]], dtype=float32)

        If fewer indices are given, they are automatically supplemented by an
        appropriate number of ``slice(None)`` ("``:``") to the right. For
        instance, a single integer indexes along the first axis:

        >>> x = mx.nd.arange(0, 6).reshape((2, 3))
        >>> x[0].asnumpy()
        array([0., 1., 2.], dtype=float32)
        >>> x[1:].asnumpy()
        array([[3., 4., 5.]], dtype=float32)

        To omit a range of axes that should be kept as-is, an `Ellipsis`
        ("``...``") can be used:

        >>> x = mx.nd.arange(0, 16).reshape((2, 2, 2, 2))
        >>> x[0, ..., 1].asnumpy()
        array([[1., 3.],
               [5., 7.]], dtype=float32)
        >>> x[0, :, :, 1].asnumpy()  # equivalent
        array([[1., 3.],
               [5., 7.]], dtype=float32)

        New axes of length 1 can be created by inserting ``None``
        (`numpy.newaxis`) in the index:

        >>> x = mx.nd.arange(0, 6).reshape((2, 3))
        >>> x[None, :, :].asnumpy()
        array([[[0., 1., 2.],
                [3., 4., 5.]]], dtype=float32)
        >>> x[None, :, :].shape
        (1, 2, 3)

        If the indexed portion of the array is contiguous in memory, no data
        is copied. Instead, a shared-memory view of the original array is
        returned, and changes to that view affect the original array:

        >>> x = mx.nd.arange(0, 8).reshape((2, 2, 2))
        >>> y = x[0]  # contiguous
        >>> y.asnumpy()
        array([[0., 1.],
               [2., 3.]], dtype=float32)
        >>> y[:] = -1
        >>> x.asnumpy()
        array([[[-1., -1.],
                [-1., -1.]],
        <BLANKLINE>
               [[ 4.,  5.],
                [ 6.,  7.]]], dtype=float32)
        >>> x = mx.nd.arange(0, 8).reshape((2, 2, 2))
        >>> y = x[1, :1, :]  # contiguous
        >>> y.asnumpy()
        array([[4., 5.]], dtype=float32)
        >>> y[:] = -1
        >>> x.asnumpy()
        array([[[ 0.,  1.],
                [ 2.,  3.]],
        <BLANKLINE>
               [[-1., -1.],
                [ 6.,  7.]]], dtype=float32)
        >>> x = mx.nd.arange(0, 8).reshape((2, 2, 2))
        >>> y = x[:, :, 1]  # not contiguous
        >>> y.asnumpy()
        array([[1., 3.],
               [5., 7.]], dtype=float32)
        >>> y[:] = -1
        >>> x.asnumpy()
        array([[[0., 1.],
                [2., 3.]],
        <BLANKLINE>
               [[4., 5.],
                [6., 7.]]], dtype=float32)

        If the indexing key contains `list`, `numpy.ndarray` or `NDArray`
        objects, advanced indexing is triggered, which always returns a
        copy:

        >>> x = mx.nd.arange(0, 8).reshape((2, 2, 2))
        >>> x[[0, 1]].asnumpy()
        array([[[0., 1.],
                [2., 3.]],
        <BLANKLINE>
               [[4., 5.],
                [6., 7.]]], dtype=float32)
        >>> x[[0, 1], :].asnumpy()  # equivalent
        array([[[0., 1.],
                [2., 3.]],
        <BLANKLINE>
               [[4., 5.],
                [6., 7.]]], dtype=float32)
        >>> y = np.array([0, 1], dtype='int32')
        >>> x[1:, y].asnumpy()
        array([[[4., 5.],
                [6., 7.]]], dtype=float32)
        >>> y = mx.nd.array([0, 1], dtype='int32')
        >>> x[1:, y].asnumpy()
        array([[[4., 5.],
                [6., 7.]]], dtype=float32)
        """
        ndim = self.ndim
        shape = self.shape

        if ndim == 0 and (key == () or key == slice(None, None, None)):
            return self

        # Handle simple cases for higher speed
        if isinstance(key, tuple) and len(key) == 0:
            return self
        if isinstance(key, tuple) and len(key) == ndim\
                and all(isinstance(idx, integer_types) for idx in key):
            out = self
            for idx in key:
                out = out[idx]
            return out
        if isinstance(key, integer_types):
            if key > shape[0] - 1:
                raise IndexError(
                    'index {} is out of bounds for axis 0 with size {}'.format(
                        key, shape[0]))
            return self._at(key)
        elif isinstance(key, py_slice):
            if (key.step is None or key.step == 1):
                if  key.start is not None or key.stop is not None:
                    return self._slice(key.start, key.stop)
                else:
                    return self
            elif key.step == 0:
                raise ValueError("slice step cannot be zero")

        key, _ = indexing_key_expand_implicit_axes(key, self.shape)
        if len(key) == 0:
            raise ValueError('indexing key cannot be an empty tuple')

        indexing_dispatch_code = get_indexing_dispatch_code(key)
        if indexing_dispatch_code == _NDARRAY_BASIC_INDEXING:
            return self._get_nd_basic_indexing(key)
        elif indexing_dispatch_code == _NDARRAY_ADVANCED_INDEXING:
            return self._get_nd_advanced_indexing(key)
        else:
            raise RuntimeError

    def _prepare_value_nd(self, value, bcast_shape, squeeze_axes=None):
        """Return a broadcast `NDArray` with same context and dtype as ``self``.
        For setting item, The returned `ndarray` is squeezed according to squeeze_axes since the
        value_nd is assigned to not yet expanded space in original array.
        `value`: numeric types or array like.
        `bcast_shape`: a shape tuple.
        `squeeze_axes`: a sequence of axes to squeeze in the value array.
        """
        if isinstance(value, numeric_types):
            value_nd = full(bcast_shape, value, ctx=self.ctx, dtype=self.dtype)
        elif type(value) == self.__class__:  # pylint: disable=unidiomatic-typecheck
            value_nd = value.as_in_context(self.ctx)
            if value_nd.dtype != self.dtype:
                value_nd = value_nd.astype(self.dtype)
        else:
            try:
                value_nd = array(value, ctx=self.ctx, dtype=self.dtype)
            except:
                raise TypeError('{} does not support assignment with non-array-like '
                                'object {} of type {}'.format(self.__class__, value, type(value)))

        # For setitem, if there is None in indices, we need to squeeze the assigned value_nd
        # since None is also ignored in slicing the  original array.
        if squeeze_axes and value_nd.ndim > len(bcast_shape):
            squeeze_axes = tuple([ax for ax in squeeze_axes if ax < len(value_nd.shape)])
            value_nd = value_nd.squeeze(axis=tuple(squeeze_axes))

        # handle the cases like the following
        # a = nd.zeros((3, 3)), b = nd.ones((1, 1, 1, 1, 3)), a[0] = b
        # b cannot broadcast directly to a[0].shape unless its leading 1-size axes are trimmed
        if value_nd.ndim > len(bcast_shape):
            squeeze_axes = []
            for i in range(value_nd.ndim - len(bcast_shape)):
                if value_nd.shape[i] == 1:
                    squeeze_axes.append(i)
                else:
                    break
            if squeeze_axes:
                value_nd = value_nd.squeeze(squeeze_axes)

        if value_nd.shape != bcast_shape:
            if value_nd.size == 0:
                value_nd = value_nd.reshape(bcast_shape)
            else:
                value_nd = value_nd.broadcast_to(bcast_shape)
        return value_nd

    # pylint: disable=invalid-name
    @staticmethod
    def _basic_indexing_key_to_begin_end_step(idcs, shape, keep_none=True):
        """Map a tuple of ``slice`` and ``None`` (ignored) to begin, end, step tuples."""
        idcs = [idx for idx in idcs if idx is not None]
        idcs = [idx if isinstance(idx, py_slice) else _int_to_slice(idx)
                for idx in idcs]

        if keep_none:
            sss_list = [(slc.start, slc.stop, slc.step) for slc, n in zip(idcs, shape)]
        else:
            sss_list = [slc.indices(n) for slc, n in zip(idcs, shape)]
        return tuple(zip(*sss_list))
    # pylint: enable=invalid-name

    # pylint: disable=invalid-name
    @staticmethod
    def _basic_indexing_key_int_to_slice(idcs):
        """Return the converted indexing tuple and the integer axes."""
        int_axes = []
        conv_idcs = []
        for ax, idx in enumerate(idcs):
            if isinstance(idx, integer_types):
                conv_idcs.append(_int_to_slice(idx))
                int_axes.append(ax)
            else:
                conv_idcs.append(idx)

        return tuple(conv_idcs), tuple(int_axes)
    # pylint: enable=invalid-name

    @staticmethod
    def _new_axes_after_basic_indexing(axes, key):
        """Return indices of ``axes`` after slicing with ``key``.

        This function is used to calculate the positions where new axes should
        end up after indexing, taking into account the removal of axes by
        integer indexing.

        The ``key`` sequence should be the exapanded key including slices, integer types
        and ``None``.
        """
        steps = [0] + [0 if isinstance(idx, integer_types) else 1 for idx in key]
        cum_steps = np.cumsum(steps)
        axes_after = tuple(cum_steps[axes])
        return axes_after

    @staticmethod
    def _new_axes_after_advanced_indexing(key, adv_axs, bcast_adv_ndim, adv_are_adjacent):  # pylint: disable=invalid-name
        """
        Return indices of ``axes`` after slicing with ``key_nd``.

        This function is used to calculate the positions where new axes should
        end up after indexing, taking into account the removal of axes by
        integer indexing.

        The ``key`` sequence should be the exapanded key including slices, array like objects,
        integer types and ``None``.
        ``adv_axes`` is the sequence of indices of advanced axes.
        ``bcast_adv_ndim`` is the number of dimensions of advanced indexing subspace.
        ``adv_are_adjacent`` is a boolean value. Value being True means all advanced indicies are adjacent.

        Note: integer indices are also considered advanced indices here.
        """
        new_axes = [ax for ax in range(len(key)) if key[ax] is None]
        adv_axs_set = set(adv_axs)
        if not adv_are_adjacent:
            steps = [bcast_adv_ndim] + [0 if ax in adv_axs_set else 1 for ax in range(len(key))]
        else:
            steps = [0] + [0 if ax in adv_axs_set else 1 for ax in range(len(key))]
        cum_steps = np.cumsum(steps)
        axes_after = tuple(cum_steps[new_axes])
        return axes_after

    # pylint: disable=invalid-name
    @staticmethod
    def _basic_indexing_slice_is_contiguous(slc_key, shape):
        """Whether indexing with the given key results in a contiguous array.

        The rule is: From right to left, if in an axis, a slice produces a
        proper subset, the later slice must have <=1 elements.

        The ``slc_key`` sequence must have the same length as ``shape`` and
        only contain `slice` objects.
        """
        assert len(slc_key) == len(shape)
        is_subset = False
        total_sliced_elements = np.prod([_get_slice_len(slc, n)
                                         for slc, n in zip(slc_key, shape)])
        if total_sliced_elements in (0, 1):
            return True
        for idx, n in zip(reversed(slc_key), reversed(shape)):
            _, _, step = idx.indices(n)
            num_elements = _get_slice_len(idx, n)
            if num_elements == 0:
                return True
            elif num_elements > 1 and (step > 1 or step < 0):
                # We do not support the case of reverse slicing of multiple elements and
                # forward slicing of #elements > 1 and step > 1
                return False
            elif is_subset:
                if num_elements > 1:
                    return False
            else:
                if num_elements < n:
                    is_subset = True
        return True
    # pylint: enable=invalid-name

    @staticmethod
    def _basic_indexing_sliced_shape(slc_key, shape):
        """Return the shape after slicing with the given key."""
        assert len(slc_key) == len(shape)
        sliced_shape = []
        for slc, n in zip(slc_key, shape):
            num_elements = _get_slice_len(slc, n)
            sliced_shape.append(num_elements)
        return tuple(sliced_shape)

    # pylint: disable=invalid-name
    @staticmethod
    def _basic_indexing_contiguous_flat_begin_end(slc_key, shape):
        """Return the flat indices of begin and end for contiguous slicing."""
        assert len(slc_key) == len(shape)
        flat_begin, flat_end = 0, 0
        for slc, n in zip(slc_key, shape):
            flat_begin *= n
            flat_end *= n
            begin, _, _ = slc.indices(n)
            num_elements = _get_slice_len(slc, n)
            if num_elements == 0:
                return 0, 0
            else:
                flat_begin += begin
                flat_end += begin + num_elements - 1
        return flat_begin, flat_end + 1
    # pylint: enable=invalid-name

    @staticmethod
    def _drop_int_axes(indexed_shape, int_axes):
        """drop the axis of indexed_shape corresponding to int axes"""
        bcast_shape = []
        for i, size in enumerate(indexed_shape):
            if i not in int_axes:
                bcast_shape.append(size)
        if not bcast_shape:
            bcast_shape = [1]
        return tuple(bcast_shape)

    def _set_nd_basic_indexing(self, key, value):
        """This function indexes ``self`` with a tuple of ``slice`` objects only."""
        for idx in key:
            if idx is not None and not isinstance(idx, (py_slice, integer_types)):
                raise RuntimeError(
                    '`key` may only contain `slice` or integer objects in the '
                    'basic implementation, got object of type {}. '
                    'This is a bug, please report it!'
                    ''.format(type(idx)))
        key_nd = tuple(idx for idx in key if idx is not None)
        int_axes = [
            ax for ax in range(len(key_nd)) if isinstance(key_nd[ax], integer_types)
        ]

        # Check bounds for integer axes
        for ax in int_axes:  # pylint: disable=invalid-name
            if not -self.shape[ax] <= key_nd[ax] < self.shape[ax]:
                raise IndexError(
                    'index {} is out of bounds for axis {} with size {}'
                    ''.format(key_nd[ax], ax, self.shape[ax]))

        begin, end, step = self._basic_indexing_key_to_begin_end_step(
            key, self.shape, keep_none=False
        )
        indexed_shape = tuple(
            _get_dim_size(b, e, s) for b, e, s in zip(begin, end, step)
        )
        can_assign_directly = (
            (indexed_shape == self.shape) and all(s > 0 for s in step)
        )
        begin, end, step = self._basic_indexing_key_to_begin_end_step(
            key, self.shape, keep_none=True
        )
        none_axes = [ax for ax in range(len(key)) if key[ax] is None]
        new_axes = self._new_axes_after_basic_indexing(none_axes, key)

        if can_assign_directly:
            # Easy case, overwrite whole array.
            if type(value) == self.__class__:  # pylint: disable=unidiomatic-typecheck
                if value.handle is not self.handle:
                    # Need to do this before `broadcast_to`.
                    bcast_shape = self._drop_int_axes(indexed_shape, int_axes)
                    value_nd = self._prepare_value_nd(value, bcast_shape=bcast_shape, squeeze_axes=new_axes)
                    value_nd = value_nd.reshape(indexed_shape)
                    value_nd.copyto(self)

            elif isinstance(value, numeric_types):
                if isinstance(value, bool):
                    self._full(int(value))
                else:
                    self._full(value)

            elif isinstance(value, (np.ndarray, np.generic)):
                tmp_shape = _shape_for_bcast(
                    value.shape, target_ndim=self.ndim, new_axes=int_axes
                )
                value = value.reshape(tmp_shape)
                if isinstance(value, np.generic) or value.shape != self.shape:
                    value = np.broadcast_to(value, self.shape)
                self._sync_copyfrom(value)

            else:
                # Other array-like
                # drop the axis of indexed_shape corresponding to int axes
                bcast_shape = self._drop_int_axes(indexed_shape, int_axes)
                value_nd = self._prepare_value_nd(value, bcast_shape=bcast_shape, squeeze_axes=new_axes)
                value_nd = value_nd.reshape(indexed_shape)
                value_nd.copyto(self)

        elif isinstance(value, numeric_types):
            self.slice_assign_scalar(float(value), begin, end, step)

        else:
            # drop the axis of indexed_shape corresponding to int axes
            bcast_shape = self._drop_int_axes(indexed_shape, int_axes)
            value_nd = self._prepare_value_nd(value, bcast_shape=bcast_shape, squeeze_axes=new_axes)
            value_nd = value_nd.reshape(indexed_shape)
            self.slice_assign(value_nd, begin, end, step)

    def _get_nd_basic_indexing(self, key):
        """This function indexes ``self`` with a tuple of `slice` objects only."""
        key_nd = tuple(idx for idx in key if idx is not None)
        if len(key_nd) < self.ndim:
            raise RuntimeError(
                'too few indices after normalization: expected `ndim` ({}) '
                'but got {}. This is a bug, please report it!'
                ''.format(self.ndim, len(key_nd))
            )
        if len(key_nd) > self.ndim:
            raise IndexError(
                'too many indices ({}) for array with {} dimensions'
                ''.format(len(key_nd), self.ndim)
            )
        slc_key, int_axes = self._basic_indexing_key_int_to_slice(key_nd)
        none_axes = [ax for ax in range(len(key)) if key[ax] is None]
        if none_axes:
            new_axes = self._new_axes_after_basic_indexing(none_axes, key)
        else:
            new_axes = []

        # Check bounds for integer axes
        for ax in int_axes:  # pylint: disable=invalid-name
            if not -self.shape[ax] <= key_nd[ax] < self.shape[ax]:
                raise IndexError(
                    'index {} is out of bounds for axis {} with size {}'
                    ''.format(key_nd[ax], ax, self.shape[ax]))

        # Convert to begin, end and step, and return immediately if the slice
        # is empty
        begin, end, step = self._basic_indexing_key_to_begin_end_step(
            slc_key, self.shape, keep_none=False
        )

        if self._basic_indexing_slice_is_contiguous(slc_key, self.shape):
            # Create a shared-memory view by using low-level flat slicing
            flat_begin, flat_end = self._basic_indexing_contiguous_flat_begin_end(
                slc_key, self.shape
            )
            handle = NDArrayHandle()
            flat_self = self.reshape(-1)
            if _int64_enabled():
                check_call(
                    _LIB.MXNDArraySlice64(
                        flat_self.handle,
                        ctypes.c_int64(flat_begin),
                        ctypes.c_int64(flat_end),
                        ctypes.byref(handle),
                    )
                )
            else:
                check_call(
                    _LIB.MXNDArraySlice(
                        flat_self.handle,
                        ctypes.c_uint32(flat_begin),
                        ctypes.c_uint32(flat_end),
                        ctypes.byref(handle),
                    )
                )
            sliced_shape = self._basic_indexing_sliced_shape(slc_key, self.shape)
            sliced = NDArray(handle=handle, writable=self.writable).reshape(sliced_shape)
        else:
            begin, end, step = self._basic_indexing_key_to_begin_end_step(
                slc_key, self.shape, keep_none=True
            )
            sliced = op.slice(self, begin, end, step)

        # Reshape to final shape due to integer and `None` entries in `key`.
        final_shape = [sliced.shape[i] for i in range(sliced.ndim)
                       if i not in int_axes]
        for ax in new_axes:  # pylint: disable=invalid-name
            final_shape.insert(ax, 1)

        if len(final_shape) == 0:
            # Override for single element indexing
            final_shape = [1]
        return sliced.reshape(final_shape)

    @staticmethod
    def _advanced_index_to_array(idx, ax_len, ctx):
        """Convert ``idx`` to `NDArray` for advanced indexing.

        The ``ax_len`` is used to convert `slice` objects to integer arrays.
        """
        if _int64_enabled():
            idx_dtype = 'int64'
        else:
            idx_dtype = 'int32'
        if isinstance(idx, NDArray):
            if idx.dtype != idx_dtype:
                idx = idx.astype(idx_dtype)
            return idx.as_in_context(ctx)
        elif isinstance(idx, (np.ndarray, list, tuple)):
            return array(idx, ctx, idx_dtype)
        elif isinstance(idx, integer_types):
            return array([idx], ctx, idx_dtype)
        elif isinstance(idx, py_slice):
            start, stop, step = idx.indices(ax_len)
            return arange(start, stop, step, ctx=ctx, dtype=idx_dtype)
        elif isinstance(idx, range):
            return arange(idx.start, idx.stop, idx.step, ctx=ctx, dtype=idx_dtype)
        else:
            raise RuntimeError('illegal index type {}'.format(type(idx)))

    # pylint: disable=invalid-name
    @staticmethod
    def _broadcast_advanced_indices(arrays, block_axes):
        """Broadcast arrays according to position in the sequence.

        Here, "according to position" means that an array of dimension 1
        (which is the case for all except ``block_axes``) will have shape
        ``(1, ..., 1, N, 1, ..., 1)``, where ``N`` is the length, and the
        position of ``N`` in the shape is the same as the position of the
        array in the ``arrays`` sequence, plus extra dimensions of the
        advanced block if it is left of the array.

        The arrays at ``block_axes`` are the advanced indices. They are assumed to
        be ready for mutual broadcasting to produce the advanced indexing block.
        It is further assumed that the numbers in ``block_axes`` are consecutive.

        The return value is a tuple containing the arrays with broadcast shapes.
        """
        block_shape = _broadcast_shapes([arrays[ax] for ax in block_axes])
        ndim_blk = len(block_shape)
        ndim_blk_delta = ndim_blk - len(block_axes)
        ndim_lead = block_axes[0]
        ndim_trail = len(arrays) - (block_axes[-1] + 1)

        bcast_shape = (
            tuple(arrays[ax].shape[0] for ax in range(ndim_lead)) +
            block_shape +
            tuple(arrays[ax].shape[0] for ax in range(block_axes[-1] + 1, len(arrays)))
        )

        bcast_arrays = [None] * len(arrays)
        for ax in block_axes:
            arr = arrays[ax].broadcast_to(block_shape)
            shp = (1,) * ndim_lead + block_shape + (1,) * ndim_trail
            bcast_arrays[ax] = arr.reshape(shp).broadcast_to(bcast_shape)

        for ax in set(range(len(arrays))) - set(block_axes):
            shp = [1] * len(bcast_shape)
            if ax < ndim_lead:
                shp[ax] = arrays[ax].shape[0]
            else:
                shp[ax + ndim_blk_delta] = arrays[ax].shape[0]
            bcast_arrays[ax] = arrays[ax].reshape(shp).broadcast_to(bcast_shape)

        return tuple(bcast_arrays)
    # pylint: enable=invalid-name

    @staticmethod
    def _drop_slice_none_at_end(key):
        """Remove ``slice(None)`` at the end of a key.

        This is used for efficiency in advanced indexing, to avoid generating
        ``arange(n)`` arrays for these axes. The `gather_nd` and `scatter_nd`
        handle implicit full trailing axes automatically.
        """
        key = list(key)
        while isinstance(key[-1], py_slice) and key[-1] == slice(None):
            key.pop()
        return tuple(key)

    def _get_index_nd(self, key):
        """
        Return an index array for use in `scatter_nd` and `gather_nd`,
        and a list of positions of new_axes in ouptut shape.
        """
        key_nd = tuple(idx for idx in key if idx is not None)
        if len(key_nd) < self.ndim:
            raise RuntimeError(
                'too few indices after normalization: expected `ndim` ({}) '
                'but got {}. This is a bug, please report it!'
                ''.format(self.ndim, len(key_nd))
            )
        if len(key_nd) > self.ndim:
            raise IndexError(
                'too many indices ({}) for array with {} dimensions'
                ''.format(len(key_nd), self.ndim)
            )
        ndim = len(key_nd)

        # --- Preparation --- #

        # - Make lists for bookkeeping of advanced indices & axes
        # - Drop trailing `slice(None)` entries in `key` for efficiency
        # - Determine whether the advanced indices are adjacent in `key`
        # - Depending on that, make index permutations to move around indices

        adv_axs = [ax for ax, idx in enumerate(key) if _is_advanced_index(idx)]
        adv_axs_nd = [ax for ax, idx in enumerate(key_nd) if _is_advanced_index(idx)]
        adv_idcs_are_adjacent = bool(np.all(np.diff(adv_axs) == 1))
        nonadv_axs_nd = [ax for ax in range(ndim) if ax not in adv_axs_nd]
        adv_idcs_nd = [key_nd[ax] for ax in adv_axs_nd]
        idcs_short = self._drop_slice_none_at_end(key_nd)
        dropped_axs = list(range(len(idcs_short), ndim))

        if adv_idcs_are_adjacent:
            # The easy case: the advanced block can stay at its position, and no
            # permutation needs to be done (identity permutation)
            axs_nd_permut = axs_nd_permut_inv = tuple(range(ndim))
            idcs_permut_short = idcs_short
            block_axs_nd = adv_axs_nd
        else:
            # The more complicated case: during broadcasting, we need to use the
            # indices in the *permuted* order, where the advanced block is
            # at the beginning, while the final index for `gather_nd` is stacked
            # in the *original* order, so that the association of index with
            # array axis remains the same.

            # This order is used for broadcasting: advanced block at the beginning
            idcs_permut_short = (
                adv_idcs_nd +
                [key_nd[ax] for ax in range(ndim)
                 if ax not in adv_axs_nd and ax not in dropped_axs]
            )
            block_axs_nd = list(range(len(adv_axs_nd)))
            axs_nd_permut = adv_axs_nd + nonadv_axs_nd
            axs_nd_permut_inv = list(np.argsort(axs_nd_permut))

        # --- Conversion, broadcasting and index stacking --- #

        # - Convert all indices in `key` to arrays: integers to 1-element arrays,
        #   `slice` objects to arrays with explicit indices
        # - Reshape arrays for broadcasting according to their position in the
        #   *permuted* key
        # - Broadcast and stack the indices in the *original* order

        shape_nd_permut = tuple(self.shape[ax] for ax in axs_nd_permut)
        converted_idcs_short = [
            self._advanced_index_to_array(idx, ax_len, self.ctx)
            for idx, ax_len in zip(idcs_permut_short, shape_nd_permut)
        ]
        bcast_idcs_permut_short = self._broadcast_advanced_indices(
            converted_idcs_short, block_axes=block_axs_nd
        )

        # Get the ndim of advanced indexing subspace
        converted_advanced_idcs = [
            self._advanced_index_to_array(idx, ax_len, self.ctx)
            for idx, ax_len in zip(adv_idcs_nd, [self.shape[ax] for ax in adv_axs_nd])
        ]
        bcast_advanced_shape = _broadcast_shapes(converted_advanced_idcs)

        # Undo the permutation to restore the original order
        bcast_idcs_short = [
            bcast_idcs_permut_short[ax]
            for ax in axs_nd_permut_inv
            if axs_nd_permut[ax] not in dropped_axs
        ]

        # Calculate where the newaxes are inserted after advanced indexing
        new_axes_positions = self._new_axes_after_advanced_indexing(key, adv_axs,\
                                len(bcast_advanced_shape), adv_idcs_are_adjacent)

                                # if any array is numpy.ndarray, stack in numpy ndarray class.
        for idcs in bcast_idcs_short:
            if type(idcs) != NDArray:  # pylint: disable=unidiomatic-typecheck
                return bcast_idcs_short, new_axes_positions

        return op.stack(*bcast_idcs_short), new_axes_positions

    def _set_nd_advanced_indexing(self, key, value):
        """This function is called by __setitem__ when key is an advanced index."""
        indices, new_axes = self._get_index_nd(key)
        vshape = get_oshape_of_gather_nd_op(self.shape, indices.shape)
        value_nd = self._prepare_value_nd(value, bcast_shape=vshape, squeeze_axes=new_axes)
        self._scatter_set_nd(value_nd, indices)

    def _get_nd_advanced_indexing(self, key):
        """Get item when key is a tuple of any objects of the following types:
        NDArray, np.ndarray, list, tuple, slice, and integer."""
        slc_key, new_axes = self._get_index_nd(key)
        sliced = op.gather_nd(self, slc_key)

        # Reshape due to `None` entries in `key`.
        if new_axes:
            final_shape = [sliced.shape[i] for i in range(sliced.ndim)]
            for ax in new_axes:  # pylint: disable=invalid-name
                final_shape.insert(ax, 1)
            return sliced.reshape(final_shape)
        else:
            return sliced

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
                                f'type {str(type(array))} is not supported')
        source_array = np.asarray(source_array, dtype=self.dtype, order='C')
        if source_array.shape != self.shape:
            raise ValueError(f'Shape inconsistent: expected {str(source_array.shape)} vs got {str(self.shape)}')
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
        start, stop, _ = _get_index_range(start, stop, self.shape[0])

        check_call(_LIB.MXNDArraySlice(
            self.handle, mx_uint(start), mx_uint(stop), ctypes.byref(handle)))
        return self.__class__(handle=handle, writable=self.writable)

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
        if idx < 0:
            length = self.shape[0]
            idx += length
            if idx < 0:
                raise IndexError(f'index {idx-length} is out of bounds for axis 0 with size {length}')
        if _int64_enabled():
            check_call(_LIB.MXNDArrayAt64(
                self.handle, ctypes.c_int64(idx), ctypes.byref(handle)))
        else:
            check_call(_LIB.MXNDArrayAt(
                self.handle, ctypes.c_uint32(idx), ctypes.byref(handle)))
        return self.__class__(handle=handle, writable=self.writable)

    def reshape(self, *shape, **kwargs):
        """Returns a **view** of this array with a new shape without altering any data.

        Parameters
        ----------
        shape : tuple of int, or n ints
            The new shape should not change the array size, namely
            ``np.prod(new_shape)`` should be equal to ``np.prod(self.shape)``.
            Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}.
            The significance of each is explained below:

            - ``0``  copy this dimension from the input to the output shape.

              Example::

              - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
              - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)

            - ``-1`` infers the dimension of the output shape by using the remainder of the
              input dimensions keeping the size of the new array same as that of the input array.
              At most one dimension of shape can be -1.

              Example::

              - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
              - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
              - input shape = (2,3,4), shape=(-1,), output shape = (24,)

            - ``-2`` copy all/remainder of the input dimensions to the output shape.

              Example::

              - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
              - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
              - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)

            - ``-3`` use the product of two consecutive dimensions of the input shape as the
              output dimension.

              Example::

              - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
              - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
              - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
              - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)

            - ``-4`` split one dimension of the input into two dimensions passed subsequent to
              -4 in shape (can contain -1).

              Example::

              - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
              - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)

            - If the argument `reverse` is set to 1, then the special values are inferred from right
              to left.

              Example::

              - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be \
                (40,5).
              - with reverse=1, output shape will be (50,4).

        reverse : bool, default False
            If true then the special values are inferred from right to left. Only supported as
            keyword argument.


        Returns
        -------
        NDArray
            An array with desired shape that shares data with this array.

        Examples
        --------
        >>> x = mx.nd.arange(0,6).reshape(2,3)
        >>> x.asnumpy()
        array([[ 0.,  1.,  2.],
               [ 3.,  4.,  5.]], dtype=float32)
        >>> y = x.reshape(3,2)
        >>> y.asnumpy()
        array([[ 0.,  1.],
               [ 2.,  3.],
               [ 4.,  5.]], dtype=float32)
        >>> y = x.reshape(3,-1)
        >>> y.asnumpy()
        array([[ 0.,  1.],
               [ 2.,  3.],
               [ 4.,  5.]], dtype=float32)
        >>> y = x.reshape(3,2)
        >>> y.asnumpy()
        array([[ 0.,  1.],
               [ 2.,  3.],
               [ 4.,  5.]], dtype=float32)
        >>> y = x.reshape(-3)
        >>> y.asnumpy()
        array([ 0.  1.  2.  3.  4.  5.], dtype=float32)
        >>> y[:] = -1
        >>> x.asnumpy()
        array([[-1., -1., -1.],
               [-1., -1., -1.]], dtype=float32)
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        elif not shape:
            shape = kwargs.get('shape')
            assert shape, "Shape must be provided."
        if not all(k in ['shape', 'reverse'] for k in kwargs):
            raise TypeError(
                "Got unknown keywords in reshape: {}. " \
                "Accepted keyword arguments are 'shape' and 'reverse'.".format(
                    ', '.join([k for k in kwargs if k not in ['shape', 'reverse']])))
        reverse = kwargs.get('reverse', False)
        handle = NDArrayHandle()

        # Actual reshape
        check_call(_LIB.MXNDArrayReshape64(self.handle,
                                           len(shape),
                                           c_array(ctypes.c_int64, shape),
                                           reverse,
                                           ctypes.byref(handle)))
        res = self.__class__(handle=handle, writable=self.writable)

        # Array size should not change
        if np.prod(res.shape) != np.prod(self.shape):
            raise ValueError('Cannot reshape array of size {} into shape {}'.format(np.prod(self.shape), shape))
        return res

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

    def split_v2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split_v2`.

        The arguments are the same as for :py:func:`split_v2`, with
        this array as data.
        """
        return split_v2(self, *args, **kwargs)

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

    def slice_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_like`.

        The arguments are the same as for :py:func:`slice_like`, with
        this array as data.
        """
        return op.slice_like(self, *args, **kwargs)

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

    def flatten(self, inplace=False):
        """Flatten this array without altering any data.

        Parameters
        ----------
        inplace : bool, default False
            If True, this method returns a **view** of this array
            that shares data with this array. Otherwise, a copy is returned.

        Returns
        -------
        NDArray
            An array with flattened shape `(d1, d2*...*dk)` that shares data with
            this array with shape `(d1, d2, ..., dk)`.

        Examples
        --------
        >>> x = mx.nd.arange(30).reshape(5,2,3)
        >>> y = x.flatten(inplace=True)
        >>> z = x.flatten()
        >>> y.shape
        (5, 6)
        >>> y[0].asnumpy()
        array([0., 1., 2., 3., 4., 5.], dtype=float32)
        >>> y[:] = -1
        >>> x[0].asnumpy()
        array([[-1., -1., -1.],
               [-1., -1., -1.]], dtype=float32)
        >>> z[0].asnumpy()
        array([0., 1., 2., 3., 4., 5.], dtype=float32)
        """
        return op.flatten(self) if not inplace else self.reshape((0, -1))

    def shape_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`shape_array`.

        The arguments are the same as for :py:func:`shape_array`, with
        this array as data.
        """
        return op.shape_array(self, *args, **kwargs)

    def size_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`size_array`.

        The arguments are the same as for :py:func:`size_array`, with
        this array as data.
        """
        return op.size_array(self, *args, **kwargs)

    def expand_dims(self, axis, inplace=False):
        """Adds an additional dimension to the current array without altering any data.

        Parameters
        ----------
        axis : int
            Position where new axis is to be inserted.
            Suppose that the input NDArray's dimension is ndim,
            the range of the inserted axis is [-ndim, ndim].
        inplace : bool, default False
            If True, this method returns a **view** of this array
            that shares data with this array. Otherwise, a copy is returned.

        Returns
        -------
        NDArray
            An array with expanded shape `(d1, d2, ..., 1, di, ..., dk)`
            that shares data with this array with shape `(d1, d2, ..., dk)`,
            given input axis `i`.

        Examples
        --------
        >>> x = mx.nd.arange(6).reshape(2,3)
        >>> y = x.expand_dims(1, inplace=True)
        >>> z = x.expand_dims(1)
        >>> y.shape
        (2, 1, 3)
        >>> y[0].asnumpy()
        array([[0., 1., 2.]], dtype=float32)
        >>> y[:] = -1
        >>> x.asnumpy()
        array([[-1., -1., -1.],
               [-1., -1., -1.]], dtype=float32)
        >>> z[0].asnumpy()
        array([[0., 1., 2.]], dtype=float32)
        """
        if not inplace:
            return op.expand_dims(self, axis=axis)
        else:
            new_shape = list(self.shape)
            assert -len(new_shape)-1 <= axis <= len(new_shape), \
                    "axis {} is out of range for {}d array".format(axis, len(new_shape))
            if axis < 0:
                axis += len(new_shape) + 1
            new_shape.insert(axis, 1)
            return self.reshape(new_shape)

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

    def depth_to_space(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`depth_to_space`.

        The arguments are the same as for :py:func:`depth_to_space`, with
        this array as data.
        """
        return op.depth_to_space(self, *args, **kwargs)

    def space_to_depth(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`space_to_depth`.

        The arguments are the same as for :py:func:`space_to_depth`, with
        this array as data.
        """
        return op.space_to_depth(self, *args, **kwargs)

    def diag(self, k=0, **kwargs):
        """Convenience fluent method for :py:func:`diag`.

        The arguments are the same as for :py:func:`diag`, with
        this array as data.
        """
        return op.diag(self, k, **kwargs)

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

    def log_sigmoid(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log_sigmoid`.

        The arguments are the same as for :py:func:`log_sigmoid`, with
        this array as data.
        """
        return op.log_sigmoid(self, *args, **kwargs)

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

    def softmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmin`.

        The arguments are the same as for :py:func:`softmin`, with
        this array as data.
        """
        return op.softmin(self, *args, **kwargs)

    def mish(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`mish`.

        The arguments are the same as for :py:func:`mish`, with
        this array as data.
        """
        return op.mish(self, *args, **kwargs)

    def squeeze(self, axis=None, inplace=False):
        """Remove dimensions with size 1 from this array without altering any data.

        Parameters
        ----------
        axis : int, tuple of int, or None
            Selects a subset of the single-dimensional entries in the shape.
            If an axis is selected with shape entry greater than one, an error is raised.
        inplace : bool, default False
            If True, this method returns a **view** of this array
            that shares data with this array. Otherwise, a copy is returned.
        """
        if not inplace:
            return op.squeeze(self, axis=axis)
        else:
            new_shape = list(self.shape)
            axes = axis # rename variable for readability
            if isinstance(axes, int):
                axes = [axes]
            if axes:
                assert len(axes) == len(set(axes)), \
                    "axis {} contains duplicate which is not allowed.".format(axes)
                resolved_axes = [i if i >= 0 else i+len(self.shape) for i in axes]
                for arg_axis, actual_axis in zip(axes, resolved_axes):
                    assert -len(new_shape) <= arg_axis < len(new_shape), \
                        "axis {} is out of range for {}d array".format(arg_axis, len(new_shape))
                    axis_size = new_shape[actual_axis]
                    assert axis_size == 1, \
                        "Squeeze target axis {} must be size 1, got {}.".format(arg_axis, axis_size)
                for i in sorted(resolved_axes, reverse=True):
                    del new_shape[i]
            else:
                for i in reversed(range(len(new_shape))):
                    if new_shape[i] == 1:
                        del new_shape[i]
            if not new_shape:
                new_shape.append(1)

            return self.reshape(new_shape)

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

    def broadcast_like(self, other):
        """Broadcasts the input array to the shape of other.

        Broadcasting is only allowed on axes with size 1. The new shape cannot change
        the number of dimensions.
        For example, you could broadcast from shape (2, 1) to (2, 3), but not from
        shape (2, 3) to (2, 3, 3).

        Parameters
        ----------
        other : NDArray
            Array with shape of the desired array.

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
        >>> y = x.broadcast_like(mx.nd.ones((2,3,3)))
        >>> y.asnumpy()
        array([[[ 0.,  0.,  0.],
                [ 1.,  1.,  1.],
                [ 2.,  2.,  2.]],
        <BLANKLINE>
               [[ 0.,  0.,  0.],
                [ 1.,  1.,  1.],
                [ 2.,  2.,  2.]]], dtype=float32)
        """
        return self.broadcast_to(other.shape)

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
        ndim = mx_int()
        if _int64_enabled():
            pdata = ctypes.POINTER(mx_int64)()
            check_call(_LIB.MXNDArrayGetShape64(
                self.handle, ctypes.byref(ndim), ctypes.byref(pdata)))
        else:
            pdata = ctypes.POINTER(mx_int)()
            check_call(_LIB.MXNDArrayGetShape(
                self.handle, ctypes.byref(ndim), ctypes.byref(pdata)))
        if ndim.value == -1:
            return None
        else:
            return tuple(pdata[:ndim.value])  # pylint: disable=invalid-slice-index


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
        <class 'mxnet.device.Device'>
        >>> y = mx.nd.zeros((2,3), mx.gpu(0))
        >>> y.context
        gpu(0)
        """
        dev_typeid = ctypes.c_int()
        dev_id = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetContext(
            self.handle, ctypes.byref(dev_typeid), ctypes.byref(dev_id)))
        return Device(Device.devtype2str[dev_typeid.value], dev_id.value)

    @property
    def ctx(self):
        """Device context of the array. Has the same meaning as context.

        Examples
        --------
        >>> x = mx.nd.array([1, 2, 3, 4])
        >>> x.ctx
        cpu(0)
        >>> type(x.ctx)
        <class 'mxnet.context.Context'>
        >>> y = mx.nd.zeros((2,3), mx.gpu(0))
        >>> y.ctx
        gpu(0)
        """
        return self.context

    @property
    def device(self):
        """Device context of the array. Has the same meaning as context.

        Examples
        --------
        >>> x = mx.nd.array([1, 2, 3, 4])
        >>> x.device
        cpu(0)
        >>> type(x.device)
        <class 'mxnet.device.Device'>
        >>> y = mx.nd.zeros((2,3), mx.gpu(0))
        >>> y.device
        gpu(0)
        """
        return self.context

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
        return dtype_mx_to_np(mx_dtype.value)

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
        if self.dtype == bfloat16:
            return self.astype(np.float32).asnumpy()
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
        if self.size != 1:
            raise ValueError("The current array is not a scalar")
        if self.ndim == 1:
            return self.asnumpy()[0]
        else:
            return self.asnumpy()[()]

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

        Returns
        -------
        NDArray, CSRNDArray or RowSparseNDArray
            The copied array after casting to the specified type, or
            the same array if copy=False and dtype is the same as the input
            array.

        Examples
        --------
        >>> x = mx.nd.zeros((2,3), dtype='float32')
        >>> y = x.astype('int32')
        >>> y.dtype
        <type 'numpy.int32'>
        """

        if dtype is None:
            dtype = mx_real_t
        if not copy and np.dtype(dtype) == self.dtype:
            return self

        return op.cast(self, dtype=dtype)

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
                return False
            return _internal._copyto(self, out=other)
        elif isinstance(other, Device):
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
        return self.copyto(self.ctx)

    def slice_assign_scalar(self, value, begin, end, step):
        """
        Assign the scalar to a cropped subset of this NDArray. Value will broadcast to the shape of the cropped shape
        and will be cast to the same dtype of the NDArray.

        Parameters
        ----------
        value: numeric value
            Value and this NDArray should be of the same data type.
            The shape of rhs should be the same as the cropped shape of this NDArray.
        begin: tuple of begin indices
        end: tuple of end indices
        step: tuple of step lenghths

        Returns
        -------
        This NDArray.

        Examples
        --------
        >>> from mxnet import nd
        >>> x = nd.ones((2, 2, 2))
        >>> y = x.slice_assign_scalar(0, (0, 0, None), (1, 1, None), (None, None, None))
        >>> y
        [[[0. 0.]
        [1. 1.]]

        [[1. 1.]
        [1. 1.]]]
        <NDArray 2x2x2 @cpu(0)>
        >>> x
        [[[0. 0.]
        [1. 1.]]

        [[1. 1.]
        [1. 1.]]]
        <NDArray 2x2x2 @cpu(0)>

        """
        return _internal._slice_assign_scalar(self, value, begin=begin, end=end, step=step, out=self)

    def slice_assign(self, rhs, begin, end, step):
        """
        Assign the rhs to a cropped subset of this NDarray in place.
        Returns the view of this NDArray.

        Parameters
        ----------
        rhs: NDArray.
            rhs and this NDArray should be of the same data type, and on the same device.
            The shape of rhs should be the same as the cropped shape of this NDArray.
        begin: tuple of begin indices
        end: tuple of end indices
        step: tuple of step lenghths

        Returns
        -------
        This NDArray.

        Examples
        --------
        >>> x = nd.ones((2, 2, 2))
        >>> assigned = nd.zeros((1, 1, 2))
        >>> y = x.slice_assign(assigned, (0, 0, None), (1, 1, None), (None, None, None))
        >>> y
        [[[0. 0.]
        [1. 1.]]

        [[1. 1.]
        [1. 1.]]]
        <NDArray 2x2x2 @cpu(0)>
        >>> x
        [[[0. 0.]
        [1. 1.]]

        [[1. 1.]
        [1. 1.]]]
        <NDArray 2x2x2 @cpu(0)>
        """
        return _internal._slice_assign(self, rhs, begin=begin, end=end, step=step, out=self)


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

        The gradient is initialized to zeros.

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
            grad = _zeros(self.shape, stype=stype, dtype=self.dtype)
        else:
            grad = op.zeros_like(self)  # pylint: disable=undefined-variable
        grad_req = _GRAD_REQ_MAP[grad_req]
        check_call(_LIB.MXAutogradMarkVariables(
            1, ctypes.pointer(self.handle),
            ctypes.pointer(mx_uint(grad_req)),
            ctypes.pointer(grad.handle)))

    def drop_grad(self):
        """Free the memory of the marked ndarray."""
        check_call(_LIB.MXAutogradDropGrads(
            1, ctypes.pointer(self.handle)))

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
            1, c_handle_array([self]),
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
        if stype == 'csr' and len(self.shape) != 2:
            raise ValueError("To convert to a CSR, the NDArray should be 2 Dimensional. Current "
                             f"shape is {str(self.shape)}")

        return op.cast_storage(self, stype=stype)

    def to_dlpack_for_read(self):
        """Returns a reference view of NDArray that represents as DLManagedTensor until
        all previous write operations on the current array are finished.

        Returns
        -------
        PyCapsule (the pointer of DLManagedTensor)
            a reference view of NDArray that represents as DLManagedTensor.

        Examples
        --------
        >>> x = mx.nd.ones((2,3))
        >>> y = mx.nd.to_dlpack_for_read(x)
        >>> type(y)
        <class 'PyCapsule'>
        >>> z = mx.nd.from_dlpack(y)
        >>> z
        [[1. 1. 1.]
         [1. 1. 1.]]
        <NDArray 2x3 @cpu(0)>
        """
        return to_dlpack_for_read(self)

    def to_dlpack_for_write(self):
        """Returns a reference view of NDArray that represents as DLManagedTensor until
        all previous read/write operations on the current array are finished.

        Returns
        -------
        PyCapsule (the pointer of DLManagedTensor)
            a reference view of NDArray that represents as DLManagedTensor.

        Examples
        --------
        >>> x = mx.nd.ones((2,3))
        >>> w = mx.nd.to_dlpack_for_write(x)
        >>> type(w)
        <class 'PyCapsule'>
        >>> u = mx.nd.from_dlpack(w)
        >>> u += 1
        >>> x
        [[2. 2. 2.]
         [2. 2. 2.]]
        <NDArray 2x3 @cpu(0)>
        """
        return to_dlpack_for_write(self)

    def _full(self, value):
        """
        This is added as an NDArray class method in order to support polymorphism in NDArray and numpy.ndarray indexing
        """
        return _internal._full(self.shape, value=value, ctx=self.ctx, dtype=self.dtype, out=self)

    def _scatter_set_nd(self, value_nd, indices):
        """
        This is added as an NDArray class method in order to support polymorphism in NDArray and numpy.ndarray indexing
        """
        return _internal._scatter_set_nd(
            lhs=self, rhs=value_nd, indices=indices, shape=self.shape, out=self
        )

def check_boolean_array_dimension(array_shape, axis, bool_shape):
    """
    Advanced boolean indexing is implemented through the use of `nonzero`.
    Size check is necessary to make sure that the boolean array
    has exactly as many dimensions as it is supposed to work with before the conversion
    """
    for i, val in enumerate(bool_shape):
        if array_shape[axis + i] != val:
            raise IndexError('boolean index did not match indexed array along axis {};'
                             ' size is {} but corresponding boolean size is {}'
                             .format(axis + i, array_shape[axis + i], val))

def indexing_key_expand_implicit_axes(key, shape):
    """
    Make implicit axes explicit by adding ``slice(None)``
    and convert boolean array to integer array through `nonzero`.

    Examples
    --------
    >>> shape = (3, 4, 5)
    >>> indexing_key_expand_implicit_axes(np.s_[2, 1, 1], shape)
    (2, 1, 1)
    >>> indexing_key_expand_implicit_axes(np.s_[0], shape)
    (0, slice(None, None, None), slice(None, None, None))
    >>> indexing_key_expand_implicit_axes(np.s_[0, ...], shape)  # equivalent
    (0, slice(None, None, None), slice(None, None, None))
    >>> indexing_key_expand_implicit_axes(np.s_[:2, None, 0, ...], shape)
    (slice(None, 2, None), None, 0, slice(None, None, None))
    >>> bool_array = np.array([[True, False, True, False],
                               [False, True, False, True],
                               [True, False, True, False]], dtype=np.bool)
    >>> indexing_key_expand_implicit_axes(np.s_[bool_array, None, 0:2], shape)
    (array([0, 0, 1, 1, 2, 2], dtype=int64), array([0, 2, 1, 3, 0, 2], dtype=int64), None, slice(None, 2, None))
    """
    if not isinstance(key, tuple):
        key = (key,)
    # We need to loop explicitly since tuple functions like `index()` or
    # `count()` use `==` internally, which doesn't play well with fancy
    # indexing.
    ell_idx = None
    num_none = 0
    nonell_key = []

    # For 0-d boolean indices: A new axis is added,
    # but at the same time no axis is "used". So if we have True,
    # we add a new axis (a bit like with np.newaxis). If it is
    # False, we add a new axis, but this axis has 0 entries.
    # prepend is defined to handle this case.
    # prepend = _NDARRAY_NO_ZERO_DIM_BOOL_ARRAY/-1 means there is no 0-d boolean scalar
    # prepend = _NDARRAY_ZERO_DIM_BOOL_ARRAY_FALSE/0 means an zero dim must be expanded
    # prepend = _NDARRAY_ZERO_DIM_BOOL_ARRAY_TRUE/1 means a new axis must be expanded
    prepend = _NDARRAY_NO_ZERO_DIM_BOOL_ARRAY
    axis = 0
    for i, idx in enumerate(key):
        if idx is Ellipsis:
            if ell_idx is not None:
                raise IndexError(
                    'Cannot use more than one ellipsis (`...`) for indexing'
                )
            ell_idx = i
        else:
            # convert primitive type boolean value to mx.np.bool type
            # otherwise will be treated as 1/0
            if isinstance(idx, bool):
                idx = array(idx, dtype=np.bool_)
            if idx is None:
                num_none += 1
            if isinstance(idx, NDArrayBase) and idx.ndim == 0 and idx.dtype == np.bool_:
                if not idx: # array(False) has priority
                    prepend = _NDARRAY_ZERO_DIM_BOOL_ARRAY_FALSE
                else:
                    prepend = _NDARRAY_ZERO_DIM_BOOL_ARRAY_TRUE
            elif isinstance(idx, NDArrayBase) and idx.ndim == 0 and idx.dtype != np.bool_:
                # This handles ndarray of zero dim. e.g array(1)
                # while advoid converting zero dim boolean array
                # float type will be converted to int
                nonell_key.append(int(idx.item()))
                axis += 1
            elif isinstance(idx, NDArrayBase) and idx.dtype == np.bool_:
                # Necessary size check before using `nonzero`
                check_boolean_array_dimension(shape, axis, idx.shape)
                # If the whole array is false and npx.set_np() is not set_up
                # the program will throw infer shape error
                if not is_np_array():
                    raise ValueError('Cannot perform boolean indexing in legacy mode. Please activate'
                                     ' numpy semantics by calling `npx.set_np()` in the global scope'
                                     ' before calling this function.')
                # Add the arrays from the nonzero result to the index
                nonell_key.extend(idx.nonzero())
                axis += idx.ndim
            else:
                nonell_key.append(idx)
                axis += 1

    nonell_key = tuple(nonell_key)

    if ell_idx is None:
        # This handles the case of "too few" indices, e.g., `nd.zeros((2, 3))[0]`,
        # where the ellipsis is implicitly after the last entry.
        ell_idx = len(nonell_key)

    ell_ndim = len(shape) + num_none - len(nonell_key)
    expanded_key = (nonell_key[:ell_idx] +
                    (slice(None),) * ell_ndim +
                    nonell_key[ell_idx:])

    return expanded_key, prepend


def _int_to_slice(idx):
    """Return a slice that indexes the same entries as a single int."""
    if idx == -1:
        # Avoid slice(-1, 0)
        return slice(-1, None)
    else:
        return slice(idx, idx + 1)


def _shape_for_bcast(shape, target_ndim, new_axes):
    """Return shape with added axes for broadcasting in ``target_ndim`` dimensions.

    If ``shape`` is shorter than ``target_ndim``, fixed ``1`` entries are inserted
    into the returned shape, in locations indexed by ``new_axes``. The rest is
    filled from the back with ``shape`` while possible.
    """
    new_shape = [None] * target_ndim
    if len(shape) < target_ndim:
        for new_ax in new_axes:
            new_shape[new_ax] = 1

    # Replace `None` from the right with `shape` entries from the right as
    # long as possible, thereafter with 1.
    ax_s = 1
    for ax in range(1, target_ndim + 1):
        if new_shape[-ax] is None:
            try:
                new_shape[-ax] = shape[-ax_s]
                ax_s += 1
            except IndexError:
                new_shape[-ax] = 1

    return tuple(new_shape)


def _is_advanced_index(idx):
    """Return whether ``idx`` is an advanced index (array-like or integer).

    Note that in contrast to basic indexing, integers are considered advanced
    indices in the context of advanced indexing as they participate in
    broadcasting.
    """
    if isinstance(idx, (NDArray, np.ndarray, integer_types, list, tuple)):
        return True
    elif isinstance(idx, py_slice) or idx is None:
        return False
    elif isinstance(idx, range):
        return True
    else:
        raise RuntimeError('illegal index type {}'.format(type(idx)))


def get_indexing_dispatch_code(key):
    """Returns a dispatch code for calling basic or advanced indexing functions."""
    assert isinstance(key, tuple)

    for idx in key:
        if isinstance(idx, (NDArray, np.ndarray, list, tuple, range)):
            if isinstance(idx, tuple) and len(idx) == 0:
                return _NDARRAY_EMPTY_TUPLE_INDEXING
            return _NDARRAY_ADVANCED_INDEXING
        elif not (isinstance(idx, (py_slice, integer_types)) or idx is None):
            raise ValueError(
                'NDArray does not support slicing with key {} of type {}.'
                ''.format(idx, type(idx))
            )
    return _NDARRAY_BASIC_INDEXING


def _get_index_range(start, stop, length, step=1):
    """Given start, stop, step and array length, return
    absolute values of start, stop, and step for generating index range.
    The returned values have been compensated by adding length if they
    are less than zero for all the cases but slice(None, None, -1).
    Note that the returned value of stop is not necessarily >= 0, since
    absolute stop is -1 in the case of slice(None, None, -1)."""
    if step == 0:
        raise ValueError('step size cannot be zero')
    if length < 0:
        raise ValueError('array length cannot be less than zero')
    if step is None:
        step = 1
    if start is None:
        if step > 0:
            start = 0
        else:
            start = length - 1
    elif start < 0:
        start += length
        if start < 0:
            start = 0
    elif start >= length:
        start = length

    if stop is None:
        if step > 0:
            stop = length
        else:
            # this supports case such as ::-1
            # stop = -1 here refers to the element before index 0,
            # instead of the last element in the array
            stop = -1
    elif stop < 0:
        stop += length
        if stop < 0:
            stop = 0
    elif stop > length:
        stop = length

    return start, stop, step


def get_oshape_of_gather_nd_op(dshape, ishape):
    """Given data and index shapes, get the output `NDArray` shape.
    This basically implements the infer shape logic of op gather_nd."""
    assert len(dshape) > 0 and len(ishape) > 0
    oshape = list(ishape[1:])
    if ishape[0] < len(dshape):
        oshape.extend(dshape[ishape[0]:])
    return tuple(oshape)


def _get_dim_size(start, stop, step):
    """Given start, stop, and step, calculate the number of elements
    of this slice.
    """
    assert step != 0
    if stop == start:
        return 0
    if step > 0:
        assert start < stop
        dim_size = (stop - start - 1) // step + 1
    else:
        assert stop < start
        dim_size = (start - stop - 1) // (-step) + 1
    return dim_size


def _get_slice_len(slc, seq_length):
    """Given a python slice object and the length of the sequence, calculate the number of elements
     in the slice.

    Parameters
    ----------
    slc : py_slice
        The slice object
    seq_length : int
        The length of the object you are going to apply the slice on

    Returns
    -------
    ret : int
        Total number of elements in the slice
    """
    start, stop, step = slc.indices(seq_length)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def _get_broadcast_shape(shape1, shape2):
    """Given two shapes that are not identical, find the shape
    that both input shapes can broadcast to."""
    if shape1 == shape2:
        return shape1

    length1 = len(shape1)
    length2 = len(shape2)
    if length1 > length2:
        shape = list(shape1)
    else:
        shape = list(shape2)
    i = max(length1, length2) - 1
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a != 1 and b != 1 and a != b:
            raise ValueError(f'shape1={shape1} is not broadcastable to shape2={shape2}')
        shape[i] = b if a == 1 else a
        i -= 1
    return tuple(shape)


def _broadcast_shapes(seq):
    """Return the broadcast shape of all advanced indices in ``seq``.

    All entries are assumed to have a ``shape`` property.
    """
    return reduce(_get_broadcast_shape, [x.shape for x in seq], ())


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
        Defaults to the current default context (``mxnet.context.current_context()``).
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
        ctx = current_device()
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

    if source_array.shape == ():
        # In this case we can't assign, so we need to go through an auxiliary array
        arr = empty((1,), ctx, dtype)
        arr[:] = source_array
        return arr.reshape(())
    elif source_array.size == 0:
        return empty(source_array.shape, ctx, dtype)
    else:
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
    source : int or sequence of int
        Original position of the axes to move. Can be negative but must be unique.
    destination : int or sequence of int
        Destination position for each of the original axes. Can be negative but must be unique.

    Returns
    -------
    result : mx.nd.array
        Array with moved axes.

    Examples
    --------
    >>> X = mx.nd.array([[1, 2, 3], [4, 5, 6]])
    >>> mx.nd.moveaxis(X, 0, 1).shape
    (3L, 2L)

    >>> X = mx.nd.zeros((3, 4, 5))
    >>> mx.nd.moveaxis(X, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    """
    try:
        source = np.core.numeric.normalize_axis_tuple(source, tensor.ndim)
    except IndexError:
        raise ValueError('Source should verify 0 <= source < tensor.ndim'
                         f'Got {source}')
    try:
        destination = np.core.numeric.normalize_axis_tuple(destination, tensor.ndim)
    except IndexError:
        raise ValueError(f'Destination should verify 0 <= destination < tensor.ndim ({tensor.ndim}).',
                         f'Got {destination}')

    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    order = [n for n in range(tensor.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return op.transpose(tensor, order)


# pylint: disable= no-member, protected-access, too-many-arguments, redefined-outer-name
def arange(start, stop=None, step=1.0, repeat=1, infer_range=None, ctx=None, dtype=mx_real_t):
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
    infer_range : boolean, optional
        Infer the stop position from the start, step, repeat, and output tensor size.
        Deprecated. Only False is supported.
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
    if infer_range is not None:
        warnings.warn('`infer_range` argument has been deprecated',
                      DeprecationWarning)
    if ctx is None:
        ctx = current_device()
    return _internal._arange(start=start, stop=stop, step=step, repeat=repeat,
                             infer_range=False, dtype=dtype, ctx=str(ctx))
# pylint: enable= no-member, protected-access, too-many-arguments


# pylint: disable= no-member, protected-access, too-many-arguments
def linspace(start, stop, num, endpoint=True, ctx=None, dtype=mx_real_t):
    """Return evenly spaced numbers within a specified interval.

    Values are generated within the half-open interval [`start`, `stop`) or
    closed interval [start, stop] depending on whether `endpoint` is True or
    False. The function is similar to `numpy.linspace`, but returns an `NDArray`.

    Parameters
    ----------
    start : number
        Start of interval.
    stop : number
        End of interval, unless endpoint is set to False.  In that case,
        the sequence consists of all but the last of `num + 1` evenly spaced
        samples, so that stop is excluded. Note that the step size changes
        when endpoint is False.
    num : number
        Number of samples to generate. Must be non-negative.
    endpoint : bool
        If True, stop is the last sample. Otherwise, it is not included.
        The default is True.
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
    >>> mx.nd.linspace(2.0, 3.0, 5).asnumpy()
    array([ 2.,  2.25.,  2.5,  2.75,  3.], dtype=float32)
    >>> mx.nd.linspace(2.0, 3.0, 5, endpoint=False).asnumpy()
    array([ 2.,  2.2.,  2.4,  2.6,  2.8], dtype=float32)
    """
    if ctx is None:
        ctx = current_device()
    return _internal._linspace(start=start, stop=stop, num=num,
                               endpoint=endpoint, dtype=dtype, ctx=str(ctx))
# pylint: disable= no-member, protected-access, too-many-arguments


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
        raise TypeError(f'type {str(type(rhs))} not supported')
#pylint: enable= too-many-arguments, no-member, protected-access


def add(lhs, rhs):
    """Returns element-wise sum of the input arrays with broadcasting.

    Equivalent to ``lhs + rhs``, ``mx.nd.broadcast_add(lhs, rhs)`` and
    ``mx.nd.broadcast_plus(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape

    Parameters
    ----------
    lhs : scalar or mxnet.ndarray.array
        First array to be added.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be subtracted.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be multiplied.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array in division.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array in modulo.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
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
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
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

def logical_and(lhs, rhs):
    """Returns the result of element-wise **logical and** comparison
    operation with broadcasting.

    For each element in input arrays, return 1(true) if lhs elements and rhs elements
    are true, otherwise return 0(false).

    Equivalent to ``lhs and rhs`` and ``mx.nd.broadcast_logical_and(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or mxnet.ndarray.array
        First input of the function.
    rhs : scalar or mxnet.ndarray.array
         Second input of the function. If ``lhs.shape != rhs.shape``, they must be
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
    >>> mx.nd.logical_and(x, 1).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.logical_and(x, y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.logical_and(z, y).asnumpy()
    array([[ 0.,  0.],
           [ 0.,  1.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_logical_and,
        lambda x, y: 1 if x and y else 0,
        _internal._logical_and_scalar,
        None)
    # pylint: enable= no-member, protected-access

def logical_or(lhs, rhs):
    """Returns the result of element-wise **logical or** comparison
    operation with broadcasting.

    For each element in input arrays, return 1(true) if lhs elements or rhs elements
    are true, otherwise return 0(false).

    Equivalent to ``lhs or rhs`` and ``mx.nd.broadcast_logical_or(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or mxnet.ndarray.array
        First input of the function.
    rhs : scalar or mxnet.ndarray.array
         Second input of the function. If ``lhs.shape != rhs.shape``, they must be
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
    >>> mx.nd.logical_or(x, 1).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.logical_or(x, y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.logical_or(z, y).asnumpy()
    array([[ 0.,  1.],
           [ 1.,  1.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_logical_or,
        lambda x, y: 1 if x or y else 0,
        _internal._logical_or_scalar,
        None)
    # pylint: enable= no-member, protected-access

def logical_xor(lhs, rhs):
    """Returns the result of element-wise **logical xor** comparison
    operation with broadcasting.

    For each element in input arrays, return 1(true) if lhs elements or rhs elements
    are true, otherwise return 0(false).

    Equivalent to ``bool(lhs) ^ bool(rhs)`` and ``mx.nd.broadcast_logical_xor(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or mxnet.ndarray.array
        First input of the function.
    rhs : scalar or mxnet.ndarray.array
         Second input of the function. If ``lhs.shape != rhs.shape``, they must be
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
    >>> mx.nd.logical_xor(x, y).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]], dtype=float32)
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        op.broadcast_logical_xor,
        lambda x, y: 1 if bool(x) ^ bool(y) else 0,
        _internal._logical_xor_scalar,
        None)
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
    # Unsupported in deferred compute mode due to use of inplace operations.
    from .._deferred_compute import is_deferred_compute  # pylint: disable=wrong-import-position
    assert not is_deferred_compute(), 'nd.concatenate is deprecated and ' \
        'unsupported in deferred compute mode. Use nd.concat instead.'

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
    ret = empty(ret_shape, ctx=arrays[0].ctx, dtype=dtype)

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
        ctx = current_device()
    dtype = mx_real_t if dtype is None else dtype
    # pylint: disable= no-member, protected-access
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, **kwargs)
    # pylint: enable= no-member, protected-access

def eye(N, M=0, k=0, ctx=None, dtype=None, **kwargs):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N: int
        Number of rows in the output.
    M: int, optional
        Number of columns in the output. If 0, defaults to N.
    k: int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal,
        and a negative value to a lower diagonal.
    ctx: Context, optional
        An optional device context (default is the current default context)
    dtype: str or numpy.dtype, optional
        An optional value type (default is `float32`)

    Returns
    -------
    NDArray
        A created array

    Examples
    --------
    >>> mx.nd.eye(2)
    [[ 1.  0.]
     [ 0.  1.]]
    <NDArray 2x2 @cpu(0)>
    >>> mx.nd.eye(2, 3, 1)
    [[ 0.  1.  0.]
     [ 0.  0.  1.]]
    <NDArray 2x3 @cpu(0)>
    """
    # pylint: disable= unused-argument
    if ctx is None:
        ctx = current_device()
    dtype = mx_real_t if dtype is None else dtype
    # pylint: disable= no-member, protected-access
    return _internal._eye(N=N, M=M, k=k, ctx=ctx, dtype=dtype, **kwargs)
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
        ctx = current_device()
    if dtype is None:
        dtype = mx_real_t
    return NDArray(handle=_new_alloc_handle(shape, ctx, False, dtype))


# pylint: disable= redefined-builtin
def histogram(a, bins=10, range=None):
    """Compute the histogram of the input data.

    Parameters
    ----------
    a : NDArray
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the
        given range (10, by default). If bins is a sequence, it defines the bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()).
        Values outside the range are ignored. The first element of the range must be less than or
        equal to the second. range affects the automatic bin computation as well, the range will
        be equally divided by the number of bins.

    Returns
    -------
    NDArray
        A created array.

    """

    # pylint: disable= no-member, protected-access
    if isinstance(bins, NDArray):
        return _internal._histogram(data=a, bins=bins)
    elif isinstance(bins, integer_types):
        if range is None:
            warnings.warn("range is not specified, using numpy's result "
                          "to ensure consistency with numpy")
            res, bin_bounds = np.histogram(a.asnumpy(), bins=bins)
            return array(res), array(bin_bounds)
        return _internal._histogram(data=a, bin_cnt=bins, range=range)
    raise ValueError("bins argument should be either an integer or an NDArray")
    # pylint: enable= no-member, protected-access, redefined-builtin

def split_v2(ary, indices_or_sections, axis=0, squeeze_axis=False):
    """Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : NDArray
        Array to be divided into sub-arrays.
    indices_or_sections : int or tuple of ints
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.
        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in
        - ary[:2]
        - ary[2:3]
        - ary[3:]
        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.
    squeeze_axis: boolean, optional
        Whether to squeeze the axis of sub-arrays or not, only useful when size
        of the sub-arrays are 1 on the `axis`. Default is False.

    Returns
    -------
    NDArray
        A created array.

    """
    indices = []
    axis_size = ary.shape[axis]
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if axis_size % sections:
            raise ValueError('array split does not result in an equal division')
        section_size = int(axis_size / sections)
        indices = [i * section_size for i in range(sections)]
    elif isinstance(indices_or_sections, tuple):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    return _internal._split_v2(ary, indices, axis, squeeze_axis)

from_dlpack = ndarray_from_dlpack(NDArray)
from_dlpack_doc = """Returns a NDArray backed by a dlpack tensor.

    Parameters
    ----------
    dlpack: PyCapsule (the pointer of DLManagedTensor)
        input data

    Returns
    -------
    NDArray
        a NDArray backed by a dlpack tensor

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.to_dlpack_for_read(x)
    >>> type(y)
    <class 'PyCapsule'>
    >>> z = mx.nd.from_dlpack(y)
    >>> type(z)
    <class 'mxnet.ndarray.ndarray.NDArray'>
    >>> z
    [[ 1.  1.  1.]
     [ 1.  1.  1.]]
    <NDArray 2x3 @cpu(0)>

    >>> w = mx.nd.to_dlpack_for_write(x)
    >>> type(w)
    <class 'PyCapsule'>
    >>> u = mx.nd.from_dlpack(w)
    >>> u += 1
    >>> x
    [[2. 2. 2.]
     [2. 2. 2.]]
    <NDArray 2x3 @cpu(0)>
    """
from_dlpack.__doc__ = from_dlpack_doc

from_numpy = ndarray_from_numpy(NDArray, array)
from_numpy_doc = """Returns an MXNet's NDArray backed by numpy's ndarray.
    When `zero_copy` is set to be true,
    this API consumes numpy's ndarray and produces MXNet's ndarray
    without having to copy the content. In this case, we disallow
    users to modify the given numpy ndarray, and it is suggested
    not to read the numpy ndarray as well for internal correctness.

    Parameters
    ----------
    ndarray: NDArray
        input data
    zero_copy: bool
        Whether we use DLPack's zero-copy conversion to convert to MXNet's NDArray.
        This is only available for c-contiguous arrays, i.e. array.flags[C_CONTIGUOUS] == True.

    Returns
    -------
    NDArray
        a NDArray backed by a dlpack tensor
"""
from_numpy.__doc__ = from_numpy_doc


to_dlpack_for_read = ndarray_to_dlpack_for_read()
to_dlpack_for_read_doc = """Returns a reference view of NDArray that represents as DLManagedTensor until
all previous write operations on the current array are finished.

Parameters
----------
data: NDArray
    input data.

Returns
-------
PyCapsule (the pointer of DLManagedTensor)
    a reference view of NDArray that represents as DLManagedTensor.

Examples
--------
>>> x = mx.nd.ones((2,3))
>>> y = mx.nd.to_dlpack_for_read(x)
>>> type(y)
<class 'PyCapsule'>
>>> z = mx.nd.from_dlpack(y)
>>> z
[[1. 1. 1.]
 [1. 1. 1.]]
<NDArray 2x3 @cpu(0)>
"""
to_dlpack_for_read.__doc__ = to_dlpack_for_read_doc

to_dlpack_for_write = ndarray_to_dlpack_for_write()
to_dlpack_for_write_doc = """Returns a reference view of NDArray that represents as
DLManagedTensor until all previous read/write operations on the current array are finished.

Parameters
----------
data: NDArray
    input data.

Returns
-------
PyCapsule : the pointer of DLManagedTensor
    a reference view of NDArray that represents as DLManagedTensor.

Examples
--------
>>> x = mx.nd.ones((2,3))
>>> w = mx.nd.to_dlpack_for_write(x)
>>> type(w)
<class 'PyCapsule'>
>>> u = mx.nd.from_dlpack(w)
>>> u += 1
>>> x
[[2. 2. 2.]
 [2. 2. 2.]]
<NDArray 2x3 @cpu(0)>
"""
to_dlpack_for_write.__doc__ = to_dlpack_for_write_doc
