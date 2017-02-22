# coding: utf-8
# pylint: disable= too-many-lines, redefined-builtin, protected-access
# pylint: disable=import-error, no-name-in-module, undefined-variable
"""NDArray API of mxnet."""
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

import operator
import numpy as np
from .base import _LIB, string_types, numeric_types
from .base import c_array, py_str, c_str, mx_real_t
from .base import mx_uint, NDArrayHandle, check_call
from .base import ctypes2buffer
from .context import Context
from . import _ndarray_internal as _internal

# Use different verison of SymbolBase
# When possible, use cython to speedup part of computation.
try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from ._ctypes.ndarray import NDArrayBase, _init_ndarray_module
    elif _sys.version_info >= (3, 0):
        from ._cy3.ndarray import NDArrayBase, _init_ndarray_module
    else:
        from ._cy2.ndarray import NDArrayBase, _init_ndarray_module
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from ._ctypes.ndarray import NDArrayBase, _init_ndarray_module


# pylint: disable= no-member
_DTYPE_NP_TO_MX = {
    np.float32 : 0,
    np.float64 : 1,
    np.float16 : 2,
    np.uint8   : 3,
    np.int32   : 4
}

_DTYPE_MX_TO_NP = {
    0 : np.float32,
    1 : np.float64,
    2 : np.float16,
    3 : np.uint8,
    4 : np.int32
}
# pylint: enable= no-member

def _new_empty_handle():
    """Return a new empty handle.

    Empty handle can be used to hold result

    Returns
    -------
    a new empty ndarray handle
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateNone(ctypes.byref(hdl)))
    return hdl

def _new_alloc_handle(shape, ctx, delay_alloc, dtype=mx_real_t):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results

    Returns
    -------
    a new empty ndarray handle
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
    """Wait all async operation to finish in MXNet

    This function is used for benchmark only
    """
    check_call(_LIB.MXNDArrayWaitAll())

class NDArray(NDArrayBase):
    """NDArray object in mxnet.

    NDArray is basic ndarray/Tensor like data structure in mxnet.
    """
    __slots__ = []
    # pylint: disable= no-member, undefined-variable
    def __repr__(self):
        shape_info = 'x'.join(['%d' % x for x in self.shape])
        return '<%s %s @%s>' % (self.__class__.__name__,
                                shape_info, self.context)

    def __add__(self, other):
        return add(self, other)

    def __iadd__(self, other):
        if not self.writable:
            raise ValueError('trying to add to a readonly NDArray')
        if isinstance(other, NDArray):
            return broadcast_add(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._plus_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return subtract(self, other)

    def __isub__(self, other):
        if not self.writable:
            raise ValueError('trying to subtract from a readonly NDArray')
        if isinstance(other, NDArray):
            return broadcast_sub(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._minus_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __neg__(self):
        return _internal._mul_scalar(self, -1.0)

    def __imul__(self, other):
        if not self.writable:
            raise ValueError('trying to multiply to a readonly NDArray')
        if isinstance(other, NDArray):
            return broadcast_mul(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._mul_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return divide(self, other)

    def __rdiv__(self, other):
        return divide(other, self)

    def __idiv__(self, other):
        if not self.writable:
            raise ValueError('trying to divide from a readonly NDArray')
        if isinstance(other, NDArray):
            return broadcast_div(self, other, out=self)
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

    def __pow__(self, other):
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __eq__(self, other):
        return equal(self, other)

    def __ne__(self, other):
        return not_equal(self, other)

    def __gt__(self, other):
        return greater(self, other)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __lt__(self, other):
        return lesser(self, other)

    def __le__(self, other):
        return lesser_equal(self, other)

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

    def __setitem__(self, in_slice, value):
        """Set ndarray value.

        `value` can be a scalar, an `NDArray` or numpy array of compatible shape.
        The following modes are supported:

        - `array[:] = value`: set all the contents
        - `array[i] = value`: set the i-th slice. If the array is of dimension
          `(d1, d2, d3)`, it sets value of a slice of shape `(1, d2, d3)`.
        - `array[i:j] = value`: similarly, if the array is of dimension
          `(d1, d2, d3)`, it sets value of a slice of shape `(j-i, d2, d3)`.

        Fully-dimensional indexing is also supported. For example, if array is
        of shape `(d1, d2, d3)`, one can do

        - `array[:, :, :] = value`: achieving the same effect of `array[:] = value`
        - `array[:, i, j:k] = value`: each index could be a python slice or an int.
        """
        # pylint: disable=too-many-branches
        if not self.writable:
            raise ValueError('trying to assign to a readonly NDArray')
        if isinstance(in_slice, int):
            sliced_arr = self._at(in_slice)
            sliced_arr[:] = value
            return
        if isinstance(in_slice, py_slice):
            if in_slice.step is not None:
                raise ValueError('NDArray only support continuous slicing on axis 0')
            if in_slice.start is not None or in_slice.stop is not None:
                sliced_arr = self._slice(in_slice.start, in_slice.stop)
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
                raise TypeError('type %s not supported' % str(type(value)))
        if isinstance(in_slice, tuple):
            # multi-dimension indexing
            my_shape = self.shape
            assert len(in_slice) == len(my_shape)
            for slice_i in in_slice:
                assert isinstance(slice_i, (py_slice, int))
            begin = [0 for _ in my_shape]
            end = [x for x in my_shape]
            for i, slice_i in enumerate(in_slice):
                if isinstance(slice_i, int):
                    assert slice_i < my_shape[i]
                    begin[i] = slice_i
                    end[i] = slice_i + 1
                if isinstance(slice_i, py_slice):
                    # only support continuous slicing
                    assert slice_i.step is None
                    begin[i] = slice_i.start or 0
                    end[i] = slice_i.stop or my_shape[i]
                    assert begin[i] < end[i]
                    assert end[i] <= my_shape[i]
            begin = tuple(begin)
            end = tuple(end)
            if isinstance(value, NDArray):
                value = value.as_in_context(self.context)
                _internal._crop_assign(self, value, out=self,
                                       begin=begin, end=end)
            elif isinstance(value, numeric_types):
                _internal._crop_assign_scalar(self, out=self,
                                              begin=begin, end=end,
                                              scalar=value)
            elif isinstance(value, (np.ndarray, np.generic)):
                value = array(value, ctx=self.context)
                _internal._crop_assign(self, value, out=self,
                                       begin=begin, end=end)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        # pylint: enable=too-many-branches

    def __getitem__(self, in_slice):
        """Get ndarray"""
        if isinstance(in_slice, int):
            return self._at(in_slice)
        if not isinstance(in_slice, py_slice) or in_slice.step is not None:
            raise ValueError('NDArray only support continuous slicing on axis 0')
        if in_slice.start is not None or in_slice.stop is not None:
            return self._slice(in_slice.start, in_slice.stop)
        else:
            return self

    def _sync_copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(array)))
        source_array = np.ascontiguousarray(source_array, dtype=self.dtype)
        if source_array.shape != self.shape:
            raise ValueError('Shape inconsistant: expected %s vs got %s'%(
                str(self.shape), str(source_array.shape)))
        check_call(_LIB.MXNDArraySyncCopyFromCPU(
            self.handle,
            source_array.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(source_array.size)))

    def _slice(self, start, stop):
        """Return a sliced NDArray that shares memory with current one.

        Parameters
        ----------
        start : int
            Starting index of slice.
        stop : int
            Finishing index of slice.
        """
        handle = NDArrayHandle()
        start = mx_uint(start) if start else mx_uint(0)
        stop = mx_uint(stop) if stop else mx_uint(self.shape[0])
        check_call(_LIB.MXNDArraySlice(
            self.handle, start, stop, ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    def _at(self, idx):
        """Return a sub NDArray that shares memory with current one.

        Parameters
        ----------
        idx : int
            index of sub array.
        """
        handle = NDArrayHandle()
        idx = mx_uint(idx)
        check_call(_LIB.MXNDArrayAt(
            self.handle, idx, ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    def reshape(self, new_shape):
        """Return a reshaped NDArray that shares memory with current one.

        Parameters
        ----------
        new_shape : iterable of int
            new shape of NDArray
        """
        handle = NDArrayHandle()
        check_call(_LIB.MXNDArrayReshape(self.handle,
                                         len(new_shape),
                                         c_array(ctypes.c_int, new_shape),
                                         ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    # pylint: disable= undefined-variable
    def broadcast_to(self, shape):
        """ Broadcasting the current NDArray into the given shape. The semantics is
        the same with `numpy`'s broadcasting

        Parameters
        ---------
        shape : the shape to broadcast
            the broadcast shape
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
            return broadcast_to(self.reshape(cur_shape), shape=shape)
        else:
            return broadcast_to(self, shape=tuple(shape))
    # pylint: enable= undefined-variable

    def wait_to_read(self):
        """Block until all pending writes operations on current NDArray are finished.

        This function will return when all the pending writes to the current
        NDArray finishes. There can still be pending read going on when the
        function returns.
        """
        check_call(_LIB.MXNDArrayWaitToRead(self.handle))

    @property
    def shape(self):
        """Get shape of current NDArray.

        Returns
        -------
        a tuple representing shape of current ndarray
        """
        ndim = mx_uint()
        pdata = ctypes.POINTER(mx_uint)()
        check_call(_LIB.MXNDArrayGetShape(
            self.handle, ctypes.byref(ndim), ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])

    @property
    def size(self):
        """Get size of current NDArray.

        Returns
        -------
        an int representing size of current ndarray
        """
        return np.prod(self.shape)

    @property
    def context(self):
        """Get context of current NDArray.

        Returns
        -------
        context : mxnet.Context
            The context of current NDArray.
        """
        dev_typeid = ctypes.c_int()
        dev_id = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetContext(
            self.handle, ctypes.byref(dev_typeid), ctypes.byref(dev_id)))
        return Context(Context.devtype2str[dev_typeid.value], dev_id.value)

    @property
    def dtype(self):
        """Get data type of current NDArray.

        Returns
        -------
        an numpy.dtype object representing type of current ndarray
        """
        mx_dtype = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetDType(
            self.handle, ctypes.byref(mx_dtype)))
        return _DTYPE_MX_TO_NP[mx_dtype.value]

    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        """Get transpose of current NDArray"""
        if len(self.shape) != 2:
            raise ValueError('Only 2D matrix is allowed to be transposed')
        return transpose(self)
    # pylint: enable= invalid-name, undefined-variable

    def asnumpy(self):
        """Return a copied numpy array of current array.

        Returns
        -------
        array : numpy.ndarray
            A copy of array content.
        """
        data = np.empty(self.shape, dtype=self.dtype)
        check_call(_LIB.MXNDArraySyncCopyToCPU(
            self.handle,
            data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(data.size)))
        return data

    def asscalar(self):
        """Return a CPU scalar(float) of current ndarray.

        This ndarray must have shape (1,)

        Returns
        -------
        scalar : np.float
            The scalar representation of the ndarray.
        """
        if self.shape != (1,):
            raise ValueError("The current array is not a scalar")
        return self.asnumpy()[0]

    def astype(self, dtype):
        """Return a copied numpy array of current array with specified type.

        Parameters
        ----------
        dtype : str or numpy.dtype
            Desired type of result array.

        Returns
        -------
        array : numpy.ndarray
            A copy of array content.
        """
        res = empty(self.shape, ctx=self.context, dtype=dtype)
        self.copyto(res)
        return res

    def copyto(self, other):
        """Copy the content of current array to other.

        When other is NDArray, the content is copied over.
        When other is a Context, a new NDArray in the context
        will be created as target

        Parameters
        ----------
        other : NDArray or Context
            Target NDArray or context we want to copy data to.

        Returns
        -------
        dst : NDArray
            The copy target NDArray
        """
        if isinstance(other, NDArray):
            if other.handle is self.handle:
                warnings.warn('copy an array to itself, is it intended?',
                              RuntimeWarning)
                return
            return _internal._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = NDArray(_new_alloc_handle(self.shape, other, True, self.dtype))
            return _internal._copyto(self, out=hret)
        else:
            raise TypeError('copyto do not support type ' + str(type(other)))

    def copy(self):
        """Make a copy of the current ndarray on the same context

        Return
        ------
        cpy : NDArray
            The copy
        """
        return self.copyto(self.context)

    # pylint: enable= no-member

    def as_in_context(self, context):
        """Return an `NDArray` that lives in the target context. If the array
        is already in that context, `self` is returned. Otherwise, a copy is
        made.

        Parameters
        ----------
        context : Context
            The target context we want the return value to live in.

        Returns
        -------
        A copy or `self` as an `NDArray` that lives in the target context.
        """
        if self.context == context:
            return self
        return self.copyto(context)


_init_ndarray_module(NDArray, "mxnet")


def onehot_encode(indices, out):
    """One hot encoding indices into matrix out.

    Parameters
    ----------
    indices: NDArray
        An NDArray containing indices of the categorical features.

    out: NDArray
        The result holder of the encoding.

    Returns
    -------
    out: Array
        Same as out.
    """
    # pylint: disable= no-member, protected-access
    return _internal._onehot_encode(indices, out, out=out)
    # pylint: enable= no-member, protected-access


def empty(shape, ctx=None, dtype=None):
    """Create an empty uninitialized new NDArray, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NDArray.

    ctx : Context, optional
        The context of the NDArray, default to current default context.

    dtype : str or numpy.dtype, optional
        The value type of the NDArray, default to np.float32

    Returns
    -------
    out: Array
        The created NDArray.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    return NDArray(handle=_new_alloc_handle(shape, ctx, False, dtype))

#pylint: disable= too-many-arguments, no-member, protected-access
def _ufunc_helper(lhs, rhs, fn_array, fn_scalar, lfn_scalar, rfn_scalar=None):
    """ Helper function for element-wise operation
    The function will perform numpy-like broadcasting if needed and call different functions

    Parameters
    ----------
    lhs : NDArray or numeric value
        left hande side operand

    rhs : NDArray or numeric value
        right hand side operand

    fn_array : function
        function to be called if both lhs and rhs are of NDArray type

    fn_scalar : function
        function to be called if both lhs and rhs are numeric values

    lfn_scalar : function
        function to be called if lhs is NDArray while rhs is numeric value

    rfn_scalar : function
        function to be called if lhs is numeric value while rhs is NDArray;
        if none is provided, then the function is commutative, so rfn_scalar is equal to lfn_scalar

    Returns
    -------
    out: NDArray
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
    """ Perform element-wise addition

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_add,
        operator.add,
        _internal._plus_scalar,
        None)
    # pylint: enable= no-member, protected-access

def subtract(lhs, rhs):
    """ Perform element-wise subtract

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_sub,
        operator.sub,
        _internal._minus_scalar,
        _internal._rminus_scalar)
    # pylint: enable= no-member, protected-access

def multiply(lhs, rhs):
    """ Perform element-wise multiplication

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_mul,
        operator.mul,
        _internal._mul_scalar,
        None)
    # pylint: enable= no-member, protected-access

def divide(lhs, rhs):
    """ Perform element-wise divide

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_div,
        operator.truediv,
        _internal._div_scalar,
        _internal._rdiv_scalar)
    # pylint: enable= no-member, protected-access

def power(lhs, rhs):
    """ Perform power operator

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_power,
        operator.pow,
        _internal._power_scalar,
        _internal._rpower_scalar)
    # pylint: enable= no-member, protected-access

def maximum(lhs, rhs):
    """ Perform maximum operator

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_maximum,
        lambda x, y: x if x > y else y,
        _internal._maximum_scalar,
        None)
    # pylint: enable= no-member, protected-access

def minimum(lhs, rhs):
    """ Perform minimum operator

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_minimum,
        lambda x, y: x if x < y else y,
        _internal._minimum_scalar,
        None)
    # pylint: enable= no-member, protected-access

def equal(lhs, rhs):
    """Return (lhs == rhs) element-wise.

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_equal,
        lambda x, y: 1 if x == y else 0,
        _internal._equal_scalar,
        None)
    # pylint: enable= no-member, protected-access

def not_equal(lhs, rhs):
    """Return (lhs != rhs) element-wise.

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_not_equal,
        lambda x, y: 1 if x != y else 0,
        _internal._not_equal_scalar,
        None)
    # pylint: enable= no-member, protected-access

def greater(lhs, rhs):
    """Return (lhs > rhs) element-wise.

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_greater,
        lambda x, y: 1 if x > y else 0,
        _internal._greater_scalar,
        _internal._lesser_scalar)
    # pylint: enable= no-member, protected-access

def greater_equal(lhs, rhs):
    """Return (lhs >= rhs) element-wise.

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_greater_equal,
        lambda x, y: 1 if x >= y else 0,
        _internal._greater_equal_scalar,
        _internal._lesser_equal_scalar)
    # pylint: enable= no-member, protected-access

def lesser(lhs, rhs):
    """Return (lhs < rhs) element-wise.

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_lesser,
        lambda x, y: 1 if x < y else 0,
        _internal._lesser_scalar,
        _internal._greater_scalar)
    # pylint: enable= no-member, protected-access


def lesser_equal(lhs, rhs):
    """Return (lhs <= rhs) element-wise.

    Parameters
    ----------
    lhs : Array or float value
        left hand side operand

    rhs : Array of float value
        right hand side operand

    Returns
    -------
    out: Array
        result array
    """
    # pylint: disable= no-member, protected-access
    return _ufunc_helper(
        lhs,
        rhs,
        broadcast_lesser_equal,
        lambda x, y: 1 if x <= y else 0,
        _internal._lesser_equal_scalar,
        _internal._greater_equal_scalar)
    # pylint: enable= no-member, protected-access

def true_divide(lhs, rhs):
    """ Same as numpy's true_divide. It adjusts the output type to present the best answer,
    regardless of input types.
    """
    return divide(lhs, rhs)

def negative(arr):
    """ Return the negation of array values """
    return multiply(arr, -1.0)

def zeros(shape, ctx=None, dtype=None):
    """Create a new NDArray filled with 0, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NDArray.
    ctx : Context, optional.
        The context of the NDArray, default to current default context.
    dtype : str or numpy.dtype, optional
        The value type of the NDArray, default to np.float32

    Returns
    -------
    out: Array
        The created NDArray.
    """
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    # pylint: disable= no-member, protected-access
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype)
    # pylint: enable= no-member, protected-access

def ones(shape, ctx=None, dtype=None):
    """Create a new NDArray filled with 1, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NDArray.
    ctx : Context, optional
        The context of the NDArray, default to current default context.
    dtype : str or numpy.dtype, optional
        The value type of the NDArray, default to np.float32

    Returns
    -------
    out: Array
        The created NDArray.
    """
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    # pylint: disable= no-member, protected-access
    return _internal._ones(shape=shape, ctx=ctx, dtype=dtype)
    # pylint: enable= no-member, protected-access

def full(shape, val, ctx=None, dtype=None):
    """Create a new NDArray filled with given value, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NDArray.
    val : float or int
        value to be filled with.
    ctx : Context, optional
        The context of the NDArray, default to current default context.
    dtype : str or numpy.dtype, optional
        The value type of the NDArray, default to np.float32

    Returns
    -------
    out: Array
        The created NDArray.
    """
    if dtype is None:
        dtype = mx_real_t
    arr = empty(shape, ctx, dtype)
    arr[:] = val
    return arr

def array(source_array, ctx=None, dtype=None):
    """Create a new NDArray that copies content from source_array.

    Parameters
    ----------
    source_array : array_like
        Source data to create NDArray from.
    ctx : Context, optional
        The context of the NDArray, default to current default context.
    dtype : str or numpy.dtype, optional
        The value type of the NDArray, default to np.float32

    Returns
    -------
    out: Array
        The created NDArray.
    """
    if dtype is None:
        dtype = mx_real_t
    if not isinstance(source_array, np.ndarray):
        try:
            source_array = np.array(source_array, dtype=dtype)
        except:
            raise TypeError('source_array must be array like object')
    arr = empty(source_array.shape, ctx, dtype)
    arr[:] = source_array
    return arr

def concatenate(arrays, axis=0, always_copy=True):
    """Concatenate a list of NDArrays along the first dimension.

    Parameters
    ----------
    arrays : list of NDArray
        Arrays to be concatenate. They must have identical shape except
        the first dimension. They also must have the same data type.
    axis : int
        The axis along which to concatenate.
    always_copy : bool
        Default `True`. When not `True`, if the arrays only contain one
        `NDArray`, that element will be returned directly, avoid copying.

    Returns
    -------
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

# pylint: disable= no-member, protected-access, too-many-arguments
def arange(start, stop=None, step=1.0, repeat=1, ctx=None, dtype=None):
    """Simlar function in the MXNet ndarray as numpy.arange
        See Also https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html.

    Parameters
    ----------
    start : number, optional
        Start of interval. The interval includes this value. The default start value is 0.
    stop : number, optional
        End of interval. The interval does not include this value.
    step : number, optional
        Spacing between values
    repeat : number, optional
        "The repeating time of all elements.
        E.g repeat=3, the element a will be repeated three times --> a, a, a.
    ctx : Context, optional
        The context of the NDArray, default to current default context.
    dtype : str or numpy.dtype, optional
        The value type of the NDArray, default to np.float32

    Returns
    -------
    out : NDArray
        The created NDArray
    """
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    return _internal._arange(start=start, stop=stop, step=step, repeat=repeat,
                             dtype=dtype, ctx=str(ctx))
# pylint: enable= no-member, protected-access, too-many-arguments


def load(fname):
    """Load ndarray from binary file.

    You can also use pickle to do the job if you only work on python.
    The advantage of load/save is the file is language agnostic.
    This means the file saved using save can be loaded by other language binding of mxnet.
    You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)

    Parameters
    ----------
    fname : str
        The name of the file.Can be S3 or HDFS address (remember built with S3 support).
        Example of fname:

        - `s3://my-bucket/path/my-s3-ndarray`
        - `hdfs://my-bucket/path/my-hdfs-ndarray`
        - `/path-to/my-local-ndarray`

    Returns
    -------
    out : list of NDArray or dict of str to NDArray
        List of NDArray or dict of str->NDArray, depending on what was saved.
    """
    if not isinstance(fname, string_types):
        raise TypeError('fname need to be string')
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
        return [NDArray(NDArrayHandle(handles[i])) for i in range(out_size.value)]
    else:
        assert out_name_size.value == out_size.value
        return dict(
            (py_str(names[i]), NDArray(NDArrayHandle(handles[i]))) for i in range(out_size.value))


def save(fname, data):
    """Save list of NDArray or dict of str->NDArray to binary file.

    You can also use pickle to do the job if you only work on python.
    The advantage of load/save is the file is language agnostic.
    This means the file saved using save can be loaded by other language binding of mxnet.
    You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)

    Parameters
    ----------
    fname : str
        The name of the file.Can be S3 or HDFS address (remember built with S3 support).
        Example of fname:

        - `s3://my-bucket/path/my-s3-ndarray`
        - `hdfs://my-bucket/path/my-hdfs-ndarray`
        - `/path-to/my-local-ndarray`

    data : list of NDArray or dict of str to NDArray
        The data to be saved.
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

def imdecode(str_img, clip_rect=(0, 0, 0, 0), out=None, index=0, channels=3, mean=None):
    """Decode an image from string. Requires OpenCV to work.

    Parameters
    ----------
    str_img : str
        binary image data
    clip_rect : iterable of 4 int
        clip decoded image to rectangle (x0, y0, x1, y1)
    out : NDArray
        output buffer. can be 3 dimensional (c, h, w) or 4 dimensional (n, c, h, w)
    index : int
        output decoded image to i-th slice of 4 dimensional buffer
    channels : int
        number of channels to output. Decode to grey scale when channels = 1.
    mean : NDArray
        subtract mean from decode image before outputing.
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
