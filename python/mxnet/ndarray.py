# coding: utf-8
# pylint: disable= too-many-lines, redefined-builtin, protected-access
"""NDArray API of mxnet."""
from __future__ import absolute_import
from __future__ import division

import ctypes
import warnings
import sys
import functools
import operator
import numpy as np
from .base import _LIB, string_types, numeric_types
from .base import c_array, py_str, c_str, mx_real_t
from .base import mx_uint, mx_float, NDArrayHandle, FunctionHandle
from .base import ctypes2buffer
from .base import check_call, ctypes2docstring
from .context import Context
from . import _ndarray_internal as _internal

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

class NDArray(object):
    """NDArray object in mxnet.

    NDArray is basic ndarray/Tensor like data structure in mxnet.
    """
    # pylint: disable= no-member
    def __init__(self, handle, writable=True):
        """initialize a new NDArray

        Parameters
        ----------
        handle : NDArrayHandle
            NDArray handle of C API
        """
        assert isinstance(handle, NDArrayHandle)
        self.handle = handle
        self.writable = writable

    def __del__(self):
        check_call(_LIB.MXNDArrayFree(self.handle))

    def __add__(self, other):
        return add(self, other)

    def __iadd__(self, other):
        if not self.writable:
            raise ValueError('trying to add to a readonly NDArray')
        if isinstance(other, NDArray):
            return _internal._plus(self, other, out=self)
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
            return _internal._minus(self, other, out=self)
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
            return _internal._mul(self, other, out=self)
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
            return _internal._div(self, other, out=self)
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

    def __getstate__(self):
        this = self.__dict__.copy()
        handle = this['handle']
        if handle is not None:
            length = ctypes.c_size_t()
            cptr = ctypes.POINTER(ctypes.c_char)()
            check_call(_LIB.MXNDArraySaveRawBytes(self.handle,
                                                  ctypes.byref(length),
                                                  ctypes.byref(cptr)))
            this['handle'] = ctypes2buffer(cptr, length.value)
        return this

    def __setstate__(self, state):
        handle = state['handle']
        if handle is not None:
            buf = handle
            handle = NDArrayHandle()
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            length = ctypes.c_size_t(len(buf))
            check_call(_LIB.MXNDArrayLoadFromRawBytes(ptr, length, ctypes.byref(handle)))
            state['handle'] = handle
        self.__dict__.update(state)

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if not self.writable:
            raise ValueError('trying to assign to a readonly NDArray')
        if isinstance(in_slice, int):
            sliced_arr = self._at(in_slice)
            sliced_arr[:] = value
            return
        if not isinstance(in_slice, slice) or in_slice.step is not None:
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

    def __getitem__(self, in_slice):
        """Get ndarray"""
        if isinstance(in_slice, int):
            return self._at(in_slice)
        if not isinstance(in_slice, slice) or in_slice.step is not None:
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
            raise ValueError('array shape do not match the shape of NDArray')
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
        dtype : numpy.dtype or string
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


def empty(shape, ctx=None, dtype=mx_real_t):
    """Create an empty uninitialized new NDArray, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NDArray.

    ctx : Context, optional
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
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
        # check whether broadcasting is needed
        lsize = functools.reduce(operator.mul, lhs.shape)
        rsize = functools.reduce(operator.mul, rhs.shape)
        if lsize < rsize:
            lhs = lhs.broadcast_to(rhs.shape)
        elif lsize > rsize:
            rhs = rhs.broadcast_to(lhs.shape)
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
        _internal._plus,
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
        _internal._minus,
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
        _internal._mul,
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
        _internal._div,
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
        _internal._power,
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
        _internal._maximum,
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
        _internal._minimum,
        lambda x, y: x if x < y else y,
        _internal._minimum_scalar,
        None)
    # pylint: enable= no-member, protected-access

def true_divide(lhs, rhs):
    """ Same as numpy's true_divide. It adjusts the output type to present the best answer,
    regardless of input types.
    """
    return divide(lhs, rhs)

def negative(arr):
    """ Return the negation of array values """
    return multiply(arr, -1.0)

def zeros(shape, ctx=None, dtype=mx_real_t):
    """Create a new NDArray filled with 0, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NDArray.
    ctx : Context, optional.
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
    """
    arr = empty(shape, ctx, dtype)
    arr[:] = 0.0
    return arr

def ones(shape, ctx=None, dtype=mx_real_t):
    """Create a new NDArray filled with 1, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NDArray.
    ctx : Context, optional
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
    """
    arr = empty(shape, ctx, dtype)
    arr[:] = 1.0
    return arr

def full(shape, val, ctx=None):
    """Create a new NDArray filled with given value, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NDArray.
    val : float
        value to be filled with.
    ctx : Context, optional
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
    """
    arr = empty(shape, ctx)
    arr[:] = val
    return arr

def array(source_array, ctx=None, dtype=mx_real_t):
    """Create a new NDArray that copies content from source_array.

    Parameters
    ----------
    source_array : array_like
        Source data to create NDArray from.

    ctx : Context, optional
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
    """

    if not isinstance(source_array, np.ndarray):
        try:
            source_array = np.array(source_array, dtype=dtype)
        except:
            raise TypeError('source_array must be array like object')
    arr = empty(source_array.shape, ctx, dtype)
    arr[:] = source_array
    return arr

def concatenate(arrays, always_copy=True):
    """Concatenate a list of NDArrays along the first dimension.

    Parameters
    ----------
    arrays : list of NDArray
        Arrays to be concatenate. They must have identical shape except
        the first dimension. They also must have the same data type.
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

    shape0 = arrays[0].shape[0]
    shape_rest = arrays[0].shape[1:]
    dtype = arrays[0].dtype
    for arr in arrays[1:]:
        shape0 += arr.shape[0]
        assert shape_rest == arr.shape[1:]
        assert dtype == arr.dtype
    ret = empty((shape0,) + shape_rest, ctx=arrays[0].context, dtype=dtype)
    idx = 0
    for arr in arrays:
        ret[idx:idx+arr.shape[0]] = arr
        idx += arr.shape[0]

    return ret

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
        substract mean from decode image before outputing.
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

# pylint: disable=too-many-locals, invalid-name
def _make_ndarray_function(handle):
    """Create a NDArray function from the FunctionHandle."""
    NDARRAY_ARG_BEFORE_SCALAR = 1
    ACCEPT_EMPTY_MUTATE_TARGET = 1 << 2
    # Get the property of NDArray
    n_used_vars = mx_uint()
    n_scalars = mx_uint()
    n_mutate_vars = mx_uint()
    type_mask = ctypes.c_int()
    check_call(_LIB.MXFuncDescribe(
        handle,
        ctypes.byref(n_used_vars),
        ctypes.byref(n_scalars),
        ctypes.byref(n_mutate_vars),
        ctypes.byref(type_mask)))
    n_mutate_vars = n_mutate_vars.value
    n_used_vars = n_used_vars.value
    n_scalars = n_scalars.value
    type_mask = type_mask.value
    accept_empty_mutate = (type_mask & ACCEPT_EMPTY_MUTATE_TARGET) != 0
    # infer type of the function
    if (type_mask & NDARRAY_ARG_BEFORE_SCALAR) != 0:
        use_vars_range = range(0, n_used_vars)
        scalar_range = range(n_used_vars, n_used_vars + n_scalars)
    else:
        scalar_range = range(0, n_scalars)
        use_vars_range = range(n_scalars, n_used_vars + n_scalars)

    # Get the information from the function
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXFuncGetInfo(
        handle, ctypes.byref(name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(ret_type)))
    func_name = py_str(name.value)
    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)
    doc_str = ('%s\n\n' +
               '%s\n' +
               'out : NDArray, optional\n' +
               '    The output NDArray to hold the result.\n\n'+
               'Returns\n' +
               '-------\n' +
               'out : NDArray\n'+
               '    The output of binary function.')
    doc_str = doc_str % (py_str(desc.value), param_str)

    # Definition of internal functions.
    def binary_ndarray_function(lhs, rhs, out=None, **kwargs):
        """Internal binary function
        """
        if out:
            if not isinstance(out, NDArray):
                raise TypeError('out must be NDArray')
            if not out.writable:
                raise TypeError('out must be writable')
        else:
            if not accept_empty_mutate:
                raise TypeError('argument out is required to call %s' % func_name)
            out = NDArray(_new_empty_handle())
        check_call(_LIB.MXFuncInvokeEx( \
                handle, \
                c_array(NDArrayHandle, (lhs.handle, rhs.handle)), \
                c_array(mx_float, ()), \
                c_array(NDArrayHandle, (out.handle,)), \
                ctypes.c_int(len(kwargs)), \
                c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()]), \
                c_array(ctypes.c_char_p, [c_str(str(i)) for i in kwargs.values()])))
        return out

    def unary_ndarray_function(src, out=None, *args, **kwargs):
        """internal NDArray function"""
        if out:
            if not isinstance(out, NDArray):
                raise TypeError('out must be NDArray')
            if not out.writable:
                raise TypeError('out must be writable')
        else:
            if not accept_empty_mutate:
                raise TypeError('argument out is required to call %s' % func_name)
            out = NDArray(_new_empty_handle())
        check_call(_LIB.MXFuncInvokeEx( \
                handle, \
                c_array(NDArrayHandle, (src.handle,)), \
                c_array(mx_float, [args[i] for i in scalar_range]), \
                c_array(NDArrayHandle, (out.handle,)), \
                ctypes.c_int(len(kwargs)), \
                c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()]), \
                c_array(ctypes.c_char_p, [c_str(str(i)) for i in kwargs.values()])))
        return out

    def generic_ndarray_function(*args, **kwargs):
        """Invoke this function by passing in parameters

        Parameters
        ----------
        *args
            Positional arguments of input scalars and NDArray
        out : NDArray or tuple of NDArray, optional
            Output NDArray, used to hold the output result.

        Returns
        -------
        out : NDArray
            The result NDArray(tuple) of result of computation.
        """
        if 'out' in kwargs:
            mutate_vars = kwargs['out']
            if isinstance(mutate_vars, NDArray):
                mutate_vars = (mutate_vars,)
            if len(mutate_vars) != n_mutate_vars:
                raise TypeError('expect %d out in %s', n_mutate_vars, func_name)
            del kwargs['out']
        else:
            if accept_empty_mutate:
                mutate_vars = tuple(
                    NDArray(_new_empty_handle()) for i in range(n_mutate_vars))
            else:
                raise TypeError('argument out is required to call %s' % func_name)
        check_call(_LIB.MXFuncInvokeEx( \
                handle, \
                c_array(NDArrayHandle, [args[i].handle for i in use_vars_range]), \
                c_array(mx_float, [args[i] for i in scalar_range]), \
                c_array(NDArrayHandle, [v.handle for v in mutate_vars]), \
                ctypes.c_int(len(kwargs)), \
                c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()]), \
                c_array(ctypes.c_char_p, [c_str(str(i)) for i in kwargs.values()])))
        if n_mutate_vars == 1:
            return mutate_vars[0]
        else:
            return mutate_vars
    # End of function declaration
    if n_mutate_vars == 1 and n_used_vars == 2 and n_scalars == 0:
        ret_function = binary_ndarray_function
    elif n_mutate_vars == 1 and n_used_vars == 1 and n_scalars == 0:
        ret_function = unary_ndarray_function
    else:
        ret_function = generic_ndarray_function
    ret_function.__name__ = func_name
    ret_function.__doc__ = doc_str
    return ret_function



# pylint: enable=too-many-locals, invalid-name

def _init_ndarray_module():
    """List and add all the ndarray functions to current module."""
    plist = ctypes.POINTER(FunctionHandle)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListFunctions(ctypes.byref(size),
                                    ctypes.byref(plist)))

    module_obj = sys.modules[__name__]
    module_internal = sys.modules["mxnet._ndarray_internal"]
    for i in range(size.value):
        hdl = FunctionHandle(plist[i])
        function = _make_ndarray_function(hdl)
        # if function name starts with underscore, register as internal namespace
        if function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
        else:
            fname = function.__name__
            fn_obj = getattr(module_obj, fname, None)
            if fn_obj is None:
                setattr(module_obj, fname, function)
            else:
                setattr(module_obj, fname + '_internal', function)

# Initialize the NDArray module
_init_ndarray_module()
