# coding: utf-8
# pylint: disable=invalid-name
"""NArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
from .base import lib
from .base import c_array
from .base import mx_uint, mx_float, NArrayHandle
from .base import ctypes2numpy_shared
from .base import check_call
from .base import MXNetError
from .context import Context

# op is implicitly imported from .function
# as a singleton of _FunctionRegistry
global op

def _new_empty_handle():
    """Return a new empty handle

    Empty handle can be used to hold result
    Returns
    -------
    a new empty narray handle
    """
    h = NArrayHandle()
    check_call(lib.MXNArrayCreateNone(ctypes.byref(h)))
    return h

def _new_alloc_handle(shape, ctx, delay_alloc):
    """Return a new handle with specified shape, context

    Empty handle is only used to hold results
    Returns
    -------
    a new empty narray handle
    """
    h = NArrayHandle()
    check_call(lib.MXNArrayCreate(
        c_array(mx_uint, shape),
        len(shape),
        ctx.device_mask,
        ctx.device_id,
        int(delay_alloc),
        ctypes.byref(h)))
    return h

class NArray(object):
    """NArray object in mxnet

    NArray is basic ndarray like data structure in mxnet
    """
    def __init__(self, handle):
        """initialize a new NArray

        Parameters
        ----------
        handle : NArrayHandle
            NArray handle of C API
        """
        assert isinstance(handle, NArrayHandle)
        self.handle = handle

    def __del__(self):
        check_call(lib.MXNArrayFree(self.handle))

    def __add__(self, other):
        hret = _new_empty_handle()
        if isinstance(other, NArray):
            op.plus.invoke_with_handle_((other.handle, self.handle), (), (hret,))
        else:
            raise MXNetError('type %s not supported' % str(type(other)))
        return NArray(handle=hret)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        hret = _new_empty_handle()
        if isinstance(other, NArray):
            op.minus.invoke_with_handle_((other.handle, self.handle), (), (hret,))
        else:
            raise MXNetError('type %s not supported' % str(type(other)))
        return NArray(handle=hret)

    def __mul__(self, other):
        hret = _new_empty_handle()
        if isinstance(other, NArray):
            op.mul.invoke_with_handle_((other.handle, self.handle), (), (hret,))
        else:
            raise MXNetError('type %s not supported' % str(type(other)))
        return NArray(handle=hret)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        hret = _new_empty_handle()
        if isinstance(other, NArray):
            op.div.invoke_with_handle_((other.handle, self.handle), (), (hret,))
        else:
            raise MXNetError('type %s not supported' % str(type(other)))
        return NArray(handle=hret)

    def wait(self):
        """Wait until the data on current NArray is available"""
        check_call(lib.MXNArrayWait(self.handle))

    @property
    def shape(self):
        """Get shape of current NArray

        Returns
        -------
        a tuple representing shape of current narray
        """
        ndim = mx_uint()
        pdata = ctypes.POINTER(mx_uint)()
        check_call(lib.MXNArrayGetShape(
            self.handle, ctypes.byref(ndim), ctypes.byref(pdata)))
        return tuple(pdata[i] for i in range(ndim.value))

    @property
    def context(self):
        """Get context of current NArray

        Returns
        -------
        the context of current NArray
        """
        dev_mask = ctypes.c_int()
        dev_id = ctypes.c_int()
        check_call(lib.MXNArrayGetContext(
            self.handle, ctypes.byref(dev_mask), ctypes.byref(dev_id)))
        return Context(Context.devmask2type[dev_mask.value], dev_id.value)

    @property
    def numpy(self):
        """Return a numpy representation of current array

        This array have to sit on CPU

        Returns
        -------
        a numpy array view
        """
        self.wait()
        pdata = ctypes.POINTER(mx_float)()
        check_call(lib.MXNArrayGetData(self.handle, ctypes.byref(pdata)))
        return ctypes2numpy_shared(pdata, self.shape)

    def copyto(self, other):
        """copy the content of current array to othe

        When other is NArray, the content is copied over.
        When other is a Context, a new NArray in the context
        will be created as target

        Parameters
        ----------
        other : NArray or Context
            another narray we want to copy to,
            or target context we want copy the data to

        Returns
        -------
        the copy target NArray
        """
        if isinstance(other, NArray):
            op.copy.invoke_with_handle_((self.handle,), (), (other.handle,))
            return other
        elif isinstance(other, Context):
            hret = _new_alloc_handle(self.shape, other, True)
            op.copy.invoke_with_handle_((self.handle,), (), (hret,))
            return NArray(handle=hret)
        else:
            raise MXNetError('copyto do not support type ' + type(other))

def create(shape, ctx=Context.default_ctx):
    """Create a new NArray, with specified shape

    Parameters
    ----------
    shape : tuple
        shape of the NArray

    Returns
    -------
    a new NArray
    """
    return NArray(handle=_new_alloc_handle(shape, ctx, False))

def _init_function_registry(new_op):
    """Initialize the global variable op with new_op

    This function is used to resolve cyclic dependency of .narray on function

    Parameters
    ----------
    new_op : function._FunctionRegistry
        a FunctionRegistry to pass in in startup
    """
    global op
    op = new_op
    return op
