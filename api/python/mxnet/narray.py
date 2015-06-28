# coding: utf-8
"""NArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
import numpy as np
from .base import lib
from .base import op
from .base import c_array
from .base import mx_uint, mx_float, NArrayHandle
from .base import ctypes2numpy
from .base import invoke
from .base import check_call
from .base import MXNetError

def _new_empty_handle():
    """Return a new empty handle
    
    Empty handle is only used to hold results
    Returns
    -------
    a new empty narray handle
    """
    h = NArrayHandle()
    check_call(lib.MXNArrayCreateNone(ctypes.byref(h)))
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
            invoke(op.plus, (other.handle, self.handle), (), (hret,))
        else:
            raise MXNetError('type %s not supported' % str(type(other)))
        return NArray(handle = hret)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        hret = _new_empty_handle()
        if isinstance(other, NArray):
            invoke(op.minus, (other.handle, self.handle), (), (hret,))
        else:
            raise MXNetError('type %s not supported' % str(type(other)))
        return NArray(handle = hret)

    def __mul__(self, other):
        hret = _new_empty_handle()
        if isinstance(other, NArray):
            invoke(op.mul, (other.handle, self.handle), (), (hret,))
        else:
            raise MXNetError('type %s not supported' % str(type(other)))
        return NArray(handle = hret)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        hret = _new_empty_handle()
        if isinstance(other, NArray):
            invoke(op.div, (other.handle, self.handle), (), (hret,))
        else:
            raise MXNetError('type %s not supported' % str(type(other)))
        return NArray(handle = hret)
    
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

    def to_numpy(self):
        """Return a copy of numpy NArray
        
        Returns
        -------
        a tuple representing shape of current narray
        """
        self.wait()
        pdata = ctypes.POINTER(mx_float)()
        check_call(lib.MXNArrayGetData(self.handle, ctypes.byref(pdata)))
        return ctypes2numpy(pdata, self.shape)


def zeros_shared(shape):
    """Create a new CPU based narray that shares memory content with a numpy array
    
    Parameters
    ----------
    shape : tuple
        shape of the NArray

    Returns
    -------
    a new NArray that shares memory with numpy.narray
    """
    h = NArrayHandle()
    data = np.zeros(shape, dtype = np.float32)
    ndim = len(shape)
    check_call(lib.MXNArrayCreateShareMem(
        data.ctypes.data,
        c_array(mx_uint, shape), 
        ndim, ctypes.byref(h)))
    ret = NArray(handle = h)
    ret.numpy = data
    return ret
