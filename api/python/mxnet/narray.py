# coding: utf-8
"""NArray interface of mxnet"""
import ctypes
import numpy as np
from base import lib
from base import op
from base import c_array
from base import ctypes2numpy
from base import invoke
from base import check_call
from base import new_narray_handle
from base import MXNetError

# function handles
_h_plus = op.plus
_h_minus = op.minus
_h_mul = op.mul
_h_div = op.div

class NArray(object):
    """NArray object in mxnet
    
    NArray is basic ndarray like data structure in mxnet    
    """
    def __init__(self, handle):
        """initialize a new NArray

        Parameters
        ----------
        handle : ctypes.c_void_p
            NArray handle of C API        
        """
        assert isinstance(handle, ctypes.c_void_p)
        self.handle = handle

    def __del__(self):
        check_call(lib.MXNArrayFree(self.handle))

    def __lbinary__(self, handle, other):
        if isinstance(other, NArray):
            hret = new_narray_handle()
            invoke(_h_plus, (self.handle, other.handle), (), (hret,))
            return NArray(handle = hret)
        else:
            raise MXNetError('type ' + str(other) + 'not supported')            

    def __add__(self, other):
        return self.__lbinary__(_h_plus, other)

    def __sub__(self, other):
        return self.__lbinary__(_h_plus, other)
    
    def wait(self):
        """Wait until the data on current NArray is available"""
        check_call(lib.MXNArrayWait(self.handle))

    def get_shape(self):
        """Get shape of current NArray
        
        Returns
        -------
        a tuple representing shape of current narray
        """
        ndim = ctypes.c_uint()
        pdata = ctypes.POINTER(ctypes.c_uint)()
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
        pdata = ctypes.POINTER(ctypes.c_float)()
        check_call(lib.MXNArrayGetData(self.handle, ctypes.byref(pdata)))        
        shape = self.get_shape()
        return ctypes2numpy(pdata, shape)


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
    h = ctypes.c_void_p()
    data = np.zeros(shape, dtype = np.float32)
    ndim = len(shape)
    check_call(lib.MXNArrayCreateShareMem(
        data.ctypes.data,
        c_array(ctypes.c_uint, shape), 
        ndim, ctypes.byref(h)))
    ret = NArray(handle = h)
    ret.numpy = data
    return ret
