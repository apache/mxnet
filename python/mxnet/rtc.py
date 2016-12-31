"""Interface to runtime cuda kernel compile module."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, NDArrayHandle, RtcHandle, mx_uint, c_array, check_call

class Rtc(object):
    """MXRtc object in mxnet.
    This class allow you to write cuda kernel in python
    and call them with NDArray.

    Parameters
    ----------
    name : str
        name of the kernel
    inputs : tuple of (str, mxnet.ndarray)
        list of input names and ndarray
    outputs : tuple of (str, mxnet.ndarray)
        list of output names and ndarray
    kernel : str
        the actual kernel code.
        Note that this is only the body of the kernel, i.e.
        after { and before }. Rtc will decorate the kernel.
        For example, if name = "mykernel" and
        inputs = [('x', mx.nd.zeros((10,)))]
        outputs = [('y', mx.nd.zeros((10,)))]
        kernel = "y[threadIdx.x] = x[threadIdx.x];",
        the kernel that is compile will be:
        extern "C" __global__ mykernel(float *x, float *y) {
            const int x_ndim = 1;
            const int x_dims = { 10 };
            const int y_ndim = 1;
            const int y_dims = { 10 };

            y[threadIdx.x] = x[threadIdx.x];
        }
    """
    def __init__(self, name, inputs, outputs, kernel):
        self.handle = RtcHandle()
        input_names = ctypes.cast(c_array(ctypes.c_char_p, [i[0] for i in inputs]),
                                  ctypes.POINTER(ctypes.c_char_p))
        output_names = ctypes.cast(c_array(ctypes.c_char_p, [i[0] for i in outputs]),
                                   ctypes.POINTER(ctypes.c_char_p))
        input_nds = ctypes.cast(c_array(NDArrayHandle, [i[1].handle for i in inputs]),
                                ctypes.POINTER(NDArrayHandle))
        output_nds = ctypes.cast(c_array(NDArrayHandle, [i[1].handle for i in outputs]),
                                 ctypes.POINTER(NDArrayHandle))
        check_call(_LIB.MXRtcCreate(ctypes.c_char_p(name),
                                    mx_uint(len(inputs)),
                                    mx_uint(len(outputs)),
                                    input_names,
                                    output_names,
                                    input_nds,
                                    output_nds,
                                    ctypes.c_char_p(kernel),
                                    ctypes.byref(self.handle)))

    def __del__(self):
        check_call(_LIB.MXRtcFree(self.handle))

    def push(self, inputs, outputs, grid_dims, block_dims):
        """run the kernel.

        Parameters
        ----------
        inputs : list of ndarray
            list of input. Can be different ndarray then uses for constructor,
            but must have the same shape and in the same order.
        outputs : list of ndarray
            list of out. Can be different ndarray then uses for constructor,
            but must have the same shape and in the same order.
        grid_dims : tuple of 3 uint
            grid dimension for kernel launch
        block_dims : tuple of 3 uint
            block dimension for kernel launch
        """
        input_nds = ctypes.cast(c_array(NDArrayHandle, [i.handle for i in inputs]),
                                ctypes.POINTER(NDArrayHandle))
        output_nds = ctypes.cast(c_array(NDArrayHandle, [i.handle for i in outputs]),
                                 ctypes.POINTER(NDArrayHandle))
        check_call(_LIB.MXRtcPush(self.handle,
                                  mx_uint(len(inputs)),
                                  mx_uint(len(outputs)),
                                  input_nds,
                                  output_nds,
                                  mx_uint(grid_dims[0]),
                                  mx_uint(grid_dims[1]),
                                  mx_uint(grid_dims[2]),
                                  mx_uint(block_dims[0]),
                                  mx_uint(block_dims[1]),
                                  mx_uint(block_dims[2])))
