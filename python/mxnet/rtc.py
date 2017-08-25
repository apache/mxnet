"""Interface to runtime cuda kernel compile module."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, NDArrayHandle, RtcHandle, mx_uint, c_array, check_call

class Rtc(object):
    """MXRtc object in mxnet.
    This class allow you to write CUDA kernels in Python
    and call them with NDArray.

    Parameters
    ----------
    name : str
        Name of the kernel.
    inputs : tuple of (str, mxnet.ndarray)
        List of input names and ndarray.
    outputs : tuple of (str, mxnet.ndarray)
        List of output names and ndarray.
    kernel : str
        The actual kernel code.
        Note that this is only the body of the kernel, i.e.
        after { and before }. Rtc will decorate the kernel.
        For example, if ``name = "mykernel"`` and
        inputs = [('x', mx.nd.zeros((10,)))]
        outputs = [('y', mx.nd.zeros((10,)))]
        kernel = "y[threadIdx.x] = x[threadIdx.x];",
        then the compiled kernel will be:
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
        """Run the kernel.

        Parameters
        ----------
        inputs : list of NDArray
            List of inputs. Can contain different NDArrays than those used for the constructor,
            but its elements must have the same shapes and appear in the same order.
        outputs : list of NDArray
            List of outputs. Can contain different ndarrays than used for the constructor,
            but must have the same shapes and appear in the same order.
        grid_dims : tuple of 3 uint
            Grid dimension for kernel launch.
        block_dims : tuple of 3 uint
            Block dimension for kernel launch.
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
