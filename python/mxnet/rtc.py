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

"""Interface to runtime cuda kernel compile module."""

from array import array
import re
import ctypes
import numpy as np

from .base import _LIB, mx_uint, c_array, c_array_buf, c_str_array, check_call
from .base import c_str, CudaModuleHandle, CudaKernelHandle, numeric_types, string_types
from .ndarray import _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, NDArray

_DTYPE_CPP_TO_NP = {
    'float': np.float32,
    'double': np.float64,
    '__half': np.float16,
    'uint8_t': np.uint8,
    'int': np.int32,
    'int32_t': np.int32,
    'int8_t': np.int8,
    'char': np.int8,
    'int64_t': np.int64,
}

class CudaModule(object):
    r"""Compile and run CUDA code from Python.

    In CUDA 7.5, you need to prepend your kernel definitions
    with 'extern "C"' to avoid name mangling::

        source = r'''
        extern "C" __global__ void axpy(const float *x, float *y, float alpha) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            y[i] += alpha * x[i];
        }
        '''
        module = mx.rtc.CudaModule(source)
        func = module.get_kernel("axpy", "const float *x, float *y, float alpha")
        x = mx.nd.ones((10,), ctx=mx.gpu(0))
        y = mx.nd.zeros((10,), ctx=mx.gpu(0))
        func.launch([x, y, 3.0], mx.gpu(0), (1, 1, 1), (10, 1, 1))
        print(y)

    Starting from CUDA 8.0, you can instead export functions by name.
    This also allows you to use templates::

        source = r'''
        template<typename DType>
        __global__ void axpy(const DType *x, DType *y, DType alpha) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            y[i] += alpha * x[i];
        }
        '''
        module = mx.rtc.CudaModule(source, exports=['axpy<float>', 'axpy<double>'])
        func32 = module.get_kernel("axpy<float>", "const float *x, float *y, float alpha")
        x = mx.nd.ones((10,), dtype='float32', ctx=mx.gpu(0))
        y = mx.nd.zeros((10,), dtype='float32', ctx=mx.gpu(0))
        func32.launch([x, y, 3.0], mx.gpu(0), (1, 1, 1), (10, 1, 1))
        print(y)

        func64 = module.get_kernel("axpy<double>", "const double *x, double *y, double alpha")
        x = mx.nd.ones((10,), dtype='float64', ctx=mx.gpu(0))
        y = mx.nd.zeros((10,), dtype='float64', ctx=mx.gpu(0))
        func32.launch([x, y, 3.0], mx.gpu(0), (1, 1, 1), (10, 1, 1))
        print(y)


    Parameters
    ----------
    source : str
        Complete source code.
    options : tuple of str
        Compiler flags. For example, use "-I/usr/local/cuda/include" to
        add cuda headers to include path.
    exports : tuple of str
        Export kernel names.
    """
    def __init__(self, source, options=(), exports=()):
        if isinstance(options, string_types):
            options = (options,)
        if isinstance(exports, string_types):
            exports = (exports,)
        self.handle = CudaModuleHandle()
        check_call(_LIB.MXRtcCudaModuleCreate(
            c_str(source),
            len(options),
            c_str_array(options),
            len(exports),
            c_str_array(exports),
            ctypes.byref(self.handle)))

    def __del__(self):
        check_call(_LIB.MXRtcCudaModuleFree(self.handle))

    def get_kernel(self, name, signature):
        r"""Get CUDA kernel from compiled module.

        Parameters
        ----------
        name : str
            String name of the kernel.
        signature : str
            Function signature for the kernel. For example, if a kernel is
            declared as::

                extern "C" __global__ void axpy(const float *x, double *y, int alpha)

            Then its signature should be::

                const float *x, double *y, int alpha

            or::

                const float *, double *, int

            Note that `*` in signature marks an argument as array and
            `const` marks an argument as constant (input) array.

        Returns
        -------
        CudaKernel
            CUDA kernels that can be launched on GPUs.
        """
        hdl = CudaKernelHandle()
        is_ndarray = []
        is_const = []
        dtypes = []
        pattern = re.compile(r"""^(const)?\s?([\w_]+)\s?(\*)?\s?([\w_]+)?$""")
        args = re.sub(r"\s+", " ", signature).split(",")
        for arg in args:
            sanitized_arg = " ".join(arg.split())
            match = pattern.match(sanitized_arg)
            if not match or match.groups()[1] == 'const':
                raise ValueError(
                    'Invalid function prototype "%s". Must be in the '
                    'form of "(const) type (*) (name)"'%sanitized_arg)
            is_const.append(bool(match.groups()[0]))
            dtype = match.groups()[1]
            is_ndarray.append(bool(match.groups()[2]))
            if dtype not in _DTYPE_CPP_TO_NP:
                raise TypeError(
                    "Unsupported kernel argument type %s. Supported types are: %s."%(
                        sanitized_arg, ','.join(_DTYPE_CPP_TO_NP.keys())))
            dtypes.append(_DTYPE_NP_TO_MX[_DTYPE_CPP_TO_NP[dtype]])

        check_call(_LIB.MXRtcCudaKernelCreate(
            self.handle,
            c_str(name),
            len(dtypes),
            c_array_buf(ctypes.c_int, array('i', is_ndarray)),
            c_array_buf(ctypes.c_int, array('i', is_const)),
            c_array_buf(ctypes.c_int, array('i', dtypes)),
            ctypes.byref(hdl)))

        return CudaKernel(hdl, name, is_ndarray, dtypes)

class CudaKernel(object):
    """Constructs CUDA kernel. Should be created by `CudaModule.get_kernel`,
    not intended to be used by users.
    """
    def __init__(self, handle, name, is_ndarray, dtypes):
        self.handle = handle
        self._name = name
        self._is_ndarray = is_ndarray
        self._dtypes = [_DTYPE_MX_TO_NP[i] for i in dtypes]

    def __del__(self):
        check_call(_LIB.MXRtcCudaKernelFree(self.handle))

    def launch(self, args, ctx, grid_dims, block_dims, shared_mem=0):
        """Launch cuda kernel.

        Parameters
        ----------
        args : tuple of NDArray or numbers
            List of arguments for kernel. NDArrays are expected for pointer
            types (e.g. `float*`, `double*`) while numbers are expected for
            non-pointer types (e.g. `int`, `float`).
        ctx : Context
            The context to launch kernel on. Must be GPU context.
        grid_dims : tuple of 3 integers
            Grid dimensions for CUDA kernel.
        block_dims : tuple of 3 integers
            Block dimensions for CUDA kernel.
        shared_mem : integer, optional
            Size of dynamically allocated shared memory. Defaults to 0.
        """
        assert ctx.device_type == 'gpu', "Cuda kernel can only be launched on GPU"
        assert len(grid_dims) == 3, "grid_dims must be a tuple of 3 integers"
        assert len(block_dims) == 3, "grid_dims must be a tuple of 3 integers"
        assert len(args) == len(self._dtypes), \
            "CudaKernel(%s) expects %d arguments but got %d"%(
                self._name, len(self._dtypes), len(args))
        void_args = []
        ref_holder = []
        for i, (arg, is_nd, dtype) in enumerate(zip(args, self._is_ndarray, self._dtypes)):
            if is_nd:
                assert isinstance(arg, NDArray), \
                    "The %d-th argument is expected to be a NDArray but got %s"%(
                        i, type(arg))
                void_args.append(arg.handle)
            else:
                assert isinstance(arg, numeric_types), \
                    "The %d-th argument is expected to be a number, but got %s"%(
                        i, type(arg))
                ref_holder.append(np.array(arg, dtype=dtype))
                void_args.append(ref_holder[-1].ctypes.data_as(ctypes.c_void_p))

        check_call(_LIB.MXRtcCudaKernelCall(
            self.handle,
            ctx.device_id,
            c_array(ctypes.c_void_p, void_args),
            mx_uint(grid_dims[0]), mx_uint(grid_dims[1]), mx_uint(grid_dims[2]),
            mx_uint(block_dims[0]), mx_uint(block_dims[1]), mx_uint(block_dims[2]),
            mx_uint(shared_mem)))
