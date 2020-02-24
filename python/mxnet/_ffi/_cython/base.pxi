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

"""Acknowledgement: This file originates from incubator-tvm"""

from libcpp.vector cimport vector
from cpython.version cimport PY_MAJOR_VERSION
from cpython cimport pycapsule
from libc.stdint cimport int32_t, int64_t, uint64_t, uint8_t, uint16_t, uint32_t
import ctypes
from ...base import get_last_ffi_error

cdef enum MXNetTypeCode:
    kInt = 0
    kUInt = 1
    kFloat = 2
    kHandle = 3
    kNull = 4
    kMXNetType = 5
    kMXNetContext = 6
    kArrayHandle = 7
    kObjectHandle = 8
    kModuleHandle = 9
    kFuncHandle = 10
    kStr = 11
    kBytes = 12
    kNDArrayContainer = 13
    kNDArrayHandle = 14
    kExtBegin = 15

cdef extern from "mxnet/runtime/c_runtime_api.h":
    ctypedef struct MXNetValue:
        int64_t v_int64
        double v_float64
        void* v_handle
        const char* v_str

ctypedef void* MXNetRetValueHandle
ctypedef void* MXNetFunctionHandle
ctypedef void* ObjectHandle


cdef extern from "mxnet/runtime/c_runtime_api.h":
    int MXNetFuncCall(MXNetFunctionHandle func,
                      MXNetValue* arg_values,
                      int* type_codes,
                      int num_args,
                      MXNetValue* ret_val,
                      int* ret_type_code)
    int MXNetFuncFree(MXNetFunctionHandle func)


cdef inline py_str(const char* x):
    if PY_MAJOR_VERSION < 3:
        return x
    else:
        return x.decode("utf-8")


cdef inline c_str(pystr):
    """Create ctypes char * from a python string
    Parameters
    ----------
    string : string type
        python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return pystr.encode("utf-8")


cdef inline CALL(int ret):
    if ret != 0:
        raise get_last_ffi_error()


cdef inline object ctypes_handle(void* chandle):
    """Cast C handle to ctypes handle."""
    return ctypes.cast(<unsigned long long>chandle, ctypes.c_void_p)


cdef inline void* c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <void*>(v_ptr)
