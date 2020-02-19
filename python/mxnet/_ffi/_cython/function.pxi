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

import ctypes
import traceback
from ...ndarray._internal import NDArrayBase
from numbers import Number, Integral


cdef inline int make_arg(object arg,
                         MXNetValue* value,
                         int* tcode,
                         ObjectRef* temp_objs,
                         list temp_args) except -1:
    """Pack arguments into c args mxnet call accept"""
    cdef unsigned long long ptr

    if isinstance(arg, (list, tuple)):
        temp_objs[0] = convert_object(arg)
        value[0].v_handle = (<void*>(temp_objs[0].get()))
        tcode[0] = kObjectHandle
    elif isinstance(arg, NDArrayBase):
        value[0].v_handle = <void*><size_t>(arg._get_handle())
        tcode[0] = kNDArrayHandle
    elif isinstance(arg, (int, long)):
        value[0].v_int64 = arg
        tcode[0] = kInt
    elif isinstance(arg, float):
        value[0].v_float64 = arg
        tcode[0] = kFloat
    elif isinstance(arg, str):
        tstr = c_str(arg)
        value[0].v_str = tstr
        tcode[0] = kStr
        temp_args.append(tstr)
    elif arg is None:
        value[0].v_handle = NULL
        tcode[0] = kNull
    elif isinstance(arg, Number):
        value[0].v_float64 = arg
        tcode[0] = kFloat
    elif isinstance(arg, ctypes.c_void_p):
        value[0].v_handle = c_handle(arg)
        tcode[0] = kHandle
    else:
        raise TypeError("Don't know how to handle type %s" % type(arg))
    return 0


cdef inline object make_ret(MXNetValue value, int tcode):
    """convert result to return value."""
    if tcode == kNull:
        return None
    elif tcode == kInt:
        return value.v_int64
    elif tcode == kFloat:
        return value.v_float64
    elif tcode == kStr:
        return py_str(value.v_str)
    elif tcode == kHandle:
        return ctypes_handle(value.v_handle)
    elif tcode == kNDArrayHandle:
        return c_make_array(value.v_handle)
    raise ValueError("Unhandled type code %d" % tcode)


cdef inline int FuncCall3(void* chandle,
                          tuple args,
                          int nargs,
                          MXNetValue* ret_val,
                          int* ret_tcode) except -1:
    cdef MXNetValue[3] values
    cdef int[3] tcodes
    cdef ObjectRef[3] temp_objs
    nargs = len(args)
    temp_args = []
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], &temp_objs[i], temp_args)
    CALL(MXNetFuncCall(chandle, &values[0], &tcodes[0],
                     nargs, ret_val, ret_tcode))
    return 0


cdef inline int FuncCall(void* chandle,
                         tuple args,
                         MXNetValue* ret_val,
                         int* ret_tcode) except -1:
    cdef int nargs
    nargs = len(args)
    if nargs <= 3:
        FuncCall3(chandle, args, nargs, ret_val, ret_tcode)
        return 0

    cdef vector[MXNetValue] values
    cdef vector[int] tcodes
    cdef vector[ObjectRef] temp_objs
    values.resize(nargs)
    tcodes.resize(nargs)
    temp_objs.resize(nargs)

    temp_args = []
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], &temp_objs[i], temp_args)
    CALL(MXNetFuncCall(chandle, &values[0], &tcodes[0],
                     nargs, ret_val, ret_tcode))
    return 0


cdef class FunctionBase:
    cdef MXNetFunctionHandle chandle
    cdef int is_global

    cdef inline _set_handle(self, handle):
        if handle is None:
            self.chandle = NULL
        else:
            self.chandle = c_handle(handle)

    property is_global:
        def __get__(self):
            return self.c_is_global != 0

        def __set__(self, value):
            self.c_is_global = value

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.chandle, ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    def __init__(self, handle, is_global):
        self._set_handle(handle)
        self.c_is_global = is_global

    def __dealloc__(self):
        if self.is_global == 0:
            CALL(MXNetFuncFree(self.chandle))

    def __call__(self, *args):
        cdef MXNetValue ret_val
        cdef int ret_tcode
        FuncCall(self.chandle, args, &ret_val, &ret_tcode)
        return make_ret(ret_val, ret_tcode)
