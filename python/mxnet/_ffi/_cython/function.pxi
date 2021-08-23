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
import numpy as onp
import traceback
from ...ndarray._internal import NDArrayBase
from numbers import Number, Integral


cdef inline int make_arg(object arg,
                         MXNetValue* value,
                         int* tcode,
                         list temp_args) except -1:
    """Pack arguments into c args mxnet call accept"""
    cdef unsigned long long ptr

    if isinstance(arg, NDArrayBase):
        value[0].v_handle = <void*><size_t>(arg._get_handle())
        tcode[0] = kNDArrayHandle
    elif isinstance(arg, Integral):
        value[0].v_int64 = arg
        tcode[0] = kInt
    elif isinstance(arg, ObjectBase):
        value[0].v_handle = (<ObjectBase>arg).chandle
        tcode[0] = kObjectHandle
    elif isinstance(arg, float):
        value[0].v_float64 = arg
        tcode[0] = kFloat
    elif isinstance(arg, PyNativeObject):
        value[0].v_handle = (<ObjectBase>(arg.__mxnet_object__)).chandle
        tcode[0] = kObjectHandle
    elif isinstance(arg, str):
        tstr = c_str(arg)
        value[0].v_str = tstr
        tcode[0] = kStr
        temp_args.append(tstr)
    elif isinstance(arg, (list, tuple, dict)):
        arg = _FUNC_CONVERT_TO_NODE(arg)
        value[0].v_handle = (<ObjectBase>arg).chandle
        tcode[0] = kObjectHandle
        temp_args.append(arg)
    elif arg is None:
        value[0].v_handle = NULL
        tcode[0] = kNull
    elif isinstance(arg, Number):
        value[0].v_float64 = arg
        tcode[0] = kFloat
    elif isinstance(arg, ctypes.c_void_p):
        value[0].v_handle = c_handle(arg)
        tcode[0] = kHandle
    elif isinstance(arg, type):
        tstr = c_str(onp.dtype(arg).name)
        value[0].v_str = tstr
        tcode[0] = kStr
        temp_args.append(tstr)
    else:
        raise TypeError("Don't know how to handle type %s" % type(arg))
    return 0


cdef inline object make_ret(MXNetValue value, int tcode):
    """convert result to return value."""
    if tcode == kNDArrayHandle:
        return c_make_array(value.v_handle)
    elif tcode == kNull:
        return None
    elif tcode == kObjectHandle:
        return make_ret_object(value.v_handle)
    elif tcode == kInt:
        return value.v_int64
    elif tcode == kFloat:
        return value.v_float64
    elif tcode == kStr:
        return py_str(value.v_str)
    elif tcode == kHandle:
        return <unsigned long long>(value.v_handle)
    raise ValueError("Unhandled type code %d" % tcode)


cdef inline int FuncCall3(void* chandle,
                          tuple args,
                          int nargs,
                          MXNetValue* ret_val,
                          int* ret_tcode) except -1:
    cdef MXNetValue[3] values
    cdef int[3] tcodes
    nargs = len(args)
    temp_args = []
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], temp_args)
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
    values.resize(nargs)
    tcodes.resize(nargs)

    temp_args = []
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], temp_args)
    CALL(MXNetFuncCall(chandle, &values[0], &tcodes[0],
                     nargs, ret_val, ret_tcode))
    return 0


cdef inline int ConstructorCall(void* constructor_handle,
                                int type_code,
                                tuple args,
                                void** handle) except -1:
    """Call contructor of a handle function"""
    cdef MXNetValue ret_val
    cdef int ret_tcode
    FuncCall(constructor_handle, args, &ret_val, &ret_tcode)
    assert ret_tcode == type_code
    handle[0] = ret_val.v_handle
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
        if ret_tcode == kPyArg:
            return args[ret_val.v_int64]
        else:
            return make_ret(ret_val, ret_tcode)

cdef object make_packed_func(MXNetFunctionHandle chandle, int is_global):
    obj = _CLASS_PACKED_FUNC.__new__(_CLASS_PACKED_FUNC)
    (<FunctionBase>obj).chandle = chandle
    (<FunctionBase>obj).is_global = is_global
    return obj

def _get_global_func(name, allow_missing=False):
    cdef MXNetFunctionHandle chandle
    CALL(MXNetFuncGetGlobal(c_str(name), &chandle))
    if chandle != NULL:
        return make_packed_func(chandle, True)

    if allow_missing:
        return None

    raise ValueError("Cannot find global function %s" % name)

_CLASS_OBJECT = None
_CLASS_PACKED_FUNC = None
_FUNC_CONVERT_TO_NODE = None

def _set_class_object(obj_class):
    """Initialize object class defined in cython"""
    global _CLASS_OBJECT
    _CLASS_OBJECT = obj_class

def _set_class_packed_func(func_class):
    """Initialize packed function defined in cython"""
    global _CLASS_PACKED_FUNC
    _CLASS_PACKED_FUNC = func_class

def _set_node_generic(func_convert_to_node):
    """Initialize packed function type conversion function in cython"""
    global _FUNC_CONVERT_TO_NODE
    _FUNC_CONVERT_TO_NODE = func_convert_to_node
