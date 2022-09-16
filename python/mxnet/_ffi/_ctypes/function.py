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
# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-branches, global-statement, unused-import
"""
Function configuration API.
Acknowledgement: This file originates from incubator-tvm
"""
import ctypes
from numbers import Number, Integral
import numpy as onp

from ...base import get_last_ffi_error, _LIB, check_call, _MAX_VALUE_64_BIT_SIGNED_, _MAX_VALUE_64_BIT_UNSIGNED_
from ..base import c_str
from .types import MXNetValue, TypeCode
from .types import RETURN_SWITCH
from ..._ctypes.ndarray import NDArrayBase
from .object import ObjectBase, PyNativeObject, _set_class_object
from . import object as _object

ObjectHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p

def _make_packed_func(handle, is_global):
    """Make a packed function class"""
    obj = _CLASS_PACKED_FUNC.__new__(_CLASS_PACKED_FUNC)
    obj.is_global = is_global
    obj.handle = handle
    return obj

def _get_global_func(name, allow_missing=False):
    handle = FunctionHandle()
    check_call(_LIB.MXNetFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    if handle.value:
        return _make_packed_func(handle, False)

    if allow_missing:
        return None

    raise ValueError(f"Cannot find global function {name}")

def _make_mxnet_args(args, temp_args):
    """Pack arguments into c args mxnet call accept"""
    num_args = len(args)
    values = (MXNetValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if isinstance(arg, NDArrayBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NDARRAYHANDLE
        elif isinstance(arg, Integral):
            if arg > _MAX_VALUE_64_BIT_UNSIGNED_:
                raise OverflowError("Integer out of bounds")
            if arg > _MAX_VALUE_64_BIT_SIGNED_:
                values[i].v_uint64 = arg
                type_codes[i] = TypeCode.UINT
            else:
                values[i].v_int64 = arg
                type_codes[i] = TypeCode.INT
        elif isinstance(arg, ObjectBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
        elif arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, PyNativeObject):
            values[i].v_handle = arg.__mxnet_object__.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, str):
            values[i].v_str = c_str(arg)
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, (list, tuple, dict)):
            arg = _FUNC_CONVERT_TO_NODE(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, ctypes.c_void_p):
            values[i].v_handle = arg
            type_codes[i] = TypeCode.HANDLE
        elif isinstance(arg, type):
            values[i].v_str = c_str(onp.dtype(arg).name)
            type_codes[i] = TypeCode.STR
        else:
            raise TypeError(f"Don't know how to handle type {type(arg)}")
    return values, type_codes, num_args


class FunctionBase(object):
    """Function base."""
    __slots__ = ["handle", "is_global"]
    # pylint: disable=no-member
    def __init__(self, handle, is_global):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool
            Whether this is a global function in python
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB is not None:
            if _LIB.MXNetFuncFree(self.handle) != 0:
                raise get_last_ffi_error()

    def __call__(self, *args):
        """Call the function with positional arguments

        args : list
           The positional arguments to the function call.
        """
        temp_args = []
        values, tcodes, num_args = _make_mxnet_args(args, temp_args)
        ret_val = MXNetValue()
        ret_tcode = ctypes.c_int()
        if _LIB.MXNetFuncCall(
                self.handle, values, tcodes, ctypes.c_int(num_args),
                ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
            raise get_last_ffi_error()
        _ = temp_args
        _ = args
        return (RETURN_SWITCH[ret_tcode.value](ret_val) if ret_tcode.value != TypeCode.PYARG
                else RETURN_SWITCH[ret_tcode.value](ret_val, args))


def __init_handle_by_constructor__(fconstructor, args):
    """Initialize handle by constructor"""
    temp_args = []
    values, tcodes, num_args = _make_mxnet_args(args, temp_args)
    ret_val = MXNetValue()
    ret_tcode = ctypes.c_int()
    if _LIB.MXNetFuncCall(
            fconstructor.handle, values, tcodes, ctypes.c_int(num_args),
            ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
        raise get_last_ffi_error()
    _ = temp_args
    _ = args
    assert ret_tcode.value == TypeCode.OBJECT_HANDLE
    handle = ret_val.v_handle
    return handle

_object.__init_by_constructor__ = __init_handle_by_constructor__

_CLASS_PACKED_FUNC = None
_FUNC_CONVERT_TO_NODE = None

def _set_class_packed_func(packed_func_class):
    """Initialize packed function defined in cython"""
    global _CLASS_PACKED_FUNC
    _CLASS_PACKED_FUNC = packed_func_class

def _set_node_generic(func_convert_to_node):
    """Initialize packed function type conversion function in cython"""
    global _FUNC_CONVERT_TO_NODE
    _FUNC_CONVERT_TO_NODE = func_convert_to_node
