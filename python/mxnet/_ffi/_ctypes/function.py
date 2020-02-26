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

from ...base import get_last_ffi_error, _LIB
from ..base import c_str
from .types import MXNetValue, TypeCode
from .types import RETURN_SWITCH
from .object import ObjectBase
from ..node_generic import convert_to_node
from ..._ctypes.ndarray import NDArrayBase

ObjectHandle = ctypes.c_void_p


def _make_mxnet_args(args, temp_args):
    """Pack arguments into c args mxnet call accept"""
    num_args = len(args)
    values = (MXNetValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if isinstance(arg, ObjectBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
        elif arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, Integral):
            values[i].v_int64 = arg
            type_codes[i] = TypeCode.INT
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, str):
            values[i].v_str = c_str(arg)
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, (list, tuple)):
            arg = convert_to_node(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, NDArrayBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NDARRAYHANDLE
        elif isinstance(arg, ctypes.c_void_p):
            values[i].v_handle = arg
            type_codes[i] = TypeCode.HANDLE
        else:
            raise TypeError("Don't know how to handle type %s" % type(arg))
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
        return RETURN_SWITCH[ret_tcode.value](ret_val)


_CLASS_OBJECT = None

def _set_class_object(obj_class):
    global _CLASS_OBJECT
    _CLASS_OBJECT = obj_class
