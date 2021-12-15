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
"""The C Types used in API.
Acknowledgement: This file originates from incubator-tvm
"""
# pylint: disable=invalid-name
import ctypes
from ..base import py_str
from ...base import NDArrayHandle
from ... import _global_var


class TypeCode(object):
    """Type code used in API calls"""
    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    NULL = 4
    MXNET_TYPE = 5
    MXNET_CONTEXT = 6
    OBJECT_HANDLE = 7
    STR = 8
    BYTES = 9
    PYARG = 10
    NDARRAYHANDLE = 11
    EXT_BEGIN = 15


class MXNetValue(ctypes.Union):
    """MXNetValue in C API"""
    _fields_ = [("v_int64", ctypes.c_int64),
                ("v_float64", ctypes.c_double),
                ("v_handle", ctypes.c_void_p),
                ("v_str", ctypes.c_char_p),
                ("v_uint64", ctypes.c_uint64)]

RETURN_SWITCH = {
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.UINT: lambda x: x.v_uint64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.NULL: lambda x: None,
    TypeCode.STR: lambda x: py_str(x.v_str),
    TypeCode.NDARRAYHANDLE: lambda x: _global_var._np_ndarray_cls(handle=NDArrayHandle(x.v_handle)),
    TypeCode.HANDLE: lambda x: x.v_handle,
    TypeCode.PYARG: lambda x, args: args[x.v_int64],
}
