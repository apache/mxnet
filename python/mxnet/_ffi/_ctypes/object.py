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
# pylint: disable=invalid-name
"""
Runtime Object api
Acknowledgement: This file originates from incubator-tvm
"""
import ctypes
from ...base import _LIB, check_call
from . import function
from .types import RETURN_SWITCH, TypeCode

ObjectHandle = ctypes.c_void_p


def _return_object(x):
    handle = x.v_handle
    if not isinstance(handle, ObjectHandle):
        handle = ObjectHandle(handle)
    # Does not support specific cpp node class for now
    cls = function._CLASS_OBJECT
    # Avoid calling __init__ of cls, instead directly call __new__
    # This allows child class to implement their own __init__
    obj = cls.__new__(cls)
    obj.handle = handle
    return obj

RETURN_SWITCH[TypeCode.OBJECT_HANDLE] = _return_object


class ObjectBase(object):
    """Base object for all object types"""
    __slots__ = ["handle"]

    def __del__(self):
        if _LIB is not None:
            check_call(_LIB.MXNetObjectFree(self.handle))

    # Does not support creation of cpp node class via python class
