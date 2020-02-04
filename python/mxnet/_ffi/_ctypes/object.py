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
"""Runtime Object api"""
from __future__ import absolute_import
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
    # tindex = ctypes.c_uint()
    # check_call(_LIB.TVMObjectGetTypeIndex(handle, ctypes.byref(tindex)))
    # cls = OBJECT_TYPE.get(tindex.value, _CLASS_NODE)
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
    # def __init_handle_by_constructor__(self, fconstructor, *args):
    #     """Initialize the handle by calling constructor function.

    #     Parameters
    #     ----------
    #     fconstructor : Function
    #         Constructor function.

    #     args: list of objects
    #         The arguments to the constructor

    #     Note
    #     ----
    #     We have a special calling convention to call constructor functions.
    #     So the return handle is directly set into the Node object
    #     instead of creating a new Node.
    #     """
    #     # assign handle first to avoid error raising
    #     self.handle = None
    #     handle = __init_by_constructor__(fconstructor, args)
    #     if not isinstance(handle, ObjectHandle):
    #         handle = ObjectHandle(handle)
    #     self.handle = handle
