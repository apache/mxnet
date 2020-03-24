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
"""Runtime Object API
Acknowledgement: This file originates from incubator-tvm"""
import os
import ctypes
from ..base import _LIB, check_call, c_str

try:
    if int(os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from ._ctypes.function import _set_class_object
        from ._ctypes.object import ObjectBase as _ObjectBase
        from ._ctypes.object import _register_object
    else:
        from ._cy3.core import _set_class_object
        from ._cy3.core import ObjectBase as _ObjectBase
        from ._cy3.core import _register_object
except ImportError:
    if int(os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from ._ctypes.function import _set_class_object
    from ._ctypes.object import ObjectBase as _ObjectBase
    from ._ctypes.object import _register_object

class Object(_ObjectBase):
    """Base class for all mxnet's runtime objects."""


def register_object(type_key=None):
    """register object type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node

    Examples
    --------
    The following code registers MyObject
    using type key "test.MyObject"

    .. code-block:: python

      @register_object("test.MyObject")
      class MyObject(Object):
          pass
    """
    object_name = type_key if isinstance(type_key, str) else type_key.__name__

    def register(cls):
        """internal register function"""
        if hasattr(cls, "_type_index"):
            tindex = cls._type_index
        else:
            tidx = ctypes.c_uint()
            check_call(_LIB.MXNetObjectTypeKey2Index(
                c_str(object_name), ctypes.byref(tidx)))
            tindex = tidx.value
        _register_object(tindex, cls)
        return cls

    if isinstance(type_key, str):
        return register

    return register(type_key)


def getitem_helper(obj, elem_getter, length, idx):
    """Helper function to implement a pythonic getitem function.

    Parameters
    ----------
    obj: object
        The original object

    elem_getter : function
        A simple function that takes index and return a single element.

    length : int
        The size of the array

    idx : int or slice
        The argument passed to getitem

    Returns
    -------
    result : object
        The result of getitem
    """
    if isinstance(idx, slice):
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else length
        step = idx.step if idx.step is not None else 1
        if start < 0:
            start += length
        if stop < 0:
            stop += length
        return [elem_getter(obj, i) for i in range(start, stop, step)]

    if idx < -length or idx >= length:
        raise IndexError("Index out of range. size: {}, got index {}"
                         .format(length, idx))
    if idx < 0:
        idx += length
    return elem_getter(obj, idx)


_set_class_object(Object)
