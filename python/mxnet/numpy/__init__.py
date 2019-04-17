#!/usr/bin/env python

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

import ctypes
from ..base import _LIB, check_call

__all__ = []


def set_np_comp(is_np_comp):
    prev = ctypes.c_int()
    check_call(_LIB.MXSetIsNumpyCompatible(ctypes.c_int(is_np_comp), ctypes.byref(prev)))
    return bool(prev.value)


def is_np_comp():
    curr = ctypes.c_bool()
    check_call(_LIB.MXIsNumpyCompatible(ctypes.byref(curr)))
    return curr.value


class _NumpyCompatibilityStateScope(object):
    """Scope for managing numpy compatibility state.

    Example::

        with _NumpyCompatibilityStateScope(True):
            y = model(x)
            backward([y])

    """
    def __init__(self, is_np_comp): #pylint: disable=redefined-outer-name
        self._enter_is_np_comp = is_np_comp
        self._prev_is_np_comp = None

    def __enter__(self):
        if self._enter_is_np_comp is not None:
            self._prev_is_np_comp = set_np_comp(self._enter_is_np_comp)

    def __exit__(self, ptype, value, trace):
        if self._enter_is_np_comp is not None and self._prev_is_np_comp != self._enter_is_np_comp:
            set_np_comp(self._prev_is_np_comp)


def enable_np_comp():
    return _NumpyCompatibilityStateScope(True)


def disable_np_comp():
    return _NumpyCompatibilityStateScope(False)
