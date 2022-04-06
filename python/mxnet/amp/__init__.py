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
# pylint: disable=wildcard-import
"""Automatic mixed precision module."""

from .amp import *

import ctypes
from ..base import (_LIB, check_call)


def set_is_amp_disabled(value):
    prev = ctypes.c_int()
    check_call(_LIB.MXSetIsAMPDisabled(ctypes.c_int(value), ctypes.byref(prev)))
    return bool(prev.value)


def is_amp_disabled():
    curr = ctypes.c_int()
    check_call(_LIB.MXIsAMPDisabled(ctypes.byref(curr)))
    return bool(curr.value)


def disable_amp():
    return _DisableAMPScope()


class _DisableAMPScope(object):
    def __init__(self):  # pylint: disable=redefined-outer-name
        self._enter_value = None

    def __enter__(self):
        self._enter_value = set_is_amp_disabled(True)

    def __exit__(self, ptype, value, trace):
        set_is_amp_disabled(self._enter_value)
