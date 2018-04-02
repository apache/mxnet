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
"""Engine properties management."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call


def set_bulk_size(size):
    """Set size limit on bulk execution.

    Bulk execution bundles many operators to run together.
    This can improve performance when running a lot of small
    operators sequentially.

    Parameters
    ----------
    size : int
        Maximum number of operators that can be bundled in a bulk.

    Returns
    -------
    int
        Previous bulk size.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXEngineSetBulkSize(
        ctypes.c_int(size), ctypes.byref(prev)))
    return prev.value


class _BulkScope(object):
    """Scope object for bulk execution."""
    def __init__(self, size):
        self._size = size
        self._old_size = None

    def __enter__(self):
        self._old_size = set_bulk_size(self._size)
        return self

    def __exit__(self, ptype, value, trace):
        set_bulk_size(self._old_size)


def bulk(size):
    """Bulk execution bundles many operators to run together.
    This can improve performance when running a lot of small
    operators sequentially.

    Returns a scope for managing bulk size::

        with mx.engine.bulk(10):
            x = mx.nd.zeros((1,))
            for i in range(100):
                x += 1
    """
    return _BulkScope(size)
