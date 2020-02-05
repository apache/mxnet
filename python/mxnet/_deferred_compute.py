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
"""Deferred Compute for NDArray."""

import ctypes
import contextlib

from .base import _LIB, check_call, SymbolHandle, _as_list
from .symbol import Symbol

__all__ = []

def is_deferred_compute():
    """Get status of deferred compute mode."""
    curr = ctypes.c_bool()
    check_call(_LIB.MXNDArrayIsDeferredComputeEnabled(ctypes.byref(curr)))
    return curr.value

def set_deferred_compute(is_deferred_compute):
    """Enable / Disable deferred compute mode.

    Parameters
    ----------
    is_deferred_compute: bool

    Returns
    -------
    Previous deferred compute state.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXNDArraySetDeferredComputeEnabled(
        ctypes.c_int(is_deferred_compute), ctypes.byref(prev)))
    return bool(prev.value)


@contextlib.contextmanager
def context():
    # Like other MXNet context manager, this bleeds state across concurrent
    # code: "Context managers that have state should use Context Variables
    # instead of threading.local() to prevent their state from bleeding to
    # other code unexpectedly, when used in concurrent code."
    val = set_deferred_compute(True)
    yield
    set_deferred_compute(val)


def get_symbol(input_arrays, output_arrays, input_names=None):
    input_arrays = _as_list(input_arrays)
    output_arrays = _as_list(output_arrays)

    # Prepare ctypes array types
    input_handles_type = ctypes.c_void_p * len(input_arrays)
    output_handles_type = ctypes.c_void_p * len(output_arrays)
    input_names_type = ctypes.c_char_p * len(input_arrays)

    # Convert handles
    input_handles = input_handles_type(*[array.handle for array in input_arrays])
    output_handles = output_handles_type(*[array.handle for array in output_arrays])

    # Handle names arguments
    if input_names is None:
        if len(input_arrays) > 1:
            input_names = ['data{}'.format(cnt) for cnt in range(len(input_arrays))]
        elif len(input_arrays) == 1:
            input_names = ['data']
        else:
            input_names = []
    else:
        input_names = _as_list(input_names)
        assert len(input_names) == len(input_arrays), \
            'If input_names is specified, it must have equal length as input_arrays'
    # Convert names
    input_names = input_names_type(
        *[ctypes.c_char_p(ctypes.create_string_buffer(name.encode()).raw) for name in input_names])

    handle = SymbolHandle()
    check_call(
        _LIB.MXNDArrayGetDeferredComputeSymbol(input_handles, output_handles, input_names,
                                               len(input_arrays), len(output_arrays),
                                               ctypes.byref(handle)))
    return Symbol(handle)
