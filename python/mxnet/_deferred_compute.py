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
    check_call(_LIB.MXNDArrayIsDeferredCompute(ctypes.byref(curr)))
    return curr.value

def set_deferred_compute(state):
    """Enable / Disable deferred compute mode.

    Parameters
    ----------
    state: bool

    Returns
    -------
    Previous deferred compute state.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXNDArraySetIsDeferredCompute(ctypes.c_int(state), ctypes.byref(prev)))
    return bool(prev.value)


@contextlib.contextmanager
def context(state=True):
    """Set deferred compute state to `state` within context. Reset afterwards to previous value."""
    # Like other MXNet context manager, this bleeds state across concurrent
    # code: "Context managers that have state should use Context Variables
    # instead of threading.local() to prevent their state from bleeding to
    # other code unexpectedly, when used in concurrent code."
    # https://github.com/apache/incubator-mxnet/issues/17495#issuecomment-585461965
    val = set_deferred_compute(state)
    try:
        yield
    finally:
        set_deferred_compute(val)


def get_symbol(output_arrays, *, sym_cls=Symbol):
    """Get symbolic representation of computation recorded in deferred compute mode.

    Parameters
    ----------
    output_arrays: NDArray or List[NDArray]
    sym_cls: class used to construct Symbol

    Returns
    -------
    Symbol of sym_cls
    """
    output_arrays = _as_list(output_arrays)
    # Prepare ctypes array types
    output_handles_type = ctypes.c_void_p * len(output_arrays)
    # Convert handles
    output_handles = output_handles_type(*[array.handle for array in output_arrays])
    handle = SymbolHandle()
    check_call(_LIB.MXNDArrayGetDeferredComputeSymbol(output_handles, len(output_arrays),
                                                      ctypes.byref(handle)))
    return sym_cls(handle)


def set_variable(arrays, variables):
    """Associate variables with arrays.

    Parameters
    ----------
    arrays: NDArray or List[NDArray]
    variables: Symbol or List[Symbol] of variables
    """

    arrays = _as_list(arrays)
    variables = _as_list(variables)

    # Prepare ctypes array types
    arrays_type = variables_type = ctypes.c_void_p * len(arrays)

    # Convert handles
    arrays = arrays_type(*[array.handle for array in arrays])
    variables = variables_type(*[symbol.handle for symbol in variables])

    check_call(_LIB.MXNDArraySetDeferredComputeVariable(arrays, variables, len(arrays)))
