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
"""Library management API of mxnet."""
import ctypes
import sys
import os
from .base import _LIB, check_call, MXNetError, _init_op_module, mx_uint
from .ndarray.register import _make_ndarray_function
from .symbol.register import _make_symbol_function

class MXlib:
    """Holds a pointed to a loaded shared library and closes it on destruction"""
    def __init__(self, handle):
        self.handle = handle
    def __del__(self):
        libdl = ctypes.CDLL("libdl.so")
        libdl.dlclose(self.handle)

# set of libraries loaded
loaded_libs = []

def load(path, verbose=True):
    """Loads library dynamically.

    Parameters
    ---------
    path : string
        Path to library .so/.dll file

    verbose : boolean
        defaults to True, set to False to avoid printing library info

    Returns
    ---------
    void
    """
    global loaded_libs

    #check if path exists
    if not os.path.exists(path):
        raise MXNetError(f"load path {path} does NOT exist")
    #check if path is an absolute path
    if not os.path.isabs(path):
        raise MXNetError(f"load path {path} is not an absolute path")
    #check if path is to a library file
    _, file_ext = os.path.splitext(path)
    if not file_ext in ['.so', '.dll']:
        raise MXNetError(f"load path {path} is NOT a library file")

    verbose_val = 1 if verbose else 0
    byt_obj = path.encode('utf-8')
    chararr = ctypes.c_char_p(byt_obj)
    lib_ptr = ctypes.c_void_p(0)
    check_call(_LIB.MXLoadLib(chararr, mx_uint(verbose_val), ctypes.byref(lib_ptr)))
    # add library pointer to list so it can be closed later
    loaded_libs.append(MXlib(lib_ptr))

    #regenerate operators
    _init_op_module('mxnet', 'ndarray', _make_ndarray_function)
    _init_op_module('mxnet', 'symbol', _make_symbol_function)

    #re-register mx.nd.op into mx.nd
    mx_nd = sys.modules["mxnet.ndarray"]
    mx_nd_op = sys.modules["mxnet.ndarray.op"]
    for op in dir(mx_nd_op):
        func = getattr(mx_nd_op, op)
        setattr(mx_nd, op, func)

    #re-register mx.sym.op into mx.sym
    mx_sym = sys.modules["mxnet.symbol"]
    mx_sym_op = sys.modules["mxnet.symbol.op"]
    for op in dir(mx_sym_op):
        func = getattr(mx_sym_op, op)
        setattr(mx_sym, op, func)

def compiled_with_gcc_cxx11_abi():
    """Check if the library is compiled with _GLIBCXX_USE_CXX11_ABI.

    Please see
    https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html for
    more information. When building libraries relying on MXNet C++ headers, it
    is required to use the same C++ ABI in the library as well as in libmxnet.

    Returns
    -------
    int
        1 If compiled with _GLIBCXX_USE_CXX11_ABI=1
        0 If compiled with _GLIBCXX_USE_CXX11_ABI=0
       -1 If compiled with a compiler that does not support _GLIBCXX_USE_CXX11_ABI

    """
    ret = ctypes.c_int()
    check_call(_LIB.MXLibInfoCompiledWithCXX11ABI(ctypes.byref(ret)))
    return ret.value
