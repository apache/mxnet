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
# under the License

from __future__ import absolute_import as _abs

import sys as _sys
import ctypes as _ctypes
import numpy as np
from ..ndarray_doc import _build_doc
from libc.stdint cimport uint32_t, int64_t

include "./base.pyi"

cdef class NDArrayBase:
    """Symbol is symbolic graph."""
    # handle for symbolic operator.
    cdef NDArrayHandle chandle
    cdef int cwritable

    cdef _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = handle.value
            self.chandle = <SymbolHandle>(ptr)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return _ctypes.cast(<unsigned long long>self.chandle, _ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    property writable:
        def __get__(self):
            return bool(self.cwritable)

    def __init__(self, handle, writable=True):
        self._set_handle(handle)
        self.cwritable = writable

    def __dealloc__(self):
        CALL(MXNDArrayFree(self.chandle))

    def __reduce__(self):
        return (_ndarray_cls, (None,), self.__getstate__())


_ndarray_cls = None
_np_ndarray_cls = None

def _set_ndarray_class(cls):
    global _ndarray_cls
    _ndarray_cls = cls


def _set_np_ndarray_class(cls):
    global _np_ndarray_cls
    _np_ndarray_cls = cls


cdef NewArray(NDArrayHandle handle, int stype=-1, int is_np_array=0):
    """Create a new array given handle"""
    create_array_fn = _np_ndarray_cls if is_np_array else _ndarray_cls
    return create_array_fn(_ctypes.cast(<unsigned long long>handle, _ctypes.c_void_p), stype=stype)


cdef class CachedOp:
    """Cached operator handle."""
    cdef CachedOpHandle chandle
    cdef _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = handle.value
            self.chandle = <SymbolHandle>(ptr)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return _ctypes.cast(<unsigned long long>self.chandle, _ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    cdef int is_np_sym

    def __init__(self, sym, flags=()):
        cdef vector[string] s_flag_keys
        cdef vector[string] s_flag_vals
        if flags is not None:
            for k, v in flags:
                s_flag_keys.push_back(c_str(k))
                s_flag_vals.push_back(c_str(str(v)))
        cdef vector[const char*] c_flag_keys = SVec2Ptr(s_flag_keys)
        cdef vector[const char*] c_flag_vals = SVec2Ptr(s_flag_vals)

        from ..symbol.numpy._symbol import _Symbol
        self.is_np_sym = bool(isinstance(sym, _Symbol))

        CALL(MXCreateCachedOpEx(
            <SymbolHandle>(<unsigned long long>sym.handle.value),
            len(flags),
            CBeginPtr(c_flag_keys),
            CBeginPtr(c_flag_vals),
            &self.chandle))

    def __del__(self):
        CALL(MXFreeCachedOp(self.chandle))

    def __call__(self, *args, out=None):
        """ctypes implementation of imperative invoke wrapper"""
        cdef vector[NDArrayHandle] ndvars
        cdef vector[NDArrayHandle] output_vars
        cdef NDArrayHandle* p_output_vars
        cdef NDArrayHandle ret_handle
        cdef int num_output
        cdef const int* p_output_stypes

        for i in args:
            ndvars.push_back((<NDArrayBase>i).chandle)

        original_output = None
        if out is not None:
            original_output = out
            if isinstance(out, NDArrayBase):
                output_vars.push_back((<NDArrayBase>out).chandle)
            else:
                for i in out:
                    output_vars.push_back((<NDArrayBase>i).chandle)

        num_output = output_vars.size()
        if output_vars.size() == 0:
            p_output_vars = NULL
        else:
            p_output_vars = &output_vars[0]

        CALL(MXInvokeCachedOpEx(
            self.chandle,
            <int>len(args),
            &ndvars[0] if ndvars.size() != 0 else NULL,
            &num_output,
            &p_output_vars,
            &p_output_stypes))

        if original_output is not None:
            return original_output
        if num_output == 1:
            return NewArray(p_output_vars[0], p_output_stypes[0], self.is_np_sym)
        else:
            return [NewArray(p_output_vars[i], p_output_stypes[i], self.is_np_sym) for i in range(num_output)]


def _imperative_invoke(handle, ndargs, keys, vals, out, is_np_op=0):
    """cython implementation of imperative invoke wrapper"""
    cdef unsigned long long ihandle = handle
    cdef OpHandle chandle = <OpHandle>ihandle
    cdef vector[string] ckeys
    cdef vector[string] cvals
    cdef vector[NDArrayHandle] ndvars
    cdef vector[NDArrayHandle] output_vars
    cdef NDArrayHandle* p_output_vars
    cdef NDArrayHandle ret_handle
    cdef int num_output
    cdef const int* p_output_stypes

    for i in ndargs:
        ndvars.push_back((<NDArrayBase>i).chandle)
    for i in keys:
        ckeys.push_back(c_str(i))
    for i in vals:
        cvals.push_back(c_str(str(i)))

    original_output = None
    if out is not None:
        original_output = out
        if isinstance(out, NDArrayBase):
            output_vars.push_back((<NDArrayBase>out).chandle)
        else:
            for i in out:
                output_vars.push_back((<NDArrayBase>i).chandle)

    num_output = output_vars.size()
    if output_vars.size() == 0:
        p_output_vars = NULL
    else:
        p_output_vars = &output_vars[0]

    cdef vector[const char*] param_keys = SVec2Ptr(ckeys)
    cdef vector[const char*] param_vals = SVec2Ptr(cvals)

    CALL(MXImperativeInvokeEx(
        chandle,
        <int>ndvars.size(),
        &ndvars[0] if ndvars.size() != 0 else NULL,
        &num_output,
        &p_output_vars,
        <int>param_keys.size(),
        CBeginPtr(param_keys),
        CBeginPtr(param_vals),
        &p_output_stypes))

    if original_output is not None:
        return original_output
    if num_output == 1:
        return NewArray(p_output_vars[0], p_output_stypes[0], is_np_op)
    else:
        return [NewArray(p_output_vars[i], p_output_stypes[i], is_np_op) for i in range(num_output)]
