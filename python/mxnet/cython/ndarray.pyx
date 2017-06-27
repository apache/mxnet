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


_ndarray_cls = NDArrayBase

def _set_ndarray_class(cls):
    global _ndarray_cls
    _ndarray_cls = cls


cdef NewArray(NDArrayHandle handle):
    """Create a new array given handle"""
    nd = _ndarray_cls(None)
    (<NDArrayBase>nd).chandle = handle
    (<NDArrayBase>nd).cwritable = True
    return nd


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

    def __init__(self, sym):
        cdef unsigned long long ptr = sym.handle.value
        CALL(MXCreateCachedOp(
            (<SymbolHandle>ptr),
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
            output_vars.resize(1)
            p_output_vars = NULL
        else:
            p_output_vars = &output_vars[0]

        CALL(MXInvokeCachedOp(
            (<CachedOp>self).chandle,
            <int>len(args),
            &ndvars[0] if ndvars.size() != 0 else NULL,
            &num_output,
            &p_output_vars))

        if original_output is not None:
            return original_output
        if num_output == 1:
            return NewArray(p_output_vars[0])
        else:
            return tuple(NewArray(p_output_vars[i]) for i in range(num_output))


def _imperative_invoke(handle, ndargs, keys, vals, out):
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
        output_vars.resize(1)
        p_output_vars = NULL
    else:
        p_output_vars = &output_vars[0]

    cdef vector[const char*] param_keys = SVec2Ptr(ckeys)
    cdef vector[const char*] param_vals = SVec2Ptr(cvals)

    CALL(MXImperativeInvoke(
        chandle,
        <int>ndvars.size(),
        &ndvars[0] if ndvars.size() != 0 else NULL,
        &num_output,
        &p_output_vars,
        <int>param_keys.size(),
        CBeginPtr(param_keys),
        CBeginPtr(param_vals)))

    if original_output is not None:
        return original_output
    if num_output == 1:
        return NewArray(p_output_vars[0])
    else:
        return tuple(NewArray(p_output_vars[i]) for i in range(num_output))
