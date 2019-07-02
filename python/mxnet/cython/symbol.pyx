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
import numpy as _numpy

from numbers import Number as _Number
from ..name import NameManager
from ..attribute import AttrScope
from ..symbol_doc import _build_doc

include "./base.pyi"

cdef class SymbolBase:
    """Symbol is symbolic graph."""
    # handle for symbolic operator.
    cdef SymbolHandle chandle

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

    def __init__(self, handle):
        self._set_handle(handle)

    def __dealloc__(self):
        CALL(NNSymbolFree(self.chandle))

    def _set_attr(self, **kwargs):
        """Set the attribute of the symbol.

        Parameters
        ----------
        **kwargs
            The attributes to set
        """
        SymbolSetAttr(self.chandle, kwargs)

    def __reduce__(self):
        return (_symbol_cls, (None,), self.__getstate__())


cdef SymbolSetAttr(SymbolHandle handle, dict kwargs):
    cdef string sparam_key
    cdef string sparam_val
    cdef const char* param_key
    cdef const char* param_val
    for k, v in kwargs.items():
        sparam_key = c_str(k)
        sparam_val = c_str(str(v))
        param_key = sparam_key.c_str()
        param_val = sparam_val.c_str()
        CALL(MXSymbolSetAttr(handle, param_key, param_val))


_symbol_cls = SymbolBase
_np_symbol_cls = None

def _set_symbol_class(cls):
    global _symbol_cls
    _symbol_cls = cls


def _set_np_symbol_class(cls):
    global _np_symbol_cls
    _np_symbol_cls = cls


cdef NewSymbol(SymbolHandle handle, int is_np_sym=0):
    """Create a new symbol given handle"""
    create_symbol_fn = _np_symbol_cls if is_np_sym else _symbol_cls
    sym = create_symbol_fn(None)
    (<SymbolBase>sym).chandle = handle
    return sym


def _symbol_creator(handle, args, kwargs, keys, vals, name, is_np_op=0):
    cdef unsigned long long ihandle = handle
    cdef OpHandle chandle = <OpHandle>ihandle
    cdef vector[string] ckeys
    cdef vector[string] cvals
    cdef vector[string] sym_keys
    cdef vector[SymbolHandle] sym_args
    cdef SymbolHandle ret_handle
    cdef string cname = c_str(name)

    for i in keys:
        ckeys.push_back(c_str(i))
    for i in vals:
        cvals.push_back(c_str(str(i)))

    cdef vector[const char*] param_keys = SVec2Ptr(ckeys)
    cdef vector[const char*] param_vals = SVec2Ptr(cvals)

    CALL(MXSymbolCreateAtomicSymbol(
        chandle,
        <nn_uint>param_keys.size(),
        CBeginPtr(param_keys),
        CBeginPtr(param_vals),
        &ret_handle))

    if args and kwargs:
        raise TypeError(
            'Operators with variable length input can only accept input'
            'Symbols either as positional or keyword arguments, not both')

    if args:
        for i in args:
            sym_args.push_back((<SymbolBase>i).chandle)
    elif kwargs:
        for k, v in kwargs.items():
            sym_keys.push_back(c_str(k))
            sym_args.push_back((<SymbolBase>v).chandle)

    cdef vector[const char*] csym_keys = SVec2Ptr(sym_keys)

    CALL(NNSymbolCompose(
        ret_handle,
        cname.c_str(),
        <nn_uint>sym_args.size(),
        &csym_keys[0] if csym_keys.size() != 0 else NULL,
        &sym_args[0] if sym_args.size() != 0 else NULL))

    return NewSymbol(ret_handle, is_np_op)
