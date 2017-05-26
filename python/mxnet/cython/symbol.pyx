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

cdef _set_symbol_class(cls):
    global _symbol_cls
    _symbol_cls = cls

cdef NewSymbol(SymbolHandle handle):
    """Create a new symbol given handle"""
    sym = _symbol_cls(None)
    (<SymbolBase>sym).chandle = handle
    return sym

cdef _make_atomic_symbol_function(OpHandle handle, string name):
    """Create an atomic symbol function by handle and funciton name."""
    cdef const char *real_name
    cdef const char *desc
    cdef nn_uint num_args
    cdef const char** arg_names
    cdef const char** arg_types
    cdef const char** arg_descs
    cdef const char* return_type
    cdef const char* key_var_num_args

    CALL(MXSymbolGetAtomicSymbolInfo(
        handle, &real_name, &desc,
        &num_args, &arg_names,
        &arg_types, &arg_descs,
        &key_var_num_args, &return_type))
    func_name = py_str(name.c_str())

    key_vargs = py_str(key_var_num_args)
    num_args = int(num_args)
    doc_str = _build_doc(func_name,
                         py_str(desc),
                         [py_str(arg_names[i]) for i in range(num_args)],
                         [py_str(arg_types[i]) for i in range(num_args)],
                         [py_str(arg_descs[i]) for i in range(num_args)],
                         key_vargs,
                         py_str(return_type) if return_type != NULL else '')

    func_hint = func_name.lower()

    def creator(*args, **kwargs):
        cdef vector[string] sparam_keys
        cdef vector[string] sparam_vals
        cdef vector[SymbolHandle] symbol_args
        cdef vector[string] ssymbol_keys
        cdef SymbolHandle ret_handle
        attr = kwargs.pop("attr", None)
        kwargs.update(AttrScope.current.get(attr))
        name = kwargs.pop("name", None)

        if key_vargs:
            if key_vargs not in kwargs:
                sparam_keys.push_back(c_str(key_vargs))
                sparam_vals.push_back(c_str(str(len(args))))

        if len(kwargs) != 0:
            for k, v in kwargs.items():
                if isinstance(v, SymbolBase):
                    ssymbol_keys.push_back(c_str(k))
                    symbol_args.push_back((<SymbolBase>v).chandle)
                elif k == 'dtype':
                    sparam_keys.push_back(c_str(k))
                    sparam_vals.push_back(c_str(_numpy.dtype(v).name))
                else:
                    sparam_keys.push_back(c_str(k))
                    sparam_vals.push_back(c_str(str(v)))

        if len(args) != 0:
            if symbol_args.size() != 0:
                raise TypeError("compose only accept input Symbols\
                    either as positional or keyword arguments, not both")
            for v in args:
                if not isinstance(v, SymbolBase):
                    raise TypeError('Compose expect `Symbol` as arguments')
                symbol_args.push_back((<SymbolBase>v).chandle)

        cdef vector[const char*] param_keys = SVec2Ptr(sparam_keys)
        cdef vector[const char*] param_vals = SVec2Ptr(sparam_vals)
        cdef vector[const char*] symbol_keys = SVec2Ptr(ssymbol_keys)

        CALL(MXSymbolCreateAtomicSymbol(
            handle,
            <nn_uint>param_keys.size(),
            CBeginPtr(param_keys),
            CBeginPtr(param_vals),
            &ret_handle))
        num_args = <nn_uint>(symbol_args.size())

        name = NameManager.current.get(name, func_hint)

        cdef const char* c_name = NULL

        if name:
            name = c_str(name)
            c_name = name

        CALL(NNSymbolCompose(
            ret_handle,
            c_name,
            num_args,
            &symbol_keys[0] if symbol_keys.size() != 0 else NULL,
            &symbol_args[0] if symbol_args.size() != 0 else NULL))
        return NewSymbol(ret_handle)

    creator.__name__ = func_name
    creator.__doc__ = doc_str
    creator.__module__ = 'mxnet.symbol'
    return creator


def _init_symbol_module(symbol_class, root_namespace):
    """List and add all the atomic symbol functions to current module."""
    cdef const char** op_name_ptrs
    cdef nn_uint size
    cdef vector[string] op_names
    cdef OpHandle handle

    _set_symbol_class(symbol_class)
    CALL(MXListAllOpNames(&size, &op_name_ptrs))
    for i in range(size):
        op_names.push_back(string(op_name_ptrs[i]))

    module_obj = _sys.modules["%s.symbol" % root_namespace]
    module_internal = _sys.modules["%s._symbol_internal" % root_namespace]
    module_contrib = _sys.modules["%s.contrib.symbol" % root_namespace]
    for i in range(op_names.size()):
        CALL(NNGetOpHandle(op_names[i].c_str(), &handle))
        function = _make_atomic_symbol_function(handle, op_names[i])
        if function.__name__.startswith('_contrib_'):
            function.__name__ = function.__name__[9:]
            function.__module__ = 'mxnet.contrib.symbol'
            setattr(module_contrib, function.__name__, function)
        elif function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)
