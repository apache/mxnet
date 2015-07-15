# coding: utf-8
"""Symbol support of mxnet"""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import c_array
from .base import mx_uint, mx_float, SymbolHandle
from .base import check_call, MXNetError
from .narray import NArray, _new_empty_handle

class Symbol(object):
    """SymbolCreator is a function that takes Param and return symbol"""
    _registry = None

    @staticmethod
    def _init_symbol_creator_registry(symbol_creator_registry):
        _registry = symbol_creator_registry
        return _registry

    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        self.handle = handle

    def __call__(self, *args, **kwargs):
        """Compose Symbols

        Parameters
        ----------
        args:
            provide positional arguments
        kwargs:
            provide keyword arguments
        Returns
        -------
        the resulting symbol
        """
        assert (len(args) == 0 or len(kwargs) == 0)
        for arg in args:
            assert isinstance(arg, Symbol)
        for key, val in kwargs:
            assert isinstance(val, Symbol)
        num_args = len(args) + len(kwargs)
        if len(kwargs) != 0:
            keys = c_array(ctypes.c_char_p, map(c_str, kwargs.keys()))
            args = c_array(SymbolHandle, kwargs.values())
        else:
            keys = None
            args = c_array(SymbolHandle, args)

        out = SymbolHandle()
        check_call(_LIB.MXSymbolCompose(
            self.handle,
            num_args,
            keys,
            args,
            ctypes.byref(out)))
        return Symbol(out)
