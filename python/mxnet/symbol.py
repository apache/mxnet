# coding: utf-8
# pylint: disable=invalid-name, protected-access
"""Symbol support of mxnet"""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import c_array, c_str
from .base import SymbolHandle
from .base import check_call

class Symbol(object):
    """SymbolCreator is a function that takes Param and return symbol"""
    _registry = None

    @staticmethod
    def _init_symbol_creator_registry(symbol_creator_registry):
        """Initialize symbol creator registry

        Parameters
        ----------
        symbol_creator_registry:
            pass in symbol_creator_registry
        Returns
        -------
        the passed in registry
        """
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

    def __del__(self):
        check_call(_LIB.MXSymbolFree(self.handle))

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self):
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolCopy(self.handle,
                                     ctypes.byref(handle)))
        return Symbol(handle)

    def __call__(self, *args, **kwargs):
        """Invoke symbol as function on inputs.

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
        s = self.__deepcopy__()
        s._compose(*args, **kwargs)
        return s

    def _compose(self, *args, **kwargs):
        """Compose symbol on inputs.

        This call mutates the current symbol.

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
        name = kwargs.pop('name', None)
        if name:
            name = c_str(name)
        if len(args) != 0 and len(kwargs) != 0:
            raise TypeError('compose only accept input Symbols \
                either as positional or keyword arguments, not both')

        for arg in args:
            if not isinstance(arg, Symbol):
                raise TypeError('Compose expect `Symbol` as arguments')
        for _, val in kwargs.items():
            if not isinstance(val, Symbol):
                raise TypeError('Compose expect `Symbol` as arguments')

        num_args = len(args) + len(kwargs)
        if len(kwargs) != 0:
            keys = c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()])
            args = c_array(SymbolHandle, [s.handle for s in kwargs.values()])
        else:
            keys = None
            args = c_array(SymbolHandle, [s.handle for s in args])
        check_call(_LIB.MXSymbolCompose( \
                self.handle, name, num_args, keys, args))

    def list_arguments(self):
        """List all the arguments in the symbol.

        Returns
        -------
        args : list of string
            List of all the arguments.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListArguments( \
                self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [sarr[i] for i in range(size.value)]

    def list_returns(self):
        """List all returns in the symbol.

        Returns
        -------
        args: list of string
            List of all the returns.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListReturns( \
                self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [sarr[i] for i in range(size.value)]

    def debug_str(self):
        """Get a debug string.

        Returns
        -------
        debug_str : string
            Debug string of the symbol.
        """
        debug_str = ctypes.c_char_p()
        check_call(_LIB.MXSymbolPrint( \
                self.handle, ctypes.byref(debug_str)))
        return debug_str.value
