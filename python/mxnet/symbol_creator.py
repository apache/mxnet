# coding: utf-8
# pylint: disable=invalid-name, protected-access, no-self-use
"""Symbol support of mxnet"""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import c_array, c_str, string_types
from .base import SymbolHandle
from .base import check_call
from .symbol import Symbol

class _SymbolCreator(object):
    """SymbolCreator is a function that takes Param and return symbol"""

    def __init__(self, name, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolCreatorHandle
            the function handle of the function

        name : string
            the name of the function
        """
        self.name = name
        self.handle = handle

    def __call__(self, *args, **kwargs):
        """Invoke creator of symbol by passing kwargs

        Parameters
        ----------
        name : string
            Name of the resulting symbol.

        *args
            Positional arguments

        **kwargs
            Provide the params necessary for the symbol creation.

        Returns
        -------
        the resulting symbol
        """
        param_keys = []
        param_vals = []
        symbol_kwargs = {}
        name = kwargs.pop('name', None)

        for k, v in kwargs.items():
            if isinstance(v, Symbol):
                symbol_kwargs[k] = v
            else:
                param_keys.append(k)
                param_vals.append(c_str(str(v)))

        # create atomic symbol
        param_keys = c_array(ctypes.c_char_p, param_keys)
        param_vals = c_array(ctypes.c_char_p, param_vals)
        sym_handle = SymbolHandle()
        check_call(_LIB.MXSymbolCreateFromAtomicSymbol( \
                self.handle, len(param_keys), \
                param_keys, param_vals, \
                ctypes.byref(sym_handle)))

        if len(args) != 0 and len(symbol_kwargs) != 0:
            raise TypeError('%s can only accept input \
                Symbols either as positional or keyword arguments, not both' % self.name)

        s = Symbol(sym_handle)
        s._compose(*args, name=name, **symbol_kwargs)
        return s

class _SymbolCreatorRegistry(object):
    """Function Registry"""
    def __init__(self):
        plist = ctypes.POINTER(ctypes.c_void_p)()
        size = ctypes.c_uint()
        check_call(_LIB.MXSymbolListAtomicSymbolCreators(ctypes.byref(size),
                                                         ctypes.byref(plist)))
        hmap = {}
        for i in range(size.value):
            name = ctypes.c_char_p()
            check_call(_LIB.MXSymbolGetAtomicSymbolName(plist[i], ctypes.byref(name)))
            hmap[name.value] = _SymbolCreator(name, plist[i])
        self.__dict__.update(hmap)

    def Variable(self, name):
        """Create a symbolic variable with specified name.

        Parameters
        ----------
        name : str
            Name of the variable.

        Returns
        -------
        variable : Symbol
            The created variable symbol.
        """
        if not isinstance(name, string_types):
            raise TypeError('Expect a string for variable `name`')
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolCreateVariable(name, ctypes.byref(handle)))
        return Symbol(handle)

    def Group(self, symbols):
        """Create a symbolic variable that groups several symbols together.

        Parameters
        ----------
        symbols : list
            List of symbols to be grouped.

        Returns
        -------
        sym : Symbol
            The created group symbol.
        """
        ihandles = []
        for sym in symbols:
            if not isinstance(sym, Symbol):
                raise TypeError('Expect Symbols in the list input')
            ihandles.append(sym.handle)
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolCreateGroup(
            len(ihandles), c_array(SymbolHandle, ihandles), ctypes.byref(handle)))
        return Symbol(handle)
