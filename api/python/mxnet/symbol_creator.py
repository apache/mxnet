# coding: utf-8
"""Symbol support of mxnet"""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import c_array, c_str
from .base import mx_uint, SymbolHandle
from .base import check_call
from .symbol import Symbol

class _SymbolCreator(object):
    """SymbolCreator is a function that takes Param and return symbol"""

    def __init__(self, name):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolCreatorHandle
            the function handle of the function

        name : string
            the name of the function
        """
        self.name = name
        use_param = mx_uint()
        check_call(_LIB.MXSymDescribe(
            c_str(self.name),
            ctypes.byref(use_param)))
        self.use_param = use_param.value

    def __call__(self, **kwargs):
        """Invoke creator of symbol by passing kwargs

        Parameters
        ----------
        params : kwargs
            provide the params necessary for the symbol creation
        Returns
        -------
        the resulting symbol
        """
        keys = c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()])
        vals = c_array(ctypes.c_char_p, [c_str(str(val)) for val in kwargs.values()])
        sym_handle = SymbolHandle()
        check_call(_LIB.MXSymCreate(
            c_str(self.name),
            mx_uint(len(kwargs)),
            keys,
            vals,
            ctypes.byref(sym_handle)))
        return Symbol(sym_handle)

class _SymbolCreatorRegistry(object):
    """Function Registry"""
    def __init__(self):
        plist = ctypes.POINTER(ctypes.c_char_p)()
        size = ctypes.c_uint()
        check_call(_LIB.MXListSyms(ctypes.byref(size),
                                   ctypes.byref(plist)))
        hmap = {}
        for i in range(size.value):
            name = plist[i]
            hmap[name.value] = _SymbolCreator(name.value)
        self.__dict__.update(hmap)
