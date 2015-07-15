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

    def __init__(self, handle, name):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolCreatorHandle
            the function handle of the function

        name : string
            the name of the function
        """
        self.handle = handle
        self.name = name
        use_param = mx_uint()
        check_call(_LIB.MXSymCreatorDescribe(
            self.handle,
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
        check_call(_LIB.MXSymCreatorInvoke(
            self.handle,
            mx_uint(len(kwargs)),
            keys,
            vals,
            ctypes.byref(sym_handle)))
        return Symbol(sym_handle)

class _SymbolCreatorRegistry(object):
    """Function Registry"""
    def __init__(self):
        plist = ctypes.POINTER(ctypes.c_void_p)()
        size = ctypes.c_uint()
        check_call(_LIB.MXListSymCreators(ctypes.byref(size),
                                          ctypes.byref(plist)))
        hmap = {}
        for i in range(size.value):
            hdl = plist[i]
            name = ctypes.c_char_p()
            check_call(_LIB.MXSymCreatorGetName(hdl, ctypes.byref(name)))
            hmap[name.value] = _SymbolCreator(hdl, name.value)
        self.__dict__.update(hmap)
