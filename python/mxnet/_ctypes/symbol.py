# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments,  global-statement
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
from ..base import _LIB
from ..base import c_array, c_str, mx_uint
from ..base import SymbolHandle
from ..base import check_call

_symbol_cls = None

class SymbolBase(object):
    """Symbol is symbolic graph."""
    __slots__ = ["handle"]
    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        self.handle = handle

    def __del__(self):
        check_call(_LIB.NNSymbolFree(self.handle))

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
            if not isinstance(arg, SymbolBase):
                raise TypeError('Compose expect `Symbol` as arguments')
        for val in kwargs.values():
            if not isinstance(val, SymbolBase):
                raise TypeError('Compose expect `Symbol` as arguments')

        num_args = len(args) + len(kwargs)
        if len(kwargs) != 0:
            keys = c_array(ctypes.c_char_p, [c_str(key) for key in kwargs])
            args = c_array(SymbolHandle, [s.handle for s in kwargs.values()])
        else:
            keys = None
            args = c_array(SymbolHandle, [s.handle for s in args])
        check_call(_LIB.NNSymbolCompose(
            self.handle, name, num_args, keys, args))

    def _set_attr(self, **kwargs):
        """Set the attribute of the symbol.

        Parameters
        ----------
        **kwargs
            The attributes to set
        """
        keys = c_array(ctypes.c_char_p,
                       [c_str(key) for key in kwargs])
        vals = c_array(ctypes.c_char_p,
                       [c_str(str(val)) for val in kwargs.values()])
        num_args = mx_uint(len(kwargs))
        check_call(_LIB.MXSymbolSetAttrs(
            self.handle, num_args, keys, vals))

    def _set_handle(self, handle):
        """Set handle."""
        self.handle = handle

    def __reduce__(self):
        return (_symbol_cls, (None,), self.__getstate__())


def _set_symbol_class(cls):
    """Set the symbolic class to be cls"""
    global _symbol_cls
    _symbol_cls = cls


def _symbol_creator(handle, args, kwargs, keys, vals, name):
    sym_handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateAtomicSymbol(
        ctypes.c_void_p(handle),
        mx_uint(len(keys)),
        c_array(ctypes.c_char_p, [c_str(i) for i in keys]),
        c_array(ctypes.c_char_p, [c_str(str(i)) for i in vals]),
        ctypes.byref(sym_handle)))

    if args and kwargs:
        raise TypeError(
            'Operators with variable length input can only accept input'
            'Symbols either as positional or keyword arguments, not both')
    s = _symbol_cls(sym_handle)
    if args:
        s._compose(*args, name=name)
    elif kwargs:
        s._compose(name=name, **kwargs)
    else:
        s._compose(name=name)
    return s
