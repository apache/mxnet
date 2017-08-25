# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments
# pylint: disable=global-statement, unused-import
"""NDArray configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import sys as _sys
import numpy as np

from ..base import _LIB
from ..base import c_array, py_str, c_str, mx_uint, _Null
from ..base import NDArrayHandle, OpHandle, CachedOpHandle
from ..base import check_call
from ..ndarray_doc import _build_doc


class NDArrayBase(object):
    """Base data structure for ndarray"""
    __slots__ = ["handle", "writable"]
    # pylint: disable= no-member
    def __init__(self, handle, writable=True):
        """initialize a new NDArray

        Parameters
        ----------
        handle : NDArrayHandle
            NDArray handle of C API
        """
        if handle is not None:
            assert isinstance(handle, NDArrayHandle)
        self.handle = handle
        self.writable = writable

    def __del__(self):
        check_call(_LIB.MXNDArrayFree(self.handle))

    def __reduce__(self):
        return (_ndarray_cls, (None,), self.__getstate__())


_ndarray_cls = None

def _set_ndarray_class(cls):
    """Set the symbolic class to be cls"""
    global _ndarray_cls
    _ndarray_cls = cls


def _imperative_invoke(handle, ndargs, keys, vals, out):
    """ctypes implementation of imperative invoke wrapper"""
    if out is not None:
        original_output = out
        if isinstance(out, NDArrayBase):
            out = (out,)
        num_output = ctypes.c_int(len(out))
        output_vars = c_array(NDArrayHandle, [i.handle for i in out])
        output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
    else:
        original_output = None
        output_vars = ctypes.POINTER(NDArrayHandle)()
        num_output = ctypes.c_int(0)

    check_call(_LIB.MXImperativeInvoke(
        ctypes.c_void_p(handle),
        ctypes.c_int(len(ndargs)),
        c_array(NDArrayHandle, [arr.handle for arr in ndargs]),
        ctypes.byref(num_output),
        ctypes.byref(output_vars),
        ctypes.c_int(len(keys)),
        c_array(ctypes.c_char_p, [c_str(key) for key in keys]),
        c_array(ctypes.c_char_p, [c_str(str(val)) for val in vals])))

    if original_output is not None:
        return original_output
    if num_output.value == 1:
        return _ndarray_cls(ctypes.cast(output_vars[0], NDArrayHandle))
    else:
        return [_ndarray_cls(ctypes.cast(output_vars[i], NDArrayHandle))
                for i in range(num_output.value)]


class CachedOp(object):
    """Cached operator handle."""
    __slots__ = ["handle"]
    def __init__(self, sym):
        self.handle = CachedOpHandle()
        check_call(_LIB.MXCreateCachedOp(
            sym.handle,
            ctypes.byref(self.handle)))

    def __del__(self):
        check_call(_LIB.MXFreeCachedOp(self.handle))

    def __call__(self, *args, **kwargs):
        """ctypes implementation of imperative invoke wrapper"""
        out = kwargs.pop('out', None)
        if out is not None:
            original_output = out
            if isinstance(out, NDArrayBase):
                out = (out,)
            num_output = ctypes.c_int(len(out))
            output_vars = c_array(NDArrayHandle, [i.handle for i in out])
            output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
        else:
            original_output = None
            output_vars = ctypes.POINTER(NDArrayHandle)()
            num_output = ctypes.c_int(0)
        if kwargs:
            raise TypeError(
                "CachedOp.__call__ got unexpected keyword argument(s): " + \
                ', '.join(kwargs.keys()))

        check_call(_LIB.MXInvokeCachedOp(
            self.handle,
            ctypes.c_int(len(args)),
            c_array(NDArrayHandle, [arr.handle for arr in args]),
            ctypes.byref(num_output),
            ctypes.byref(output_vars)))

        if original_output is not None:
            return original_output
        if num_output.value == 1:
            return _ndarray_cls(ctypes.cast(output_vars[0], NDArrayHandle))
        else:
            return [_ndarray_cls(ctypes.cast(output_vars[i], NDArrayHandle))
                    for i in range(num_output.value)]
