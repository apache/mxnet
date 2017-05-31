# coding: utf-8
"""Common code between symbolic and ndarray."""
from __future__ import absolute_import as _abs

import ctypes

from ..base import _LIB
from ..base import c_array, c_str
from ..base import OpHandle, CachedOpHandle
from ..base import check_call


class CachedOp(object):
    """Cached operator handle."""
    __slots__ = ["handle", "op"]
    def __init__(self, op, num_input, **kwargs):
        self.op = op
        op_handle = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(op), ctypes.byref(op_handle)))
        self.handle = CachedOpHandle()
        check_call(_LIB.MXCachedCreateOp(
            op_handle,
            ctypes.c_int(num_input),
            ctypes.c_int(len(kwargs)),
            c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()]),
            c_array(ctypes.c_char_p, [c_str(str(val)) for val in kwargs.values()]),
            ctypes.byref(self.handle)))

    def __del__(self):
        check_call(_LIB.MXCachedFree(self.handle))
