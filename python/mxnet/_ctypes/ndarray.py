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
# under the License.

# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments
# pylint: disable=global-statement, unused-import
"""NDArray configuration API."""
from __future__ import absolute_import as _abs

import ctypes

from ..base import _LIB
from ..base import c_str_array, c_handle_array
from ..base import NDArrayHandle, CachedOpHandle
from ..base import check_call


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
_np_ndarray_cls = None


def _set_ndarray_class(cls):
    """Set the symbolic class to be cls"""
    global _ndarray_cls
    _ndarray_cls = cls


def _set_np_ndarray_class(cls):
    """Set the symbolic class to be cls"""
    global _np_ndarray_cls
    _np_ndarray_cls = cls


def _imperative_invoke(handle, ndargs, keys, vals, out, is_np_op):
    """ctypes implementation of imperative invoke wrapper"""
    if out is not None:
        original_output = out
        if isinstance(out, NDArrayBase):
            out = (out,)
        num_output = ctypes.c_int(len(out))
        output_vars = c_handle_array(out)
        output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
    else:
        original_output = None
        output_vars = ctypes.POINTER(NDArrayHandle)()
        num_output = ctypes.c_int(0)

    # return output stypes to avoid the c_api call for checking
    # a handle's stype in _ndarray_cls
    out_stypes = ctypes.POINTER(ctypes.c_int)()

    check_call(_LIB.MXImperativeInvokeEx(
        ctypes.c_void_p(handle),
        ctypes.c_int(len(ndargs)),
        c_handle_array(ndargs),
        ctypes.byref(num_output),
        ctypes.byref(output_vars),
        ctypes.c_int(len(keys)),
        c_str_array(keys),
        c_str_array([str(s) for s in vals]),
        ctypes.byref(out_stypes)))

    create_ndarray_fn = _np_ndarray_cls if is_np_op else _ndarray_cls
    if original_output is not None:
        return original_output
    if num_output.value == 1:
        return create_ndarray_fn(ctypes.cast(output_vars[0], NDArrayHandle),
                                 stype=out_stypes[0])
    else:
        return [create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle),
                                  stype=out_stypes[i]) for i in range(num_output.value)]


class CachedOp(object):
    """Cached operator handle."""
    __slots__ = ["handle", "is_np_sym"]

    def __init__(self, sym, flags=()):
        self.handle = CachedOpHandle()

        from ..symbol.numpy._symbol import _Symbol
        self.is_np_sym = bool(isinstance(sym, _Symbol))

        check_call(_LIB.MXCreateCachedOpEx(
            sym.handle,
            len(flags),
            c_str_array([key for key, _ in flags]),
            c_str_array([str(val) for _, val in flags]),
            ctypes.byref(self.handle)))

    def __del__(self):
        check_call(_LIB.MXFreeCachedOp(self.handle))

    def _is_from_np_compat_op(self, idx):
        """Check if the CachedOp's idx-th output is directly from a numpy op."""
        is_from_np_op = ctypes.c_int(0)
        check_call(_LIB.MXIsCachedOpOutputFromNumpyCompatOp(self.handle, ctypes.c_int(idx),
                                                            ctypes.byref(is_from_np_op)))
        return is_from_np_op.value != 0

    def __call__(self, *args, **kwargs):
        """ctypes implementation of imperative invoke wrapper"""
        out = kwargs.pop('out', None)
        if out is not None:
            original_output = out
            if isinstance(out, NDArrayBase):
                out = (out,)
            num_output = ctypes.c_int(len(out))
            output_vars = c_handle_array(out)
            output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
        else:
            original_output = None
            output_vars = ctypes.POINTER(NDArrayHandle)()
            num_output = ctypes.c_int(0)
        if kwargs:
            raise TypeError(
                "CachedOp.__call__ got unexpected keyword argument(s): " + \
                ', '.join(kwargs.keys()))

        # return output stypes to avoid the c_api call for checking
        # a handle's stype in _ndarray_cls
        out_stypes = ctypes.POINTER(ctypes.c_int)()

        check_call(_LIB.MXInvokeCachedOpEx(
            self.handle,
            ctypes.c_int(len(args)),
            c_handle_array(args),
            ctypes.byref(num_output),
            ctypes.byref(output_vars),
            ctypes.byref(out_stypes)))

        if original_output is not None:
            return original_output
        create_ndarray_fn = _np_ndarray_cls if self.is_np_sym else _ndarray_cls
        if num_output.value == 1:
            return create_ndarray_fn(ctypes.cast(output_vars[0], NDArrayHandle),
                                     stype=out_stypes[0])
        else:
            return [create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle),
                                      stype=out_stypes[i]) for i in range(num_output.value)]
