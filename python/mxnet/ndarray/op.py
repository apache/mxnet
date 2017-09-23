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

"""Register backend ops in mxnet.ndarray namespace"""
__all__ = ['CachedOp']

import sys as _sys
import os as _os
import ctypes
import numpy as np  # pylint: disable=unused-import

from ..ndarray_doc import _build_doc

# Use different version of SymbolBase
# When possible, use cython to speedup part of computation.
# pylint: disable=unused-import
try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from .._ctypes.ndarray import NDArrayBase
        from .._ctypes.ndarray import CachedOp, _imperative_invoke
    elif _sys.version_info >= (3, 0):
        from .._cy3.ndarray import NDArrayBase, _imperative_invoke
        from .._cy3.ndarray import CachedOp, _imperative_invoke
    else:
        from .._cy2.ndarray import NDArrayBase, _imperative_invoke
        from .._cy2.ndarray import CachedOp, _imperative_invoke
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from .._ctypes.ndarray import NDArrayBase, _imperative_invoke
    from .._ctypes.ndarray import CachedOp, _imperative_invoke

from ..base import mx_uint, check_call, _LIB, py_str, _init_op_module, _Null
# pylint: enable=unused-import


# pylint: disable=too-many-locals, invalid-name
def _make_ndarray_function(handle, name):
    """Create a NDArray function from the FunctionHandle."""
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    key_var_num_args = ctypes.c_char_p()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
        handle, ctypes.byref(real_name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(key_var_num_args),
        ctypes.byref(ret_type)))
    narg = int(num_args.value)
    arg_names = [py_str(arg_names[i]) for i in range(narg)]
    arg_types = [py_str(arg_types[i]) for i in range(narg)]
    func_name = name
    key_var_num_args = py_str(key_var_num_args.value)
    ret_type = py_str(ret_type.value) if ret_type.value is not None else ''
    doc_str = _build_doc(func_name,
                         py_str(desc.value),
                         arg_names,
                         arg_types,
                         [py_str(arg_descs[i]) for i in range(narg)],
                         key_var_num_args,
                         ret_type)

    dtype_name = None
    arr_name = None
    ndsignature = []
    signature = []
    ndarg_names = []
    kwarg_names = []
    for i in range(narg):
        name, atype = arg_names[i], arg_types[i]
        if name == 'dtype':
            dtype_name = name
            signature.append('%s=_Null'%name)
        elif atype.startswith('NDArray') or atype.startswith('Symbol'):
            assert not arr_name, \
                "Op can only have one argument with variable " \
                "size and it must be the last argument."
            if atype.endswith('[]'):
                ndsignature.append('*%s'%name)
                arr_name = name
            else:
                ndsignature.append('%s=None'%name)
                ndarg_names.append(name)
        else:
            signature.append('%s=_Null'%name)
            kwarg_names.append(name)
    signature.append('out=None')
    signature.append('name=None')
    signature.append('**kwargs')
    signature = ndsignature + signature

    code = []
    if arr_name:
        code.append("""
def %s(*%s, **kwargs):"""%(func_name, arr_name))
        code.append("""
    ndargs = []
    for i in {}:
        assert isinstance(i, NDArrayBase), \\
            "Positional arguments must have NDArray type, " \\
            "but got %s"%str(i)
        ndargs.append(i)""".format(arr_name))
        if dtype_name is not None:
            code.append("""
    if '%s' in kwargs:
        kwargs['%s'] = np.dtype(kwargs['%s']).name"""%(
            dtype_name, dtype_name, dtype_name))
        code.append("""
    _ = kwargs.pop('name', None)
    out = kwargs.pop('out', None)
    keys = list(kwargs.keys())
    vals = list(kwargs.values())""")
    else:
        code.append("""
def %s(%s):
    ndargs = []
    keys = list(kwargs.keys())
    vals = list(kwargs.values())"""%(func_name, ', '.join(signature)))
        # NDArray args
        for name in ndarg_names: # pylint: disable=redefined-argument-from-local
            code.append("""
    if {name} is not None:
        assert isinstance({name}, NDArrayBase), \\
            "Argument {name} must have NDArray type, but got %s"%str({name})
        ndargs.append({name})""".format(name=name))
        # kwargs
        for name in kwarg_names: # pylint: disable=redefined-argument-from-local
            code.append("""
    if %s is not _Null:
        keys.append('%s')
        vals.append(%s)"""%(name, name, name))
        # dtype
        if dtype_name is not None:
            code.append("""
    if %s is not _Null:
        keys.append('%s')
        vals.append(np.dtype(%s).name)"""%(dtype_name, dtype_name, dtype_name))

    code.append("""
    return _imperative_invoke(%d, ndargs, keys, vals, out)"""%(
        handle.value))

    local = {}
    exec(''.join(code), None, local)  # pylint: disable=exec-used
    ndarray_function = local[func_name]
    ndarray_function.__name__ = func_name
    ndarray_function.__doc__ = doc_str
    ndarray_function.__module__ = 'mxnet.ndarray'
    return ndarray_function


_init_op_module('mxnet', 'ndarray', _make_ndarray_function)
