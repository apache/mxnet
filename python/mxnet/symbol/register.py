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

# pylint: disable=unused-import
"""Register backend ops in mxnet.symbol namespace."""
import os as _os
import ctypes
import numpy as _np

from . import _internal
from ._internal import SymbolBase, _symbol_creator
from ..attribute import AttrScope
from ..base import mx_uint, check_call, _LIB, py_str
from ..symbol_doc import _build_doc
from ..base import _Null, _init_op_module, _is_np_op, _output_is_list
from ..name import NameManager
# pylint: enable=unused-import


def _verify_np_symbol(op_name, func_name, sym):
    """Verify if the sym is a numpy symbol.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    sym : symbol to be verified
    """
    from .numpy._symbol import _Symbol as np_symbol
    if not isinstance(sym, np_symbol):
        raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                        'This is a numpy operator which can only accept '
                        'MXNet numpy ndarrays, while received a legacy ndarray. '
                        'Please ensure that you have activated numpy semantics by calling '
                        '`npx.set_np()` in your code. If you still see this error with numpy '
                        'semantics activated, please call `as_np_ndarray()` upon the legacy '
                        'ndarray to convert it to an MXNet numpy ndarray, and then feed the '
                        'converted array to this operator.'
                        .format(op_name, func_name))


def _verify_legacy_symbol(op_name, func_name, sym):
    """Verify if the sym is a legacy symbol.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    sym : symbol to be verified
    """
    from .numpy._symbol import _Symbol as np_symbol
    if isinstance(sym, np_symbol):
        raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. '
                        'This is a legacy operator which can only accept '
                        'legacy ndarrays, while received an MXNet numpy ndarray. '
                        'Please call `as_nd_ndarray()` upon the numpy ndarray to '
                        'convert it to a legacy ndarray, and then feed the converted '
                        'array to this operator.'
                        .format(op_name, func_name))


def _generate_symbol_function_code(handle, op_name, func_name, signature_only=False):
    """Generate function for symbol op by handle and function name."""
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
    key_var_num_args = py_str(key_var_num_args.value)
    ret_type = py_str(ret_type.value) if ret_type.value is not None else ''
    doc_str = _build_doc(op_name,
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
    #signature.append('is_train=False')
    signature.append('name=None')
    signature.append('attr=None')
    signature.append('out=None')
    signature.append('**kwargs')
    signature = ndsignature + signature

    is_np_op = _is_np_op(op_name)
    output_is_list = _output_is_list(op_name)
    verify_symbol_fn = _verify_np_symbol.__name__ if is_np_op else _verify_legacy_symbol.__name__
    code = []
    if arr_name:
        code.append("""
def %s(*%s, **kwargs):"""%(func_name, arr_name))
        if not signature_only:
            code.append("""
    sym_args = []
    for i in {}:
        assert isinstance(i, SymbolBase), \\
            "Positional arguments must be Symbol instances, " \\
            "but got %s"%str(i)
        {}('{}', '{}', i)
        sym_args.append(i)""".format(arr_name, verify_symbol_fn, op_name, func_name))
            if dtype_name is not None:
                code.append("""
    if '%s' in kwargs:
        if _np.dtype(kwargs['%s']).names:
            kwargs['%s'] = _np.dtype(kwargs['%s']).names[0]
        else:
            kwargs['%s'] = _np.dtype(kwargs['%s']).name """%(
                dtype_name, dtype_name, dtype_name,
                dtype_name, dtype_name, dtype_name))
            code.append("""
    attr = kwargs.pop('attr', None)
    if not hasattr(AttrScope._current, "value"):
        AttrScope._current.value = AttrScope()
    kwargs.update(AttrScope._current.value.get(attr))
    name = kwargs.pop('name', None)
    if not hasattr(NameManager._current, "value"):
        NameManager._current.value = NameManager()
    name = NameManager._current.value.get(name, '%s')
    _ = kwargs.pop('out', None)
    keys = []
    vals = []
    sym_kwargs = dict()
    for k, v in kwargs.items():
        if isinstance(v, SymbolBase):
            sym_kwargs[k] = v
            %s('%s', '%s', v)
        else:
            keys.append(k)
            vals.append(v)"""%(func_name.lower(), verify_symbol_fn, op_name, func_name))
            if key_var_num_args: # pylint: disable=using-constant-test
                code.append("""
    if '%s' not in kwargs:
        keys.append('%s')
        vals.append(len(sym_args) + len(sym_kwargs))"""%(
            key_var_num_args, key_var_num_args))

            code.append("""
    return _symbol_creator(%d, sym_args, sym_kwargs, keys, vals, name, %s, %s)"""%(
                handle.value, str(is_np_op), str(output_is_list)))
    else:
        code.append("""
def %s(%s):"""%(func_name, ', '.join(signature)))
        if not signature_only:
            code.append("""
    if not hasattr(AttrScope._current, "value"):
        AttrScope._current.value = AttrScope()
    kwargs.update(AttrScope._current.value.get(attr))
    sym_kwargs = dict()
    _keys = []
    _vals = []
    for _k, _v in kwargs.items():
        if isinstance(_v, SymbolBase):
            sym_kwargs[_k] = _v
            {}('{}', '{}', _v)
        else:
            _keys.append(_k)
            _vals.append(_v)""".format(verify_symbol_fn, op_name, func_name))
            # NDArray args
            for name in ndarg_names: # pylint: disable=redefined-argument-from-local
                code.append("""
    if {name} is not None:
        assert isinstance({name}, SymbolBase), \\
            "Argument {name} must be Symbol instances, but got %s"%str({name})
        sym_kwargs['{name}'] = {name}""".format(name=name))
                code.append("""
        {}('{}', '{}', {name})
                """.format(verify_symbol_fn, op_name, func_name, name=name))
            # kwargs
            for name in kwarg_names: # pylint: disable=redefined-argument-from-local
                code.append("""
    if %s is not _Null:
        _keys.append('%s')
        _vals.append(%s)"""%(name, name, name))
            # dtype
            if dtype_name is not None:
                if is_np_op:
                    code.append("""
    if %s is not _Null and %s is not None:
        _keys.append('%s')
        _vals.append(_np.dtype(%s).name)"""%(dtype_name, dtype_name, dtype_name, dtype_name))
                else:
                    code.append("""
    if %s is not _Null:
        _keys.append('%s')
        if _np.dtype(%s).names:
            _vals.append(_np.dtype(%s).names[0])
        else:
            _vals.append(_np.dtype(%s).name) """%(dtype_name, dtype_name, dtype_name,
                                                  dtype_name, dtype_name))

            code.append("""
    if not hasattr(NameManager._current, "value"):
        NameManager._current.value = NameManager()
    name = NameManager._current.value.get(name, '%s')
    return _symbol_creator(%d, None, sym_kwargs, _keys, _vals, name, %s, %s)"""%(
        func_name.lower(), handle.value, str(is_np_op), str(output_is_list)))

    if signature_only:
        code.append("""
    return (0,)""")

    doc_str_lines = _os.linesep+''.join(['    '+s if s.strip() else s
                                         for s in 'r"""{doc_str}"""'.format(doc_str=doc_str)
                                         .splitlines(True)])
    code.insert(1, doc_str_lines)
    return ''.join(code), doc_str


def _make_symbol_function(handle, name, func_name):
    """Create a symbol function by handle and function name."""
    code, doc_str = _generate_symbol_function_code(handle, name, func_name)

    local = {}
    exec(code, None, local)  # pylint: disable=exec-used
    symbol_function = local[func_name]
    symbol_function.__name__ = func_name
    symbol_function.__doc__ = doc_str
    symbol_function.__module__ = 'mxnet.symbol'
    return symbol_function

_init_op_module('mxnet', 'symbol', _make_symbol_function)

# Update operator documentation with added float support
# Note that we can only do this after the op module is initialized
# Otherwise the backend operators cannot be found
# pylint: disable=wrong-import-position
from .contrib import adamw_update, mp_adamw_update
from ._internal import _adamw_update, _mp_adamw_update
adamw_update.__doc__ = _adamw_update.__doc__.replace("rescale_grad : Symbol",
                                                     "rescale_grad : Symbol or float")
mp_adamw_update.__doc__ = _mp_adamw_update.__doc__.replace("rescale_grad : Symbol",
                                                           "rescale_grad : Symbol or float")
