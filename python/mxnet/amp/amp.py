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
"""Functions for enabling AMP (automatic mixed precision)."""
__all__ = ['init', 'init_trainer', 'scale_loss', 'unscale', 'convert_model',
           'convert_hybrid_block', 'list_lp16_ops', 'list_fp32_ops',
           'list_lp16_fp32_ops', 'list_conditional_fp32_ops',
           'list_widest_type_cast', 'list_loss_output_functions', 'list_lp16_use_fp32_params',
           'convert_symbol']

from array import array
import ctypes
import inspect
import logging
import contextlib
import sys
import numpy as np

from mxnet import numpy
from .. import symbol
from ..device import gpu
from ..symbol import Symbol
from ..symbol import contrib as symbol_contrib
from .. import ndarray
from ..ndarray import NDArray, dtype_np_to_mx, get_dtype_type, get_dtype_name, bfloat16
from . import lists
from ..gluon import Block, HybridBlock, trainer
from .. import base
from ..base import (_NP_OP_PREFIX, _NP_OP_SUBMODULE_LIST, _NP_EXT_OP_PREFIX,
                    _NP_EXT_OP_SUBMODULE_LIST, _NP_INTERNAL_OP_PREFIX,
                    c_str_array, c_str, c_array_buf, SymbolHandle, check_call, _LIB)
from .. import optimizer as opt
from .loss_scaler import LossScaler
from ..operator import get_all_registered_operators_grouped
from ..util import wrap_ctx_to_device_func

OFFLINE_CAST_DTYPE_ATTR = '__amp_dtype__'

float_types_gpu = (np.float16, np.float32)
float_types_cpu = (bfloat16, np.float32)

def _cast_symbol_NDArray(s, dtype, is_numpy_module=False):
    if isinstance(s, Symbol):
        amp_cast = symbol.numpy._internal.amp_cast if is_numpy_module else symbol.amp_cast
        return amp_cast(s, dtype=dtype)
    if isinstance(s, NDArray):
        amp_cast = ndarray.numpy._internal.amp_cast if is_numpy_module else ndarray.amp_cast
        if s.dtype != dtype and (s.dtype in float_types_gpu and s.context.device_type != 'cpu' or
                                 s.dtype in float_types_cpu and s.context.device_type == 'cpu'):
            return amp_cast(s, dtype=dtype)
    return s

def _get_nd_fun_to_wrap(name, module, submodule_dict):
    module_internal = getattr(module, "_internal")
    prefix = base._get_op_name_prefix(name)
    if prefix:
        if prefix != '_random_' or name.endswith('_like'):
            func_name = name[len(prefix):]
            cur_module = submodule_dict[prefix]
        else:
            func_name = name
            cur_module = module_internal
    elif name.startswith('_'):
        func_name = name
        cur_module = module_internal
    else:
        func_name = name
        cur_module = module
    return func_name, [cur_module]

def _get_np_fun_to_wrap(name, ns_prefix):
    for pre, mod, subs in ((_NP_OP_PREFIX, 'numpy', _NP_OP_SUBMODULE_LIST),
                           (_NP_EXT_OP_PREFIX, 'numpy_extension', _NP_EXT_OP_SUBMODULE_LIST),
                           (_NP_INTERNAL_OP_PREFIX, 'numpy._internal', [])):
        if name.startswith(pre):
            nm = name[len(pre):]
            for sub in subs:
                if nm.startswith(sub):
                    func, modules = nm[len(sub):], [sys.modules[f'{ns_prefix}.{mod}.{sub[1:-1]}']]
                    break
            else:
                func, modules = nm, [sys.modules[f'{ns_prefix}.{mod}']]
                break
    else:
        assert False, f'Unable to find target module for {name} in {ns_prefix}'
    if name.startswith(_NP_INTERNAL_OP_PREFIX) and ns_prefix == 'mxnet.ndarray':
        if hasattr(ndarray.numpy._api_internal, func):
            modules.append(ndarray.numpy._api_internal)
    return func, modules

def _wrap_module_functions(module, is_numpy_module, target_dtype, get_aliases, get_cond_aliases,
                           get_fun_to_wrap, target_precision_ops=None, conditional_fp32_ops=None,
                           fp32_ops=None):

    nd_mod = ndarray.numpy._internal if is_numpy_module else ndarray
    sy_mod = symbol.numpy._internal if is_numpy_module else symbol

    def _ndarray_wrapper(f, target_dtype, fp32_param=None, cond_arg=None):
        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                        kwargs[cond_arg[0]] not in cond_arg[1]):
                    return f(*args, **kwargs)
            if fp32_param:
                new_args = []
                for i, x in enumerate(args):
                    if fp32_param[i]:
                        new_args.append(x)
                    else:
                        new_args.append(_cast_symbol_NDArray(x, target_dtype, is_numpy_module))
            else:
                new_args = list(map(
                    lambda x: _cast_symbol_NDArray(x, target_dtype, is_numpy_module), args))
            args = tuple(new_args)
            if fp32_param:
                new_kwargs = {}
                for k, v in kwargs.items():
                    if k in fp32_param:
                        new_kwargs[k] = v
                    else:
                        new_kwargs[k] = _cast_symbol_NDArray(v, target_dtype, is_numpy_module)
                    kwargs = new_kwargs
            else:
                kwargs = {k: _cast_symbol_NDArray(v, target_dtype, is_numpy_module)
                          for k, v in kwargs.items()}
            return f(*args, **kwargs)
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    def _symbol_wrapper(f, target_dtype, fp32_param=None, cond_arg=None):
        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                        kwargs[cond_arg[0]] not in cond_arg[1]):
                    return f(*args, **kwargs)
            sym = f(*args, **kwargs)
            inputs = sym.get_children()
            aux = sym.list_auxiliary_states()
            if fp32_param:
                new_inputs = []
                for i, x in enumerate(inputs):
                    if (x.name in aux) or fp32_param[i]:
                        new_inputs.append(x)
                    else:
                        new_inputs.append(_cast_symbol_NDArray(x, target_dtype, is_numpy_module))
                inputs = new_inputs
            else:
                inputs = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype, is_numpy_module)
                                  if x.name not in aux else x, inputs))
            atomic_sym = sym._gen_atomic_symbol()
            wrapped_sym = atomic_sym(*inputs)
            wrapped_sym._set_attr(name=sym.name)
            return wrapped_sym
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    def _symbol_widest_wrapper(f):
        def _new_fun(*args, **kwargs):
            symbols = []
            is_symbol = False
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, (Symbol, NDArray)):
                    symbols.append((args, i, arg))
                    is_symbol = is_symbol or isinstance(arg, Symbol)
            for k, arg in kwargs.items():
                if isinstance(arg, (Symbol, NDArray)):
                    symbols.append((kwargs, k, arg))
                    is_symbol = is_symbol or isinstance(arg, Symbol)
            if not is_symbol:
                # NDArray case
                widest_type = target_dtype
                for _, _, arg in symbols:
                    if isinstance(arg, NDArray):
                        if arg.dtype == np.float32:
                            widest_type = np.float32
                for arr, index, arg in symbols:
                    if arg.dtype != widest_type and arg.dtype == target_dtype:
                        arr[index] = nd_mod.amp_cast(arg, dtype=widest_type)
            else:
                # Symbol case
                sym_to_check = list(map(lambda x: x[2], symbols))
                casted_syms = sy_mod.amp_multicast(*sym_to_check, num_outputs=len(sym_to_check))
                symbols = list(map(lambda x_y: (x_y[0][0], x_y[0][1], x_y[1]),
                                   zip(symbols, casted_syms)))
                for arr, index, arg in symbols:
                    arr[index] = arg

            return f(*args, **kwargs)
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    _wrapper = _symbol_wrapper if module in (symbol, Symbol, symbol_contrib) else _ndarray_wrapper

    fp32_param_list = list_lp16_use_fp32_params(target_dtype)
    wrap_list = target_precision_ops if target_precision_ops is not None \
                    else list_lp16_ops(target_dtype)
    for fun_name in get_aliases(wrap_list):
        fun_name, modules = get_fun_to_wrap(fun_name, module)
        for cur_module in modules:
            f_to_wrap = getattr(cur_module, fun_name)
            fp32_param = fp32_param_list[fun_name] if (fp32_param_list and fun_name in fp32_param_list) else None
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, target_dtype, fp32_param=fp32_param))
            if not is_numpy_module and cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, target_dtype, fp32_param=fp32_param))

    wrap_list = fp32_ops if fp32_ops is not None else list_fp32_ops(target_dtype)
    for fun_name in get_aliases(wrap_list):
        fun_name, modules = get_fun_to_wrap(fun_name, module)
        for cur_module in modules:
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, np.float32))
            if not is_numpy_module and cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, np.float32))

    wrap_list = conditional_fp32_ops if conditional_fp32_ops is not None \
                    else list_conditional_fp32_ops(target_dtype)
    for fun_name, arg, arg_values in get_cond_aliases(wrap_list):
        fun_name, modules = get_fun_to_wrap(fun_name, module)
        for cur_module in modules:
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, np.float32, cond_arg=(arg, arg_values)))
            if not is_numpy_module and cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, np.float32, cond_arg=(arg, arg_values)))

    for fun_name in get_aliases(list_widest_type_cast(target_dtype)):
        fun_name, modules = get_fun_to_wrap(fun_name, module)
        for cur_module in modules:
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _symbol_widest_wrapper(f_to_wrap))
            if not is_numpy_module and cur_module == module:
                setattr(module.op, fun_name, _symbol_widest_wrapper(f_to_wrap))

def _wrap_loss_output_functions(module, ls, target_dtype):
    if module == ndarray:
        def _wrapper(f):
            def _scaling_wrapper(*args, **kwargs):
                if 'grad_scale' in kwargs:
                    kwargs['grad_scale'] = kwargs['grad_scale'] * ls.loss_scale
                else:
                    kwargs['grad_scale'] = ls.loss_scale
                return f(*args, **kwargs)
            _scaling_wrapper.__name__ = f.__name__
            _scaling_wrapper.__module__ = f.__module__
            _scaling_wrapper.__doc__ = f.__doc__
            return _scaling_wrapper
    else:
        def _wrapper(f):
            def _warning_wrapper(*args, **kwargs):
                logging.warning("%s does not support dynamic loss scaling "
                                "in symbolic and hybridized execution.", f.__name__)
                return f(*args, **kwargs)
            _warning_wrapper.__name__ = f.__name__
            _warning_wrapper.__module__ = f.__module__
            _warning_wrapper.__doc__ = f.__doc__
            return _warning_wrapper

    for fun_name in list_loss_output_functions(target_dtype):
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap))
        except AttributeError:
            pass

_amp_initialized = False
_amp_loss_scale_initialized = False
_loss_scaler = None

@contextlib.contextmanager
def scale_loss(loss, optimizer_or_trainer):
    assert optimizer_or_trainer._amp_loss_scaler is not None, \
        'Loss scaler is not initialized, did you forget to call amp.init_trainer()?'
    optimizer_or_trainer._scale = (optimizer_or_trainer._amp_original_scale /
                                   optimizer_or_trainer._amp_loss_scaler.loss_scale)
    if isinstance(loss, (list, tuple)):
        yield [l * optimizer_or_trainer._amp_loss_scaler.loss_scale for l in loss]
    else:
        yield optimizer_or_trainer._amp_loss_scaler.loss_scale * loss

def warn_if_model_exists():
    for f in inspect.stack():
        for k, v in f.frame.f_locals.items():
            if isinstance(v, Block):
                logging.warning('Block %s created in [%s:%d] before AMP init.',
                                k, f.filename, f.lineno)
                return

def init(target_dtype='float16', target_precision_ops=None,
         conditional_fp32_ops=None, fp32_ops=None, layout_optimization=False):
    """Initialize AMP (automatic mixed precision).

    This needs to be done before model creation.

    Parameters
    ----------
    target_dtype : {'float16', 'bfloat16'}
        Target low precision type for AMP. Currently only float16 and bfloat16 are supported.
    target_precision_ops : list of string
        Override the list of functions casted to target_dtype. Entries in this list
        are names of the functions casted to target_dtype.
    conditional_fp32_ops : list of (string, string, list of string)
        Override the list of functions conditionally casted to FP32. The format
        of the list is (name of the function, name of the parameter, list of
        values of the parameter that make the function be casted to FP32).
    fp32_ops : list of string
        Override the list of functions casted to FP32. Entries in this list
        are names of the functions casted to FP32.
    """
    global _amp_initialized
    global _loss_scaler
    if not _amp_initialized:
        assert target_dtype in ['float16', np.float16, 'bfloat16', bfloat16], \
               "AMP currently supports only float16 or bfloat16 as a target_dtype"
        _amp_initialized = True
        log_msg = "Using AMP"
        if layout_optimization:
            log_msg += "\n - layout optimization: enabled"
            check_call(_LIB.MXSetOptimizeLayout(ctypes.c_bool(True)))
        logging.info(log_msg)
        if target_dtype == "bfloat16":
            target_dtype = bfloat16
        else:
            target_dtype = np.dtype(target_dtype)

        warn_if_model_exists()

        ops = get_all_registered_operators_grouped()
        get_aliases_nd = lambda l: [a for op in l for a in ops[op] if not base._is_np_op(a)]
        get_aliases_np = lambda l: [a for op in l for a in ops[op] if base._is_np_op(a)]
        get_aliases_np_pub = lambda l: [a for op in l for a in ops[op]
                                        if a.startswith(('_np_', '_npx_'))]
        get_cond_aliases_nd = lambda l: [(a, *rest) for op, *rest in l for a in ops[op]
                                         if not base._is_np_op(a)]
        get_cond_aliases_np = lambda l: [(a, *rest) for op, *rest in l for a in ops[op]
                                         if base._is_np_op(a)]
        get_cond_aliases_np_pub = lambda l: [(a, *rest) for op, *rest in l for a in ops[op]
                                             if a.startswith(('_np_', '_npx_'))]
        sy_submodules = {p:getattr(symbol, p[1:-1]) for p in base._OP_NAME_PREFIX_LIST}
        get_sy_fun = lambda fun, mod: _get_nd_fun_to_wrap(fun, mod, sy_submodules)
        nd_submodules = {p:getattr(ndarray, p[1:-1]) for p in base._OP_NAME_PREFIX_LIST}
        get_nd_fun = lambda fun, mod: _get_nd_fun_to_wrap(fun, mod, nd_submodules)
        get_np_sy_fun = lambda fun, mod: _get_np_fun_to_wrap(fun, "mxnet.symbol")
        get_np_nd_fun = lambda fun, mod: _get_np_fun_to_wrap(fun, "mxnet.ndarray")
        get_np_fun = lambda fun, mode: _get_np_fun_to_wrap(fun, "mxnet")
        todo = [
            (symbol, False, get_aliases_nd, get_cond_aliases_nd, get_sy_fun),
            (ndarray, False, get_aliases_nd, get_cond_aliases_nd, get_nd_fun),
            (symbol.numpy, True, get_aliases_np, get_cond_aliases_np, get_np_sy_fun),
            (ndarray.numpy, True, get_aliases_np, get_cond_aliases_np, get_np_nd_fun),
            (numpy, True, get_aliases_np_pub, get_cond_aliases_np_pub, get_np_fun),
        ]
        _loss_scaler = LossScaler()
        for module, is_numpy, get_aliases, get_cond_aliases, get_fun in todo:
            _wrap_module_functions(module, is_numpy, target_dtype, get_aliases, get_cond_aliases,
                                   get_fun, target_precision_ops, conditional_fp32_ops, fp32_ops)
            _wrap_loss_output_functions(module, _loss_scaler, target_dtype)

def init_trainer(optimizer_or_trainer):
    """Initialize trainer or optimizer to work with AMP dynamic loss scaling.

    Parameters
    ----------
    optimizer_or_trainer : Optimizer or Trainer
        MXNet Optimizer or Gluon trainer to initialize with AMP
    """
    global _amp_loss_scale_initialized
    global _amp_initialized
    global _loss_scaler
    assert _amp_initialized, "AMP not initialized, did you forget to call amp.init()?"
    if not _amp_loss_scale_initialized:
        _amp_loss_scale_initialized = True
        loss_scaler = _loss_scaler
    else:
        loss_scaler = LossScaler()
    #_wrap_output
    if isinstance(optimizer_or_trainer, trainer.Trainer):
        optimizer_or_trainer._amp_loss_scaler = loss_scaler
        optimizer_or_trainer._amp_original_scale = optimizer_or_trainer._scale
        trainer.Trainer.amp_loss_scale = property(lambda self: self._amp_loss_scaler.loss_scale)
    elif isinstance(optimizer_or_trainer, opt.Optimizer):
        raise TypeError("AMP is currently only compatible with Gluon Trainer")
    else:
        raise TypeError("optimizer_or_trainer should be a Gluon Trainer or "
                        f"an optimizer, instead is {type(optimizer_or_trainer)}")

def unscale(optimizer_or_trainer):
    """Check and unscale the gradients manually. This function should only be used
    if accessing gradients is necessary, e.g. for gradient clipping.

    Parameters
    ----------
    optimizer_or_trainer : Optimizer or Trainer
        MXNet optimizer or Gluon Trainer used when scaling the gradients
    """
    if isinstance(optimizer_or_trainer, trainer.Trainer):
        valid_grads = [p._grad for p in optimizer_or_trainer._params if p._grad is not None]
        for grads in valid_grads:
            # TODO(ptredak): make a bulked unscale
            for g in grads:
                g[:] *= optimizer_or_trainer._scale
        optimizer_or_trainer._scale = 1.
    elif isinstance(optimizer_or_trainer, opt.Optimizer):
        # TODO(ptredak): make it work with the optimizer
        raise TypeError("AMP is currently only compatible with Gluon Trainer")
    else:
        raise TypeError("optimizer_or_trainer should be a Gluon Trainer or "
                        f"an optimizer, instead is {type(optimizer_or_trainer)}")


def convert_symbol(sym, input_dtypes, param_dtypes, target_dtype, target_dtype_ops=None,
                   fp32_ops=None, conditional_fp32_ops=None, excluded_sym_names=[],
                   cast_params_offline=False):
    """Given a symbol object representing a neural network of data type FP32 and target_dtype,
    add cast layers according to the op lists (target_dtype_ops, fp32_ops,
    conditional_fp32_ops) if provided, otherwise use the default
    lists provided by the framework.

    Parameters
    ----------
    sym : Symbol
        FP32 neural network symbol
    input_dtypes: dict
        Dictionary mapping names of model inputs to their dtypes
    param_dtypes: dict
        Dictionary mapping names of model parameters to their dtypes
    target_dtype : str or numpy, optional defaults to float16
        currently only supports float16 and bfloat16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_dtype_ops : list of strs, optional
        Override the list of operator names casted to the target_dtype.
        If None, uses the framework's default list to be casted to target_dtype.
    fp32_ops : list of strs, optional
        Override the list of operator names casted to FP32.
        If None, uses the framework's default list to be casted to FP32.
    conditional_fp32_ops : list of (string, string, list of string), optional
        Override the list of functions to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to FP32)
    excluded_sym_names : list of strs, optional
        A list of strings that represent the names of symbols that users want to exclude
        from being casted to LP16 or FP32.
    data_names : list of strs, optional
        A list of strings that represent input data tensor names to the model
    cast_params_offline : bool, default False
        Whether to cast arg_params and aux_params now, instead of doing it every time at runtime.
    """
    import json

    assert isinstance(sym, Symbol), "First argument to convert_symbol should be a Symbol"
    assert target_dtype_ops is None or isinstance(target_dtype_ops, list), \
        "target_dtype_ops should be a list of strings"
    assert fp32_ops is None or isinstance(fp32_ops, list), \
        "fp32_ops should be a list of strings"
    assert conditional_fp32_ops is None or isinstance(conditional_fp32_ops, list), \
        "conditional_fp32_ops should be a list of strings"

    target_dtype = get_dtype_name(target_dtype)
    assert target_dtype in ['float16', *bfloat16.names], \
        "Only float16 and bfloat16 types are currently supported as target_dtype"

    if target_dtype_ops is None:
        target_dtype_ops = list_lp16_ops(target_dtype)
    if fp32_ops is None:
        fp32_ops = list_fp32_ops(target_dtype)

    # conditional ops
    if conditional_fp32_ops is None:
        conditional_fp32_ops = list_conditional_fp32_ops(target_dtype)
    cond_ops = {cond_op[0]: {} for cond_op in conditional_fp32_ops}
    for cond_op in conditional_fp32_ops:
        op_name, attr_name, attr_vals = cond_op
        assert isinstance(op_name, str) and isinstance(attr_name, str) and isinstance(attr_vals, list), \
            "conditional_fp32_ops should be a list of (str, str, list of str)"
        cond_ops[op_name].setdefault(attr_name, []).extend(attr_vals)

    nodes_attrs = sym.attr_dict()
    nodes_op = {n['name']: n['op'] for n in json.loads(sym.tojson())['nodes']}
    for node_name, node_op in nodes_op.items():
        if node_op not in cond_ops:
            continue
        node_attrs = nodes_attrs[node_name]
        for attr_name, attr_vals in cond_ops[node_op].items():
            assert attr_name in node_attrs
            if node_attrs[attr_name] in attr_vals:
                excluded_sym_names.append(node_name)
                break

    excluded_sym_names = set(excluded_sym_names)
    for node in sym.get_internals():
        if node.name in excluded_sym_names:
            excluded_sym_names.remove(node.name)
            opt_constraints = node.attr('__opt_constraint__')
            opt_constraints = 0 if opt_constraints is None else int(opt_constraints)
            opt_constraints |= HybridBlock.OptConstraint.Flag.DisableAMP.value
            node._set_attr(__opt_constraint__=str(opt_constraints))

    if len(excluded_sym_names) > 0:
        logging.warning("excluded_sym_names are not present in the network. Missing nodes: {}".format(
            excluded_sym_names))

    # Op lists should not intersect
    common_ops = set(target_dtype_ops) & set(fp32_ops)
    assert len(common_ops) == 0, "Common ops in target_dtype_ops and fp32_ops: {}".format(common_ops)
    common_ops = set(target_dtype_ops) & set(cond_ops)
    assert len(common_ops) == 0, "Common ops in target_dtype_ops and conditional_fp32_ops: {}".format(
        common_ops)
    common_ops = set(cond_ops) & set(fp32_ops)
    assert len(common_ops) == 0, "Common ops in fp32_ops and conditional_fp32_ops: {}".format(common_ops)

    combined_ops = set(target_dtype_ops + fp32_ops + list(cond_ops.keys()))
    original_cond_ops = [cond_op[0] for cond_op in list_conditional_fp32_ops(target_dtype)]
    all_lp16_fp32_ops = set(list_lp16_ops(target_dtype) + list_fp32_ops(target_dtype) +
                            list_lp16_fp32_ops(target_dtype) + original_cond_ops)

    illegal_ops = combined_ops - all_lp16_fp32_ops
    assert len(illegal_ops) == 0, f'''Can only choose ops from one of the four lists
                            for lp16_ops and fp32_ops
                            1. amp.list_lp16_ops(target_dtype)
                            2. amp.list_fp32_ops(target_dtype)
                            3. amp.list_lp16_fp32_ops(target_dtype)
                            4. amp.list_conditional_fp32_ops(target_dtype)
                            Op {illegal_ops} not in any of them'''

    widest_dtype_ops = list_widest_type_cast(target_dtype)

    input_names = list(input_dtypes.keys())
    all_arg_names, all_arg_types = [], []

    for name, dtype in {**input_dtypes, **param_dtypes}.items():
        all_arg_names.append(name)
        all_arg_types.append(dtype_np_to_mx(dtype))
    out = SymbolHandle()
    check_call(_LIB.MXReducePrecisionSymbol(sym.handle,
                                            ctypes.byref(out),
                                            ctypes.c_int(dtype_np_to_mx(target_dtype)),
                                            ctypes.c_int(cast_params_offline),
                                            c_str(OFFLINE_CAST_DTYPE_ATTR),
                                            ctypes.c_uint(len(input_names)),
                                            c_str_array(input_names),
                                            ctypes.c_uint(len(all_arg_names)),
                                            c_str_array(all_arg_names),
                                            c_array_buf(ctypes.c_int, array('i', all_arg_types)),
                                            ctypes.c_uint(len(target_dtype_ops)),
                                            c_str_array(target_dtype_ops),
                                            ctypes.c_uint(len(fp32_ops)),
                                            c_str_array(fp32_ops),
                                            ctypes.c_uint(len(widest_dtype_ops)),
                                            c_str_array(widest_dtype_ops)))
    return type(sym)(out)


def convert_model(sym, arg_params, aux_params, input_dtypes, target_dtype,
                  target_dtype_ops=None, fp32_ops=None, conditional_fp32_ops=None,
                  excluded_sym_names=[], cast_params_offline=False):
    """API for converting a model from FP32 model to a mixed precision model.
    MXNet tries to convert the FP32 model to mixed precision model by adding
    cast layers using amp_cast and amp_multicast operators which can be used for inference use cases.
    The decision on which cast layer to add is based on hardcoded lists for Automatic Mixed Precision
    in MXNet. These lists can be overridden by the user by providing their own lists
    using : targe_precision_ops, fp32_ops, widest_precision_ops, conditional_fp32_ops

    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    input_dtypes: dict
        Dictionary mapping names of model inputs to their dtypes
    target_dtype : str
        Currently only supports float16 and bfloat 16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_dtype_ops : list of strs
        Override the list of operator names casted to target_dtype.
        If None, uses the framework's default list to be casted to target dtype.
    fp32_ops : list of strs
        Override the lists of operator names casted to FP32.
        If None, uses the framework's default list to be casted to FP32.
    widest_dtype_ops : list of strs
        A list of op names provided by user which should run in widest precision among its inputs.
        If None, uses the framework's default list of widest_precision_ops.
    conditional_fp32_ops : list of (string, string, list of string)
        Override the list of operators to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to
        fp32)
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being executed in lower precision.
    cast_params_offline : bool, default False
        Whether to cast arg_params and aux_params now, instead of doing it every time at runtime.
    """
    assert isinstance(sym, Symbol), "First argument to convert_model should be a Symbol"
    assert isinstance(
        arg_params, dict), "Second argument to convert_model should be a dict of name to ndarray"
    assert isinstance(
        aux_params, dict), "Third argument to convert_model should be a dict of name to ndarray"

    arg_params = arg_params.copy()
    aux_params = aux_params.copy()
    param_dtypes = {name: data.dtype for name, data in arg_params.items()}
    param_dtypes.update({name: data.dtype for name, data in aux_params.items()})
    sym = convert_symbol(sym, input_dtypes, param_dtypes, target_dtype, target_dtype_ops,
                         fp32_ops, conditional_fp32_ops, excluded_sym_names, cast_params_offline)

    # If dtype is set for params, cast the param to that dtype
    attr_dict = sym.attr_dict()
    for sym_name in sym.list_arguments():
        if attr_dict.get(sym_name, {}).get(OFFLINE_CAST_DTYPE_ATTR, '') != '' and sym_name in arg_params:
            typ = get_dtype_type(attr_dict[sym_name][OFFLINE_CAST_DTYPE_ATTR])
            if arg_params[sym_name].dtype != typ:
                arg_params[sym_name] = arg_params[sym_name].astype(typ)

    for sym_name in sym.list_auxiliary_states():
        if attr_dict.get(sym_name, {}).get(OFFLINE_CAST_DTYPE_ATTR, '') != '' and sym_name in aux_params:
            typ = get_dtype_type(attr_dict[sym_name][OFFLINE_CAST_DTYPE_ATTR])
            if aux_params[sym_name].dtype != typ:
                aux_params[sym_name] = aux_params[sym_name].astype(typ)

    # Return the converted symbol and casted params
    return sym, arg_params, aux_params


@wrap_ctx_to_device_func
def convert_hybrid_block(block, data_example, target_dtype, target_dtype_ops=None,
                         fp32_ops=None, conditional_fp32_ops=None,
                         excluded_sym_names=[], device=None,
                         cast_params_offline=False):
    """Given a hybrid block/symbol block representing a FP32 model and a target_dtype,
    return a block with mixed precision support which can be used for inference use cases.

    Parameters
    ----------
    block : HybridBlock or SymbolBlock object
        FP32 HybridBlock or SymbolBlock object
    data_example: tuple or list of NDArrays
        Data example, representing the data that this model will work with during the inference.
    target_dtype : str or numpy
        currently only supports float16 and bfloat16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_precision_ops : list of strs
        Override the list of operator names casted to target_dtype.
        If None, uses the framework's default list to be casted to FP32.
    conditional_fp32_ops : list of (str, str, list of str)
        Override the list of functions to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to FP32
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being quantized
    device : Device
        Device on which model parameters should live. Default value: current device.
    cast_params_offline : bool, default False
        Whether to cast arg_params and aux_params now, instead of doing it every time at runtime.
    """
    from ..gluon import SymbolBlock
    from ..ndarray import NDArray as ND_NDArray, waitall
    from ..numpy import ndarray as NP_NDArray

    assert isinstance(block, HybridBlock), "block input should be a HybridBlock"
    if not isinstance(data_example, (list, tuple)):
        data_example = [data_example]
    for data in data_example:
        assert isinstance(data, (ND_NDArray, NP_NDArray)), "Data example must be composed of " \
            "mxnet.numpy.ndarray or mxnet.ndarray.NDArray instances"
    if not block._active:
        block.hybridize(static_alloc=False, static_shape=False)
    block(*data_example)
    waitall()

    sym, params = block.export(None, remove_amp_cast=False)
    args, auxs = {}, {}
    for name, data in params.items():
        if name.startswith('arg:'):
            arg_name = name[len('arg:'):]
            args[arg_name] = data
        else:
            assert name.startswith('aux:')
            aux_name = name[len('aux:'):]
            auxs[aux_name] = data

    input_names = set(sym.list_arguments()) - (set(args.keys()) | set(auxs.keys()))
    input_names_ordered = HybridBlock.generate_arg_names(len(data_example))
    assert input_names == set(input_names_ordered)

    input_dtypes = {name: data.dtype for name, data in zip(input_names_ordered, data_example)}
    lp_sym, lp_args, lp_auxs = convert_model(sym, args, auxs, input_dtypes, target_dtype,
                                             target_dtype_ops, fp32_ops, conditional_fp32_ops,
                                             excluded_sym_names, cast_params_offline)

    inputs = [in_sym for in_sym in lp_sym.get_inputs() if in_sym.name in input_names]
    param_dict = lp_args
    param_dict.update(lp_auxs)

    ret = SymbolBlock(lp_sym, inputs)
    ret.load_dict(param_dict, device=device, cast_dtype=True, dtype_source='saved')
    return ret


def list_lp16_ops(target_dtype):
    """Get the default list of LP16 ops for AMP
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.FP16_FUNCS
    else:
        assert get_dtype_name(target_dtype) in bfloat16.names, "not supported type"
        return lists.symbol_bf16.BF16_FUNCS

def list_fp32_ops(target_dtype):
    """Get the default list of FP32 ops for AMP
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.FP32_FUNCS
    else:
        assert get_dtype_name(target_dtype) in bfloat16.names, "not supported type"
        return lists.symbol_bf16.FP32_FUNCS

def list_lp16_fp32_ops(target_dtype):
    """Get the default list of ops which run in both LP16 and FP32
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.FP16_FP32_FUNCS
    else:
        assert get_dtype_name(target_dtype) in bfloat16.names, "not supported type"
        return lists.symbol_bf16.BF16_FP32_FUNCS

def list_conditional_fp32_ops(target_dtype):
    """Get the conditional fp32 ops list
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.CONDITIONAL_FP32_FUNCS
    else:
        assert get_dtype_name(target_dtype) in bfloat16.names, "not supported type"
        return lists.symbol_bf16.CONDITIONAL_FP32_FUNCS

def list_widest_type_cast(target_dtype):
    """Get the widest type cast ops list
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.WIDEST_TYPE_CASTS
    else:
        assert get_dtype_name(target_dtype) in bfloat16.names, "not supported type"
        return lists.symbol_bf16.WIDEST_TYPE_CASTS

def list_loss_output_functions(target_dtype):
    """Get loss function list
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.LOSS_OUTPUT_FUNCTIONS
    else:
        assert get_dtype_name(target_dtype) in bfloat16.names, "not supported type"
        return lists.symbol_bf16.LOSS_OUTPUT_FUNCTIONS

def list_lp16_use_fp32_params(target_dtype):
    """ Get the params restrict for LP16

    """
    if target_dtype in ['float16', np.float16]:
        return None
    else:
        assert get_dtype_name(target_dtype) in bfloat16.names, "not supported type"
        return lists.symbol_bf16.BF16_USE_FP32_PARAMS
