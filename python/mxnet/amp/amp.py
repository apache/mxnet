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
__all__ = ['init', 'init_trainer', 'scale_loss', 'unscale']

from types import MethodType
import logging
import contextlib
import numpy as np

from .. import symbol
from ..symbol import Symbol
from ..symbol import contrib as symbol_contrib
from .. import ndarray
from ..ndarray import NDArray
from ..ndarray import contrib as ndarray_contrib
from . import lists
from ..gluon import trainer
from .. import optimizer as opt
from .loss_scaler import LossScaler

def _cast_symbol_NDArray(s, dtype):
    if isinstance(s, Symbol):
        return symbol.amp_cast(s, dtype=dtype)
    elif isinstance(s, NDArray):
        if (s.dtype != dtype and
                (s.dtype == np.float16 or s.dtype == np.float32) and
                s.context.device_type != 'cpu'):
            return ndarray.amp_cast(s, dtype=dtype)
        else:
            return s
    else:
        return s

def _wrap_symbol_functions(module, fp16_list=None, conditional_fp32_list=None, fp32_list=None):
    def _ndarray_wrapper(f, target_dtype, cond_arg=None):
        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                        kwargs[cond_arg[0]] not in cond_arg[1]):
                    return f(*args, **kwargs)
            new_args = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype), args))
            args = tuple(new_args)
            kwargs = {k: _cast_symbol_NDArray(v, target_dtype) for k, v in kwargs.items()}
            return f(*args, **kwargs)
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    def _symbol_wrapper(f, target_dtype, cond_arg=None):
        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                        kwargs[cond_arg[0]] not in cond_arg[1]):
                    return f(*args, **kwargs)
            sym = f(*args, **kwargs)
            inputs = sym.get_children()
            aux = sym.list_auxiliary_states()
            inputs = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype)
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
                widest_type = np.float16
                for _, _, arg in symbols:
                    if isinstance(arg, NDArray):
                        if arg.dtype == np.float32:
                            widest_type = np.float32
                for arr, index, arg in symbols:
                    if arg.dtype != widest_type and arg.dtype == np.float16:
                        arr[index] = ndarray.amp_cast(arg, dtype=widest_type)
            else:
                # Symbol case
                sym_to_check = list(map(lambda x: x[2], symbols))
                casted_syms = symbol.amp_multicast(*sym_to_check, num_outputs=len(sym_to_check))
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

    wrap_list = fp16_list if fp16_list is not None else lists.symbol.FP16_FUNCS
    for fun_name in wrap_list:
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap, np.float16))
        except AttributeError:
            pass

    wrap_list = fp32_list if fp32_list is not None else lists.symbol.FP32_FUNCS
    for fun_name in wrap_list:
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap, np.float32))
        except AttributeError:
            pass

    wrap_list = conditional_fp32_list if conditional_fp32_list is not None \
                    else lists.symbol.CONDITIONAL_FP32_FUNCS
    for fun_name, arg, arg_values in wrap_list:
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap, np.float32, (arg, arg_values)))
        except AttributeError:
            pass

    for fun_name in lists.symbol.WIDEST_TYPE_CASTS:
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _symbol_widest_wrapper(f_to_wrap))
        except AttributeError:
            pass

def _wrap_loss_output_functions(module, ls):
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

    for fun_name in lists.symbol.LOSS_OUTPUT_FUNCTIONS:
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
    optimizer_or_trainer._scale = 1. / optimizer_or_trainer._amp_loss_scaler.loss_scale
    if isinstance(loss, (list, tuple)):
        yield [l * optimizer_or_trainer._amp_loss_scaler.loss_scale for l in loss]
    else:
        yield optimizer_or_trainer._amp_loss_scaler.loss_scale * loss

def init(fp16_list=None, conditional_fp32_list=None, fp32_list=None):
    """Initialize AMP (automatic mixed precision).

    This needs to be done before model creation.

    Parameters
    ----------
    fp16_list : list of string
        Override the list of functions casted to FP16. Entries in this list
        are names of the functions casted to FP16.
    conditional_fp32_list : list of (string, string, list of string)
        Override the list of functions conditionally casted to FP32. The format
        of the list is (name of the function, name of the parameter, list of
        values of the parameter that make the function be casted to FP32).
    fp32_list : list of string
        Override the list of functions casted to FP32. Entries in this list
        are names of the functions casted to FP32.
    """
    global _amp_initialized
    global _loss_scaler
    if not _amp_initialized:
        _amp_initialized = True
        logging.info("Using AMP")
        _wrap_symbol_functions(symbol, fp16_list, conditional_fp32_list, fp32_list)
        _wrap_symbol_functions(symbol._internal, fp16_list, conditional_fp32_list, fp32_list)
        _wrap_symbol_functions(Symbol, fp16_list, conditional_fp32_list, fp32_list)
        _wrap_symbol_functions(symbol_contrib, fp16_list, conditional_fp32_list, fp32_list)
        _wrap_symbol_functions(ndarray, fp16_list, conditional_fp32_list, fp32_list)
        _wrap_symbol_functions(ndarray._internal, fp16_list, conditional_fp32_list, fp32_list)
        _wrap_symbol_functions(NDArray, fp16_list, conditional_fp32_list, fp32_list)
        _wrap_symbol_functions(ndarray_contrib, fp16_list, conditional_fp32_list, fp32_list)
        _loss_scaler = LossScaler()
        _wrap_loss_output_functions(ndarray, _loss_scaler)
        _wrap_loss_output_functions(symbol, _loss_scaler)

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
        skip_update = optimizer_or_trainer._amp_loss_scaler.wait_and_update
        optimizer_or_trainer._optimizer.old_update_multi_precision = \
                optimizer_or_trainer._optimizer.update_multi_precision
        def new_update_multi_precision(self, index, weight, grad, state):
            if not skip_update():
                self.old_update_multi_precision(index, weight, grad, state)
        optimizer_or_trainer._optimizer.update_multi_precision = \
            MethodType(new_update_multi_precision, optimizer_or_trainer._optimizer)
        launch_check_overflow = optimizer_or_trainer._amp_loss_scaler.launch_check_overflow
        optimizer_or_trainer._old_update = optimizer_or_trainer._update
        def new_update(self, ignore_stale_grad=False):
            launch_check_overflow(self._params)
            self._old_update(ignore_stale_grad)
        optimizer_or_trainer._update = MethodType(new_update, optimizer_or_trainer)

    elif isinstance(optimizer_or_trainer, opt.Optimizer):
        # TODO(ptredak): make it work with the optimizer
        raise TypeError("AMP is currently only compatible with Gluon Trainer")
    else:
        raise TypeError("optimizer_or_trainer should be a Gluon Trainer or "
                        "an optimizer, instead is %s" % type(optimizer_or_trainer))

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
                        "an optimizer, instead is %s" % type(optimizer_or_trainer))
