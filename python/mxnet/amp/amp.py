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
__all__ = ['init', 'init_trainer', 'scale_loss']

import logging
import contextlib
import numpy as np
from types import MethodType
from functools import partial

from .. import symbol
from ..symbol import Symbol
from ..symbol import contrib as symbol_contrib
from .. import ndarray
from ..ndarray import NDArray
from ..ndarray import contrib as ndarray_contrib
from . import lists
from ..gluon import trainer
from .. import optimizer as opt
from .loss_scaler import *

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

def _wrap_symbol_functions(module):
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
            attr = sym.list_attr()
            inputs = sym.get_children()
            aux = sym.list_auxiliary_states()
            inputs = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype)
                              if x.name not in aux else x, inputs))
            wrapped_sym = f(*inputs, **attr)
            #wrapped_sym_argnames = wrapped_sym.list_arguments()
            #wrapped_sym._compose(**dict(zip(wrapped_sym_argnames, inputs)))
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

    for fun_name in lists.symbol.FP16_FUNCS:
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap, np.float16))
        except AttributeError:
            pass

    for fun_name in lists.symbol.FP32_FUNCS:
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap, np.float32))
        except AttributeError:
            pass

    for fun_name, arg, arg_values in lists.symbol.CONDITIONAL_FP32_FUNCS:
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
    pass

_amp_initialized = False
_amp_loss_scale_initialized = False

@contextlib.contextmanager
def scale_loss(loss, optimizer_or_trainer, params=None):
    assert optimizer_or_trainer._amp_loss_scaler is not None, \
        'Loss scaler is not initialized, did you forget to call amp.init_trainer()?'
    optimizer_or_trainer._scale = 1. / optimizer_or_trainer._amp_loss_scaler.loss_scale
    if isinstance(loss, (list, tuple)):
        yield [l * optimizer_or_trainer._amp_loss_scaler.loss_scale for l in loss]
    else:
        yield optimizer_or_trainer._amp_loss_scaler.loss_scale * loss

def init():
    global _amp_initialized
    if not _amp_initialized:
        _amp_initialized = True
        print("AMP init!")
        _wrap_symbol_functions(symbol)
        _wrap_symbol_functions(Symbol)
        _wrap_symbol_functions(symbol_contrib)
        _wrap_symbol_functions(ndarray)
        _wrap_symbol_functions(NDArray)
        _wrap_symbol_functions(ndarray_contrib)

def init_trainer(optimizer_or_trainer, params=None):
    global _amp_loss_scale_initialized
    global _amp_initialized
    assert _amp_initialized, "AMP not initialized, did you forget to call amp.init()?"
    loss_scaler = LossScaler()
    if not _amp_loss_scale_initialized:
        _wrap_loss_output_functions(ndarray, loss_scaler)
        _wrap_loss_output_functions(symbol, loss_scaler)
        _amp_loss_scale_initialized = True
    #_wrap_output
    if isinstance(optimizer_or_trainer, trainer.Trainer):
        assert params == None, "optimizer_or_trainer is a trainer so params should be None."
        optimizer_or_trainer._amp_loss_scaler = loss_scaler
        skip_update = optimizer_or_trainer._amp_loss_scaler.wait_and_update
        optimizer_or_trainer._optimizer.old_update_multi_precision = optimizer_or_trainer._optimizer.update_multi_precision
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
        assert params == None, "optimizer_or_trainer is an optimizer so params should be None."
        raise TypeError("AMP is currently only compatible with Gluon Trainer")  # TODO(ptredak): make it work with the optimizer
    # TODO(cfujitsang): but not important because unlikely to be used
    #elif hasattr(optimizer_or_trainer, '__call__'):
    #    assert isinstance(params, dict), "optimizer_or_trainer is a function "
    #                                     "so params should be defined."
    #    raise NotImplementedError()
    else:
        raise TypeError("optimizer_or_trainer should be a Gluon Trainer, "
                        "an optimizer or a function, instead is %s" %
                        type(optimizer_or_trainer))
