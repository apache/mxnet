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
__all__ = ['init']

import numpy as np

from .. import symbol
from ..symbol import Symbol
from .. import ndarray
from ..ndarray import NDArray
from . import lists

def _cast_symbol_NDArray(s, dtype):
    print("Trying to cast... " + str(type(s)))
    if isinstance(s, Symbol):
        print("Encountered symbol, casting to " + str(dtype))
        return symbol.amp_cast(s, dtype=dtype)
    elif isinstance(s, NDArray):
        print("Encountered NDArray, with dtype = " + str(s.dtype))
        if s.dtype != dtype and (s.dtype == np.float16 or s.dtype == np.float32):
            print("Casting to " + str(dtype))
            return ndarray.amp_cast(s, dtype=dtype)
        else:
            return s
    else:
        return s

def _wrap_symbol_functions(module):
    def symbol_wrapper(f, target_dtype, cond_arg=None):
        def new_fun(*args, **kwargs):
            print("Wrapper of " + f.__name__ + " to " + str(target_dtype))
            print("Cond_arg: " + str(cond_arg))
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                    kwargs[cond_arg[0]] not in cond_arg[1]):
                    print("No match for " + str(cond_arg))
                    return f(*args, **kwargs)
            print("Casting *args")
            new_args = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype), args))
            args = tuple(new_args)
            print("Casting **kwargs")
            kwargs = {k: _cast_symbol_NDArray(v, target_dtype) for k, v in kwargs.items()}
            return f(*args, **kwargs)
        return new_fun

    def symbol_widest_wrapper(f):
        def new_fun(*args, **kwargs):
            print("Wrapper of " + f.__name__ + " to widest type")
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
        return new_fun

    for fun_name in lists.symbol.FP16_FUNCS:
        print("Wrapping fp16 func " + fun_name + " in " + module.__name__)
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, symbol_wrapper(f_to_wrap, np.float16))
        except AttributeError:
            print("Function " + fun_name + " does not exist in " + module.__name__ + ".")
            pass

    for fun_name in lists.symbol.FP32_FUNCS:
        print("Wrapping fp32 func " + fun_name + " in " + module.__name__)
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, symbol_wrapper(f_to_wrap, np.float32))
        except AttributeError:
            print("Function " + fun_name + " does not exist in " + module.__name__ + ".")
            pass

    for fun_name, arg, arg_values in lists.symbol.CONDITIONAL_FP32_FUNCS:
        print("Wrapping fp32 func " + fun_name + " in " + module.__name__)
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, symbol_wrapper(f_to_wrap, np.float32, (arg, arg_values)))
        except AttributeError:
            print("Function " + fun_name + " does not exist in " + module.__name__ + ".")
            pass

    for fun_name in lists.symbol.WIDEST_TYPE_CASTS:
        print("Wrapping widest cast func " + fun_name + " in " + module.__name__)
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, symbol_widest_wrapper(f_to_wrap))
        except AttributeError:
            print("Function " + fun_name + " does not exist in " + module.__name__ + ".")
            pass

class AMPHandle(object):
    def __init__(self):
        super(AMPHandle, self).__init__()
        self.loss_scale = 128.0

def init():
    print("AMP init!")
    _wrap_symbol_functions(symbol)
    _wrap_symbol_functions(Symbol)
    _wrap_symbol_functions(ndarray)
    _wrap_symbol_functions(NDArray)
    return AMPHandle()
