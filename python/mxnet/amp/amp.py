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
import traceback

from .. import symbol
from ..symbol import Symbol
from .. import ndarray
from ..ndarray import NDArray
from . import lists

def _cast_symbol_NDArray(s, dtype):
    print("Trying to cast... " + str(type(s)))
    if isinstance(s, Symbol):
        print("Encountered symbol, casting to " + str(dtype))
        return symbol.cast(s, dtype=dtype)
    elif isinstance(s, NDArray):
        print("Encountered NDArray, with dtype = " + str(s.dtype))
        if s.dtype != dtype:
            print("Casting to " + str(dtype))
            return ndarray.cast(s, dtype=dtype)
        else:
            return s
    else:
        return s

def _wrap_symbol_functions(module):
    def symbol_wrapper(f, target_dtype):
        def new_fun(*args, **kwargs):
            print("Wrapper of " + f.__name__ + " to " + str(target_dtype))
            print(locals())
            print("Called from:")
            traceback.print_stack()
            print("Casting *args")
            print(args)
            new_args = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype), args))
            print(new_args)
            args = tuple(new_args)
            print(args)
            print("Casting **kwargs")
            print(kwargs)
            kwargs = {k: _cast_symbol_NDArray(v, target_dtype) for k, v in kwargs.items()}
            print(kwargs)
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
