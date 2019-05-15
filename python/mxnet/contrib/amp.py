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
"""Contrib AMP API to demonstrate conversion to mixed precision model"""

import ctypes
import numpy as np
from .. import ndarray as nd
from array import array
from ..base import _LIB, check_call
from ..base import mx_uint, c_str_array, c_array_buf
from ..base import SymbolHandle
from ..symbol import Symbol
from ..ndarray import _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from ..model import load_checkpoint
import copy
from ..context import cpu

def convert_symbol(sym, target_dtype="float16", target_dtype_ops=None,
                   fp32_ops=None, widest_dtype_ops=None, conditional_fp32_ops=None,
                   excluded_sym_names=None, input_names=["data"]):
    """Given a symbol object representing a neural network of data type FP32 and target_dtype,
    add cast layers according to the op lists (target_dtype_ops, fp32_ops,
    widest_dtype_ops, conditional_fp32_ops) if provided, otherwise use the default
    lists provided by the framework.

    Parameters
    ----------
    sym : Symbol
        FP32 neural network symbol
    target_dtype : str or numpy
        currently only supports float16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_dtype_ops : list of strs
        Override the list of operator names casted to the target_dtype.
        If None, uses the framework's default list to be casted to target_dtype.
    fp32_ops : list of strs
        Override the list of operator names casted to FP32.
        If None, uses the framework's default list to be casted to FP32.
    widest_dtype_ops : list of strs
        Override the list of operator names which should run in widest precision among its
        input arguments.
        If None, uses the framework's default list of widest_dtype_ops.
    conditional_fp32_ops : list of (string, string, list of string)
        Override the list of functions to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to FP32)
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being casted.
    input_names : list of strs
        A list of strings containing input variable names
    """
    if target_dtype != "float16":
        raise ValueError("Only target_dtype float16 is supported currently")
    num_target_dtype_ops = 0
    num_fp32_ops = 0
    num_widest_dtype_ops = 0
    num_conditional_fp32_ops = 0
    num_excluded_syms = 0

    if (input_names is None or len(input_names) <= 0):
        raise ValueError("input_names cannot be None or empty list")

    if target_dtype_ops is not None:
        assert isinstance(target_dtype_ops, list)
        num_target_dtype_ops = len(target_dtype_ops)
    else:
        target_dtype_ops = []

    if fp32_ops is not None:
        assert isinstance(fp32_ops, list)
        num_fp32_ops = len(fp32_ops)
    else:
        fp32_ops = []

    if widest_dtype_ops is not None:
        assert isinstance(widest_dtype_ops, list)
        num_widest_dtype_ops = len(widest_dtype_ops)
    else:
        widest_dtype_ops = []

    if conditional_fp32_ops is not None:
        assert isinstance(conditional_fp32_ops, list)
        num_conditional_fp32_ops = len(conditional_fp32_ops)
    else:
        conditional_fp32_ops = []

    if excluded_sym_names is not None:
        assert isinstance(excluded_sym_names, list)
        num_excluded_syms = len(excluded_sym_names)
    else:
        excluded_sym_names = []

    target_dtype = _DTYPE_NP_TO_MX[np.dtype(target_dtype).type]

    str_keys = []
    sdata = []
    for k in input_names:
        str_keys.append(k)
        sdata.append(0)
    keys = c_str_array(str_keys)

    out = SymbolHandle()
    check_call(_LIB.MXReducePrecisionSymbol(sym.handle,
                                            ctypes.byref(out),
                                            mx_uint(len(sdata)),
                                            c_array_buf(ctypes.c_int, array('i', sdata)),
                                            ctypes.byref(ctypes.c_int(target_dtype)),
                                            mx_uint(num_target_dtype_ops),
                                            mx_uint(num_fp32_ops),
                                            mx_uint(num_widest_dtype_ops),
                                            mx_uint(num_conditional_fp32_ops),
                                            mx_uint(num_excluded_syms),
                                            c_str_array(target_dtype_ops),
                                            c_str_array(fp32_ops),
                                            c_str_array(widest_dtype_ops),
                                            c_str_array(conditional_fp32_ops),
                                            c_str_array(excluded_sym_names),
                                            keys))
    return Symbol(out)

def convert_model(sym, arg_params, aux_params, target_dtype="float16", target_dtype_ops=None,
                  fp32_ops=None, widest_dtype_ops=None,
                  conditional_fp32_ops=None, excluded_sym_names=None, input_names=['data']):
    """API for converting a model from FP32 model to a mixed precision model.
    MXNet tries to convert the FP32 model to mixed precision model by adding
    cast layers using amp_cast and amp_multicast operators. The decision on
    which cast layer to add is based on hardcoded lists for Automatic Mixed Precision
    in MXNet. These lists can be overridden by the user by providing their own lists
    using : targe_precision_ops, fp32_ops, widest_precision_ops, conditional_fp32_ops

    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    target_dtype : str
        Currently only supports float16. The target dtype indicates to add cast layers
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
        from being quantized.
    input_names : list of strs
        A list of strings containing input variable names
    """
    if excluded_sym_names is None:
        excluded_sym_names = []
        if not isinstance(excluded_sym_names, list):
            raise ValueError('excluded_sym_names must be a list of strings representing'
                             ' the names of the symbols that should not be casted,'
                             ' while received type %s' % str(type(excluded_sym_names)))

    if target_dtype != "float16":
        raise ValueError("Only target_dtype float16 is supported currently")

    sym = convert_symbol(sym, target_dtype, target_dtype_ops,
                         fp32_ops, widest_dtype_ops, conditional_fp32_ops,
                         excluded_sym_names)
    return sym, arg_params, aux_params
