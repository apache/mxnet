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

import os
import sys
import mxnet as mx
import numpy as np
from random import randint
import warnings
import collections
import ctypes
from mxnet import amp
import pytest
from mxnet.test_utils import set_default_context, same_symbol_structure, assert_almost_equal
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock, nn, rnn
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))

bfloat16 = np.dtype([('bfloat16', np.uint16)])

def test_amp_coverage():
    conditional = [item[0] for item in amp.lists.symbol_bf16.CONDITIONAL_FP32_FUNCS]

    # Check for duplicates
    for a in [amp.lists.symbol_bf16.BF16_FUNCS,
          amp.lists.symbol_bf16.BF16_FP32_FUNCS,
          amp.lists.symbol_bf16.FP32_FUNCS,
          amp.lists.symbol_bf16.WIDEST_TYPE_CASTS,
          conditional]:
        ret = [item for item, count in collections.Counter(a).items() if count > 1]
        assert ret == [], "Elements " + str(ret) + " are duplicated in the AMP lists."

    t = []
    for a in [amp.lists.symbol_bf16.BF16_FUNCS,
              amp.lists.symbol_bf16.BF16_FP32_FUNCS,
              amp.lists.symbol_bf16.FP32_FUNCS,
              amp.lists.symbol_bf16.WIDEST_TYPE_CASTS,
              conditional]:
        t += a
    ret = [item for item, count in collections.Counter(t).items() if count > 1]
    assert ret == [], "Elements " + str(ret) + " exist in more than 1 AMP list."

    # Check the coverage
    py_str = lambda x: x.decode('utf-8')

    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    mx.base._LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist))
    op_names = []
    for i in range(size.value):
        s = py_str(plist[i])
        if not s.startswith("_backward") \
           and not s.startswith("_contrib_backward_"):
            op_names.append(s)

    ret1 = set(op_names) - set(t)

    if ret1 != set():
        warnings.warn("Operators " + str(ret1) + " do not exist in AMP lists (in "
                       "python/mxnet/amp/lists/symbol_bf16.py) - please add them. "
                       """Please follow these guidelines for choosing a proper list:
                       - if your operator is not to be used in a computational graph
                         (e.g. image manipulation operators, optimizers) or does not have
                         inputs, put it in BF16_FP32_FUNCS list,
                       - if your operator requires FP32 inputs or is not safe to use with lower
                         precision, put it in FP32_FUNCS list,
                       - if your operator supports both FP32 and lower precision, has
                         multiple inputs and expects all inputs to be of the same
                         type, put it in WIDEST_TYPE_CASTS list,
                       - if your operator supports both FP32 and lower precision and has
                         either a single input or supports inputs of different type,
                         put it in BF16_FP32_FUNCS list,
                       - if your operator is both safe to use in lower precision and
                         it is highly beneficial to use it in lower precision, then
                         put it in BF16_FUNCS (this is unlikely for new operators)
                       - If you are not sure which list to choose, FP32_FUNCS is the
                         safest option""")

def test_bf16_casting():
    data = mx.sym.var("data")
    out1 = mx.sym.amp_cast(data, dtype=bfloat16)
    out2 = mx.sym.amp_cast(data, dtype="float32")
    out3 = mx.sym.amp_cast(data, dtype=bfloat16)
    # When two ops from data, with different dtypes,
    # data should be float32
    res = mx.sym.Group([out1, out2])
    final_res = amp.convert_symbol(res, data_names=[], target_dtype="bfloat16", cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.cpu(), data=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float32

    # When two ops from data, both casted to bfloat16,
    # data should be bfloat16
    res = mx.sym.Group([out1, out3])
    final_res = amp.convert_symbol(res, data_names=[], target_dtype="bfloat16", cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.cpu(), data=(1, 2))
    assert exe.arg_arrays[0].dtype == bfloat16

    # AMP Multicast test where one node is float32, another is bfloat16
    data = mx.sym.var("data", dtype="float32")
    data2 = mx.sym.var("data2", dtype=bfloat16)
    out4 = mx.sym.amp_multicast(data, data2, num_outputs=2)
    final_res = amp.convert_symbol(out4, target_dtype="bfloat16", cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.cpu(), data2=(1, 2), data=(1, 2))
    assert exe.arg_arrays[0].dtype == bfloat16

    # AMP Multicast test where two non input nodes are bfloat16,
    # and one input node is float32
    data = mx.sym.var("data", dtype="float32")
    data2 = mx.sym.var("data2", dtype=bfloat16)
    data3 = mx.sym.var("data3", dtype=bfloat16)
    out5 = mx.sym.amp_multicast(data,
                                mx.sym.elemwise_add(data2, data3),
                                num_outputs=2)
    final_res = amp.convert_symbol(out5, target_dtype_ops=[], target_dtype="bfloat16",
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.cpu(), data=(1, 2), data2=(1, 2), data3=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float32

    # AMP Multicast test where three input nodes one bf16, one fp32
    # one unknown
    data = mx.sym.var("data", dtype=bfloat16)
    data2 = mx.sym.var("data2", dtype="float32")
    data3 = mx.sym.var("data3")
    out6 = mx.sym.amp_multicast(data, data2, data3, num_outputs=3)
    final_res = amp.convert_symbol(out6, target_dtype_ops=[], target_dtype="bfloat16",
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.cpu(), data=(1, 2), data2=(1, 2),
                                data3=(1, 2))
    assert exe.arg_arrays[2].dtype == np.float32

    # Input node to amp_multicast and amp_cast, if dtypes conflict
    # and input node is already bf16, it should still be bf16
    data = mx.sym.var("data", dtype=bfloat16)
    data2 = mx.sym.var("data2", dtype="float32")
    out7 = mx.sym.Group([mx.sym.amp_multicast(data, data2, num_outputs=2), mx.sym.amp_cast(data, dtype=bfloat16)])
    final_res = amp.convert_symbol(out7, target_dtype_ops=[], target_dtype="bfloat16",
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.cpu(), data=(1, 2), data2=(1, 2))
    assert exe.arg_arrays[0].dtype == bfloat16

    # Input node to amp_multicast and amp_cast, if dtypes conflict
    # and input node is already fp32, it should be changed to bf16
    data = mx.sym.var("data", dtype="float32")
    data2 = mx.sym.var("data2", dtype=bfloat16)
    out8 = mx.sym.Group([mx.sym.amp_multicast(data, data2, num_outputs=2), mx.sym.amp_cast(data, dtype=bfloat16)])
    final_res = amp.convert_symbol(out8, target_dtype_ops=[], target_dtype="bfloat16",
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.cpu(), data=(1, 2), data2=(1, 2))
    assert exe.arg_arrays[0].dtype == bfloat16
