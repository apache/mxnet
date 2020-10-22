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
from mxnet.test_utils import set_default_context, same_symbol_structure
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock, nn, rnn
from mxnet.operator import get_all_registered_operators_grouped
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import assert_raises_cudnn_not_satisfied
sys.path.insert(0, os.path.join(curr_path, '../train'))
set_default_context(mx.gpu(0))

@pytest.fixture()
def amp_tests(request):
    def teardown():
        mx.nd.waitall()

    request.addfinalizer(teardown)

def test_amp_coverage(amp_tests):
    conditional = [item[0] for item in amp.lists.symbol_fp16.CONDITIONAL_FP32_FUNCS]

    # Check for duplicates
    for a in [amp.lists.symbol_fp16.FP16_FUNCS,
          amp.lists.symbol_fp16.FP16_FP32_FUNCS,
          amp.lists.symbol_fp16.FP32_FUNCS,
          amp.lists.symbol_fp16.WIDEST_TYPE_CASTS,
          conditional]:
        ret = [item for item, count in collections.Counter(a).items() if count > 1]
        assert ret == [], "Elements " + str(ret) + " are duplicated in the AMP lists."

    t = []
    for a in [amp.lists.symbol_fp16.FP16_FUNCS,
              amp.lists.symbol_fp16.FP16_FP32_FUNCS,
              amp.lists.symbol_fp16.FP32_FUNCS,
              amp.lists.symbol_fp16.WIDEST_TYPE_CASTS,
              conditional]:
        t += a
    ret = [item for item, count in collections.Counter(t).items() if count > 1]
    assert ret == [], "Elements " + str(ret) + " exist in more than 1 AMP list."

    # Check the coverage
    covered = set(t)
    ops = get_all_registered_operators_grouped()
    required = set(k for k in ops
                   if not k.startswith(("_backward", "_contrib_backward", "_npi_backward")) and
                   not k.endswith("_backward"))

    extra = covered - required
    assert not extra, f"{len(extra)} operators are not needed in the AMP lists: {sorted(extra)}"

    guidelines = """Please follow these guidelines for choosing a proper list:
    - if your operator is not to be used in a computational graph
      (e.g. image manipulation operators, optimizers) or does not have
      inputs, put it in FP16_FP32_FUNCS list,
    - if your operator requires FP32 inputs or is not safe to use with lower
      precision, put it in FP32_FUNCS list,
    - if your operator supports both FP32 and lower precision, has
      multiple inputs and expects all inputs to be of the same
      type, put it in WIDEST_TYPE_CASTS list,
    - if your operator supports both FP32 and lower precision and has
      either a single input or supports inputs of different type,
      put it in FP16_FP32_FUNCS list,
    - if your operator is both safe to use in lower precision and
      it is highly beneficial to use it in lower precision, then
      put it in FP16_FUNCS (this is unlikely for new operators)
    - If you are not sure which list to choose, FP32_FUNCS is the
                     safest option"""
    diff = required - covered
    assert not diff, f"{len(diff)} operators {sorted(diff)} do not exist in AMP lists (in " \
        f"python/mxnet/amp/lists/symbol_fp16.py) - please add them. " \
        f"\n{guidelines}"

@pytest.mark.skip(reason='Error during waitall(). Tracked in #18099')
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_amp_conversion_rnn(amp_tests):
    with mx.Context(mx.gpu(0)):
        model = nn.HybridSequential()
        model.add(rnn.LSTM(hidden_size=10, num_layers=2, bidirectional=True))
        model.add(nn.Dense(2))
        model.initialize()
        model.hybridize()
        out = model(mx.nd.ones((2, 3, 4)))
        new_model = amp.convert_hybrid_block(model)
        out2 = new_model(mx.nd.ones((2, 3, 4)))
        mx.test_utils.assert_almost_equal(out.asnumpy(), out2.asnumpy(), atol=1e-2, rtol=1e-2)


def test_fp16_casting(amp_tests):
    data = mx.sym.var("data")
    out1 = mx.sym.amp_cast(data, dtype="float16")
    out2 = mx.sym.amp_cast(data, dtype="float32")
    out3 = mx.sym.amp_cast(data, dtype="float16")
    # When two ops from data, with different dtypes,
    # data should be float32
    res = mx.sym.Group([out1, out2])
    final_res = amp.convert_symbol(res, data_names=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.gpu(), data=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float32

    # When two ops from data, both casted to float16,
    # data should be float16
    res = mx.sym.Group([out1, out3])
    final_res = amp.convert_symbol(res, data_names=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.gpu(), data=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float16

    # AMP Multicast test where one node is float32, another is float16
    data = mx.sym.var("data", dtype=np.float32)
    data2 = mx.sym.var("data2", dtype=np.float16)
    out4 = mx.sym.amp_multicast(data, data2, num_outputs=2)
    final_res = amp.convert_symbol(out4, cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.gpu(), data2=(1, 2), data=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float16

    # AMP Multicast test where two non input nodes are float16,
    # and one input node is float32
    data = mx.sym.var("data", dtype=np.float32)
    data2 = mx.sym.var("data2", dtype=np.float16)
    data3 = mx.sym.var("data3", dtype=np.float16)
    out5 = mx.sym.amp_multicast(data,
                                mx.sym.elemwise_add(data2, data3),
                                num_outputs=2)
    final_res = amp.convert_symbol(out5, target_dtype_ops=[],
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.gpu(), data=(1, 2), data2=(1, 2), data3=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float32

    # AMP Multicast test where three input nodes one fp16, one fp32
    # one unknown
    data = mx.sym.var("data", dtype=np.float16)
    data2 = mx.sym.var("data2", dtype=np.float32)
    data3 = mx.sym.var("data3")
    out6 = mx.sym.amp_multicast(data, data2, data3, num_outputs=3)
    final_res = amp.convert_symbol(out6, target_dtype_ops=[],
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.gpu(), data=(1, 2), data2=(1, 2),
                                data3=(1, 2))
    assert exe.arg_arrays[2].dtype == np.float32

    # Input node to amp_multicast and amp_cast, if dtypes conflict
    # and input node is already fp16, it should still be fp16
    data = mx.sym.var("data", dtype=np.float16)
    data2 = mx.sym.var("data2", dtype=np.float32)
    out7 = mx.sym.Group([mx.sym.amp_multicast(data, data2, num_outputs=2), mx.sym.amp_cast(data, dtype="float16")])
    final_res = amp.convert_symbol(out7, target_dtype_ops=[],
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.gpu(), data=(1, 2), data2=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float16

    # Input node to amp_multicast and amp_cast, if dtypes conflict
    # and input node is already fp32, it should be changed to fp16
    data = mx.sym.var("data", dtype=np.float32)
    data2 = mx.sym.var("data2", dtype=np.float16)
    out8 = mx.sym.Group([mx.sym.amp_multicast(data, data2, num_outputs=2), mx.sym.amp_cast(data, dtype="float16")])
    final_res = amp.convert_symbol(out8, target_dtype_ops=[],
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res._simple_bind(ctx=mx.gpu(), data=(1, 2), data2=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float16

    # Check for symbol which has slice channel
    data = mx.sym.var("data")
    data2 = mx.sym.var("data2")
    data._set_attr(__dtype__="-1")
    data2._set_attr(__dtype__="-1")
    concat_res = mx.sym.concat(data, data2)
    out = mx.sym.split(concat_res, axis=1, num_outputs=2)
    final_res = amp.convert_symbol(out)

