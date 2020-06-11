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
import mxnet.contrib.amp as amp
import pytest
from mxnet.test_utils import set_default_context, download_model, same_symbol_structure, assert_almost_equal
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock, nn, rnn
from mxnet.contrib.amp import amp
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed

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
                       "python/mxnet/contrib/amp/lists/symbol_bf16.py) - please add them. "
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

@with_seed()
def test_amp_conversion():
    def check_amp_convert_symbol():
        x = mx.sym.var("x")
        y = mx.sym.var("y")
        z = mx.sym.FullyConnected(x, y, num_hidden=10, no_bias=True)
        siny = mx.sym.sin(y)
        res = z + siny
        # Compare symbols with similar computation graphs created using convert_symbol and manually.
        res_converted = amp.convert_symbol(res, target_dtype="bfloat16",
                                           target_dtype_ops=["FullyConnected"],
                                           fp32_ops=["sin"])
        x_bf16 = mx.sym.amp_cast(x, dtype=bfloat16)
        y_bf16 = mx.sym.amp_cast(y, dtype=bfloat16)
        siny = mx.sym.sin(y)
        z = mx.sym.FullyConnected(x_bf16, y_bf16, num_hidden=10, no_bias=True)
        amp_casted_z = mx.sym.amp_cast(z, dtype="float32")
        res_expected = amp_casted_z + siny
        assert same_symbol_structure(res_converted, res_expected), \
            "convert_symbol generating wrong computation graph"

        # convert_symbol called with incorrect inputs
        pytest.raises(AssertionError, amp.convert_symbol, res,
                      target_dtype="bfloat16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["elemwise_add"])
        pytest.raises(AssertionError, amp.convert_symbol, res,
                      target_dtype="bfloat16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["Activation"],
                      conditional_fp32_ops=[('Activation', 'act_type', ['selu'])])
        pytest.raises(AssertionError, amp.convert_symbol, res,
                      target_dtype="bfloat16", target_dtype_ops=["Activation"],
                      fp32_ops=["Activation"],
                      conditional_fp32_ops=[('Activation', 'act_type', ['selu'])])
        pytest.raises(AssertionError, amp.convert_symbol, res,
                      target_dtype="bfloat16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["FullyConnected"])

        # Test for op in conditional ops with condition not satisfied
        x = mx.sym.var("x")
        y = mx.sym.var("y")
        fc_cond = mx.sym.FullyConnected(x, y, num_hidden=10, no_bias=True)
        res_converted = amp.convert_symbol(fc_cond, target_dtype="bfloat16",
                                           target_dtype_ops=[],
                                           fp32_ops=["sin"],
                                           conditional_fp32_ops=[("FullyConnected", "no_bias", ["False"])])

        res_expected = mx.sym.FullyConnected(x, y, num_hidden=10, no_bias=True)
        assert same_symbol_structure(res_converted, res_expected), \
           "convert_symbol generating wrong computation graph when conditional ops is used"

        # Test for op in conditional ops with condition satisfied
        res_converted = amp.convert_symbol(fc_cond, target_dtype="bfloat16", target_dtype_ops=[],
                                           fp32_ops=["sin"],
                                           conditional_fp32_ops=[("FullyConnected", "no_bias", ["True"])])
        x_fp32 = mx.sym.amp_cast(x, dtype="float32")
        y_fp32 = mx.sym.amp_cast(y, dtype="float32")
        res_expected = mx.sym.FullyConnected(x_fp32, y_fp32, num_hidden=10, no_bias=True)
        assert same_symbol_structure(res_converted, res_expected), \
           "convert_symbol generating wrong computation graph when conditional ops used with satisfying condition"

        # Test with a real world model, default inputs for convert_symbol
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, 'model')
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        prefix, epoch = download_model("imagenet1k-resnet-18", dst_dir=model_path)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        inputs = {}
        inputs['data'] = mx.nd.ones((1, 3, 224, 224))
        inputs.update(arg_params)
        converted_sym = amp.convert_symbol(sym, target_dtype="bfloat16")
        exe = converted_sym.simple_bind(mx.cpu(), data=(1, 3, 224, 224), grad_req='null')
        exe.forward(is_train=False, **inputs)
        exe.outputs[0].asnumpy()

        inputs_bf16 = {}
        inputs_bf16['data'] = mx.nd.ones((1, 3, 224, 224))
        inputs_bf16['fc1_weight'] = mx.nd.amp_cast(inputs['fc1_weight'], dtype=bfloat16)
        inputs_bf16['fc1_bias'] = mx.nd.amp_cast(inputs['fc1_bias'], dtype=bfloat16)

        # Test with a real world model, tweak inputs for convert_symbol
        converted_sym = amp.convert_symbol(sym, target_dtype="bfloat16",
                                           target_dtype_ops=["Convolution"], data_names=["data"],
                                           cast_optional_params=True)
        converted_sym2 = amp.convert_symbol(sym, target_dtype="bfloat16",
                                            target_dtype_ops=["Convolution"], data_names=["data"],
                                            cast_optional_params=False)

        exe = converted_sym.simple_bind(mx.cpu(), data=(1, 3, 224, 224), grad_req='null')
        exe2 = converted_sym2.simple_bind(mx.cpu(), data=(1, 3, 224, 224), grad_req='null')

        converted_args = converted_sym.list_arguments()
        converted_auxs = converted_sym.list_auxiliary_states()
        for i, key in enumerate(exe.arg_arrays):
            if converted_args[i] in arg_params:
                arg_dtype = exe.arg_arrays[i].dtype
                if arg_dtype == bfloat16:
                    arg_params[converted_args[i]] = mx.nd.amp_cast(arg_params[converted_args[i]], dtype=bfloat16)
                else:
                    arg_params[converted_args[i]] = arg_params[converted_args[i]].astype(arg_dtype)
        for i, key in enumerate(exe.aux_arrays):
            aux_dtype = exe.aux_arrays[i].dtype
            if converted_auxs[i] in aux_params:
                if arg_dtype == bfloat16:
                    aux_params[converted_auxs[i]] = mx.nd.amp_cast(aux_params[converted_auxs[i]], dtype=bfloat16)
                else:
                    aux_params[converted_auxs[i]] = aux_params[converted_auxs[i]].astype(aux_dtype)

        inputs_bf16.update(arg_params)
        exe.forward(is_train=False, **inputs_bf16)
        exe.outputs[0].wait_to_read()

        exe2.forward(is_train=False, **inputs)
        exe2.outputs[0].wait_to_read()

    def check_amp_convert_hybrid_block():
        # Test conversion for hybrid block on CPU
        model_cpu = get_model("resnet50_v1")
        model_cpu.initialize(ctx=mx.cpu())
        model_cpu.hybridize()
        model_cpu(mx.nd.random.uniform(0, 1, shape=(1, 3, 224, 224), ctx=mx.cpu()))
        converted_model_cpu = amp.convert_hybrid_block(model_cpu, target_dtype="bfloat16", ctx=mx.cpu())

        # Test with real world model, default inputs for convert_hybrid_block
        model = get_model("resnet50_v1")
        model.initialize(ctx=mx.cpu())
        model.hybridize()
        model(mx.nd.zeros((1, 3, 224, 224)))
        converted_model = amp.convert_hybrid_block(model, target_dtype="bfloat16", ctx=mx.cpu())
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224),
                                                     dtype=np.float32))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224),
                                                     dtype=np.float32))

        # Test with real world model, tweak inputs for convert_hybrid_block
        converted_model = amp.convert_hybrid_block(model, target_dtype="bfloat16",
                                                   target_dtype_ops=["Convolution"], ctx=mx.cpu())
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224),
                                                      dtype=np.float32))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224),
                                                     dtype=np.float32))

        # Check symbolic block
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, 'model')
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        prefix, epoch = download_model("imagenet1k-resnet-18", dst_dir=model_path)
        net = SymbolBlock.imports(os.path.join(model_path, "imagenet1k-resnet-18-symbol.json"),
                                  input_names=["data", "softmax_label"],
                                  param_file=os.path.join(model_path, "imagenet1k-resnet-18-0000.params"))
        net.reset_ctx(ctx=mx.cpu())
        net.hybridize()
        net(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1,)))
        converted_model = amp.convert_hybrid_block(net, target_dtype="bfloat16", ctx=mx.cpu())
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1,)))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1,)))

        # Check symbolic block, tweaked inputs
        converted_model = amp.convert_hybrid_block(net, target_dtype="bfloat16", target_dtype_ops=["Convolution"], ctx=mx.cpu())
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1, )))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1, )))
        params = converted_model.collect_params()
        assert params["stage2_unit1_conv2_weight"].dtype == np.float32

        # Pass cast_optional_params as True to convert_hybrid_block
        converted_model = amp.convert_hybrid_block(net, target_dtype="bfloat16", target_dtype_ops=["Convolution"],
                                                   cast_optional_params=True, ctx=mx.cpu())
        params = converted_model.collect_params()
        assert params["stage2_unit1_conv2_weight"].dtype == bfloat16

    check_amp_convert_symbol()
    check_amp_convert_hybrid_block()

@with_seed()
def test_bf16_casting():
    data = mx.sym.var("data")
    out1 = mx.sym.amp_cast(data, dtype=bfloat16)
    out2 = mx.sym.amp_cast(data, dtype="float32")
    out3 = mx.sym.amp_cast(data, dtype=bfloat16)
    # When two ops from data, with different dtypes,
    # data should be float32
    res = mx.sym.Group([out1, out2])
    final_res = amp.convert_symbol(res, data_names=[], target_dtype="bfloat16", cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.cpu(), data=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float32

    # When two ops from data, both casted to bfloat16,
    # data should be bfloat16
    res = mx.sym.Group([out1, out3])
    final_res = amp.convert_symbol(res, data_names=[], target_dtype="bfloat16", cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.cpu(), data=(1, 2))
    assert exe.arg_arrays[0].dtype == bfloat16

    # AMP Multicast test where one node is float32, another is bfloat16
    data = mx.sym.var("data", dtype="float32")
    data2 = mx.sym.var("data2", dtype=bfloat16)
    out4 = mx.sym.amp_multicast(data, data2, num_outputs=2)
    final_res = amp.convert_symbol(out4, target_dtype="bfloat16", cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.cpu(), data2=(1, 2), data=(1, 2))
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
    exe = final_res.simple_bind(ctx=mx.cpu(), data=(1, 2), data2=(1, 2), data3=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float32

    # AMP Multicast test where three input nodes one bf16, one fp32
    # one unknown
    data = mx.sym.var("data", dtype=bfloat16)
    data2 = mx.sym.var("data2", dtype="float32")
    data3 = mx.sym.var("data3")
    out6 = mx.sym.amp_multicast(data, data2, data3, num_outputs=3)
    final_res = amp.convert_symbol(out6, target_dtype_ops=[], target_dtype="bfloat16",
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.cpu(), data=(1, 2), data2=(1, 2),
                                data3=(1, 2))
    assert exe.arg_arrays[2].dtype == np.float32

    # Input node to amp_multicast and amp_cast, if dtypes conflict
    # and input node is already bf16, it should still be bf16
    data = mx.sym.var("data", dtype=bfloat16)
    data2 = mx.sym.var("data2", dtype="float32")
    out7 = mx.sym.Group([mx.sym.amp_multicast(data, data2, num_outputs=2), mx.sym.amp_cast(data, dtype=bfloat16)])
    final_res = amp.convert_symbol(out7, target_dtype_ops=[], target_dtype="bfloat16",
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.cpu(), data=(1, 2), data2=(1, 2))
    assert exe.arg_arrays[0].dtype == bfloat16

    # Input node to amp_multicast and amp_cast, if dtypes conflict
    # and input node is already fp32, it should be changed to bf16
    data = mx.sym.var("data", dtype="float32")
    data2 = mx.sym.var("data2", dtype=bfloat16)
    out8 = mx.sym.Group([mx.sym.amp_multicast(data, data2, num_outputs=2), mx.sym.amp_cast(data, dtype=bfloat16)])
    final_res = amp.convert_symbol(out8, target_dtype_ops=[], target_dtype="bfloat16",
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.cpu(), data=(1, 2), data2=(1, 2))
    assert exe.arg_arrays[0].dtype == bfloat16

