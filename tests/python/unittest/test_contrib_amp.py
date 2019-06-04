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
import warnings
import collections
import ctypes
import mxnet.contrib.amp as amp
from nose.tools import assert_raises
from mxnet.test_utils import set_default_context, download_model, compare_symbol_structure
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock
from mxnet.contrib.amp import amp
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed, teardown

def test_amp_coverage():
    conditional = [item[0] for item in amp.lists.symbol.CONDITIONAL_FP32_FUNCS]

    # Check for duplicates
    for a in [amp.lists.symbol.FP16_FUNCS,
          amp.lists.symbol.FP16_FP32_FUNCS,
          amp.lists.symbol.FP32_FUNCS,
          amp.lists.symbol.WIDEST_TYPE_CASTS,
          conditional]:
        ret = [item for item, count in collections.Counter(a).items() if count > 1]
        assert ret == [], "Elements " + str(ret) + " are duplicated in the AMP lists."

    t = []
    for a in [amp.lists.symbol.FP16_FUNCS,
              amp.lists.symbol.FP16_FP32_FUNCS,
              amp.lists.symbol.FP32_FUNCS,
              amp.lists.symbol.WIDEST_TYPE_CASTS,
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
                       "python/mxnet/contrib/amp/lists/symbol.py) - please add them. "
                       """Please follow these guidelines for choosing a proper list:
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
        res_converted = amp.convert_symbol(res, target_dtype="float16", target_dtype_ops=["FullyConnected"],
                                           fp32_ops=["sin"])

        x_fp16 = mx.sym.amp_cast(x, dtype="float16")
        y_fp16 = mx.sym.amp_cast(y, dtype="float16")
        amp_casted_siny = mx.sym.sin(mx.sym.amp_cast(y, dtype="float32"))
        z = mx.sym.FullyConnected(x_fp16, y_fp16, num_hidden=10, no_bias=True)
        outs = mx.sym.amp_multicast(z, amp_casted_siny, num_outputs=2)
        res_expected = outs[0] + outs[1]
        assert compare_symbol_structure(res_converted, res_expected), "convert_symbol generating wrong computation graph"

        # convert_symbol called with incorrect inputs
        assert_raises(AssertionError, amp.convert_symbol, res, target_dtype="float16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["elemwise_add"])
        assert_raises(AssertionError, amp.convert_symbol, res, target_dtype="float16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["Activation"], conditional_fp32_ops=[('Activation', 'act_type', ['selu'])])
        assert_raises(AssertionError, amp.convert_symbol, res, target_dtype="float16", target_dtype_ops=["Activation"],
                      fp32_ops=["Activation"], conditional_fp32_ops=[('Activation', 'act_type', ['selu'])])
        assert_raises(AssertionError, amp.convert_symbol, res, target_dtype="float16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["FullyConnected"])

        # Test for op in conditional ops with condition not satisfied
        x = mx.sym.var("x")
        y = mx.sym.var("y")
        fc_cond = mx.sym.FullyConnected(x, y, num_hidden=10, no_bias=True)
        res_converted = amp.convert_symbol(fc_cond, target_dtype="float16", target_dtype_ops=[],
                                           fp32_ops=["sin"], conditional_fp32_ops=[("FullyConnected", "no_bias", ["False"])])

        res_expected = mx.sym.FullyConnected(x, y, num_hidden=10, no_bias=True)
        assert compare_symbol_structure(res_converted, res_expected), "convert_symbol generating wrong computation graph when conditional ops is used"

        # Test for op in conditional ops with condition satisfied
        res_converted = amp.convert_symbol(fc_cond, target_dtype="float16", target_dtype_ops=[],
                                           fp32_ops=["sin"], conditional_fp32_ops=[("FullyConnected", "no_bias", ["True"])])
        x_fp32 = mx.sym.amp_cast(x, dtype="float32")
        y_fp32 = mx.sym.amp_cast(y, dtype="float32")
        res_expected = mx.sym.FullyConnected(x_fp32, y_fp32, num_hidden=10, no_bias=True)
        assert compare_symbol_structure(res_converted, res_expected), "convert_symbol generating wrong computation graph when conditional ops used with satisfying condition"

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
        converted_sym = amp.convert_symbol(sym)
        exe = converted_sym.simple_bind(mx.gpu(0), data=(1, 3, 224, 224), grad_req='null')
        exe.forward(is_train=False, **inputs)
        exe.outputs[0].asnumpy()

        inputs['fc1_weight'] = inputs['fc1_weight'].astype(np.float16)
        inputs['fc1_bias'] = inputs['fc1_bias'].astype(np.float16)

        # Test with a real world model, tweak inputs for convert_symbol
        converted_sym = amp.convert_symbol(sym, target_dtype="float16", target_dtype_ops=["Convolution"], data_names=["data"])
        exe = converted_sym.simple_bind(mx.gpu(0), data=(1, 3, 224, 224), grad_req='null')
        exe.forward(is_train=False, **inputs)
        exe.outputs[0].wait_to_read()

    def check_amp_convert_model():
        # Test with real world model, default inputs for convert_model
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, 'model')
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        prefix, epoch = download_model("imagenet1k-resnet-18", dst_dir=model_path)

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

        # Test with real world model, tweak inputs for convert_model
        result_sym, result_arg_params, result_aux_params = amp.convert_model(sym, arg_params, aux_params, target_dtype="float16", target_dtype_ops=["Convolution"])
        mod = mx.mod.Module(result_sym, data_names=["data"], label_names=["softmax_label"], context=mx.gpu(0))
        mod.bind(data_shapes=[['data', (1, 3, 224, 224)]], label_shapes=[['softmax_label', (1,)]])

        mod.set_params(result_arg_params, result_aux_params)
        mod.forward(mx.io.DataBatch(data=[mx.nd.ones((1, 3, 224, 224), ctx=mx.gpu(0))],
                                    label=[mx.nd.ones((1,), ctx=mx.gpu(0))]))
        mod.get_outputs()[0].asnumpy()

    def check_amp_convert_hybrid_block():
        # Test with real world model, default inputs for convert_hybrid_block
        model = get_model("resnet50_v1")
        model.collect_params().initialize(ctx=mx.gpu())
        model.hybridize()
        model(mx.nd.zeros((1, 3, 224, 224)))
        converted_model = amp.convert_hybrid_block(model)
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224), dtype=np.float32, ctx=mx.gpu(0)))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224), dtype=np.float32, ctx=mx.gpu(0)))

        # Test with real world model, tweak inputs for convert_hybrid_block
        converted_model = amp.convert_hybrid_block(model, target_dtype="float16",
                                                   target_dtype_ops=["Convolution"])
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224), dtype=np.float32, ctx=mx.gpu(0)))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224), dtype=np.float32, ctx=mx.gpu(0)))

        # Check symbolic block
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, 'model')
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        prefix, epoch = download_model("imagenet1k-resnet-18", dst_dir=model_path)
        net = SymbolBlock.imports(os.path.join(model_path, "imagenet1k-resnet-18-symbol.json"),
                                  input_names=["data", "softmax_label"],
                                  param_file=os.path.join(model_path, "imagenet1k-resnet-18-0000.params"))
        net.collect_params().reset_ctx(ctx=mx.gpu())
        net.hybridize()
        net(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1,)))
        converted_model = amp.convert_hybrid_block(net)
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1,)))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1,)))

        # Check symbolic block, tweaked inputs
        converted_model = amp.convert_hybrid_block(net, target_dtype="float16", target_dtype_ops=["Convolution"])
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1, )))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224)), mx.nd.zeros((1, )))

    with mx.Context(mx.gpu(0)):
        check_amp_convert_symbol()
        check_amp_convert_model()
        check_amp_convert_hybrid_block()


if __name__ == '__main__':
    import nose
    nose.runmodule()
