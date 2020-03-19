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
from nose.tools import assert_raises
from mxnet.test_utils import set_default_context, download_model, same_symbol_structure
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock, nn, rnn
from mxnet.contrib.amp import amp
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed, teardown, assert_raises_cudnn_not_satisfied
sys.path.insert(0, os.path.join(curr_path, '../train'))
from test_bucketing import train_model
set_default_context(mx.gpu(0))

def test_amp_coverage():
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
                       "python/mxnet/contrib/amp/lists/symbol_fp16.py) - please add them. "
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
        res_converted = amp.convert_symbol(res, target_dtype="float16",
                                           target_dtype_ops=["FullyConnected"],
                                           fp32_ops=["sin"])

        x_fp16 = mx.sym.amp_cast(x, dtype="float16")
        y_fp16 = mx.sym.amp_cast(y, dtype="float16")
        siny = mx.sym.sin(y)
        z = mx.sym.FullyConnected(x_fp16, y_fp16, num_hidden=10, no_bias=True)
        amp_casted_z = mx.sym.amp_cast(z, dtype="float32")
        res_expected = amp_casted_z + siny
        assert same_symbol_structure(res_converted, res_expected), \
            "convert_symbol generating wrong computation graph"

        # convert_symbol called with incorrect inputs
        assert_raises(AssertionError, amp.convert_symbol, res,
                      target_dtype="float16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["elemwise_add"])
        assert_raises(AssertionError, amp.convert_symbol, res,
                      target_dtype="float16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["Activation"],
                      conditional_fp32_ops=[('Activation', 'act_type', ['selu'])])
        assert_raises(AssertionError, amp.convert_symbol, res,
                      target_dtype="float16", target_dtype_ops=["Activation"],
                      fp32_ops=["Activation"],
                      conditional_fp32_ops=[('Activation', 'act_type', ['selu'])])
        assert_raises(AssertionError, amp.convert_symbol, res,
                      target_dtype="float16", target_dtype_ops=["FullyConnected"],
                      fp32_ops=["FullyConnected"])

        # Test for op in conditional ops with condition not satisfied
        x = mx.sym.var("x")
        y = mx.sym.var("y")
        fc_cond = mx.sym.FullyConnected(x, y, num_hidden=10, no_bias=True)
        res_converted = amp.convert_symbol(fc_cond, target_dtype="float16",
                                           target_dtype_ops=[],
                                           fp32_ops=["sin"],
                                           conditional_fp32_ops=[("FullyConnected", "no_bias", ["False"])])

        res_expected = mx.sym.FullyConnected(x, y, num_hidden=10, no_bias=True)
        assert same_symbol_structure(res_converted, res_expected), \
            "convert_symbol generating wrong computation graph when conditional ops is used"

        # Test for op in conditional ops with condition satisfied
        res_converted = amp.convert_symbol(fc_cond, target_dtype="float16", target_dtype_ops=[],
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
        converted_sym = amp.convert_symbol(sym)
        exe = converted_sym.simple_bind(mx.gpu(0), data=(1, 3, 224, 224), grad_req='null')
        exe.forward(is_train=False, **inputs)
        exe.outputs[0].asnumpy()

        inputs2 = {}
        inputs2['data'] = mx.nd.ones((1, 3, 224, 224))
        inputs2['fc1_weight'] = inputs['fc1_weight'].astype(np.float16)
        inputs2['fc1_bias'] = inputs['fc1_bias'].astype(np.float16)

        # Test with a real world model, tweak inputs for convert_symbol
        converted_sym = amp.convert_symbol(sym, target_dtype="float16",
                                           target_dtype_ops=["Convolution"], data_names=["data"],
                                           cast_optional_params=True)
        converted_sym2 = amp.convert_symbol(sym, target_dtype="float16",
                                            target_dtype_ops=["Convolution"], data_names=["data"],
                                            cast_optional_params=False)

        exe = converted_sym.simple_bind(mx.gpu(0), data=(1, 3, 224, 224), grad_req='null')
        exe2 = converted_sym2.simple_bind(mx.gpu(), data=(1, 3, 224, 224), grad_req='null')

        converted_args = converted_sym.list_arguments()
        converted_auxs = converted_sym.list_auxiliary_states()
        for i, key in enumerate(exe.arg_arrays):
            if converted_args[i] in arg_params:
                arg_params[converted_args[i]] = arg_params[converted_args[i]].astype(exe.arg_arrays[i].dtype)
        for i, key in enumerate(exe.aux_arrays):
            if converted_auxs[i] in aux_params:
                aux_params[converted_auxs[i]] = aux_params[converted_auxs[i]].astype(exe.aux_arrays[i].dtype)

        inputs2.update(arg_params)
        exe.forward(is_train=False, **inputs2)
        exe.outputs[0].wait_to_read()

        inputs['fc1_weight'] = inputs['fc1_weight'].astype(np.float16)
        inputs['fc1_bias'] = inputs['fc1_bias'].astype(np.float16)
        exe2.forward(is_train=False, **inputs)
        exe2.outputs[0].wait_to_read()


    def check_amp_convert_model():
        # Test with real world model, default inputs for convert_model
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, 'model')
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        prefix, epoch = download_model("imagenet1k-resnet-18", dst_dir=model_path)

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

        # Test with real world model, tweak inputs for convert_model
        result_sym, result_arg_params, result_aux_params = amp.convert_model(sym,
                                                                             arg_params,
                                                                             aux_params,
                                                                             target_dtype="float16",
                                                                             target_dtype_ops=["Convolution"])
        mod = mx.mod.Module(result_sym, data_names=["data"], label_names=["softmax_label"], context=mx.gpu())
        mod.bind(data_shapes=[['data', (1, 3, 224, 224)]], label_shapes=[['softmax_label', (1,)]])

        mod.set_params(result_arg_params, result_aux_params)
        mod.forward(mx.io.DataBatch(data=[mx.nd.ones((1, 3, 224, 224))],
                                    label=[mx.nd.ones((1,))]))
        mod.get_outputs()[0].asnumpy()
        assert mod._arg_params["stage2_unit1_conv2_weight"].dtype == np.float32

        # Call convert_model with cast_optional_params set to True
        result_sym, result_arg_params, result_aux_params = amp.convert_model(sym,
                                                                             arg_params,
                                                                             aux_params,
                                                                             target_dtype="float16",
                                                                             target_dtype_ops=["Convolution"], cast_optional_params=True)
        mod = mx.mod.Module(result_sym, data_names=["data"], label_names=["softmax_label"], context=mx.gpu())
        mod.bind(data_shapes=[['data', (1, 3, 224, 224)]], label_shapes=[['softmax_label', (1,)]])
        mod.set_params(result_arg_params, result_aux_params)
        mod.forward(mx.io.DataBatch(data=[mx.nd.ones((1, 3, 224, 224))],
                                    label=[mx.nd.ones((1,))]))
        mod.get_outputs()[0].asnumpy()
        assert mod._arg_params["stage2_unit1_conv2_weight"].dtype == np.float16


    def check_amp_convert_hybrid_block():
        # Test conversion for hybrid block on CPU
        model_cpu = get_model("resnet50_v1")
        model_cpu.collect_params().initialize(ctx=mx.cpu())
        model_cpu.hybridize()
        model_cpu(mx.nd.random.uniform(0, 1, shape=(1, 3, 224, 224), ctx=mx.cpu()))
        converted_model_cpu = amp.convert_hybrid_block(model_cpu)

        # Test with real world model, default inputs for convert_hybrid_block
        model = get_model("resnet50_v1")
        model.collect_params().initialize(ctx=mx.gpu())
        model.hybridize()
        model(mx.nd.zeros((1, 3, 224, 224)))
        converted_model = amp.convert_hybrid_block(model)
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224),
                                                     dtype=np.float32))
        result = converted_model.forward(mx.nd.zeros((1, 3, 224, 224),
                                                     dtype=np.float32))

        # Test with real world model, tweak inputs for convert_hybrid_block
        converted_model = amp.convert_hybrid_block(model, target_dtype="float16",
                                                   target_dtype_ops=["Convolution"])
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
        params = converted_model.collect_params()
        assert params["stage2_unit1_conv2_weight"].dtype == np.float32

        # Pass cast_optional_params as True to convert_hybrid_block
        converted_model = amp.convert_hybrid_block(net, target_dtype="float16", target_dtype_ops=["Convolution"],
                                                   cast_optional_params=True)
        params = converted_model.collect_params()
        assert params["stage2_unit1_conv2_weight"].dtype == np.float16


    def check_amp_convert_bucketing_module():
        model = train_model(context=mx.current_context())
        result_model = amp.convert_bucketing_module(model)
        val_sent = []
        batch_size = 128
        invalid_label = -1
        num_sentence = 1000
        buckets = [5, 10, 20, 30, 40]
        len_vocab = 50

        for _ in range(num_sentence):
            len_sentence = randint(6, max(buckets)-1) # leave out the two last buckets empty
            val_sentence = []
            for _ in range(len_sentence):
                val_sentence.append(randint(1, len_vocab))
            val_sent.append(val_sentence)

        data_val =  mx.rnn.BucketSentenceIter(val_sent, batch_size, buckets=buckets,
                                     invalid_label=invalid_label)
        result_model.bind(data_val.provide_data, data_val.provide_label, for_training=False)
        result_model.score(data_val, mx.metric.Perplexity(invalid_label),
                           batch_end_callback=mx.callback.Speedometer(batch_size, 1))

        # AMP conversion with cast_optional_params set to true
        # Flaky test when cast_optional_params set to True : https://github.com/apache/incubator-mxnet/issues/16030
        '''
        result_model = amp.convert_bucketing_module(model, cast_optional_params=True)
        result_model.bind(data_val.provide_data, data_val.provide_label, for_training=False)
        result_model.score(data_val, mx.metric.Perplexity(invalid_label),
                           batch_end_callback=mx.callback.Speedometer(batch_size, 1))
        '''


    with mx.Context(mx.gpu(0)):
        check_amp_convert_symbol()
        check_amp_convert_model()
        check_amp_convert_hybrid_block()
        check_amp_convert_bucketing_module()

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_amp_conversion_rnn():
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


@with_seed()
def test_module_backward_compatibility():
    channel_num = 10
    conv_layer_filter_dims = [2, 3]
    conv_layer_strides = [1, 1]
    dimension = 5
    data_len = 10

    data = mx.sym.var("data")
    conv = mx.sym.Convolution(data,
                              num_filter=channel_num,
                              kernel=tuple(conv_layer_filter_dims),
                              stride=tuple(conv_layer_strides))

    bn = mx.sym.BatchNorm(conv,
                          eps=0.001,
                          momentum=0.9,
                          fix_gamma=False,
                          use_global_stats=False,
                          output_mean_var=False,
                          name="conv0_batchnorm")
    fc = mx.sym.FullyConnected(bn, num_hidden=10, name="fullyconnected")
    mod = mx.mod.Module(fc, data_names=["data"], context=mx.gpu(0))
    mod.bind(data_shapes=[['data', (1, 3, 224, 224)]])
    mod.init_params()

    arg_params, aux_params = mod.get_params()
    for param_key, param_val in arg_params.items():
        assert param_val.dtype == np.float32, "Incorrect inference type for arg_params," \
                                               "please check simple_bind for module executor"
    for param_key, param_val in aux_params.items():
        assert param_val.dtype == np.float32, "Incorrect inference type for aux_params," \
                                               "please check simple_bind for module executor"


    sym, arg_params, aux_params = amp.convert_model(mod._symbol, mod._arg_params, mod._aux_params, target_dtype_ops=["Convolution"])
    mod = mx.mod.Module(sym, data_names=["data"], context=mx.gpu(0))
    mod.bind(data_shapes=[['data', (1, 3, 224, 224)]])
    mod.set_params(arg_params, aux_params)
    assert arg_params["fullyconnected_weight"].dtype == np.float16, \
        "Module API is overwriting the inferred dtype for a mixed precision model"


@with_seed()
def test_fp16_casting():
    data = mx.sym.var("data")
    out1 = mx.sym.amp_cast(data, dtype="float16")
    out2 = mx.sym.amp_cast(data, dtype="float32")
    out3 = mx.sym.amp_cast(data, dtype="float16")
    # When two ops from data, with different dtypes,
    # data should be float32
    res = mx.sym.Group([out1, out2])
    final_res = amp.convert_symbol(res, data_names=[], cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.gpu(), data=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float32

    # When two ops from data, both casted to float16,
    # data should be float16
    res = mx.sym.Group([out1, out3])
    final_res = amp.convert_symbol(res, data_names=[], cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.gpu(), data=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float16

    # AMP Multicast test where one node is float32, another is float16
    data = mx.sym.var("data", dtype=np.float32)
    data2 = mx.sym.var("data2", dtype=np.float16)
    out4 = mx.sym.amp_multicast(data, data2, num_outputs=2)
    final_res = amp.convert_symbol(out4, cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.gpu(), data2=(1, 2), data=(1, 2))
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
    exe = final_res.simple_bind(ctx=mx.gpu(), data=(1, 2), data2=(1, 2), data3=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float32

    # AMP Multicast test where three input nodes one fp16, one fp32
    # one unknown
    data = mx.sym.var("data", dtype=np.float16)
    data2 = mx.sym.var("data2", dtype=np.float32)
    data3 = mx.sym.var("data3")
    out6 = mx.sym.amp_multicast(data, data2, data3, num_outputs=3)
    final_res = amp.convert_symbol(out6, target_dtype_ops=[],
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.gpu(), data=(1, 2), data2=(1, 2),
                                data3=(1, 2))
    assert exe.arg_arrays[2].dtype == np.float32

    # Input node to amp_multicast and amp_cast, if dtypes conflict
    # and input node is already fp16, it should still be fp16
    data = mx.sym.var("data", dtype=np.float16)
    data2 = mx.sym.var("data2", dtype=np.float32)
    out7 = mx.sym.Group([mx.sym.amp_multicast(data, data2, num_outputs=2), mx.sym.amp_cast(data, dtype="float16")])
    final_res = amp.convert_symbol(out7, target_dtype_ops=[],
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.gpu(), data=(1, 2), data2=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float16

    # Input node to amp_multicast and amp_cast, if dtypes conflict
    # and input node is already fp32, it should be changed to fp16
    data = mx.sym.var("data", dtype=np.float32)
    data2 = mx.sym.var("data2", dtype=np.float16)
    out8 = mx.sym.Group([mx.sym.amp_multicast(data, data2, num_outputs=2), mx.sym.amp_cast(data, dtype="float16")])
    final_res = amp.convert_symbol(out8, target_dtype_ops=[],
                                   fp32_ops=[], cast_optional_params=True)
    exe = final_res.simple_bind(ctx=mx.gpu(), data=(1, 2), data2=(1, 2))
    assert exe.arg_arrays[0].dtype == np.float16

    # Check for symbol which has slice channel
    data = mx.sym.var("data")
    data2 = mx.sym.var("data2")
    data._set_attr(__dtype__="-1")
    data2._set_attr(__dtype__="-1")
    concat_res = mx.sym.concat(data, data2)
    out = mx.sym.split(concat_res, axis=1, num_outputs=2)
    final_res = amp.convert_symbol(out)


if __name__ == '__main__':
    import nose
    nose.runmodule()
