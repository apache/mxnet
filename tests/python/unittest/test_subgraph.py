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

# pylint: skip-file
from __future__ import print_function
import numpy as np
import mxnet as mx
import copy
import os
import math
import ctypes
import random
import itertools
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.base import py_str, MXNetError, _as_list, SymbolHandle, check_call, _LIB, c_handle_array, mx_uint, c_str, c_str_array
from common import setup_module, with_seed, teardown
import unittest
from mxnet.gluon.model_zoo.vision import get_model
from collections import namedtuple
import gluoncv

def make_subgraph(subg, *args):
    js = subg.tojson()
    return mx.sym._internal._CachedOp(*args, subgraph=js)

@with_seed()
def test_make_subgraph():
    def make_subgraph1(stype):
        a = mx.symbol.Variable(name='a', stype=stype)
        b = mx.symbol.Variable(name='b', stype=stype)
        c = a * b
        d = c * 2

        a1 = mx.symbol.Variable(name='a', stype=stype)
        b1 = mx.symbol.Variable(name='b', stype=stype)
        y = make_subgraph(c, a1, b1)
        y = y * 2

        s = (10, 10)
        a_arr = mx.nd.array(np.random.normal(-0.1, 0.1, size=s),
                ctx=default_context()).tostype(stype)
        b_arr = mx.nd.array(np.random.normal(-0.1, 0.1, size=s),
                ctx=default_context()).tostype(stype)
        return (d, y, {'a': a_arr, 'b': b_arr}, {})

    def create_weights(shapes, names):
        nd_dict = {}
        sym_dict = {}
        assert len(shapes) == len(names)
        for i in range(len(shapes)):
            sym_dict[names[i]] = mx.symbol.Variable(names[i])
            nd_dict[names[i]] = mx.nd.array(np.ones(shapes[i]), ctx=default_context())
        return (nd_dict, sym_dict)

    def make_subgraph_weight(orig, shape, stype):
        arg_shapes, out_shapes, aux_shapes = orig.infer_shape(data=shape)
        weight_shapes = arg_shapes[1:]
        weight_names = orig.list_arguments()[1:]
        weight_dict, weight_sym_dict = create_weights(weight_shapes, weight_names)
        aux_dict, aux_sym_dict = create_weights(aux_shapes, orig.list_auxiliary_states())

        input_dict = copy.deepcopy(weight_sym_dict)
        input_dict.update(aux_sym_dict)
        input_dict['data'] = mx.symbol.Variable('data', stype=stype)
        input_list = []
        for name in orig.list_inputs():
            assert name in input_dict.keys()
            input_list.append(input_dict[name])
        subg = make_subgraph(orig, *input_list)

        arr = mx.nd.random.uniform(-1, 1, shape=shape, ctx=default_context()).tostype(stype)
        arg_dict = weight_dict
        arg_dict['data'] = arr
        return (orig, subg, arg_dict, aux_dict)

    def make_subgraph2(stype, out_mean_var):
        data = mx.symbol.Variable('data', stype=stype)
        orig = mx.symbol.BatchNorm(data, fix_gamma=False,
                output_mean_var=out_mean_var, name="batchnorm")
        s = (10, 10)
        return make_subgraph_weight(orig, s, stype)

    def make_subgraph3(stype):
        data = mx.symbol.Variable('data', stype=stype)
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=16, no_bias=True)
        bn1 = mx.symbol.BatchNorm(conv1, fix_gamma=False, output_mean_var=False)
        conv2 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=16, no_bias=True)
        bn2 = mx.symbol.BatchNorm(conv2, fix_gamma=False, output_mean_var=False)
        orig = bn1 + bn2
        s = (1, 3, 32, 32)
        return make_subgraph_weight(orig, s, stype)

    def make_subgraph4(stype):
        model = get_model('resnet18_v1')
        model.hybridize()
        model.initialize()
        s = (1, 3, 32, 32)
        data = mx.nd.random.normal(shape=s)
        out = model(data)
        model.export('resnet18')
        orig = mx.sym.load('resnet18-symbol.json')
        return make_subgraph_weight(orig, s, stype)

    make_subgraphs = [make_subgraph1,
            lambda stype: make_subgraph2(stype, False),
            lambda stype: make_subgraph2(stype, True),
            make_subgraph3, make_subgraph4]
    stypes = ['default', 'row_sparse']
    for make_subg in make_subgraphs:
        for stype in stypes:
            orig, subg, inputs, aux_states = make_subg(stype)
            all_inputs = copy.deepcopy(inputs)
            all_inputs.update(aux_states)
            args_grad = {key : mx.nd.empty(shape=all_inputs[key].shape) for key in all_inputs.keys()}
            e1 = orig.bind(ctx=default_context(), args=all_inputs, args_grad=args_grad,
                    aux_states=all_inputs)
            args_grad = {key : mx.nd.empty(shape=all_inputs[key].shape) for key in all_inputs.keys()}
            e2 = subg.bind(ctx=default_context(), args=all_inputs, args_grad=args_grad,
                    aux_states=all_inputs)
            e1.forward(is_train=True)
            e2.forward(is_train=True)
            for i in range(len(e1.outputs)):
                assert_almost_equal(e1.outputs[i].asnumpy(), e2.outputs[i].asnumpy(),
                        rtol=0.001, atol=0.0001)

            out_grads = [mx.nd.random.uniform(-1, 1, shape=out.shape, ctx=default_context())
                    for out in e1.outputs]
            e1.backward(out_grads)
            e2.backward(out_grads)
            for i in range(len(e1.grad_arrays)):
                assert_almost_equal(e1.grad_arrays[i].asnumpy(), e2.grad_arrays[i].asnumpy(),
                        rtol=0.001, atol=0.0001)

# this test checks for the problem reported in #14727
def test_input_order():
    # get model from model-zoo
    model = gluoncv.model_zoo.faster_rcnn_resnet50_v1b_coco(pretrained=True)
    im_fname = gluoncv.utils.download('https://github.com/dmlc/web-data/blob/master/gluoncv/detection/biking.jpg?raw=true', path='biking.jpg')
    # hybridize and export
    x, orig_img = gluoncv.data.transforms.presets.rcnn.load_test(im_fname)
    model.hybridize()
    box_ids, scores, bboxes = model(x)
    model.export('faster-rcnn')
    
    # set partitioning config
    op_names = [
        "_add",
        "_contrib_MultiBoxDetection",
        "_contrib_MultiBoxPrior",
        "_contrib_MultiBoxTarget",
        "_copy",
        "_div_scalar",
        "_DivScalar",
        "_minus",
        "_Minus",
        "_minus_scalar",
        "_MinusScalar",
        "_mul",
        "_Mul",
        "_mul_scalar",
        "_MulScalar",
        "_plus",
        "_Plus",
        "_plus_scalar",
        "_PlusScalar",
        "_rdiv_scalar",
        "_RDivScalar",
        "_rminus_scalar",
        "_RMinusScalar",
        "_rnn_param_concat",
        "_sub",
        "abs",
        "Activation",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctanh",
        "argmax",
        "argmin",
        "BatchNorm",
        "BatchNorm_v1",
        "BlockGrad",
        "broadcast_add",
        "broadcast_equal",
        "broadcast_greater",
        "broadcast_greater_equal",
        "broadcast_lesser",
        "broadcast_lesser_equal",
        "broadcast_mul",
        "broadcast_not_equal",
        "broadcast_plus",
        "cast",
        "Cast",
        "clip",
        "concat",
        "Concat",
        "Convolution",
        "Convolution_v1",
        "cos",
        "cosh",
        "crop",
        "Deconvolution",
        "Dropout",
        "elemwise_add",
        "elemwise_mul",
        "elemwise_sub",
        "Embedding",
        "exp",
        "expand_dims",
        "flatten",
        "Flatten",
        "flip",
        "FullyConnected",
        "identity",
        "identity",
        "LeakyReLU",
        "LinearRegressionOutput",
        "log",
        "log_softmax",
        "LRN",
        "make_loss",
        "MakeLoss",
        "max",
        "max_axis",
        "mean",
        "min",
        "min_axis",
        "negative",
        "one_hot",
        "pad",
        "Pad",
        "pick",
        "Pooling",
        "Pooling_v1",
        "prod",
        "reciprocal",
        "relu",
        "repeat",
        "reshape",
        "Reshape",
        "reverse",
        "RNN",
        "rsqrt",
        "sigmoid",
        "sin",
        "sinh",
        "slice",
        "SliceChannel",
        "softmax",
        "SoftmaxActivation",
        "SoftmaxOutput",
        "softmin",
        "split",
        "sqrt",
        "sum",
        "sum_axis",
        "tan",
        "tanh",
        "tile",
        "topk",
        "transpose",
        "zeros_like"
    ]
    check_call(_LIB.MXSetSubgraphPropertyOpNames(c_str("default"),
                                                 mx_uint(len(op_names)),
                                                 c_str_array(op_names)))
    os.environ['MXNET_SUBGRAPH_BACKEND'] = 'default'

    # load model in module API
    sym, arg_params, aux_params = mx.model.load_checkpoint('faster-rcnn', 0)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    fname = mx.test_utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')
    img = mx.image.imread(fname)

    # convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    img = img.expand_dims(axis=0) # batchify

    Batch = namedtuple('Batch', ['data'])
    mod.forward(Batch([img]))
    
    # wait for all outputs to be ready
    for o in mod.get_outputs():
        o.wait_to_read()

def test_subgraph_with_customOp():
    class MyAdd(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            self.assign(out_data[0], req[0], in_data[0] + 1)

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            self.assign(in_grad[0], req[0], out_grad[0])

    @mx.operator.register('MyAdd1')
    class MyAdd1Prop(mx.operator.CustomOpProp):
        def __init__(self):
            super(MyAdd1Prop, self).__init__(need_top_grad=True)

        def list_arguments(self):
            return ['data']

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shape):
            # inputs, outputs, aux
            return [in_shape[0]], [in_shape[0]], []

        def create_operator(self, ctx, shapes, dtypes):
            return MyAdd()

    @mx.operator.register('MyAdd2')
    class MyAdd2Prop(mx.operator.CustomOpProp):
        def __init__(self):
            super(MyAdd2Prop, self).__init__(need_top_grad=True)

        def list_arguments(self):
            return ['data']

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shape):
            # inputs, outputs, aux
            return [in_shape[0]], [in_shape[0]], []

        def create_operator(self, ctx, shapes, dtypes):
            return MyAdd()

    inp = mx.nd.zeros(shape=(100, 100))
    a = mx.symbol.Variable('a')
    b = a + 1
    b = mx.symbol.Custom(data=a, op_type='MyAdd1')
    c = mx.symbol.Custom(data=a, op_type='MyAdd2')
    b.bind(mx.cpu(), {'a': inp}).forward()
    c.bind(mx.cpu(), {'a': inp}).forward()
    mx.nd.waitall()

if __name__ == '__main__':
    import nose
    nose.runmodule()
