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
from mxnet.test_utils import *
from common import setup_module, with_seed, teardown
from mxnet.gluon.model_zoo.vision import get_model

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
