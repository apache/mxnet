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

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal
import numpy as np
from nose.tools import raises
from copy import deepcopy
import warnings
import json


def test_parameter():
    p = gluon.Parameter('weight', shape=(10, 10))
    p.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assert len(p.list_data()) == 2
    assert len(p.list_grad()) == 2
    assert p.data(mx.cpu(1)).context == mx.cpu(1)
    assert p.data(mx.cpu(0)).shape == (10, 10)
    assert p.var().name == 'weight'

    p.reset_ctx(ctx=[mx.cpu(1), mx.cpu(2)])
    assert p.list_ctx() == [mx.cpu(1), mx.cpu(2)]


def test_paramdict():
    params = gluon.ParameterDict('net_')
    params.get('weight', shape=(10, 10))
    assert list(params.keys()) == ['net_weight']
    params.initialize(ctx=mx.cpu())
    params.save('test.params')
    params.load('test.params', mx.cpu())


def test_parameter_sharing():
    class Net(gluon.Block):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.dense0 = nn.Dense(5, in_units=5)
                self.dense1 = nn.Dense(5, in_units=5)

        def forward(self, x):
            return self.dense1(self.dense0(x))

    net1 = Net(prefix='net1_')
    net2 = Net(prefix='net2_', params=net1.collect_params())
    net1.collect_params().initialize()
    net2(mx.nd.zeros((3, 5)))

    net1.save_params('net1.params')

    net3 = Net(prefix='net3_')
    net3.load_params('net1.params', mx.cpu())


def test_parameter_str():
    class Net(gluon.Block):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.dense0 = nn.Dense(10, in_units=5, use_bias=False)

    net = Net(prefix='net1_')
    lines = str(net.collect_params()).splitlines()

    assert lines[0] == 'net1_ ('
    assert 'net1_dense0_weight' in lines[1]
    assert '(10, 5)' in lines[1]
    assert 'numpy.float32' in lines[1]
    assert lines[2] == ')'

def test_collect_paramters():
    net = nn.HybridSequential(prefix="test_")
    with net.name_scope():
        net.add(nn.Conv2D(10, 3))
        net.add(nn.Dense(10, activation='relu'))
    assert set(net.collect_params().keys()) == \
        set(['test_conv0_weight', 'test_conv0_bias','test_dense0_weight','test_dense0_bias'])
    assert set(net.collect_params('.*weight').keys()) == \
        set(['test_conv0_weight', 'test_dense0_weight'])
    assert set(net.collect_params('test_conv0_bias|test_dense0_bias').keys()) == \
        set(['test_conv0_bias', 'test_dense0_bias'])

def test_basic():
    model = nn.Sequential()
    model.add(nn.Dense(128, activation='tanh', in_units=10, flatten=False))
    model.add(nn.Dropout(0.5))
    model.add(nn.Dense(64, activation='tanh', in_units=256),
              nn.Dense(32, in_units=64))
    model.add(nn.Activation('relu'))

    # symbol
    x = mx.sym.var('data')
    y = model(x)
    assert len(y.list_arguments()) == 7

    # ndarray
    model.collect_params().initialize(mx.init.Xavier(magnitude=2.24))
    x = model(mx.nd.zeros((32, 2, 10)))
    assert x.shape == (32, 32)
    x.wait_to_read()

    model.collect_params().setattr('grad_req', 'null')
    assert list(model.collect_params().values())[0]._grad is None
    model.collect_params().setattr('grad_req', 'write')
    assert list(model.collect_params().values())[0]._grad is not None


def test_dense():
    model = nn.Dense(128, activation='tanh', in_units=10, flatten=False, prefix='test_')
    inputs = mx.sym.Variable('data')
    outputs = model(inputs)
    assert set(model.collect_params().keys()) == set(['test_weight', 'test_bias'])
    assert outputs.list_outputs() == ['test_tanh_fwd_output']
    args, outs, auxs = outputs.infer_shape(data=(2, 3, 10))
    assert outs == [(2, 3, 128)]

    model = nn.Dense(128, activation='relu', in_units=30, flatten=True, prefix='test2_')
    inputs = mx.sym.Variable('data')
    outputs = model(inputs)
    assert set(model.collect_params().keys()) == set(['test2_weight', 'test2_bias'])
    assert outputs.list_outputs() == ['test2_relu_fwd_output']
    args, outs, auxs = outputs.infer_shape(data=(17, 2, 5, 3))
    assert outs == [(17, 128)]


def test_symbol_block():
    model = nn.HybridSequential()
    model.add(nn.Dense(128, activation='tanh'))
    model.add(nn.Dropout(0.5))
    model.add(nn.Dense(64, activation='tanh'),
              nn.Dense(32, in_units=64))
    model.add(nn.Activation('relu'))

    model.initialize()

    inputs = mx.sym.var('data')
    outputs = model(inputs).get_internals()

    smodel = gluon.SymbolBlock(outputs, inputs, params=model.collect_params())

    assert len(smodel(mx.nd.zeros((16, 10)))) == 14

    out = smodel(mx.sym.var('in'))
    assert len(out) == len(outputs.list_outputs())

    class Net(nn.HybridBlock):
        def __init__(self, model):
            super(Net, self).__init__()
            self.model = model

        def hybrid_forward(self, F, x):
            out = self.model(x)
            return F.add_n(*[i.sum() for i in out])

    net = Net(smodel)
    net.hybridize()
    assert isinstance(net(mx.nd.zeros((16, 10))), mx.nd.NDArray)

    inputs = mx.sym.var('data')
    outputs = model(inputs)
    smodel = gluon.SymbolBlock(outputs, inputs, params=model.collect_params())
    net = Net(smodel)
    net.hybridize()
    assert isinstance(net(mx.nd.zeros((16, 10))), mx.nd.NDArray)


def check_layer_forward(layer, dshape):
    layer.collect_params().initialize()
    x = mx.nd.ones(shape=dshape)
    x.attach_grad()
    with mx.autograd.record():
        out = layer(x)
    out.backward()

    np_out = out.asnumpy()
    np_dx = x.grad.asnumpy()

    layer.hybridize()

    x = mx.nd.ones(shape=dshape)
    x.attach_grad()
    with mx.autograd.record():
        out = layer(x)
    out.backward()

    mx.test_utils.assert_almost_equal(np_out, out.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(np_dx, x.grad.asnumpy(), rtol=1e-5, atol=1e-6)

def test_conv():
    layers1d = [
        nn.Conv1D(16, 3, in_channels=4),
        nn.Conv1D(16, 3, groups=2, in_channels=4),
        nn.Conv1D(16, 3, strides=3, groups=2, in_channels=4),
        ]
    for layer in layers1d:
        check_layer_forward(layer, (1, 4, 10))


    layers2d = [
        nn.Conv2D(16, (3, 4), in_channels=4),
        nn.Conv2D(16, (5, 4), in_channels=4),
        nn.Conv2D(16, (3, 4), groups=2, in_channels=4),
        nn.Conv2D(16, (3, 4), strides=4, in_channels=4),
        nn.Conv2D(16, (3, 4), dilation=4, in_channels=4),
        nn.Conv2D(16, (3, 4), padding=4, in_channels=4),
        ]
    for layer in layers2d:
        check_layer_forward(layer, (1, 4, 20, 20))


    layers3d = [
        nn.Conv3D(16, (1, 8, 4), in_channels=4, activation='relu'),
        nn.Conv3D(16, (5, 4, 3), in_channels=4),
        nn.Conv3D(16, (3, 3, 3), groups=2, in_channels=4),
        nn.Conv3D(16, 4, strides=4, in_channels=4),
        nn.Conv3D(16, (3, 3, 3), padding=4, in_channels=4),
        ]
    for layer in layers3d:
        check_layer_forward(layer, (1, 4, 10, 10, 10))


    layer = nn.Conv2D(16, (3, 3), layout='NHWC', in_channels=4)
    # check_layer_forward(layer, (1, 10, 10, 4))

    layer = nn.Conv3D(16, (3, 3, 3), layout='NDHWC', in_channels=4)
    # check_layer_forward(layer, (1, 10, 10, 10, 4))


def test_deconv():
    # layers1d = [
    #     nn.Conv1DTranspose(16, 3, in_channels=4),
    #     nn.Conv1DTranspose(16, 3, groups=2, in_channels=4),
    #     nn.Conv1DTranspose(16, 3, strides=3, groups=2, in_channels=4),
    #     ]
    # for layer in layers1d:
    #     check_layer_forward(layer, (1, 4, 10))


    layers2d = [
        nn.Conv2DTranspose(16, (3, 4), in_channels=4),
        nn.Conv2DTranspose(16, (5, 4), in_channels=4),
        nn.Conv2DTranspose(16, (3, 4), groups=2, in_channels=4),
        nn.Conv2DTranspose(16, (3, 4), strides=4, in_channels=4),
        nn.Conv2DTranspose(16, (3, 4), dilation=4, in_channels=4),
        nn.Conv2DTranspose(16, (3, 4), padding=4, in_channels=4),
        nn.Conv2DTranspose(16, (3, 4), strides=4, output_padding=3, in_channels=4),
        ]
    for layer in layers2d:
        check_layer_forward(layer, (1, 4, 20, 20))


    # layers3d = [
    #     nn.Conv3DTranspose(16, (1, 8, 4), in_channels=4),
    #     nn.Conv3DTranspose(16, (5, 4, 3), in_channels=4),
    #     nn.Conv3DTranspose(16, (3, 3, 3), groups=2, in_channels=4),
    #     nn.Conv3DTranspose(16, 4, strides=4, in_channels=4),
    #     nn.Conv3DTranspose(16, (3, 3, 3), padding=4, in_channels=4),
    #     ]
    # for layer in layers3d:
    #     check_layer_forward(layer, (1, 4, 10, 10, 10))
    #
    #
    # layer = nn.Conv2DTranspose(16, (3, 3), layout='NHWC', in_channels=4)
    # # check_layer_forward(layer, (1, 10, 10, 4))
    #
    # layer = nn.Conv3DTranspose(16, (3, 3, 3), layout='NDHWC', in_channels=4)
    # # check_layer_forward(layer, (1, 10, 10, 10, 4))


def test_pool():
    layers1d = [
        nn.MaxPool1D(),
        nn.MaxPool1D(3),
        nn.MaxPool1D(3, 2),
        nn.AvgPool1D(),
        nn.GlobalAvgPool1D(),
        ]
    for layer in layers1d:
        check_layer_forward(layer, (1, 2, 10))


    layers2d = [
        nn.MaxPool2D(),
        nn.MaxPool2D((3, 3)),
        nn.MaxPool2D(3, 2),
        nn.AvgPool2D(),
        nn.GlobalAvgPool2D(),
        ]
    for layer in layers2d:
        check_layer_forward(layer, (1, 2, 10, 10))

    layers3d = [
        nn.MaxPool3D(),
        nn.MaxPool3D((3, 3, 3)),
        nn.MaxPool3D(3, 2),
        nn.AvgPool3D(),
        nn.GlobalAvgPool3D(),
        ]
    for layer in layers3d:
        check_layer_forward(layer, (1, 2, 10, 10, 10))

    # test ceil_mode
    x = mx.nd.zeros((2, 2, 10, 10))

    layer = nn.MaxPool2D(3, ceil_mode=False)
    layer.collect_params().initialize()
    assert (layer(x).shape==(2, 2, 3, 3))

    layer = nn.MaxPool2D(3, ceil_mode=True)
    layer.collect_params().initialize()
    assert (layer(x).shape==(2, 2, 4, 4))


def test_batchnorm():
    layer = nn.BatchNorm(in_channels=10)
    check_layer_forward(layer, (2, 10, 10, 10))


def test_instancenorm():
    layer = nn.InstanceNorm(in_channels=10)
    check_layer_forward(layer, (2, 10, 10, 10))


def test_reflectionpad():
    layer = nn.ReflectionPad2D(3)
    check_layer_forward(layer, (2, 3, 24, 24))


def test_reshape():
    x = mx.nd.ones((2, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.collect_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x.reshape((-1,))
        x = x + 10
    x.backward()


def test_slice():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.collect_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x[1:3]
        x = x + 10
    x.backward()


def test_at():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.collect_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x[1]
        x = x + 10
    x.backward()


def test_deferred_init():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2)
    layer.collect_params().initialize()
    layer(x)


def check_split_data(x, num_slice, batch_axis, **kwargs):
    res = gluon.utils.split_data(x, num_slice, batch_axis, **kwargs)
    assert len(res) == num_slice
    mx.test_utils.assert_almost_equal(mx.nd.concat(*res, dim=batch_axis).asnumpy(),
                                      x.asnumpy())


def test_split_data():
    x = mx.nd.random.uniform(shape=(128, 33, 64))

    check_split_data(x, 8, 0)
    check_split_data(x, 3, 1)
    check_split_data(x, 4, 1, even_split=False)
    check_split_data(x, 15, 1, even_split=False)
    try:
        check_split_data(x, 4, 1)
    except ValueError:
        return
    assert False, "Should have failed"


def test_flatten():
    flatten = nn.Flatten()
    x = mx.nd.zeros((3,4,5,6))
    assert flatten(x).shape == (3, 4*5*6)
    x = mx.nd.zeros((3,6))
    assert flatten(x).shape == (3, 6)
    x = mx.nd.zeros((3,))
    assert flatten(x).shape == (3, 1)


def test_trainer():
    def dict_equ(a, b):
        assert set(a) == set(b)
        for k in a:
            assert (a[k].asnumpy() == b[k].asnumpy()).all()
    x = gluon.Parameter('x', shape=(10,))
    x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
    trainer = gluon.Trainer([x], 'sgd', {'learning_rate': 1.0, 'momentum': 0.5})
    with mx.autograd.record():
        for w in x.list_data():
            y = w + 1
            y.backward()
    trainer.step(1)

    assert (x.data(mx.cpu(1)).asnumpy() == -2).all()

    x.lr_mult = 0.5

    with mx.autograd.record():
        for w in x.list_data():
            y = w + 1
            y.backward()
    trainer.step(1)

    assert (x.data(mx.cpu(1)).asnumpy() == -4).all()

    trainer.save_states('test.states')
    states = deepcopy(trainer._kvstore._updater.states) if trainer._update_on_kvstore \
             else deepcopy(trainer._updaters[0].states)
    trainer.load_states('test.states')
    if trainer._update_on_kvstore:
        dict_equ(trainer._kvstore._updater.states, states)
        assert trainer._optimizer == trainer._kvstore._updater.optimizer
    else:
        for updater in trainer._updaters:
            dict_equ(updater.states, states)
        assert trainer._optimizer == trainer._updaters[0].optimizer


def test_block_attr_hidden():
    b = gluon.Block()

    # regular attributes can change types
    b.a = None
    b.a = 1

@raises(TypeError)
def test_block_attr_block():
    b = gluon.Block()

    # regular variables can't change types
    b.b = gluon.Block()
    b.b = (2,)

@raises(TypeError)
def test_block_attr_param():
    b = gluon.Block()

    # regular variables can't change types
    b.b = gluon.Parameter()
    b.b = (2,)

def test_block_attr_regular():
    b = gluon.Block()

    # set block attribute also sets _children
    b.c = gluon.Block()
    c2 = gluon.Block()
    b.c = c2
    assert b.c is c2 and b._children[0] is c2

def test_block_attr_list_of_block():
    class Model1(gluon.Block):
        def __init__(self, **kwargs):
            super(Model1, self).__init__(**kwargs)
            with self.name_scope():
                self.layers = [nn.Dense(i * 10) for i in range(6)]

    class Model2(gluon.Block):
        def __init__(self, **kwargs):
            super(Model2, self).__init__(**kwargs)
            with self.name_scope():
                self.layers = dict()
                self.layers['a'] = [nn.Dense(10), nn.Dense(10)]

    class Model3(gluon.Block):
        def __init__(self, **kwargs):
            super(Model3, self).__init__(**kwargs)
            with self.name_scope():
                self.layers = nn.Sequential()
                self.layers.add(*[nn.Dense(i * 10) for i in range(6)])

    class Model4(gluon.Block):
        def __init__(self, **kwargs):
            super(Model4, self).__init__(**kwargs)
            with self.name_scope():
                self.data = {'a': '4', 'b': 123}

    with warnings.catch_warnings(record=True) as w:
        model = Model1()
        model.collect_params()
        assert len(w) > 0
    with warnings.catch_warnings(record=True) as w:
        model = Model2()
        model.collect_params()
        assert len(w) > 0
    with warnings.catch_warnings(record=True) as w:
        model = Model3()
        model.collect_params()
        assert len(w) == 0
    with warnings.catch_warnings(record=True) as w:
        model = Model4()
        model.collect_params()
        assert len(w) == 0

def test_sequential_warning():
    with warnings.catch_warnings(record=True) as w:
        b = gluon.nn.Sequential()
        b.add(gluon.nn.Dense(20))
        b.hybridize()
        assert len(w) == 1

def test_global_norm_clip():
    x1 = mx.nd.ones((3,3))
    x2 = mx.nd.ones((4,4))
    norm = gluon.utils.clip_global_norm([x1, x2], 1.0)
    assert norm == 5.0
    assert_almost_equal(x1.asnumpy(), np.ones((3,3))/5)
    assert_almost_equal(x2.asnumpy(), np.ones((4,4))/5)

    x3 = mx.nd.array([1.0, 2.0, float('nan')])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gluon.utils.clip_global_norm([x1, x3], 2.0)
        assert len(w) == 1


def test_embedding():
    layer = gluon.nn.Embedding(10, 100)
    layer.initialize()
    x = mx.nd.array([3,4,2,0,1])
    with mx.autograd.record():
        y = layer(x)
        y.backward()
    assert (layer.weight.grad()[:5] == 1).asnumpy().all()
    assert (layer.weight.grad()[5:] == 0).asnumpy().all()


def test_export():
    ctx = mx.context.current_context()
    model = gluon.model_zoo.vision.resnet18_v1(
        prefix='resnet', ctx=ctx, pretrained=True)
    model.hybridize()
    data = mx.nd.random.normal(shape=(1, 3, 224, 224))
    out = model(data)

    model.export('gluon')

    module = mx.mod.Module.load('gluon', 0, label_names=None, context=ctx)
    module.bind(data_shapes=[('data', data.shape)])
    module.forward(mx.io.DataBatch([data], None), is_train=False)
    mod_out, = module.get_outputs()

    assert_almost_equal(out.asnumpy(), mod_out.asnumpy())

    model2 = gluon.model_zoo.vision.resnet18_v1(prefix='resnet', ctx=ctx)
    model2.collect_params().load('gluon-0000.params', ctx)
    out2 = model2(data)

    assert_almost_equal(out.asnumpy(), out2.asnumpy())


def test_hybrid_stale_cache():
    net = mx.gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(mx.gluon.nn.Dense(10, weight_initializer='zeros', bias_initializer='ones', flatten=False))

    net.hybridize()
    net.initialize()
    net(mx.nd.ones((2,3,5)))

    net.add(mx.gluon.nn.Flatten())
    assert net(mx.nd.ones((2,3,5))).shape == (2, 30)

    net = mx.gluon.nn.HybridSequential()
    with net.name_scope():
        net.fc1 = mx.gluon.nn.Dense(10, weight_initializer='zeros',
                                    bias_initializer='ones', flatten=False)
        net.fc2 = mx.gluon.nn.Dense(10, weight_initializer='zeros',
                                    bias_initializer='ones', flatten=False)
    net.hybridize()
    net.initialize()
    net(mx.nd.ones((2,3,5)))

    net.fc2 = mx.gluon.nn.Dense(10, weight_initializer='zeros',
                                bias_initializer='ones', flatten=True)
    net.initialize()
    assert net(mx.nd.ones((2,3,5))).shape == (2, 10)


def test_lambda():
    net1 = mx.gluon.nn.HybridSequential()
    net1.add(nn.Activation('tanh'),
             nn.LeakyReLU(0.1))

    net2 = mx.gluon.nn.HybridSequential()
    op3 = lambda F, x, *args: F.LeakyReLU(x, *args, slope=0.1)
    net2.add(nn.HybridLambda('tanh'),
             nn.HybridLambda(op3))

    op4 = lambda x: mx.nd.LeakyReLU(x, slope=0.1)
    net3 = mx.gluon.nn.Sequential()
    net3.add(nn.Lambda('tanh'),
             nn.Lambda(op4))

    input_data = mx.nd.random.uniform(shape=(2, 3, 5, 7))
    out1, out2, out3 = net1(input_data), net2(input_data), net3(input_data)
    assert_almost_equal(out1.asnumpy(), out2.asnumpy())
    assert_almost_equal(out1.asnumpy(), out3.asnumpy())


def test_fill_shape_deferred():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Conv2D(64, kernel_size=2, padding=1),
                nn.BatchNorm(),
                nn.Dense(10))
    net.hybridize()
    net.initialize()
    net(mx.nd.ones((2,3,5,7)))
    assert net[0].weight.shape[1] == 3, net[0].weight.shape[1]
    assert net[1].gamma.shape[0] == 64, net[1].gamma.shape[0]
    assert net[2].weight.shape[1] == 3072, net[2].weight.shape[1]


def test_dtype():
    net = mx.gluon.model_zoo.vision.resnet18_v1()
    net.initialize()
    net.cast('float64')
    with mx.autograd.record():
        y = net(mx.nd.ones((16, 3, 32, 32), dtype='float64'))
        y.backward()

    net = mx.gluon.model_zoo.vision.resnet18_v1()
    net.initialize()
    net.hybridize()
    net(mx.nd.ones((16, 3, 32, 32), dtype='float32'))

    net.cast('float64')
    net(mx.nd.ones((16, 3, 32, 32), dtype='float64'))

    mx.nd.waitall()


def test_fill_shape_load():
    ctx = mx.context.current_context()
    net1 = nn.HybridSequential()
    with net1.name_scope():
        net1.add(nn.Conv2D(64, kernel_size=2, padding=1),
                 nn.BatchNorm(),
                 nn.Dense(10))
    net1.hybridize()
    net1.initialize(ctx=ctx)
    net1(mx.nd.ones((2,3,5,7), ctx))
    net1.save_params('net_fill.params')

    net2 = nn.HybridSequential()
    with net2.name_scope():
        net2.add(nn.Conv2D(64, kernel_size=2, padding=1),
                 nn.BatchNorm(),
                 nn.Dense(10))
    net2.hybridize()
    net2.initialize()
    net2.load_params('net_fill.params', ctx)
    assert net2[0].weight.shape[1] == 3, net2[0].weight.shape[1]
    assert net2[1].gamma.shape[0] == 64, net2[1].gamma.shape[0]
    assert net2[2].weight.shape[1] == 3072, net2[2].weight.shape[1]


def test_inline():
    net = mx.gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(mx.gluon.nn.Dense(10))
        net.add(mx.gluon.nn.Dense(10))
        net.add(mx.gluon.nn.Dense(10))

    net.initialize()
    net.hybridize(inline_limit=3)
    with mx.autograd.record():
        y = net(mx.nd.zeros((1,10)))

    len_1 = len(json.loads(mx.autograd.get_symbol(y).tojson())['nodes'])
    y.backward()

    net.hybridize(inline_limit=0)
    with mx.autograd.record():
        y = net(mx.nd.zeros((1,10)))

    len_2 = len(json.loads(mx.autograd.get_symbol(y).tojson())['nodes'])
    y.backward()

    assert len_1 == len_2 + 2


def test_activations():
    point_to_validate = mx.nd.array([-0.1, 0.1] * 3)

    swish = mx.gluon.nn.Swish()
    def swish_test(x):
        return x * mx.nd.sigmoid(x)

    for test_point, ref_point in zip(swish_test(point_to_validate), swish(point_to_validate)):
        assert test_point == ref_point

    elu = mx.gluon.nn.ELU()
    def elu_test(x):
        def elu(x):
            return 1.0 * (mx.nd.exp(x) - 1) if x < 0 else x
        return [elu(x_i) for x_i in x]

    for test_point, ref_point in zip(elu_test(point_to_validate), elu(point_to_validate)):
        assert test_point == ref_point

    selu = mx.gluon.nn.SELU()
    def selu_test(x):
        def selu(x):
            scale, alpha = 1.0507009873554804934193349852946, 1.6732632423543772848170429916717
            return scale * x if x >= 0 else alpha * mx.nd.exp(x) - alpha
        return [selu(x_i) for x_i in x]

    for test_point, ref_point in zip(selu(point_to_validate), selu(point_to_validate)):
        assert test_point == ref_point

    prelu = mx.gluon.nn.PReLU()
    prelu.initialize()
    x = point_to_validate.reshape((1, 3, 2))
    assert_almost_equal(prelu(x).asnumpy(), mx.nd.where(x >= 0, x, 0.25 * x).asnumpy())



if __name__ == '__main__':
    import nose
    nose.runmodule()
