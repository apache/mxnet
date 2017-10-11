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
    model = gluon.model_zoo.vision.resnet18_v1(prefix='resnet', pretrained=True)
    model.hybridize()
    data = mx.nd.random.normal(shape=(1, 3, 224, 224))
    out = model(data)

    model.export('gluon')

    module = mx.mod.Module.load('gluon', 0, label_names=None)
    module.bind(data_shapes=[('data', data.shape)])
    module.forward(mx.io.DataBatch([data], None), is_train=False)
    mod_out, = module.get_outputs()

    assert_almost_equal(out.asnumpy(), mod_out.asnumpy())

    model2 = gluon.model_zoo.vision.resnet18_v1(prefix='resnet')
    model2.collect_params().load('gluon-0000.params', mx.cpu())
    out2 = model2(data)

    assert_almost_equal(out.asnumpy(), out2.asnumpy())


if __name__ == '__main__':
    import nose
    nose.runmodule()
