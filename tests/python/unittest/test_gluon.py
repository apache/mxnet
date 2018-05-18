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
from mxnet.ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from common import setup_module, with_seed, assertRaises
import numpy as np
from nose.tools import raises, assert_raises
from copy import deepcopy
import warnings
import json
import unittest

@with_seed()
def test_parameter():
    p = gluon.Parameter('weight', shape=(10, 10))
    p.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assert len(p.list_data()) == 2
    assert len(p.list_grad()) == 2
    assert p.data(mx.cpu(1)).context == mx.cpu(1)
    assert p.data(mx.cpu(0)).shape == (10, 10)
    assert p.var().name == 'weight'
    assert p.grad(mx.cpu(0)).stype == 'default'
    assert p.data().stype == 'default'

    p.reset_ctx(ctx=[mx.cpu(1), mx.cpu(2)])
    assert p.list_ctx() == [mx.cpu(1), mx.cpu(2)]

@with_seed()
def test_sparse_parameter():
    p = gluon.Parameter('weight', shape=(10, 10), stype='row_sparse', grad_stype='row_sparse')
    p.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    row_id = mx.nd.arange(0, 10, ctx=mx.cpu(1))
    assert len(p.list_row_sparse_data(row_id)) == 2
    assert len(p.list_grad()) == 2
    weight = p.row_sparse_data(row_id)
    assert weight.context == mx.cpu(1)
    assert weight.shape == (10, 10)
    assert weight.stype == 'row_sparse'
    assert p.var().name == 'weight'
    assert p.var().attr('__storage_type__') == str(_STORAGE_TYPE_STR_TO_ID['row_sparse'])
    assert p.grad(mx.cpu(0)).stype == 'row_sparse'

    p.reset_ctx(ctx=[mx.cpu(1), mx.cpu(2)])
    assert p.list_ctx() == [mx.cpu(1), mx.cpu(2)]

@with_seed()
def test_parameter_invalid_access():
    # cannot call data on row_sparse parameters
    p0 = gluon.Parameter('weight', shape=(10, 10), stype='row_sparse', grad_stype='row_sparse')
    p0.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assertRaises(ValueError, p0.data)
    assertRaises(ValueError, p0.list_data)
    row_id = mx.nd.arange(0, 10)
    # cannot call row_sparse_data on dense parameters
    p1 = gluon.Parameter('weight', shape=(10, 10))
    p1.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assertRaises(ValueError, p1.row_sparse_data, row_id.copyto(mx.cpu(0)))
    assertRaises(ValueError, p1.list_row_sparse_data, row_id)

@with_seed()
def test_paramdict():
    params0 = gluon.ParameterDict('net_')
    params0.get('w0', shape=(10, 10))
    params0.get('w1', shape=(10, 10), stype='row_sparse')
    all_row_ids = mx.nd.arange(0, 10)
    assert list(params0.keys()) == ['net_w0', 'net_w1']
    params0.initialize(ctx=mx.cpu())
    prev_w0 = params0.get('w0').data(mx.cpu())
    prev_w1 = params0.get('w1').row_sparse_data(all_row_ids)

    params0.save('test.params')
    params0.load('test.params', mx.cpu())
    # compare the values before and after save/load
    cur_w0 = params0.get('w0').data(mx.cpu())
    cur_w1 = params0.get('w1').row_sparse_data(all_row_ids)
    mx.test_utils.assert_almost_equal(prev_w0.asnumpy(), cur_w0.asnumpy())
    mx.test_utils.assert_almost_equal(prev_w1.asnumpy(), cur_w1.asnumpy())

    # create a new param dict with dense params, and load from the checkpoint
    # of sparse & dense params
    params1 = gluon.ParameterDict('net_')
    params1.get('w0', shape=(10, 10))
    params1.get('w1', shape=(10, 10))
    assertRaises(RuntimeError, params1.load, 'test.params', mx.cpu())
    params1.load('test.params', mx.cpu(), cast_stype=True)
    # compare the values before and after save/load
    cur_w0 = params1.get('w0').data(mx.cpu())
    cur_w1 = params1.get('w1').data(mx.cpu())
    mx.test_utils.assert_almost_equal(prev_w0.asnumpy(), cur_w0.asnumpy())
    mx.test_utils.assert_almost_equal(prev_w1.asnumpy(), cur_w1.asnumpy())


@with_seed()
def test_parameter_row_sparse_data():
    def check_parameter_row_sparse_data(with_trainer):
        ctx0 = mx.cpu(1)
        ctx1 = mx.cpu(2)
        dim0 = 4
        x = gluon.Parameter('x', shape=(dim0, 2), stype='row_sparse')
        x.initialize(init='xavier', ctx=[ctx0, ctx1])
        if with_trainer:
            trainer = gluon.Trainer([x], 'sgd')
        x_param = x._data[0].copy()
        assert x_param.stype == 'row_sparse'
        row_id_0 = mx.nd.array([0,1], ctx=ctx0)
        retained_0 = x.row_sparse_data(row_id_0)
        retained_target_0 = mx.nd.sparse.retain(x_param, row_id_0.as_in_context(ctx0))
        mx.test_utils.assert_almost_equal(retained_0.asnumpy(), retained_target_0.asnumpy())
        assert retained_0.context == ctx0
        row_id_1 = mx.nd.arange(0, dim0, ctx=ctx1)
        retained_1 = x.row_sparse_data(row_id_1)
        retained_target_1 = x_param
        mx.test_utils.assert_almost_equal(retained_1.asnumpy(), retained_target_1.asnumpy())
        assert retained_1.context == ctx1
        row_id_2 = mx.nd.array([0,1,2])
        retained_2 = x.list_row_sparse_data(row_id_2)
        retained_target_2 = mx.nd.sparse.retain(x_param, row_id_2.as_in_context(ctx0))
        mx.test_utils.assert_almost_equal(retained_2[0].asnumpy(), retained_target_2.asnumpy())

    check_parameter_row_sparse_data(True)
    check_parameter_row_sparse_data(False)

@with_seed()
def test_constant():
    class Test(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Test, self).__init__(**kwargs)
            self.value = np.asarray([[1,2], [3,4]])
            self.const = self.params.get_constant('const', self.value)

        def hybrid_forward(self, F, x, const):
            return x + const

    test = Test()
    test.initialize()
    trainer = gluon.Trainer(test.collect_params(), 'sgd',
                            {'learning_rate': 1.0, 'momentum': 0.5})

    with mx.autograd.record():
        x = mx.nd.ones((2,2))
        x.attach_grad()
        y = test(x)
        y.backward()

    trainer.step(1)

    assert (test.const.data().asnumpy() == test.value).all()
    assert (x.grad.asnumpy() == 1).all()


@with_seed()
def test_parameter_sharing():
    class Net(gluon.Block):
        def __init__(self, in_units=0, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.dense0 = nn.Dense(5, in_units=in_units)
                self.dense1 = nn.Dense(5, in_units=in_units)

        def forward(self, x):
            return self.dense1(self.dense0(x))

    net1 = Net(prefix='net1_', in_units=5)
    net2 = Net(prefix='net2_', params=net1.collect_params())
    net1.collect_params().initialize()
    net2(mx.nd.zeros((3, 5)))

    net1.save_params('net1.params')

    net3 = Net(prefix='net3_')
    net3.load_params('net1.params', mx.cpu())

    net4 = Net(prefix='net4_')
    net5 = Net(prefix='net5_', in_units=5, params=net4.collect_params())
    net4.collect_params().initialize()
    net5(mx.nd.zeros((3, 5)))

    net4.save_params('net4.params')

    net6 = Net(prefix='net6_')
    net6.load_params('net4.params', mx.cpu())


@with_seed()
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
    assert 'float32' in lines[1]
    assert lines[2] == ')'


@with_seed()
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

@with_seed()
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


@with_seed()
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


@with_seed()
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

@with_seed()
@raises(AssertionError)
def test_sparse_symbol_block():
    data = mx.sym.var('data')
    weight = mx.sym.var('weight', stype='row_sparse')
    bias = mx.sym.var('bias')
    out = mx.sym.broadcast_add(mx.sym.dot(data, weight), bias)
    # an exception is expected when creating a SparseBlock w/ sparse param
    net = gluon.SymbolBlock(out, data)

@with_seed()
@raises(ValueError)
def test_sparse_hybrid_block():
    params = gluon.ParameterDict('net_')
    params.get('weight', shape=(5, 5), stype='row_sparse')
    params.get('bias', shape=(5,))
    # an exception is expected when creating a HybridBlock w/ sparse param
    net = gluon.nn.Dense(5, params=params)

@with_seed()
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

@with_seed()
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


@with_seed()
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


@with_seed()
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


@with_seed()
def test_batchnorm():
    layer = nn.BatchNorm(in_channels=10)
    check_layer_forward(layer, (2, 10, 10, 10))


@with_seed()
def test_instancenorm():
    layer = nn.InstanceNorm(in_channels=10)
    check_layer_forward(layer, (2, 10, 10, 10))

@with_seed()
def test_layernorm():
    layer = nn.LayerNorm(in_channels=10)
    check_layer_forward(layer, (2, 10, 10, 10))


@with_seed()
def test_reflectionpad():
    layer = nn.ReflectionPad2D(3)
    check_layer_forward(layer, (2, 3, 24, 24))


@with_seed()
def test_reshape():
    x = mx.nd.ones((2, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.collect_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x.reshape((-1,))
        x = x + 10
    x.backward()


@with_seed()
def test_slice():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.collect_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x[1:3]
        x = x + 10
    x.backward()


@with_seed()
def test_at():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.collect_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x[1]
        x = x + 10
    x.backward()


@with_seed()
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


@with_seed()
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


@with_seed()
def test_flatten():
    flatten = nn.Flatten()
    x = mx.nd.zeros((3,4,5,6))
    assert flatten(x).shape == (3, 4*5*6)
    x = mx.nd.zeros((3,6))
    assert flatten(x).shape == (3, 6)
    x = mx.nd.zeros((3,))
    assert flatten(x).shape == (3, 1)


if __name__ == '__main__':
    import nose
    nose.runmodule()
