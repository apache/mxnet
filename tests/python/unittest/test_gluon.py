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
import gc

import mxnet as mx
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn
from mxnet.base import py_str, MXNetError
from mxnet.test_utils import assert_almost_equal, default_context, assert_allclose
from mxnet.util import is_np_array
from mxnet.ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from mxnet.test_utils import use_np
from common import assertRaises, assert_raises_cudnn_not_satisfied, \
    xfail_when_nonstandard_decimal_separator, environment
import numpy as onp
from numpy.testing import assert_array_equal
import pytest
from copy import deepcopy
import warnings
import json
import random
import tempfile

mx.npx.reset_np()

def test_parameter():
    p = gluon.Parameter('weight', shape=(10, 10))
    p.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assert len(p.list_data()) == 2
    assert len(p.list_grad()) == 2
    assert p.data(mx.cpu(1)).context == mx.cpu(1)
    assert p.data(mx.cpu(0)).shape == (10, 10)
    assert p.grad(mx.cpu(0)).stype == 'default'
    assert p.data(mx.cpu(0)).stype == 'default'

    p.reset_ctx(ctx=[mx.cpu(1), mx.cpu(2)])
    assert p.list_ctx() == [mx.cpu(1), mx.cpu(2)]

def test_invalid_parameter_stype():
    with pytest.raises(AssertionError):
        p = gluon.Parameter('weight', shape=(10, 10), stype='invalid')

def test_invalid_parameter_grad_stype():
    with pytest.raises(AssertionError):
        p = gluon.Parameter('weight', shape=(10, 10), grad_stype='invalid')

def test_sparse_parameter():
    p = gluon.Parameter('weight', shape=(10, 10), stype='row_sparse', grad_stype='row_sparse')
    p.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    row_id = mx.np.arange(0, 10, ctx=mx.cpu(1))
    assert len(p.list_grad()) == 2
    # getting row_sparse data without trainer throws an exception
    assertRaises(RuntimeError, p.list_row_sparse_data, row_id)
    trainer = mx.gluon.Trainer([p], 'sgd')
    assert len(p.list_row_sparse_data(row_id)) == 2
    weight = p.row_sparse_data(row_id)
    assert weight.context == mx.cpu(1)
    assert weight.shape == (10, 10)
    assert weight.stype == 'row_sparse'
    assert p.var().attr('__storage_type__') == str(_STORAGE_TYPE_STR_TO_ID['row_sparse'])
    assert p.grad(mx.cpu(0)).stype == 'row_sparse'

    p.reset_ctx(ctx=[mx.cpu(1), mx.cpu(2)])
    assert p.list_ctx() == [mx.cpu(1), mx.cpu(2)]

def test_parameter_invalid_access():
    # cannot call data on row_sparse parameters
    p0 = gluon.Parameter('weight', shape=(10, 10), stype='row_sparse', grad_stype='row_sparse')
    p0.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assertRaises(RuntimeError, p0.data)
    assertRaises(RuntimeError, p0.list_data)
    row_id = mx.np.arange(0, 10)
    # cannot call row_sparse_data on dense parameters
    p1 = gluon.Parameter('weight', shape=(10, 10))
    p1.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assertRaises(RuntimeError, p1.row_sparse_data, row_id.copyto(mx.cpu(0)))
    assertRaises(RuntimeError, p1.list_row_sparse_data, row_id)


def test_parameter_row_sparse_data():
    ctx0 = mx.cpu(1)
    ctx1 = mx.cpu(2)
    dim0 = 4
    x = gluon.Parameter('x', shape=(dim0, 2), stype='row_sparse')
    x.initialize(init='xavier', ctx=[ctx0, ctx1])
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


@use_np
def test_constant():
    class Test(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Test, self).__init__(**kwargs)
            self.value = onp.asarray([[1,2], [3,4]])
            self.const = gluon.Constant(self.value)

        def forward(self, x):
            return x + self.const.data()

    test = Test()
    test.initialize()
    trainer = gluon.Trainer(test.collect_params(), 'sgd',
                            {'learning_rate': 1.0, 'momentum': 0.5})

    with mx.autograd.record():
        x = mx.np.ones((2,2))
        x.attach_grad()
        y = test(x)
        y.backward()

    trainer.step(1)

    assert (test.const.data().asnumpy() == test.value).all()
    assert (x.grad.asnumpy() == 1).all()


@use_np
def test_parameter_sharing():
    class Net(gluon.Block):
        def __init__(self, in_units=0, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.dense0 = nn.Dense(5, in_units=in_units)
            self.dense1 = nn.Dense(5, in_units=in_units)

        def forward(self, x):
            return self.dense1(self.dense0(x))

    net1 = Net(in_units=5)
    net2 = Net().share_parameters(net1.collect_params())
    net1.initialize()
    net2(mx.np.zeros((3, 5)))

    net1.save_parameters('net1.params')

    net3 = Net()
    net3.load_parameters('net1.params', mx.cpu())

    net4 = Net()
    net5 = Net(in_units=5).share_parameters(net4.collect_params())
    net4.initialize()
    net5(mx.np.zeros((3, 5)))

    net4.save_parameters('net4.params')

    net6 = Net()
    net6.load_parameters('net4.params', mx.cpu())


def test_parameter_str():
    class Net(gluon.Block):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.dense0 = nn.Dense(10, in_units=5, use_bias=False)

    net = Net()
    lines = str(net.collect_params()).splitlines()

    assert 'dense0.weight' in lines[0]
    assert '(10, 5)' in lines[0]
    assert 'float32' in lines[0]


def test_collect_parameters():
    net = nn.HybridSequential()
    net.add(nn.Conv2D(10, 3))
    net.add(nn.Dense(10, activation='relu'))
    assert set(net.collect_params().keys()) == \
        set(['0.weight', '0.bias','1.weight','1.bias'])
    assert set(net.collect_params('.*weight').keys()) == \
        set(['0.weight', '1.weight'])
    assert set(net.collect_params('0.bias|1.bias').keys()) == \
        set(['0.bias', '1.bias'])

@use_np
def test_basic():
    model = nn.Sequential()
    model.add(nn.Dense(128, activation='tanh', in_units=10, flatten=False))
    model.add(nn.Dropout(0.5))
    model.add(nn.Dense(64, activation='tanh', in_units=256),
              nn.Dense(32, in_units=64))
    model.add(nn.Activation('relu'))

    # ndarray
    model.initialize(mx.init.Xavier(magnitude=2.24))
    x = model(mx.np.zeros((32, 2, 10)))
    assert x.shape == (32, 32)
    x.wait_to_read()

    model.setattr('grad_req', 'null')
    assert list(model.collect_params().values())[0]._grad is None
    model.setattr('grad_req', 'write')
    assert list(model.collect_params().values())[0]._grad is not None


def test_sparse_symbol_block():
    data = mx.sym.var('data')
    weight = mx.sym.var('weight', stype='row_sparse')
    bias = mx.sym.var('bias')
    out = mx.sym.broadcast_add(mx.sym.dot(data, weight), bias)
    with pytest.raises(AssertionError):
        # an exception is expected when creating a SparseBlock w/ sparse param
        net = gluon.SymbolBlock(out, data)

def test_sparse_hybrid_block():
    params = {}
    params['weight'] = gluon.Parameter('weight', shape=(5,5), stype='row_sparse', dtype='float32')
    params['bias'] = gluon.Parameter('bias', shape=(5), dtype='float32')
    net = gluon.nn.Dense(5).share_parameters(params)
    net.initialize()
    x = mx.np.ones((2,5))
    with pytest.raises(RuntimeError):
        # an exception is expected when forwarding a HybridBlock w/ sparse param
        y = net(x)


@use_np
def test_hybrid_block_none_args():
    class Foo(gluon.HybridBlock):
        def forward(self, a, b):
            if a is None and b is not None:
                return b
            elif b is None and a is not None:
                return a
            elif a is not None and b is not None:
                return a + b
            else:
                raise NotImplementedError

    class FooDefault(gluon.HybridBlock):
        def forward(self, a, b=None):
            if a is None and b is not None:
                return b
            elif b is None and a is not None:
                return a
            elif a is not None and b is not None:
                return a + b
            else:
                raise NotImplementedError


    class FooNested(gluon.HybridBlock):
        def __init__(self):
            super(FooNested, self).__init__()
            self.f1 = Foo()
            self.f2 = Foo()
            self.f3 = Foo()

        def forward(self, a, b):
            data = self.f1(a, b)
            data = self.f2(a, data)
            data = self.f3(data, b)
            return data

    for arg_inputs in [(None, mx.np.ones((10,))),
                       (mx.np.ones((10,)), mx.np.ones((10,))),
                       (mx.np.ones((10,)), None)]:
        foo1 = FooNested()
        foo1.hybridize()
        foo2 = FooNested()
        for _ in range(2): # Loop for 2 times to trigger forwarding of the cached version
            out1 = foo1(*arg_inputs)
            out2 = foo2(*arg_inputs)
            if isinstance(out1, tuple):
                for lhs, rhs in zip(out1, out2):
                    assert_almost_equal(lhs.asnumpy(), rhs.asnumpy())
            else:
                assert_almost_equal(out1.asnumpy(), out2.asnumpy())

    for do_hybridize in [True, False]:
        foo = FooNested()
        if do_hybridize:
            foo.hybridize()
        pytest.raises(ValueError, foo, None, None)

    # Make sure the ValueError is correctly raised
    foo = FooNested()
    foo.hybridize()
    foo(None, mx.np.ones((10,)))  # Pass for the first time to initialize the cached op
    pytest.raises(ValueError, lambda: foo(mx.np.ones((10,)), mx.np.ones((10,))))
    foo = FooNested()
    pytest.raises(TypeError, lambda: foo(mx.np.ones((10,)), mx.sym.var('a')))
    foo = FooNested()
    pytest.raises(TypeError, lambda: foo(mx.sym.var('a'), mx.np.ones((10,))))

    # Test the case of the default values
    foo1 = FooDefault()
    foo1.hybridize()
    foo2 = FooDefault()
    out1 = foo1(mx.np.ones((10,)))
    out2 = foo2(mx.np.ones((10,)))
    out3 = foo1(mx.np.ones((10,)), None)
    out4 = foo2(mx.np.ones((10,)), None)
    assert_almost_equal(out1.asnumpy(), out2.asnumpy())
    assert_almost_equal(out1.asnumpy(), out3.asnumpy())
    assert_almost_equal(out1.asnumpy(), out4.asnumpy())
    foo1 = FooDefault()
    foo1.hybridize()
    out1 = foo1(mx.np.ones((10,)), None)
    out2 = foo1(mx.np.ones((10,)))
    assert_almost_equal(out1.asnumpy(), out2.asnumpy())
    pytest.raises(ValueError, lambda: foo1(mx.np.ones((10,)), mx.np.ones((10,))))


@use_np
def test_hybrid_block_hybrid_no_hybrid():
    class FooHybrid(gluon.HybridBlock):
        def forward(self, a, b):
            if isinstance(a, (list, tuple)):
                a = sum(a)
            if isinstance(b, (list, tuple)):
                b = sum(b)
            return a + b

    class Foo(gluon.Block):
        def forward(self, a, b):
            if isinstance(a, (list, tuple)):
                a = sum(a)
            if isinstance(b, (list, tuple)):
                b = sum(b)
            return a + b
    # When hybridize is not called, HybridBlock acts the same as Block
    foo_hybrid = FooHybrid()
    foo = Foo()
    for a, b in [(mx.np.ones((10,)), 1),
                 (mx.np.ones((20,)), 2),
                 ([mx.np.ones((10,)), mx.np.ones((10,))],
                  [mx.np.ones((10)), mx.np.ones((10,)), mx.np.ones((10,))]),
                 ([mx.np.ones((10,)), mx.np.ones((10,))], 3)]:
        hybrid_block_out = foo_hybrid(a, b)
        block_out = foo(a, b)
        assert_almost_equal(hybrid_block_out.asnumpy(), block_out.asnumpy())
    # When hybridize is called, we need to make sure that the model raises for the unsupported cases
    # 1. Scalar values in the input
    # 2. No sym in the input
    # 3. No mixing of cpu ndarray and gpu ndarray  (Tested in gpu/test_gluon_gpu.py)
    # 4. Allow mixing of cpu_pinned and cpu
    foo_hybrid = FooHybrid()
    foo_hybrid.hybridize()
    pytest.raises(ValueError, lambda: foo_hybrid(mx.np.ones((10,)), 1))
    foo_hybrid = FooHybrid()
    foo_hybrid.hybridize()
    pytest.raises(TypeError, lambda: foo_hybrid(mx.np.ones((10,)), mx.sym.var('a')))
    foo_hybrid = FooHybrid()
    foo_hybrid.hybridize()
    pytest.raises(ValueError, lambda: foo_hybrid(mx.np.ones((10,), ctx=mx.cpu(1)),
                                                 mx.np.ones((10,), ctx=mx.cpu(2))))


def check_layer_forward(layer, dshape):
    print("checking layer {}\nshape: {}.".format(layer, dshape))
    layer.initialize()
    x = mx.np.ones(shape=dshape)
    x.attach_grad()
    with mx.autograd.record():
        out = layer(x)
    out.backward()

    np_out = out.asnumpy()
    np_dx = x.grad.asnumpy()

    layer.hybridize()

    x = mx.np.ones(shape=dshape)
    x.attach_grad()
    with mx.autograd.record():
        out = layer(x)
    out.backward()

    mx.test_utils.assert_almost_equal(np_out, out.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(np_dx, x.grad.asnumpy(), rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize('layer,shape', [
    (nn.Conv1D(16, 3, in_channels=4), (1, 4, 10)),
    (nn.Conv1D(16, 3, groups=2, in_channels=4), (1, 4, 10)),
    (nn.Conv1D(16, 3, strides=3, groups=2, in_channels=4), (1, 4, 10)),
    (nn.Conv2D(16, (3, 4), in_channels=4), (1, 4, 20, 20)),
    (nn.Conv2D(16, (5, 4), in_channels=4), (1, 4, 20, 20)),
    (nn.Conv2D(16, (3, 4), groups=2, in_channels=4), (1, 4, 20, 20)),
    (nn.Conv2D(16, (3, 4), strides=4, in_channels=4), (1, 4, 20, 20)),
    (nn.Conv2D(16, (3, 4), dilation=4, in_channels=4), (1, 4, 20, 20)),
    (nn.Conv2D(16, (3, 4), padding=4, in_channels=4), (1, 4, 20, 20)),
    (nn.Conv3D(16, (1, 8, 4), in_channels=4, activation='relu'), (1, 4, 10, 10, 10)),
    (nn.Conv3D(16, (5, 4, 3), in_channels=4), (1, 4, 10, 10, 10)),
    (nn.Conv3D(16, (3, 3, 3), groups=2, in_channels=4), (1, 4, 10, 10, 10)),
    (nn.Conv3D(16, 4, strides=4, in_channels=4), (1, 4, 10, 10, 10)),
    (nn.Conv3D(16, (3, 3, 3), padding=4, in_channels=4), (1, 4, 10, 10, 10)),
])
def test_conv(layer, shape):
    check_layer_forward(layer, shape)

@pytest.mark.parametrize('layer,shape', [
    (nn.Conv2D(16, (3, 3), layout='NHWC', in_channels=4), (1, 10, 10, 4)),
    # (nn.Conv3D(16, (3, 3, 3), layout='NDHWC', in_channels=4), (1, 10, 10, 10, 4)),
])
@pytest.mark.skipif(mx.context.current_context().device_type!='gpu' or
                    not mx.runtime.Features().is_enabled('CUDNN'),
                    reason='nhwc/ndhwc layout is only supported with CUDNN.')
def test_conv_nhwc(layer, shape):
    check_layer_forward(layer, shape)


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
    #   nn.Conv2DTranspose(16, (3, 4), padding=4, in_channels=4),
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
    # transpose shape to bring feature dimension 'c' from 2nd position to last
    def transpose(shape):
        return (shape[0],) + shape[2:] + (shape[1],)

    for layout in ['NCW', 'NWC']:
        shape1d = (1, 2, 10)
        if layout == 'NWC':
            shape1d = transpose(shape1d)
        layers1d = [
            nn.MaxPool1D(layout=layout),
            nn.MaxPool1D(3, layout=layout),
            nn.MaxPool1D(3, 2, layout=layout),
            nn.AvgPool1D(layout=layout),
            nn.AvgPool1D(count_include_pad=False, layout=layout),
            nn.GlobalAvgPool1D(layout=layout),
            ]
        for layer in layers1d:
            check_layer_forward(layer, shape1d)


    for layout in ['NCHW', 'NHWC']:
        shape2d = (1, 2, 10, 10)
        if layout == 'NHWC':
            shape2d = transpose(shape2d)
        layers2d = [
            nn.MaxPool2D(layout=layout),
            nn.MaxPool2D((3, 3), layout=layout),
            nn.MaxPool2D(3, 2, layout=layout),
            nn.AvgPool2D(layout=layout),
            nn.AvgPool2D(count_include_pad=False, layout=layout),
            nn.GlobalAvgPool2D(layout=layout),
            ]
        for layer in layers2d:
            check_layer_forward(layer, shape2d)

    for layout in ['NCDHW', 'NDHWC']:
        shape3d = (1, 2, 10, 10, 10)
        if layout == 'NDHWC':
            shape3d = transpose(shape3d)
        layers3d = [
            nn.MaxPool3D(layout=layout),
            nn.MaxPool3D((3, 3, 3), layout=layout),
            nn.MaxPool3D(3, 2, layout=layout),
            nn.AvgPool3D(layout=layout),
            nn.AvgPool3D(count_include_pad=False, layout=layout),
            nn.GlobalAvgPool3D(layout=layout),
            ]
        for layer in layers3d:
            check_layer_forward(layer, shape3d)

    # test ceil_mode
    for layout in ['NCHW', 'NHWC']:
        xshape = (2, 2, 10, 10)
        noceil_out_shape = (2, 2, 3, 3)
        ceil_out_shape = (2, 2, 4, 4)
        if layout == 'NHWC':
            xshape = transpose(xshape)
            noceil_out_shape = transpose(noceil_out_shape)
            ceil_out_shape = transpose(ceil_out_shape)

        x = mx.np.zeros(xshape)

        layer = nn.MaxPool2D(3, ceil_mode=False, layout=layout)
        layer.initialize()
        assert (layer(x).shape==noceil_out_shape)

        layer = nn.MaxPool2D(3, ceil_mode=True, layout=layout)
        layer.initialize()
        assert (layer(x).shape==ceil_out_shape)


@pytest.mark.parametrize('variable', ['running_var', 'running_mean'])
def test_batchnorm_backward_synchronization(variable):
    """
    Tests if synchronization of BatchNorm running variables is done correctly.
    If not, the test sometimes fails - depending on the timing.
    """
    ctx = mx.test_utils.default_context()

    for _ in range(20):
        layer = nn.BatchNorm()
        layer.initialize(ctx=ctx)
        for _ in range(3):
            data = mx.np.random.normal(loc=10, scale=2, size=(1, 3, 10, 10), ctx=ctx)
            with mx.autograd.record():
                out = layer(data)
            out.backward()

        # check if each read give the same value
        var1 = getattr(layer, variable).data().asnumpy()
        for _ in range(10):
            var2 = getattr(layer, variable).data().asnumpy()
            if (var1 != var2).any():
                raise AssertionError("Two consecutive reads of " + variable + " give different results")


def test_batchnorm():
    layer = nn.BatchNorm(in_channels=10)
    check_layer_forward(layer, (2, 10, 10, 10))


@use_np
@xfail_when_nonstandard_decimal_separator
def test_sync_batchnorm():
    def _check_batchnorm_result(input, num_devices=1, cuda=False):
        from mxnet.gluon.utils import split_and_load

        def _find_bn(module):
            if isinstance(module, (mx.gluon.nn.BatchNorm, mx.gluon.nn.SyncBatchNorm)):
                return module
            elif isinstance(module.module, (mx.gluon.nn.BatchNorm, mx.gluon.nn.SyncBatchNorm)):
                return module.module

            raise RuntimeError('BN not found')

        def _syncParameters(bn1, bn2, ctx):
            ctx = input.context
            bn2.gamma.set_data(bn1.gamma.data(ctx))
            bn2.beta.set_data(bn1.beta.data(ctx))
            bn2.running_mean.set_data(bn1.running_mean.data(ctx))
            bn2.running_var.set_data(bn1.running_var.data(ctx))

        input1 = input.copy()
        input2 = input.copy()

        if cuda:
            input1 = input.as_in_context(mx.gpu(0))
            ctx_list = [mx.gpu(i) for i in range(num_devices)]
        else:
            ctx_list = [mx.cpu(0) for _ in range(num_devices)]

        nch = input.shape[1] if input.ndim > 1 else 1
        bn1 = mx.gluon.nn.BatchNorm(in_channels=nch)
        bn2 = mx.gluon.nn.SyncBatchNorm(
            in_channels=nch, num_devices=num_devices)

        bn1.initialize(ctx=ctx_list[0])
        bn2.initialize(ctx=ctx_list)

        # using the same values for gamma and beta
        #_syncParameters(_find_bn(bn1), _find_bn(bn2), ctx_list[0])

        input1.attach_grad()
        inputs2 = split_and_load(input2, ctx_list, batch_axis=0)
        for xi in inputs2:
            xi.attach_grad()

        with mx.autograd.record():
            output1 = bn1(input1)
            output2 = [bn2(xi) for xi in inputs2]
            loss1 = (output1 ** 2).sum()
            loss2 = [(output ** 2).sum() for output in output2]
            mx.autograd.backward(loss1)
            mx.autograd.backward(loss2)

        output2 = mx.np.concatenate([output.as_in_context(input.context)
                                    for output in output2], axis=1)
        # check bn1

        momentum = 0.9
        epsilon = 1e-5
        axis = 1
        data = input1
        running_mean = mx.np.zeros(nch, ctx=data.context)
        running_var = mx.np.ones(nch, ctx=data.context)

        axes = list(range(data.ndim))
        del axes[axis]
        data_mean = data.mean(axis=axes, keepdims=True)
        data_var = mx.np.square(data - data_mean).mean(axis=axes, keepdims=True)

        target_output = (data - data_mean) / mx.np.sqrt(data_var + epsilon)

        # squeeze data_mean and data_var
        data_mean_flat = data_mean.squeeze()
        data_var_flat = data_var.squeeze()

        running_mean = running_mean * momentum + \
            data_mean_flat * (1 - momentum)
        running_var = running_var * momentum + \
            data_var_flat * (1 - momentum)

        atol = 1e-2
        rtol = 1e-2
        assert_almost_equal(output1.asnumpy(), target_output.asnumpy(),
                            atol=atol, rtol=rtol)
        assert_almost_equal(_find_bn(bn1).running_mean.data(ctx_list[0]).asnumpy(),
                            running_mean.asnumpy(),
                            atol=atol, rtol=rtol)
        assert_almost_equal(_find_bn(bn1).running_var.data(ctx_list[0]).asnumpy(),
                            running_var.asnumpy(),
                            atol=atol, rtol=rtol)
        # assert forwarding
        assert_almost_equal(input1.asnumpy(), input2.asnumpy(),
                            atol=atol, rtol=rtol)
        assert_almost_equal(output1.asnumpy(),
                            output2.asnumpy(), atol=atol, rtol=rtol)
        assert_almost_equal(_find_bn(bn1).running_mean.data(ctx_list[0]).asnumpy(),
                            _find_bn(bn2).running_mean.data(ctx_list[0]).asnumpy(),
                            atol=atol, rtol=rtol)
        assert_almost_equal(_find_bn(bn1).running_var.data(ctx_list[0]).asnumpy(),
                            _find_bn(bn2).running_var.data(ctx_list[0]).asnumpy(),
                            atol=atol, rtol=rtol)
        input2grad = mx.np.concatenate(
            [output.grad.as_in_context(input.ctx) for output in inputs2], axis=0)
        assert_almost_equal(input1.grad.asnumpy(),
                            input2grad.asnumpy(), atol=atol, rtol=rtol)

    cfgs = [(1, False)]
    num_gpus = 0 if default_context().device_type != 'gpu' else mx.context.num_gpus()
    batch_size = 24
    for i in range(1, num_gpus + 1):
        if batch_size % i == 0:
            cfgs.append((i, True))
    for ndev, cuda in cfgs:
        # check with unsync version
        for shape in [(batch_size, 2), (batch_size, 3, 4), (batch_size, 4, 4, 4), (batch_size, 5, 6, 4, 4)]:
            print(str((ndev, cuda, shape)))
            for i in range(10):
                _check_batchnorm_result(mx.np.random.uniform(size=shape,
                                                             ctx=mx.cpu(0)),
                                        num_devices=ndev, cuda=cuda)


def test_instancenorm():
    layer = nn.InstanceNorm(in_channels=10)
    check_layer_forward(layer, (2, 10, 10, 10))

def test_layernorm():
    layer = nn.LayerNorm(in_channels=10)
    check_layer_forward(layer, (2, 10, 10, 10))
    # Check for the case of error raising
    for hybridize in [False, True]:
        layer = nn.LayerNorm(in_channels=10)
        layer.initialize()
        if hybridize:
            layer.hybridize()
        pytest.raises(AssertionError, lambda: layer(mx.np.ones((2, 11))))

def test_groupnorm():
    layer = nn.GroupNorm()
    check_layer_forward(layer, (2, 10, 10, 10))
    layer = nn.GroupNorm(num_groups=2)
    check_layer_forward(layer, (2, 10, 10, 10))
    layer = nn.GroupNorm(num_groups=5)
    check_layer_forward(layer, (2, 10, 10, 10))

def test_reflectionpad():
    layer = nn.ReflectionPad2D(3)
    check_layer_forward(layer, (2, 3, 24, 24))


def test_reshape():
    x = mx.np.ones((2, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x.reshape((-1,))
        x = x + 10
    x.backward()


def test_slice():
    x = mx.np.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x[1:3]
        x = x + 10
    x.backward()


def test_at():
    x = mx.np.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_channels=4)
    layer.initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x[1]
        x = x + 10
    x.backward()


def test_deferred_init():
    x = mx.np.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2)
    layer.initialize()
    layer(x)



@use_np
def check_split_data(x, num_slice, batch_axis, **kwargs):
    res = gluon.utils.split_data(x, num_slice, batch_axis, **kwargs)
    assert len(res) == num_slice
    mx.test_utils.assert_almost_equal(mx.np.concatenate(res, axis=batch_axis).asnumpy(),
                                      x.asnumpy())
    np_res = onp.array_split(x.asnumpy(), num_slice, axis=batch_axis)
    res_asnp = [s.asnumpy() for s in res]
    for r1, r2 in zip(np_res, res_asnp):
        assert all(r1.reshape(-1) == r2.reshape(-1))


@use_np
def test_split_data_np():
    x = mx.np.random.uniform(size=(128, 33, 64))
    check_split_data(x, 8, 0)
    check_split_data(x, 3, 1)
    check_split_data(x, 4, 1, even_split=False)
    check_split_data(x, 15, 1, even_split=False)
    try:
        check_split_data(x, 4, 1)
    except ValueError:
        return
    assert False, "Should have failed"

def test_split_data():
    x = mx.np.random.uniform(size=(128, 33, 64))
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
    x = mx.np.zeros((3,4,5,6))
    assert flatten(x).shape == (3, 4*5*6)
    x = mx.np.zeros((3,6))
    assert flatten(x).shape == (3, 6)
    x = mx.np.zeros((3,))
    assert flatten(x).shape == (3, 1)

def test_block_attr_hidden():
    b = gluon.Block()

    # regular attributes can change types
    b.a = None
    b.a = 1


def test_block_attr_block():
    b = gluon.Block()

    with pytest.raises(TypeError):
        # regular variables can't change types
        b.b = gluon.Block()
        b.b = (2,)


def test_block_attr_param():
    b = gluon.Block()

    with pytest.raises(TypeError):
        # regular variables can't change types
        b.b = gluon.Parameter()
        b.b = (2,)


def test_block_attr_regular():
    b = gluon.Block()

    # set block attribute also sets a weakref in _children
    b.c = gluon.Block()
    c2 = gluon.Block()
    b.c = c2
    assert b.c is c2 and list(b._children.values())[0]() is c2


def test_block_attr_list_of_block():
    class Model1(gluon.Block):
        def __init__(self, **kwargs):
            super(Model1, self).__init__(**kwargs)
            self.layers = [nn.Dense(i * 10) for i in range(6)]

    class Model2(gluon.Block):
        def __init__(self, **kwargs):
            super(Model2, self).__init__(**kwargs)
            self.layers = dict()
            self.layers['a'] = [nn.Dense(10), nn.Dense(10)]

    class Model3(gluon.Block):
        def __init__(self, **kwargs):
            super(Model3, self).__init__(**kwargs)
            self.layers = nn.Sequential()
            self.layers.add(*[nn.Dense(i * 10) for i in range(6)])

    class Model4(gluon.Block):
        def __init__(self, **kwargs):
            super(Model4, self).__init__(**kwargs)
            self.data = {'a': '4', 'b': 123}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        model = Model1()
        model.collect_params()
        assert len(w) > 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        model = Model2()
        model.collect_params()
        assert len(w) > 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        model = Model3()
        model.collect_params()
        assert len(w) == 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        model = Model4()
        model.collect_params()
        assert len(w) == 0

def check_sequential(net):
    dense1 = gluon.nn.Dense(10)
    net.add(dense1)
    dense2 = gluon.nn.Dense(10)
    net.add(dense2)
    dense3 = gluon.nn.Dense(10)
    net.add(dense3)
    net.initialize()

    net(mx.np.zeros((10, 10)))
    net.hybridize()
    assert net[1] is dense2
    assert net[-1] is dense3
    slc = net[1:3]
    assert len(slc) == 2 and slc[0] is dense2 and slc[1] is dense3
    assert isinstance(slc, type(net))

@use_np
def check_sequential_dc(net):
    class MyBlock(mx.gluon.HybridBlock):
        def __init__(self):
            super().__init__()
            self.dense = mx.gluon.nn.Dense(units=10, in_units=10)
            self.weight = mx.gluon.Parameter('weight', shape=(10, ))

        def forward(self, x):
            return self.dense(x) + self.weight.data()

    dense1 = MyBlock()
    net.add(dense1)
    dense2 = MyBlock()
    net.add(dense2)
    dense3 = MyBlock()
    net.add(dense3)

    net.initialize()
    net.hybridize()
    net(mx.np.zeros((10, 10)))
    assert net[1] is dense2
    assert net[-1] is dense3
    slc = net[1:3]
    assert len(slc) == 2 and slc[0] is dense2 and slc[1] is dense3
    assert isinstance(slc, type(net))

@use_np
@pytest.mark.garbage_expected
def test_sequential():
    check_sequential(gluon.nn.Sequential())
    check_sequential(gluon.nn.HybridSequential())
    check_sequential_dc(gluon.nn.HybridSequential())

def test_sequential_warning():
    with warnings.catch_warnings(record=True) as w:
        # The following line permits the test to pass if run multiple times
        warnings.simplefilter('always')
        b = gluon.nn.Sequential()
        b.add(gluon.nn.Dense(20))
        b.hybridize()
        assert len(w) == 1


@use_np
def test_global_norm_clip():
    def check_global_norm_clip(check_isfinite):
        x1 = mx.np.ones((3,3))
        x2 = mx.np.ones((4,4))
        norm = gluon.utils.clip_global_norm([x1, x2], 1.0, check_isfinite=check_isfinite)
        assert norm == 5.0
        assert_almost_equal(x1.asnumpy(), onp.ones((3,3))/5)
        assert_almost_equal(x2.asnumpy(), onp.ones((4,4))/5)

        x3 = mx.np.array([1.0, 2.0, float('nan')])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gluon.utils.clip_global_norm([x1, x3], 2.0, check_isfinite=check_isfinite)
            assert len(w) == check_isfinite

    for check_isfinite in [True, False]:
        check_global_norm_clip(check_isfinite)


def test_embedding():
    def check_embedding():
        layer = gluon.nn.Embedding(10, 100)
        layer.initialize()
        x = mx.np.array([3,4,2,0,1])
        with mx.autograd.record():
            y = layer(x)
            y.backward()
        assert (layer.weight.grad().asnumpy()[:5] == 1).all()
        assert (layer.weight.grad().asnumpy()[5:] == 0).all()

    def check_embedding_large_input():
        embedding = mx.gluon.nn.Embedding(10, 1)
        embedding.initialize()
        embedding.hybridize()
        shape = (20481,)
        with mx.autograd.record():
            emb_in = embedding(mx.np.ones(shape))
            loss = emb_in.sum()
        loss.backward()
        assert embedding.weight.grad().sum().item() == 20481

    check_embedding()
    check_embedding_large_input()

def test_export(tmpdir):
    tmpfile = os.path.join(str(tmpdir), 'gluon')
    ctx = mx.context.current_context()
    model = gluon.model_zoo.vision.resnet18_v1(
        ctx=ctx, pretrained=False)
    model.initialize()
    model.hybridize()
    data = mx.np.random.normal(size=(1, 3, 32, 32))
    out = model(data)

    symbol_filename, params_filename = model.export(tmpfile)
    assert symbol_filename == tmpfile+'-symbol.json'
    assert params_filename == tmpfile+'-0000.params'

@use_np
def test_import():
    ctx = mx.context.current_context()
    net1 = gluon.model_zoo.vision.resnet18_v1(
        ctx=ctx, pretrained=False)
    net1.initialize()
    net1.hybridize()
    data = mx.np.random.normal(size=(1, 3, 32, 32))
    out1 = net1(data)

    net1.export('net1', epoch=1)

    net2 = gluon.SymbolBlock.imports(
        'net1-symbol.json', ['data'], 'net1-0001.params', ctx)
    out2 = net2(data)
    lines = str(net2).splitlines()

    assert_almost_equal(out1.asnumpy(), out2.asnumpy())
    assert lines[0] == 'SymbolBlock('
    assert lines[1]
    assert lines[2] == ')'


def test_hybrid_stale_cache():
    net = mx.gluon.nn.HybridSequential()
    net.add(mx.gluon.nn.Dense(10, weight_initializer='zeros', bias_initializer='ones', flatten=False))

    net.hybridize()
    net.initialize()
    net(mx.np.ones((2,3,5)))

    net.add(mx.gluon.nn.Flatten())
    assert net(mx.np.ones((2,3,5))).shape == (2, 30)

    net = mx.gluon.nn.HybridSequential()
    net.fc1 = mx.gluon.nn.Dense(10, weight_initializer='zeros',
                                bias_initializer='ones', flatten=False)
    net.fc2 = mx.gluon.nn.Dense(10, weight_initializer='zeros',
                                bias_initializer='ones', flatten=False)
    net.hybridize()
    net.initialize()
    net(mx.np.ones((2,3,5)))

    net.fc2 = mx.gluon.nn.Dense(10, weight_initializer='zeros',
                                bias_initializer='ones', flatten=True)
    net.initialize()
    assert net(mx.np.ones((2,3,5))).shape == (2, 10)


def test_lambda():
    net1 = mx.gluon.nn.HybridSequential()
    net1.add(nn.Activation('tanh'),
             nn.LeakyReLU(0.1))

    net2 = mx.gluon.nn.HybridSequential()
    op3 = lambda x, *args: mx.npx.leaky_relu(x, *args, slope=0.1)
    net2.add(nn.HybridLambda('tanh'),
             nn.HybridLambda(op3))

    op4 = lambda x: mx.npx.leaky_relu(x, slope=0.1)
    net3 = mx.gluon.nn.Sequential()
    net3.add(nn.Lambda('tanh'),
             nn.Lambda(op4))

    input_data = mx.np.random.uniform(size=(2, 3, 5, 7))
    out1, out2, out3 = net1(input_data), net2(input_data), net3(input_data)
    assert_almost_equal(out1.asnumpy(), out2.asnumpy(), rtol=1e-3, atol=1e-3)
    assert_almost_equal(out1.asnumpy(), out3.asnumpy(), rtol=1e-3, atol=1e-3)


@use_np
def test_fill_shape_deferred():
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=2, padding=1),
            nn.BatchNorm(),
            nn.Dense(10))
    net
    net.hybridize()
    net.initialize()
    net(mx.np.ones((2,3,5,7)))
    assert net[0].weight.shape[1] == 3, net[0].weight.shape[1]
    assert net[1].gamma.shape[0] == 64, net[1].gamma.shape[0]
    assert net[2].weight.shape[1] == 3072, net[2].weight.shape[1]


@use_np
def test_dtype():
    net = mx.gluon.model_zoo.vision.resnet18_v1()
    net.initialize()
    net.cast('float64')
    with mx.autograd.record():
        y = net(mx.np.ones((16, 3, 32, 32), dtype='float64'))
        y.backward()

    net = mx.gluon.model_zoo.vision.resnet18_v1()
    net.initialize()
    net.hybridize()
    net(mx.np.ones((16, 3, 32, 32), dtype='float32'))

    net.cast('float64')
    net(mx.np.ones((16, 3, 32, 32), dtype='float64'))

    mx.npx.waitall()

    class Net(gluon.Block):
        def __init__(self, in_dim, output_dim):
            super(Net, self).__init__()
            self.embed = gluon.nn.Embedding(input_dim=in_dim, output_dim=output_dim,dtype=onp.float64)
            self.dense = gluon.nn.Dense(2, dtype=onp.float64)

        def forward(self, x):
            e = self.embed(x)
            assert(e.dtype == onp.float64)
            y = self.dense(e)
            assert(y.dtype == onp.float64)
            return y

    net = Net(5, 10)
    net.initialize()
    out = net(mx.np.ones((3,), dtype=onp.float64))
    mx.npx.waitall()

def test_fill_shape_load():
    ctx = mx.context.current_context()
    net1 = nn.HybridSequential()
    net1.add(nn.Conv2D(64, kernel_size=2, padding=1),
             nn.BatchNorm(),
             nn.Dense(10))
    net1
    net1.hybridize()
    net1.initialize(ctx=ctx)
    net1(mx.np.ones((2,3,5,7), ctx=ctx))
    net1.save_parameters('net_fill.params')

    net2 = nn.HybridSequential()
    net2.add(nn.Conv2D(64, kernel_size=2, padding=1),
             nn.BatchNorm(),
             nn.Dense(10))
    net2.hybridize()
    net2.initialize()
    net2.load_parameters('net_fill.params', ctx)
    assert net2[0].weight.shape[1] == 3, net2[0].weight.shape[1]
    assert net2[1].gamma.shape[0] == 64, net2[1].gamma.shape[0]
    assert net2[2].weight.shape[1] == 3072, net2[2].weight.shape[1]


def test_inline():
    net = mx.gluon.nn.HybridSequential()
    net.add(mx.gluon.nn.Dense(10))
    net.add(mx.gluon.nn.Dense(10))
    net.add(mx.gluon.nn.Dense(10))

    net.initialize()
    net.hybridize(inline_limit=3)
    with mx.autograd.record():
        y = net(mx.np.zeros((1,10)))

    len_1 = len(json.loads(mx.autograd.get_symbol(y).tojson())['nodes'])
    y.backward()

    net.hybridize(inline_limit=0)
    with mx.autograd.record():
        y = net(mx.np.zeros((1,10)))

    len_2 = len(json.loads(mx.autograd.get_symbol(y).tojson())['nodes'])
    y.backward()

    assert len_1 == len_2 + 2


@xfail_when_nonstandard_decimal_separator
def test_activations():
    point_to_validate = mx.np.array([-0.1, 0.1] * 3)

    swish = mx.gluon.nn.Swish()
    def swish_test(x):
        return x * mx.npx.sigmoid(x)

    for test_point, ref_point in zip(swish_test(point_to_validate), swish(point_to_validate)):
        assert test_point == ref_point

    silu = mx.gluon.nn.SiLU()
    def silu_test(x):
        return x * mx.npx.sigmoid(x)

    for test_point, ref_point in zip(silu_test(point_to_validate), silu(point_to_validate)):
        assert test_point == ref_point

    elu = mx.gluon.nn.ELU()
    def elu_test(x):
        def elu(x):
            return mx.np.expm1(x) if x <= 0.0 else x
        return [elu(x_i) for x_i in x]

    for test_point, ref_point in zip(elu_test(point_to_validate), elu(point_to_validate)):
        assert_almost_equal(test_point.asnumpy(), ref_point.asnumpy())

    selu = mx.gluon.nn.SELU()
    def selu_test(x):
        def selu(x):
            scale, alpha = 1.0507009873554804934193349852946, 1.6732632423543772848170429916717
            return scale * x if x >= 0 else scale * alpha * mx.np.expm1(x)
        return [selu(x_i) for x_i in x]

    for test_point, ref_point in zip(selu_test(point_to_validate), selu(point_to_validate)):
        assert test_point == ref_point

    prelu = mx.gluon.nn.PReLU()
    prelu.initialize()
    x = point_to_validate.reshape((1, 3, 2))
    assert_almost_equal(prelu(x).asnumpy(), mx.np.where(x >= 0, x, 0.25 * x).asnumpy())

    multichannel_init = mx.initializer.Constant(mx.np.array([0.1, 0.25, 0.5]))
    prelu_multichannel = mx.gluon.nn.PReLU(alpha_initializer=multichannel_init, in_channels=3)
    prelu_multichannel.initialize()
    assert_almost_equal(prelu_multichannel(x).asnumpy(), onp.array([[-0.01, 0.1], [-0.025, 0.1], [-0.05, 0.1]]))

    # https://github.com/apache/incubator-mxnet/issues/18381
    # gelu = mx.gluon.nn.GELU()
    # def gelu_test(x):
    #     CUBE_CONSTANT = 0.044715
    #     ROOT_TWO_OVER_PI = 0.7978845608028654
    #     def g(x):
    #         return ROOT_TWO_OVER_PI * (x + CUBE_CONSTANT * x * x * x)
    #     def f(x):
    #         return 1.0 + mx.nd.tanh(g(x))
    #     def gelu(x):
    #         return 0.5 * x * f(x)
    #     return [gelu(x_i) for x_i in x]

    # for test_point, ref_point in zip(gelu_test(point_to_validate), gelu(point_to_validate)):
    #     assert test_point == ref_point


@use_np
def test_dropout():
    def get_slice(x, axis, idx):
        ix = ()
        for i in range(x.ndim):
            if i == axis:
                ix += (idx,)
            else:
                ix += (slice(None, None, None),)
        return x[ix]

    def check_dropout_axes(ratio, shape, axes):
        compactshape = list(shape)
        for axis in axes:
            compactshape[axis] = 1
        compactx = mx.np.random.uniform(size=tuple(compactshape))
        broadcastx = compactx.broadcast_to(shape)
        dropouty = mx.gluon.nn.Dropout(rate=ratio, axes=axes)(broadcastx)
        for axis in axes:
            target = get_slice(dropouty, axis, 0).asnumpy()
            for i in range(1, shape[axis]):
                assert(get_slice(dropouty, axis, i).asnumpy() == target).all()

    nshape = (10, 10, 10, 10)
    with mx.autograd.train_mode():
        check_dropout_axes(0.25, nshape, axes = (0,))
        check_dropout_axes(0.25, nshape, axes = (1,))
        check_dropout_axes(0.25, nshape, axes = (2,))
        check_dropout_axes(0.25, nshape, axes = (3,))
        check_dropout_axes(0.25, nshape, axes = (0, 1))
        check_dropout_axes(0.25, nshape, axes = (0, 2))
        check_dropout_axes(0.25, nshape, axes = (0, 3))
        check_dropout_axes(0.25, nshape, axes = (1, 2))
        check_dropout_axes(0.25, nshape, axes = (1, 3))
        check_dropout_axes(0.25, nshape, axes = (2, 3))
        check_dropout_axes(0.25, nshape, axes = (0, 1, 2))
        check_dropout_axes(0.25, nshape, axes = (0, 2, 3))
        check_dropout_axes(0.25, nshape, axes = (1, 2, 3))

def test_req():
    data = mx.np.random.uniform(size=(1,3,224,224))
    label = mx.np.random.uniform(size=(1))
    label[:] = 1
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    net = nn.HybridSequential()
    net1 = nn.HybridSequential()
    net1.add(nn.Dense(4))
    net2 = nn.HybridSequential()
    net2.add(nn.Dense(3))
    net2.add(nn.Dense(2))
    net.add(net1)
    net.add(net2)
    net.initialize()

    net.hybridize()

    for v in net.collect_params().values():
        v.grad_req = 'add'

    net.zero_grad()
    with mx.autograd.record():
        pred = net(data)
        l = loss(pred, label)
        l.backward()
        grad = net[0][0].weight.grad().mean().asnumpy()
        # run twice to check req = add
        pred = net(data)
        l = loss(pred, label)
        l.backward()

    grad_double = net[0][0].weight.grad().mean().asnumpy()
    assert_almost_equal(grad * 2, grad_double)


@use_np
def test_save_load(tmpdir):
    net = mx.gluon.model_zoo.vision.get_resnet(1, 18, pretrained=False, root=str(tmpdir))
    net.initialize()
    net(mx.np.ones((1,3,224,224)))
    net.save_parameters(os.path.join(str(tmpdir), 'test_save_load.params'))

    net = mx.gluon.model_zoo.vision.get_resnet(1, 18)
    net.output = mx.gluon.nn.Dense(1000)

    net.load_parameters(os.path.join(str(tmpdir), 'test_save_load.params'))

    class Network(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Network, self).__init__(**kwargs)
            self.encoders = gluon.nn.HybridSequential()
            for _ in range(2):
                lstm = mx.gluon.rnn.LSTM(200, 1, bidirectional=True)
                self.encoders.add(lstm)

        def forward(self, x):
            for i in range(2):
                x = self.encoders[i](x)
            return x
    net = Network()
    net.initialize(mx.init.Xavier(), ctx=mx.cpu())
    net.hybridize()
    x = onp.random.rand(32, 10, 10)
    x = mx.np.array(x).as_in_context(mx.cpu())
    net(x)
    # _, param_path = tempfile.mkstemp(suffix='.params', dir=str(tmpdir))
    param_path = os.path.join(str(tmpdir), 'test_save_load_network.params')
    net.save_parameters(param_path)
    net2 = Network()
    net2.load_parameters(param_path)

@use_np
def test_save_load_deduplicate_with_shared_params(tmpdir):
    class B(mx.gluon.Block):
        def __init__(self):
            super(B, self).__init__()
            self.weight = gluon.Parameter('weight', shape=(10, 10))

    class C(mx.gluon.Block):
        def __init__(self, b1, b2):
            super(C, self).__init__()
            self.b1 = b1
            self.b2 = b2

    b1 = B()
    b2 = B().share_parameters(b1.collect_params())
    c = C(b1, b2)
    c.initialize()
    # _, param_path = tempfile.mkstemp(suffix='.params', dir=str(tmpdir))
    param_path = os.path.join(str(tmpdir), 'test_save_load_deduplicate_with_shared_params.params')
    c.save_parameters(param_path, deduplicate=True)

    params = mx.npx.load(param_path)
    assert len(params) == 1  # Only a single copy of the shared parameter is saved

    b1 = B()
    b2 = B().share_parameters(b1.collect_params())
    c = C(b1, b2)
    c.load_parameters(param_path)

    # Test default behavior
    c.save_parameters(param_path, deduplicate=False)

    params = mx.npx.load(param_path)
    assert len(params) == 2  # Only a single copy of the shared parameter is saved

    b1 = B()
    b2 = B().share_parameters(b1.collect_params())
    c = C(b1, b2)
    c.load_parameters(param_path)


def test_hybrid_multi_context():
    net = mx.gluon.model_zoo.vision.get_resnet(1, 18)
    net.initialize(ctx=[mx.cpu(0), mx.cpu(1)])
    net.hybridize()
    net(mx.np.zeros((1, 3, 32, 32), ctx=mx.cpu(0))).asnumpy()

def test_zero_grad():
    def _test_grad_reset(ctx, dtype='float32', sparse=False, embeddingType=None):
        data = mx.np.random.uniform(size=(3,3), dtype=dtype, ctx=ctx)
        if embeddingType is None:
            embeddingType = dtype
        net = nn.Embedding(3, 4, sparse_grad=sparse, dtype=embeddingType)
        net.initialize(ctx=ctx)
        with mx.autograd.record():
            l = net(data)
            l.backward()
        net.zero_grad()
        grad = net.collect_params()['weight'].grad()
        assert_almost_equal(grad.asnumpy(), grad.asnumpy() * 0)

    def _test_multi_reset(nArrays, dtype, ctx):
        # Construct the list of non-zeros arrays with random shapes
        arr = []
        for _ in range(nArrays):
            arrType = random.choice(dtype) if isinstance(dtype, list) else dtype
            shape = ()
            for _ in range(onp.random.randint(1, 5)):
                shape = shape + (onp.random.randint(1, 10),)
            arr.append(mx.nd.random.uniform(shape=shape, dtype=arrType, ctx=ctx))

        # Reset all arrays
        mx.nd.reset_arrays(*arr, num_arrays=len(arr))

        # Check results
        for i in range(nArrays):
            grad = arr[i].asnumpy()
            assert_almost_equal(grad, grad * 0)


    # Setting context for current test
    ctx = mx.context.current_context()

    # Launching _test_multi_reset 10 times with different types & randomly chosen nArrays
    testedTypes = ['float16', 'float32', 'float64']
    for _ in range(10):
        for type in [testedTypes] + testedTypes:
            _test_multi_reset(onp.random.randint(1, 50), type, ctx)

    with environment('MXNET_STORAGE_FALLBACK_LOG_VERBOSE', '0'):
        for type in ['float16', 'float32', 'float64']:
            for embType in ['float32', 'float64']:
                _test_grad_reset(ctx, dtype=type, sparse=False, embeddingType=embType)


@pytest.mark.parametrize('static_alloc', [False, True])
@pytest.mark.parametrize('static_shape', [False, True])
def test_hybrid_static_memory(static_alloc, static_shape):
    if static_shape and not static_alloc:
        pytest.skip()
    x = mx.np.random.uniform(size=(2, 3, 32, 32))
    x.attach_grad()

    net = gluon.model_zoo.vision.get_resnet(
        1, 18, pretrained=False, ctx=mx.context.current_context())
    net.initialize()
    net(x)

    def test(net, x):
        with mx.autograd.record():
            y = net(x) + net(x)
            y.backward()

        grads = {k: v.grad() for k, v in net.collect_params().items() if v.grad_req != 'null'}

        return y, grads

    y1, grads1 = test(net, x)
    net.hybridize(static_alloc=static_alloc, static_shape=static_shape)
    y2, grads2 = test(net, x)

    assert_almost_equal(y1.asnumpy(), y2.asnumpy(), rtol=1e-3, atol=1e-5)
    for key in grads1:
        assert_almost_equal(grads1[key].asnumpy(), grads2[key].asnumpy(), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize('static_alloc', [False, True])
@pytest.mark.parametrize('static_shape', [False, True])
def test_hybrid_static_memory_switching(static_alloc, static_shape):
    if static_shape and not static_alloc:
        pytest.skip()
    net = gluon.model_zoo.vision.get_resnet(
        1, 18, pretrained=False, ctx=mx.context.current_context())
    net.initialize()
    net.hybridize(static_alloc=static_alloc, static_shape=static_shape)

    x = mx.np.random.uniform(size=(4, 3, 32, 32))
    net(x)
    with mx.autograd.record():
        y = net(x)
        y.backward()
    x = mx.np.random.uniform(size=(2, 3, 32, 32))
    net(x)
    with mx.autograd.record():
        y = net(x)
        y.backward()
    mx.npx.waitall()

def test_hook():
    global hook_call_count
    hook_call_count = 0
    global pre_hook_call_count
    pre_hook_call_count = 0

    def call_hook(block, x, y):
        global hook_call_count
        hook_call_count += 1

    def call_pre_hook(block, x):
        global pre_hook_call_count
        pre_hook_call_count += 1

    block = nn.Dense(10)
    block.initialize()
    handle = block.register_forward_hook(call_hook)
    pre_handle = block.register_forward_pre_hook(call_pre_hook)
    block(mx.np.ones((3, 5)))

    assert hook_call_count == 1
    assert pre_hook_call_count == 1

    handle.detach()
    block(mx.np.ones((3, 5)))

    assert hook_call_count == 1
    assert pre_hook_call_count == 2

    pre_handle.detach()
    block(mx.np.ones((3, 5)))
    assert hook_call_count == 1
    assert pre_hook_call_count == 2

@use_np
def test_op_hook_output_names():
    def check_name(block, expected_names, inputs=None, expected_opr_names=None, monitor_all=False):
        opr_names = []
        output_names = []

        def mon_callback(node_name, opr_name, arr):
            output_names.append(node_name)
            opr_names.append(opr_name)
            assert isinstance(arr, mx.nd.NDArray)

        block.register_op_hook(mon_callback, monitor_all)
        if not inputs:
            block(mx.np.ones((2, 3, 4)))
        else:
            block(inputs)

        for output_name, expected_name in zip(output_names, expected_names):
            output_name_list = output_name.split('_')
            output_name_list.pop(1)
            expected_name_list = expected_name.split('_')
            expected_name_list.pop(1)
            assert output_name_list == expected_name_list

        if expected_opr_names:
            for opr_name, expected_opr_name in zip(opr_names, expected_opr_names):
                assert opr_name == expected_opr_name

    # Test with Dense layer
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.Dense(2))
    model.initialize()
    model.hybridize()
    check_name(model, ["node_0_output"])

    # Test with Activation, FListInputNames not registered, input name will have _input appended
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.Activation("relu"))
    model.initialize()
    model.hybridize()
    check_name(model, ["node_1_output"])

    # Test with Pooling, monitor_all is set to True
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.AvgPool1D())
    model.initialize()
    model.hybridize()
    check_name(model, ['node_2_data', 'node_2_output'],
               expected_opr_names=["Pooling"], monitor_all=True)

    # stack two layers and test
    model = mx.gluon.nn.HybridSequential()
    model.add(mx.gluon.nn.Dense(2))
    model.add(mx.gluon.nn.Activation("relu"))
    model.initialize()
    model.hybridize()
    check_name(model,
               ['node_3_data', 'node_3_weight',
                'node_3_bias', 'node_3_output',
                'node_4_input0', 'node_4_output'], monitor_all=True)

    # check with different hybridize modes
    model.hybridize(static_alloc=True)
    check_name(model,
               ['node_5_data', 'node_5_weight',
                'node_5_bias', 'node_5_output',
                'node_6_input0', 'node_6_output'], monitor_all=True)

def test_apply():
    global called_blocks
    called_blocks = []

    def record_name(block):
        global called_blocks
        called_blocks.append(type(block))

    block = nn.HybridSequential()
    block.add(nn.Dense(10))
    block.add(nn.Dropout(0.5))
    block.apply(record_name)

    assert called_blocks == [type(block[0]), type(block[1]), type(block)]


@use_np
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_summary():
    net = gluon.model_zoo.vision.resnet50_v1()
    net.initialize()
    net.summary(mx.np.ones((32, 3, 224, 224)))

    net2 = nn.Sequential()
    net2.add(nn.Embedding(40, 30))
    net2.add(gluon.rnn.LSTM(30))
    net2.add(nn.Dense(40, flatten=False).share_parameters(net2[0].params))
    net2.initialize()
    with mx.util.np_shape(True), mx.util.np_array(True):
        net2.summary(mx.np.ones((80, 32)))

    net3 = gluon.rnn.LSTM(30)
    net3.initialize()
    begin_state = net3.begin_state(32)
    net3.summary(mx.np.ones((80, 32, 5)), begin_state)

    net.hybridize()
    pytest.raises(AssertionError, net.summary, mx.np.ones((32, 3, 224, 224)))

@use_np
@pytest.mark.skip(reason='Currently, sparse feature is not supported in Gluon2.0')
def test_sparse_hybrid_block_grad():
    class Embedding(mx.gluon.HybridBlock):
        def __init__(self, num_tokens, embedding_size):
            super(Embedding, self).__init__()
            self.num_tokens = num_tokens

            self.embedding = mx.gluon.nn.Embedding(
                num_tokens, embedding_size, sparse_grad=True)

        def forward(self, words):
            emb = self.embedding(words)
            return emb + mx.np.ones_like(emb)

    embedding = Embedding(20, 3)
    embedding.initialize()
    embedding.hybridize()

    with mx.autograd.record():
        emb0 = embedding(mx.np.arange(10)).sum()
        emb1 = embedding(mx.np.arange(10)).sum()
        loss = emb0 + emb1
    loss.backward()
    grad = embedding.embedding.weight.grad().asnumpy()
    assert (grad[:10] == 2).all()
    assert (grad[10:] == 0).all()

@use_np
@pytest.mark.skip(reason='Currently, sparse feature is not supported in Gluon2.0')
def test_sparse_hybrid_block():
    class Linear(mx.gluon.HybridBlock):
        def __init__(self, units):
            super(Linear, self).__init__()
            self.w = gluon.Parameter('w', shape=(units, units))

        def forward(self, x, w):
            return mx.np.dot(x, w)

    class SparseBlock(mx.gluon.HybridBlock):
        def __init__(self, units):
            super(SparseBlock, self).__init__()
            self.net = Linear(units)

        def forward(self, x):
            return self.net(x) * x

    block = SparseBlock(2)
    block.initialize()
    block.hybridize()
    x = mx.np.ones((2,2)).tostype('csr')
    with mx.autograd.record():
        z = block(x) + block(x)
    z.backward()
    assert (block.net.w.grad().asnumpy() == 4).all()

def test_hybrid_static_memory_recording():
    net = gluon.model_zoo.vision.get_resnet(
        1, 18, pretrained=False, ctx=mx.context.current_context())
    net.initialize()
    net.hybridize(static_alloc=True)

    x = mx.np.random.uniform(size=(1, 3, 32, 32))
    with mx.autograd.record(True):
        net(x)
    net(x)


@use_np
def test_share_inputs_outputs():
    class TestIOBackward(gluon.HybridBlock):
        def __init__(self):
            super(TestIOBackward, self).__init__()

        def forward(self, in1, in2):
            return in1 + in2

    class TestIOForward(gluon.HybridBlock):
        def __init__(self):
            super(TestIOForward, self).__init__()

        def forward(self, in1):
            return in1

    d1 = mx.np.arange(10)
    d2 = mx.np.arange(10)

    params=[{'inline_limit':0},
            {'inline_limit':0, 'static_alloc':True},
            {'inline_limit':0, 'static_alloc':True, 'static_shape':True}]
    # Test the case that inputs and outputs of a forward graph share NDArrays.
    for param in params:
        t = TestIOForward()
        t.hybridize(**param)
        for i in range(5):
            d1.attach_grad()
            out_grad = mx.np.random.uniform(size=(10))
            res = t(d1)
            assert_almost_equal(res.asnumpy(), d1.asnumpy())

    # Test the case that inputs and outputs of a backward graph share NDArrays.
    for param in params:
        t = TestIOBackward()
        t.hybridize(**param)
        for i in range(5):
            d1.attach_grad()
            d2.attach_grad()
            out_grad = mx.np.random.uniform(size=(10))
            with mx.autograd.record():
                res = t(d1, d2)
            res.backward(out_grad=out_grad)
            assert_almost_equal(out_grad.asnumpy(), d1.grad.asnumpy())
            assert_almost_equal(out_grad.asnumpy(), d2.grad.asnumpy())


@use_np
def test_grad_graph_change():
    class Model(mx.gluon.HybridBlock):
        def forward(self, array, index):
            row = array.take(index)
            return row, index
    array = mx.np.arange(3)
    index = mx.np.array([2])
    array.attach_grad()
    model = Model()
    model.hybridize(inline_limit=0)
    with mx.autograd.record(train_mode=True):
        row, _ = model(array, index)
    row.backward()


def check_layer_forward_withinput(net, x):
    x_hybrid = x.copy()
    x.attach_grad()
    x_hybrid.attach_grad()
    net.initialize()
    with mx.autograd.record():
        out1 = net(x_hybrid)
    out1.backward()
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x)
    out2.backward()
    mx.test_utils.assert_almost_equal(x.grad.asnumpy(), x_hybrid.grad.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(out1.asnumpy(), out2.asnumpy(), rtol=1e-5, atol=1e-6)

@use_np
@pytest.mark.parametrize('chn_num', [16, 256])
@pytest.mark.parametrize('kernel', [1, 3, 224])
def test_conv2d_16c(chn_num, kernel):
    batch_size = 4
    class Net(gluon.HybridBlock):
        def __init__(self,
                     chn_num,
                     kernel,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = gluon.nn.Conv2D(chn_num, (kernel, kernel))

        def forward(self, x):
            out = self.conv0(x)
            return out

    x = mx.np.random.uniform(-1.0, 1.0, size=(batch_size, 3, 224, 224))
    net = Net(chn_num, kernel)
    check_layer_forward_withinput(net, x)

@use_np
@pytest.mark.parametrize('grp', [16])
@pytest.mark.parametrize('kernel_size', [1, 3])
def test_group_conv2d_16c(grp, kernel_size):
    input_size_list = onp.random.randint(low=3, high=65, size=10).tolist()
    batch_size = 4
    class Net(gluon.HybridBlock):
        def __init__(self,
                     chn_num,
                     kernel,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = gluon.nn.Conv2D(chn_num, (1, 1))
            self.conv1 = gluon.nn.Conv2D(chn_num, (kernel, kernel), groups=chn_num)

        def forward(self, x):
            y = self.conv0(x)
            out = self.conv1(y)
            return out

    for i in range(len(input_size_list)):
        x = mx.np.random.uniform(-1.0, 1.0, size=(batch_size, 3, input_size_list[i], input_size_list[i]))
        net = Net(grp, kernel_size)
        check_layer_forward_withinput(net, x)

@use_np
@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_deconv2d_16c():
    in_chn_list = [1024, 512, 256, 128, 64, 32, 16]
    out_chn_list = [512, 256, 128, 64, 32, 16, 3]
    kernel_list = [1, 3, 5, 7]
    in_shape = [4, 8, 16, 32, 64, 224]
    batch_size = 4
    class Net(gluon.HybridBlock):
        def __init__(self, chn_num, kernel, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.deconv0 = gluon.nn.Conv2DTranspose(chn_num, (kernel, kernel))

        def forward(self, x):
            out = self.deconv0(x)
            return out
    for i in range(len(in_shape)):
        x = mx.np.random.uniform(-1.0, 1.0, size=(batch_size, in_chn_list[i], in_shape[i], in_shape[i]))
        for j in range(len(kernel_list)):
            net = Net(out_chn_list[i], kernel_list[j])
            check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_batchnorm_16c():
    chn_list = [16, 1024]
    shape = onp.random.randint(low=1, high=300, size=10)
    shape_list = []
    for i in range(len(shape)):
        shape_list.append((shape[i], shape[i]))
    batch_size = 4
    class Net(gluon.HybridBlock):
        def __init__(self,
                     chn_num,
                     kernel,
                     axis,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = gluon.nn.Conv2D(chn_num, (kernel, kernel))
            self.bn0   = gluon.nn.BatchNorm(axis=axis)

        def forward(self, x):
            conv = self.conv0(x)
            out = self.bn0(conv)
            return out

    for i in range(len(chn_list)):
        for j in range(len(shape_list)):
            shape = (batch_size, ) + (3,) + shape_list[j]
            x = mx.np.random.uniform(-1.0, 1.0, size=shape)
            net = Net(chn_list[i], 1, 1)
            check_layer_forward_withinput(net, x)


@use_np
def test_batchnorm_chnls():
    chn_list = [1024, 512, 256, 128, 64, 45, 32, 16, 3]
    class Net(gluon.HybridBlock):
        def __init__(self,
                     chn_num,
                     norm_kwargs=None,
                     in_channels=3,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.in_channels = in_channels
            self.conv1 = gluon.nn.Conv3D(
                    in_channels=self.in_channels,
                    channels=chn_num,
                    kernel_size=(1, 7, 7),
                    strides=(1, 2, 2),
                    padding=(0, 3, 3),
                    use_bias=False,
                    )
            self.bn1 = gluon.nn.BatchNorm(in_channels=chn_num, **({} if norm_kwargs is None else norm_kwargs))

        def forward(self, x):
            """Hybrid forward of R2+1D net"""
            conv = self.conv1(x)
            out = self.bn1(conv)
            return out

    for i in range(len(chn_list)):
        net = Net(chn_list[i])
        net.initialize(init=init.Constant(1))
        x = mx.np.zeros((1, 3, 8, 160, 160))
        net(x).asnumpy()


@use_np
def test_concat():
    chn_list = [16, 64]
    shapes = [1, 3, 5]
    input_num = onp.random.randint(low=2, high=11)
    shape_list = []
    for i in range(len(shapes)):
        shape_list.append((shapes[i], shapes[i]))
    batch_size = 4
    class Net(gluon.HybridBlock):
        def __init__(self,
                     check_dim,
                     input_num,
                     chn_num,
                     kernel,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.concat = nn.HybridConcatenate(axis=check_dim)
            for i in range(input_num):
                self.concat.add(gluon.nn.Conv2D(chn_num, (kernel, kernel)))

        def forward(self, x):
            return self.concat(x)

    for s in range(len(shape_list)):
        shape = (batch_size,) + (3,) + shape_list[i]
        x = mx.np.random.uniform(-1.0, 1.0, size=shape)
        for i in range(len(chn_list)):
            for axis in range(4):
                net = Net(axis, input_num, chn_list[i], 1)
                check_layer_forward_withinput(net, x)

@use_np
def test_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(64, (3, 3))

        def forward(self, x):
            x_reshape = x.reshape((-1, 3, 128, 32))
            out = self.conv0(x_reshape)
            return out
    x = mx.np.random.uniform(size=(4, 3, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_conv_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(64, (3, 3))
            self.conv1 = nn.Conv2D(128, (3, 3))

        def forward(self, x):
            x_reshape = x.reshape((0, 0, 128, 32))
            y = self.conv0(x_reshape)
            "spatial shape of y is (62, 62)"
            y_reshape = y.reshape((0, 0, 124, 31))
            out = self.conv1(y_reshape)
            return out
    x = mx.np.random.uniform(size=(4, 3, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)

@use_np
def test_slice_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(16, (3, 3))

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=(0, 2, 0, 0), end=(4, 5, 32, 32))
            out = self.conv0(x_slice)
            return out
    x = mx.np.random.uniform(size=(8, 6, 32, 32))
    net = Net()
    check_layer_forward_withinput(net, x)


@use_np
def test_slice_conv_slice_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(32, (3, 3))
            self.conv1 = nn.Conv2D(16, (1, 1))

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=(0, 0, 0, 0), end=(4, 16, 16, 16))
            y = self.conv0(x_slice)
            "shape of y is (4, 32, 14, 14)"
            y_slice = mx.npx.slice(y, begin=(0, 0, 0, 0), end=(4, 16, 3, 3))
            out = self.conv1(y_slice)
            return out
    x = mx.np.random.uniform(size=(4, 32, 32, 32))
    net = Net()
    check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_slice_conv_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(64, (3, 3))
            self.conv1 = nn.Conv2D(128, (3, 3))

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=(0, 0, 1, 1), end=(4, 16, 33, 33))
            y = self.conv0(x_slice)
            "shape of y is (4, 64, 30, 30)"
            y_reshape = y.reshape((0, 0, 60, 15))
            out = self.conv1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(4, 32, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)

@use_np
def test_reshape_conv_slice_conv():
    """
    This test will test gluon Conv2d computation with ndarray reshape and slice
    """
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(16, (3, 3))
            self.conv1 = nn.Conv2D(32, (3, 3))

        def forward(self, x):
            x_reshape = x.reshape((-1, 3, 64, 16))
            y = self.conv0(x_reshape)
            "shape of y is (4, 16, 62, 14)"
            y_slice = mx.npx.slice(y, begin=(0, 0, 0, 0), end=(2, 16, 14, 14))
            out = self.conv1(y_slice)
            return out
    x = mx.np.random.uniform(size=(4, 3, 32, 32))
    net = Net()
    check_layer_forward_withinput(net, x)

@use_np
def test_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            channel0 = onp.random.randint(1, 17)
            self.dense0 = nn.Dense(channel0)

        def forward(self, x):
            x_reshape = x.reshape((8, 64, 128, -1))
            out = self.dense0(x_reshape)
            return out

    x = mx.np.random.uniform(size=(4, 32, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)


@use_np
def test_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            channel0 = onp.random.randint(1, 17)
            self.dense0 = nn.Dense(channel0)
            self.slice = slice

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=tuple(self.slice[0]),
                              end=tuple(self.slice[1]))
            out = self.dense0(x_slice)
            return out

    x = mx.np.random.uniform(size=(16, 32, 64, 64))
    slice = [[0, 16, 0, 0], [4, 32, 32, 32]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)

@use_np
def test_slice_dense_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            channel0 = 32
            channel1 = onp.random.randint(1, 17)
            self.dense0 = nn.Dense(channel0)
            self.dense1 = nn.Dense(channel1)
            self.slice = slice

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.dense0(x_slice)
            y_slice = mx.npx.slice(y, begin=(1, 0), end=(3, 10))
            out = self.dense1(y_slice)
            return out

    x = mx.np.random.uniform(size=(16, 32, 64, 64))
    slice = [[0, 16, 0, 0], [4, 32, 32, 32]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)

@use_np
def test_reshape_dense_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            channel0 = onp.random.randint(1, 17)
            channel1 = onp.random.randint(1, 33)
            self.dense0 = nn.Dense(channel0)
            self.dense1 = nn.Dense(channel1)

        def forward(self, x):
            x_reshape = x.reshape((4, 16, 128, 32))
            y = self.dense0(x_reshape)
            y_reshape = y.reshape((1, -1))
            out = self.dense1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(4, 16, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)


@use_np
def test_slice_dense_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            channel0 = onp.random.randint(1, 17)
            channel1 = onp.random.randint(1, 17)
            self.dense0 = nn.Dense(channel0)
            self.dense1 = nn.Dense(channel1)
            self.slice = slice

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.dense0(x_slice)
            y_reshape = y.reshape((1, -1))
            out = self.dense1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(16, 32, 64, 64))
    slice = [[0, 16, 0, 0], [4, 32, 32, 32]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)


@use_np
def test_reshape_dense_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            channel0 = 64
            channel1 = onp.random.randint(1, 17)
            self.dense0 = nn.Dense(channel0)
            self.dense1 = nn.Dense(channel1)

        def forward(self, x):
            x_reshape = x.reshape((4, 16, 128, 32))
            y = self.dense0(x_reshape)
            y_slice = mx.npx.slice(y, begin=(1, 32), end=(3, 64))
            out = self.dense1(y_slice)
            return out

    x = mx.np.random.uniform(size=(4, 16, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(96, (1, 1))
            self.bn0 = nn.BatchNorm()
            self.reshape = shape

        def forward(self, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape)
            out = self.bn0(x_reshape)
            return out

    x = mx.np.random.uniform(size=(4, 32, 64, 64))
    shape = (4, 64, 64, -1)
    net = Net(shape)
    check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.serial
def test_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(128, (1, 1))
            self.bn0 = nn.BatchNorm()
            self.slice = slice

        def forward(self, x):
            x_in = self.conv0(x)
            x_slice = mx.npx.slice(x_in, begin=tuple(self.slice[0]),
                              end=tuple(self.slice[1]))
            out = self.bn0(x_slice)
            return out

    x = mx.np.random.uniform(size=(16, 128, 256, 256))
    slice = [[0, 0, 0, 0], [4, 32, 32, 32]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
@pytest.mark.serial
def test_slice_batchnorm_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(128, (1, 1))
            self.bn0 = nn.BatchNorm()
            self.bn1 = nn.BatchNorm()
            self.slice = slice

        def forward(self, x):
            x_in = self.conv0(x)
            x_slice = mx.npx.slice(x_in, begin=tuple(self.slice[0][0]), end=tuple(self.slice[0][1]))
            y = self.bn0(x_slice)
            y_slice = mx.npx.slice(y, begin=tuple(self.slice[1][0]), end=tuple(self.slice[1][1]))
            out = self.bn1(y_slice)
            return out

    x = mx.np.random.uniform(size=(16, 128, 256, 256))
    slice = [[[0, 0, 0, 0], [4, 32, 32, 32]], [[0, 0, 0, 0], [2, 64, 16, 16]]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_batchnorm_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(128, (1, 1))
            self.bn0 = nn.BatchNorm()
            self.bn1 = nn.BatchNorm()
            self.reshape = shape

        def forward(self, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape[0])
            y = self.bn0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.bn1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(4, 32, 64, 64))
    shape = [(4, 64, 64, -1), (4, 128, -1, 32)]
    net = Net(shape)
    check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.serial
def test_slice_batchnorm_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(128, (1, 1))
            self.bn0 = nn.BatchNorm()
            self.bn1 = nn.BatchNorm()
            self.reshape = shape
            self.slice = slice

        def forward(self, x):
            x_in = self.conv0(x)
            x_slice = mx.npx.slice(x_in, begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.bn0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.bn1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(16, 128, 256, 256))
    slice = [[0, 0, 0, 0], [4, 32, 32, 32]]
    shape = (1, 128, 64, -1)
    net = Net(shape, slice)
    check_layer_forward_withinput(net, x)


@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_batchnorm_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(128, (1, 1))
            self.bn0 = nn.BatchNorm()
            self.bn1 = nn.BatchNorm()
            self.reshape = shape
            self.slice = slice

        def forward(self, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape)
            y = self.bn0(x_reshape)
            y_slice = y.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            out = self.bn1(y_slice)
            return out

    x = mx.np.random.uniform(size=(4, 32, 64, 64))
    slice = [[0, 0, 0, 0], [2, 64, 32, 32]]
    shape = (4, 64, 64, -1)
    net = Net(shape, slice)
    check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_pooling2d():
    max_pooling = nn.MaxPool2D(strides=(2, 3), padding=(1, 1))
    avg_pooling = nn.AvgPool2D(strides=(2, 2), padding=(1, 1))
    global_maxpooling = nn.GlobalMaxPool2D()
    global_avgpooling = nn.GlobalAvgPool2D()
    pooling_layers = [max_pooling, avg_pooling, global_maxpooling, global_avgpooling]
    class Net(gluon.HybridBlock):
        def __init__(self,
                     shape,
                     pooling_layer,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.pool0 = pooling_layer

        def forward(self, x):
            x_reshape = x.reshape(self.reshape)
            out = self.pool0(x_reshape)
            return out

    x = mx.np.random.uniform(size=(4, 32, 32, 32))
    shape = (4, 64, 64, -1)
    for i in range(len(pooling_layers)):
        net = Net(shape, pooling_layers[i])
        check_layer_forward_withinput(net, x)

@pytest.mark.serial
def test_slice_pooling2d():
    # transpose shape to bring feature dimension 'c' from 2nd position to last
    def transpose(shape):
        return (shape[0],) + shape[2:] + (shape[1],)

    for layout in ['NCHW', 'NHWC']:
        max_pooling = nn.MaxPool2D(strides=(2, 3), padding=(1, 1), layout=layout)
        avg_pooling = nn.AvgPool2D(strides=(2, 2), padding=(1, 1), layout=layout)
        global_maxpooling = nn.GlobalMaxPool2D(layout=layout)
        global_avgpooling = nn.GlobalAvgPool2D(layout=layout)
        pooling_layers = [max_pooling, avg_pooling, global_maxpooling, global_avgpooling]
        class Net(gluon.HybridBlock):
            def __init__(self,
                         slice,
                         pooling_layer,
                         **kwargs):
                super(Net, self).__init__(**kwargs)
                self.slice = slice
                self.pool0 = pooling_layer

            def forward(self, x):
                x_slice = mx.npx.slice(x, begin=self.slice[0], end=self.slice[1])
                out = self.pool0(x_slice)
                return out

        xshape = (16, 128, 256, 256)
        slice_shape = (4, 16, 32, 64)
        if layout == 'NHWC':
            xshape = transpose(xshape)
            slice_shape = transpose(slice_shape)
        x = mx.np.random.uniform(size=xshape)
        slice = [(0, 0, 0, 0), slice_shape]
        for i in range(len(pooling_layers)):
            net = Net(slice, pooling_layers[i])
            check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_pooling2d_reshape_pooling2d():
    max_pooling = nn.MaxPool2D(strides=(2, 2), padding=(1, 1))
    avg_pooling = nn.AvgPool2D(strides=(2, 2), padding=(1, 1))
    global_maxpooling = nn.GlobalMaxPool2D()
    global_avgpooling = nn.GlobalAvgPool2D()
    pooling_layers = [max_pooling, avg_pooling, global_maxpooling, global_avgpooling]
    class Net(gluon.HybridBlock):
        def __init__(self,
                     shape,
                     pooling_layer1,
                     pooling_layer2,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.pool0 = pooling_layer1
            self.pool1 = pooling_layer2

        def forward(self, x):
            x_reshape = x.reshape(self.reshape[0])
            y = self.pool0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.pool1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(16, 128, 256, 256))
    shape = [(128, 256, 64, -1), (128, 256, 11, -1)]
    for i in range(len(pooling_layers)):
        for j in range(len(pooling_layers)):
            if isinstance(pooling_layers[i], (nn.GlobalMaxPool2D, nn.GlobalAvgPool2D)):
                shape[1] = (256, 128, 1, 1)
            net = Net(shape, pooling_layers[i], pooling_layers[j])
            check_layer_forward_withinput(net, x)

@pytest.mark.serial
def test_slice_pooling2d_slice_pooling2d():
    max_pooling = nn.MaxPool2D(strides=(2, 3), padding=(1, 1))
    avg_pooling = nn.AvgPool2D(strides=(2, 2), padding=(1, 1))
    global_maxpooling = nn.GlobalMaxPool2D()
    global_avgpooling = nn.GlobalAvgPool2D()
    pooling_layers = [max_pooling, avg_pooling, global_maxpooling, global_avgpooling]
    class Net(gluon.HybridBlock):
        def __init__(self,
                     slice,
                     pooling_layer1,
                     pooling_layer2,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.slice = slice
            self.pool0 = pooling_layer1
            self.pool1 = pooling_layer2

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=self.slice[0][0], end=self.slice[0][1])
            y = self.pool0(x_slice)
            y_slice = mx.npx.slice(y, begin=self.slice[1][0], end=self.slice[1][1])
            out = self.pool1(y_slice)
            return out

    x = mx.np.random.uniform(size=(16, 128, 256, 256))
    slice = [[(8, 0, 100, 50), (16, -1, -1, -1)], [(0, 64, 0, 50), (2, -1, -1, -1)]]
    for i in range(len(pooling_layers)):
        for j in range(len(pooling_layers)):
            if isinstance(pooling_layers[i], (nn.GlobalMaxPool2D, nn.GlobalAvgPool2D)):
                slice[1] = [(0, 64, 0, 0), (2, -1, 1, 1)]
            net = Net(slice, pooling_layers[i], pooling_layers[j])
            check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_slice_pooling2d_reshape_pooling2d():
    max_pooling = nn.MaxPool2D(strides=(2, 3), padding=(1, 1))
    avg_pooling = nn.AvgPool2D(strides=(2, 2), padding=(1, 1))
    global_maxpooling = nn.GlobalMaxPool2D()
    global_avgpooling = nn.GlobalAvgPool2D()
    pooling_layers = [max_pooling, avg_pooling, global_maxpooling, global_avgpooling]
    class Net(gluon.HybridBlock):
        def __init__(self,
                     shape,
                     slice,
                     pooling_layer1,
                     pooling_layer2,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.slice = slice
            self.pool0 = pooling_layer1
            self.pool1 = pooling_layer2

        def forward(self, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            y = self.pool0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.pool1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(16, 128, 256, 256))
    slice = [(8, 0, 100, 50), (16, 128, 256, 256)]
    shape = (32, -1, 0, 0)
    for i in range(len(pooling_layers)):
        for j in range(len(pooling_layers)):
            net = Net(shape, slice, pooling_layers[i], pooling_layers[j])
            check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
@pytest.mark.serial
def test_reshape_pooling2d_slice_pooling2d():
    max_pooling = nn.MaxPool2D(strides=(2, 3), padding=(1, 1))
    avg_pooling = nn.AvgPool2D(strides=(2, 2), padding=(1, 1))
    global_maxpooling = nn.GlobalMaxPool2D()
    global_avgpooling = nn.GlobalAvgPool2D()
    pooling_layers = [max_pooling, avg_pooling, global_maxpooling, global_avgpooling]
    class Net(gluon.HybridBlock):
        def __init__(self,
                     shape,
                     slice,
                     pooling_layer1,
                     pooling_layer2,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.slice = slice
            self.pool0 = pooling_layer1
            self.pool1 = pooling_layer2

        def forward(self, x):
            x_reshape = x.reshape(self.reshape)
            y = self.pool0(x_reshape)
            y_slice = y.slice(begin=self.slice[0], end=self.slice[1])
            out = self.pool1(y_slice)
            return out

    x = mx.np.random.uniform(size=(16, 128, 256, 256))
    shape = (0, 512, 64, -1)
    slice = [(8, 256, 10, 20), (-1, -1, -1, 70)]
    for i in range(len(pooling_layers)):
        for j in range(len(pooling_layers)):
            if isinstance(pooling_layers[i], (nn.GlobalMaxPool2D, nn.GlobalAvgPool2D)):
                slice = [(8, 256, 0, 0), (-1, -1, 1, 1)]
            net = Net(shape, slice, pooling_layers[i], pooling_layers[j])
            check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
@pytest.mark.serial
def test_reshape_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.conv0 = nn.Conv2DTranspose(64, (3, 3))

        def forward(self, x):
            x_reshape = x.reshape(self.reshape)
            out = self.conv0(x_reshape)
            return out
    x = mx.np.random.uniform(size=(4, 16, 32, 32))
    shape = (4, 16, 64, -1)
    net = Net(shape)
    check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
@pytest.mark.serial
def test_slice_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.slice = slice
            self.conv0 = nn.Conv2DTranspose(64, (3, 3))

        def forward(self, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            out = self.conv0(x_slice)
            return out
    x = mx.np.random.uniform(size=(8, 32, 64, 64))
    slice = [(0, 16, 0, 0), (4, 32, 32, 32)]
    net = Net(slice)
    check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
@pytest.mark.serial
def test_reshape_deconv_reshape_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.conv0 = nn.Conv2DTranspose(32, (3, 3))
            self.conv1 = nn.Conv2DTranspose(64, (3, 3), strides=(2, 2))

        def forward(self, x):
            x_reshape = x.reshape(self.reshape[0])
            y = self.conv0(x_reshape)
            "shape of y is (4, 32, 66, 18)"
            y_reshape = y.reshape(self.reshape[1])
            out = self.conv1(y_reshape)
            return out
    x = mx.np.random.uniform(size=(4, 16, 32, 32))
    shape = [(4, 16, 64, -1), (4, 32, 33, -1)]
    net = Net(shape)
    check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
@pytest.mark.serial
def test_slice_deconv_slice_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.slice = slice
            self.conv0 = nn.Conv2DTranspose(32, (3, 3))
            self.conv1 = nn.Conv2DTranspose(64, (3, 3), strides=(2, 2))

        def forward(self, x):
            x_slice = x.slice(begin=self.slice[0][0], end=self.slice[0][1])
            y = self.conv0(x_slice)
            "shape of y is (4, 32, 66, 18)"
            y_slice = y.slice(begin=self.slice[1][0], end=self.slice[1][1])
            out = self.conv1(y_slice)
            return out
    x = mx.np.random.uniform(size=(8, 32, 64, 64))
    slice = [[(0, 0, 0, 0), (4, 16, 32, 32)], [(0, 0, 0, 0), (2, 16, 16, 16)]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
@pytest.mark.serial
def test_reshape_deconv_slice_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.slice = slice
            self.conv0 = nn.Conv2DTranspose(32, (3, 3))
            self.conv1 = nn.Conv2DTranspose(64, (3, 3), strides=(2, 2))

        def forward(self, x):
            x_reshape = x.reshape(self.reshape)
            y = self.conv0(x_reshape)
            "shape of y is (4, 32, 66, 18)"
            y_slice = y.slice(begin=self.slice[0], end=self.slice[1])
            out = self.conv1(y_slice)
            return out
    x = mx.np.random.uniform(size=(4, 16, 32, 32))
    shape = (4, 16, 64, -1)
    slice = [(0, 0, 0, 0), (2, 16, 16, 16)]
    net = Net(shape, slice)
    check_layer_forward_withinput(net, x)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
@pytest.mark.serial
def test_slice_deconv_reshape_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.slice = slice
            self.conv0 = nn.Conv2DTranspose(32, (3, 3))
            self.conv1 = nn.Conv2DTranspose(96, (3, 3), strides=(2, 2))

        def forward(self, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            y = self.conv0(x_slice)
            "shape of y is (4, 32, 34, 34)"
            y_reshape = y.reshape(self.reshape)
            out = self.conv1(y_reshape)
            return out
    x = mx.np.random.uniform(size=(8, 32, 64, 64))
    shape = (4, 64, 34, -1)
    slice = [(4, 0, 0, 0), (8, 16, 32, 32)]
    net = Net(shape, slice)
    check_layer_forward_withinput(net, x)

@use_np
@pytest.mark.serial
def test_reshape_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.act = nn.Activation(act)

        def forward(self, x):
            x_reshape = x.reshape(self.reshape)
            out = self.act(x_reshape)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu", "softsign"]
    for act in acts:
        x = mx.np.random.uniform(-1, 1, size=(4, 16, 32, 32))
        shape = (4, 32, 32, -1)
        net = Net(act, shape)
        check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.serial
def test_slice_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.slice = slice
            self.act = nn.Activation(act)

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=self.slice[0], end=self.slice[1])
            out = self.act(x_slice)
            return out

    acts = ["relu", "sigmoid", "tanh", "softrelu", "softsign"]
    for act in acts:
        x = mx.np.random.uniform(-1, 1, size=(8, 32, 64, 64))
        slice = [(0, 16, 32, 32), (4, 32, 64, 64)]
        net = Net(act, slice)
        check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.serial
def test_reshape_activation_reshape_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act0, act1, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.act0 = nn.Activation(act0)
            self.act1 = nn.Activation(act1)

        def forward(self, x):
            x_reshape = x.reshape(self.reshape[0])
            y = self.act0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.act1(y_reshape)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu", "softsign"]
    for idx0, act0 in enumerate(acts):
        for idx1, act1 in enumerate(acts):
            if idx1 == idx0:
                continue
            x = mx.np.random.uniform(-1, 1, size=(4, 16, 32, 32))
            shape = [(4, 32, 32, -1), (4, 32, 16, -1)]
            net = Net(act0, act1, shape)
            check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.serial
def test_slice_activation_slice_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act0, act1, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.slice = slice
            self.act0 = nn.Activation(act0)
            self.act1 = nn.Activation(act1)

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=self.slice[0][0], end=self.slice[0][1])
            y = self.act0(x_slice)
            y_slice = mx.npx.slice(y, begin=self.slice[1][0], end=self.slice[1][1])
            out = self.act1(y_slice)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu", "softsign"]
    for idx0, act0 in enumerate(acts):
        for idx1, act1 in enumerate(acts):
            if idx1 == idx0:
                continue
            x = mx.np.random.uniform(-1, 1, size=(8, 32, 64, 64))
            slice = [[(0, 16, 32, 32), (4, 32, 64, 64)], [(2, 0, 16, 16), (4, 16, 32, 32)]]
            net = Net(act0, act1, slice)
            check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.serial
def test_reshape_activation_slice_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act0, act1, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.slice = slice
            self.act0 = nn.Activation(act0)
            self.act1 = nn.Activation(act1)

        def forward(self, x):
            x_reshape = x.reshape(self.reshape)
            y = self.act0(x_reshape)
            y_slice = mx.npx.slice(y, begin=self.slice[0], end=self.slice[1])
            out = self.act1(y_slice)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu", "softsign"]
    for idx0, act0 in enumerate(acts):
        for idx1, act1 in enumerate(acts):
            if idx1 == idx0:
                continue
            x = mx.np.random.uniform(-1, 1, size=(4, 16, 32, 32))
            shape = (4, 32, 32, -1)
            slice = [(0, 0, 0, 0), (2, 16, 16, 16)]
            net = Net(act0, act1, shape, slice)
            check_layer_forward_withinput(net, x)


@use_np
@pytest.mark.serial
def test_slice_activation_reshape_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act0, act1, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.reshape = shape
            self.slice = slice
            self.act0 = nn.Activation(act0)
            self.act1 = nn.Activation(act1)

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=self.slice[0], end=self.slice[1])
            y = self.act0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.act1(y_reshape)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu", "softsign"]
    for idx0, act0 in enumerate(acts):
        for idx1, act1 in enumerate(acts):
            if idx1 == idx0:
                continue
            x = mx.np.random.uniform(-1, 1, size=(8, 32, 64, 64))
            slice = [(0, 16, 32, 32), (4, 32, 64, 64)]
            shape = (4, 32, 32, -1)
            net = Net(act0, act1, shape, slice)
            check_layer_forward_withinput(net, x)

@use_np
@pytest.mark.serial
def test_np_shape_parameters():
    class Foo(gluon.Block):
        def __init__(self, **kwargs):
            super(Foo, self).__init__(**kwargs)
            self.dense = gluon.nn.Dense(16)
        def forward(self, x):
            return self.dense(x)

    with mx.np_shape(True):
        z = mx.np.zeros((2,2016))
        print(z.shape)
        foo = Foo()
        foo.initialize()
        print(foo(z).shape)

def test_gluon_param_load():
    net = mx.gluon.nn.Dense(10, in_units=10)
    net.initialize()
    net.save_parameters('test_gluon_param_load.params')
    net.cast('float16')
    net.load_parameters('test_gluon_param_load.params', cast_dtype=True)
    mx.npx.waitall()

def test_gluon_param_load_dtype_source():
    net = mx.gluon.nn.Dense(10, in_units=10)
    net.initialize()
    net.cast('float16')
    net.save_parameters('test_gluon_param_load_dtype_source.params')
    net.cast('float32')
    net.load_parameters('test_gluon_param_load_dtype_source.params', cast_dtype=True, dtype_source="saved")
    assert net.weight.dtype == onp.float16
    mx.npx.waitall()

@use_np
def test_squeeze_consistency():
    class Foo(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Foo, self).__init__(**kwargs)

        def forward(self, x):
            return x.squeeze()

    block = Foo()
    block.hybridize()
    shape = (onp.random.randint(1, 10), onp.random.randint(1, 10), 1)
    block(mx.np.ones(shape))

def test_shared_parameters_with_non_default_initializer():
    class MyBlock(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(MyBlock, self).__init__(**kwargs)

            self.param = gluon.Parameter(shape=(1, ), init=mx.init.Constant(-10.0))

    bl = MyBlock()
    bl2 = MyBlock().share_parameters(bl.collect_params())
    assert bl.param is bl2.param
    bl3 = MyBlock()
    assert bl.param is not bl3.param
    assert bl.param.init == bl3.param.init

@use_np
def test_reqs_switching_training_inference():
    class Foo(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Foo, self).__init__(**kwargs)

        def forward(self, x):
            y = 2 * x
            return mx.np.sqrt(x) + mx.np.sqrt(y)

    f = Foo()
    f.hybridize(static_alloc=True)
    x = mx.np.ones(shape=(10,10))
    x.attach_grad()
    x2 = mx.np.ones(shape=x.shape) * 2
    x2.attach_grad()

    # Call first in training mode
    with mx.autograd.record():
        y = f(x)
    y.backward()

    grad1 = x.grad.asnumpy()

    # Compute the gradient with some other input
    with mx.autograd.record():
        y = f(x2)
    y.backward()

    # Call inference mode
    y = f(x)

    # Call training mode again
    with mx.autograd.record():
        y = f(x)
    y.backward()

    grad2 = x.grad.asnumpy()

    mx.test_utils.assert_almost_equal(grad1, grad2)


@pytest.mark.usefixtures("check_leak_ndarray")
def test_no_memory_leak_in_gluon():
    class MyNet(mx.gluon.Block):
        def __init__(self):
            super().__init__()
            self.net = mx.gluon.nn.Dense(10, in_units=10)
    net = MyNet()
    net.initialize()

def test_DeformableConvolution():
    """test of the deformable convolution layer with possible combinations of arguments,
    currently this layer only supports gpu
    """
    try:
        ctx = mx.gpu()
        _ = mx.np.array([0], ctx=ctx)
    except mx.base.MXNetError:
        pytest.skip("deformable_convolution only supports GPU")
    net = nn.HybridSequential()
    net.add(
        nn.DeformableConvolution(10, kernel_size=(3, 3), strides=1, padding=0),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                                  offset_use_bias=False, use_bias=False),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                                  offset_use_bias=False),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                                  use_bias=False),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, offset_use_bias=False, use_bias=False),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, offset_use_bias=False),
        nn.DeformableConvolution(12, kernel_size=(3, 2), strides=1, padding=0, use_bias=False),
        nn.DeformableConvolution(12, kernel_size=(3, 2), strides=1, padding=0, use_bias=False, num_deformable_group=4),
    )

    net.initialize(force_reinit=True, ctx=ctx)
    net.hybridize()

    x = mx.np.random.uniform(size=(8, 5, 30, 31), ctx=ctx)
    with mx.autograd.record():
        y = net(x)
        y.backward()

def test_ModulatedDeformableConvolution():
    """test of the deformable convolution layer with possible combinations of arguments,
    currently this layer only supports gpu
    """
    net = nn.HybridSequential()
    net.add(
        nn.DeformableConvolution(10, kernel_size=(3, 3), strides=1, padding=0),
        nn.DeformableConvolution(10, kernel_size=(1, 1), strides=1, padding=0),
        nn.DeformableConvolution(10, kernel_size=(5, 5), strides=1, padding=0),
        nn.DeformableConvolution(10, kernel_size=(3, 5), strides=1, padding=0),
        nn.DeformableConvolution(10, kernel_size=(5, 1), strides=1, padding=0, num_deformable_group=2),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                                 offset_use_bias=False, use_bias=False),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                                 offset_use_bias=False),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                                 use_bias=False),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, offset_use_bias=False, use_bias=False),
        nn.DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, offset_use_bias=False),
        nn.DeformableConvolution(12, kernel_size=(3, 2), strides=1, padding=0, use_bias=False),
        nn.DeformableConvolution(12, kernel_size=(3, 2), strides=1, padding=0, use_bias=False, num_deformable_group=4),
    )

    ctx = default_context()
    net.initialize(force_reinit=True, ctx=ctx)
    net.hybridize()

    x = mx.np.random.uniform(size=(8, 5, 30, 31), ctx=ctx)
    with mx.autograd.record():
        y = net(x)


@use_np
@pytest.mark.parametrize('dc', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.garbage_expected
def test_concatenate(dc, hybridize):
    if dc:
        class MyBlock(mx.gluon.HybridBlock):
            def __init__(self, units, activation=None, in_units=0):
                super().__init__()
                self.dense = mx.gluon.nn.Dense(units, activation=activation, in_units=in_units)

            def forward(self, x):
                return self.dense(x)
    else:
        MyBlock = nn.Dense

    model = nn.HybridConcatenate(axis=1)
    model.add(MyBlock(128, activation='tanh', in_units=10))
    model.add(MyBlock(64, activation='tanh', in_units=10))
    model.add(MyBlock(32, in_units=10))
    model2 = nn.Concatenate(axis=1)
    model2.add(MyBlock(128, activation='tanh', in_units=10))
    model2.add(MyBlock(64, activation='tanh', in_units=10))
    model2.add(MyBlock(32, in_units=10))

    # ndarray
    model.initialize(mx.init.Xavier(magnitude=2.24))
    model2.initialize(mx.init.Xavier(magnitude=2.24))
    if hybridize:
        model.hybridize()
        model2.hybridize()
    x = model(mx.np.zeros((32, 10)))
    x2 = model2(mx.np.zeros((32, 10)))
    assert x.shape == (32, 224)
    assert x2.shape == (32, 224)
    x.wait_to_read()
    x2.wait_to_read()

def test_identity():
    model = nn.Identity()
    x = mx.np.random.uniform(size=(128, 33, 64))
    assert_almost_equal(model(x), x)

def test_pixelshuffle1d():
    nchan = 2
    up_x = 2
    nx = 3
    shape_before = (1, nchan * up_x, nx)
    shape_after = (1, nchan, nx * up_x)
    layer = nn.PixelShuffle1D(up_x)
    x = mx.np.arange(onp.prod(shape_before)).reshape(shape_before)
    y = layer(x)
    assert y.shape == shape_after
    assert_allclose(
        y,
        [[[0, 3, 1, 4, 2, 5],
          [6, 9, 7, 10, 8, 11]]]
    )

def test_pixelshuffle2d():
    nchan = 2
    up_x = 2
    up_y = 3
    nx = 2
    ny = 3
    shape_before = (1, nchan * up_x * up_y, nx, ny)
    shape_after = (1, nchan, nx * up_x, ny * up_y)
    layer = nn.PixelShuffle2D((up_x, up_y))
    x = mx.np.arange(onp.prod(shape_before)).reshape(shape_before)
    y = layer(x)
    assert y.shape == shape_after
    # - Channels are reshaped to form 2x3 blocks
    # - Within each block, the increment is `nx * ny` when increasing the column
    #   index by 1
    # - Increasing the block index adds an offset of 1
    # - Increasing the channel index adds an offset of `nx * up_x * ny * up_y`
    assert_allclose(
        y,
        [[[[ 0,  6, 12,  1,  7, 13,  2,  8, 14],
           [18, 24, 30, 19, 25, 31, 20, 26, 32],
           [ 3,  9, 15,  4, 10, 16,  5, 11, 17],
           [21, 27, 33, 22, 28, 34, 23, 29, 35]],

          [[36, 42, 48, 37, 43, 49, 38, 44, 50],
           [54, 60, 66, 55, 61, 67, 56, 62, 68],
           [39, 45, 51, 40, 46, 52, 41, 47, 53],
           [57, 63, 69, 58, 64, 70, 59, 65, 71]]]]
    )

def test_pixelshuffle3d():
    nchan = 1
    up_x = 2
    up_y = 1
    up_z = 2
    nx = 2
    ny = 3
    nz = 4
    shape_before = (1, nchan * up_x * up_y * up_z, nx, ny, nz)
    shape_after = (1, nchan, nx * up_x, ny * up_y, nz * up_z)
    layer = nn.PixelShuffle3D((up_x, up_y, up_z))
    x = mx.np.arange(onp.prod(shape_before)).reshape(shape_before)
    y = layer(x)
    assert y.shape == shape_after
    # - Channels are reshaped to form 2x1x2 blocks
    # - Within each block, the increment is `nx * ny * nz` when increasing the
    #   column index by 1, e.g. the block [[[ 0, 24]], [[48, 72]]]
    # - Increasing the block index adds an offset of 1
    assert_allclose(
        y,
        [[[[[ 0, 24,  1, 25,  2, 26,  3, 27],
            [ 4, 28,  5, 29,  6, 30,  7, 31],
            [ 8, 32,  9, 33, 10, 34, 11, 35]],

           [[48, 72, 49, 73, 50, 74, 51, 75],
            [52, 76, 53, 77, 54, 78, 55, 79],
            [56, 80, 57, 81, 58, 82, 59, 83]],

           [[12, 36, 13, 37, 14, 38, 15, 39],
            [16, 40, 17, 41, 18, 42, 19, 43],
            [20, 44, 21, 45, 22, 46, 23, 47]],

           [[60, 84, 61, 85, 62, 86, 63, 87],
            [64, 88, 65, 89, 66, 90, 67, 91],
            [68, 92, 69, 93, 70, 94, 71, 95]]]]]
    )
