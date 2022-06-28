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
from __future__ import absolute_import
from __future__ import division

import os
from uuid import uuid4
import numpy as _np
import mxnet as mx
from mxnet import gluon, autograd, np, npx
from mxnet.test_utils import use_np, assert_almost_equal, check_gluon_hybridize_consistency, assert_allclose
from mxnet.gluon import nn
from mxnet.base import MXNetError
import random
import pytest


def test_create_np_param():
    M, K, N = 10, 9, 20

    def check_block_params(x, TestBlock, hybridize, expected_type, initializer):
        net = TestBlock()
        net.initialize(initializer())
        if hybridize:
            net.hybridize()
        net(x)
        params = net.collect_params()
        for _, v in params.items():
            assert type(v.data()) is expected_type

    @use_np
    class TestBlock1(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock1, self).__init__()
            self.w = gluon.Parameter('w', shape=(K, N), allow_deferred_init=True)

        def forward(self, x):
            device = x.device
            return np.dot(x, self.w.data(device))

    x = mx.np.random.uniform(size=(M, K))
    for initializer in [mx.initializer.Uniform, mx.initializer.Normal]:
        check_block_params(x, TestBlock1, False, mx.np.ndarray, initializer)
        check_block_params(x, TestBlock1, True, mx.np.ndarray, initializer)


@use_np
def test_optimizer_with_np_ndarrays():
    class LinearRegression(gluon.HybridBlock):
        def __init__(self, num_input_dim=0, num_hidden_dim=100, num_output_dim=10):
            super(LinearRegression, self).__init__()
            self.w1 = gluon.Parameter('w1', shape=(num_input_dim, num_hidden_dim),
                                      allow_deferred_init=True)
            self.w2 = gluon.Parameter('w2', shape=(num_hidden_dim, num_output_dim),
                                      allow_deferred_init=True)

        def forward(self, x):
            device = x.device
            h = x.dot(self.w1.data(device))  # equivalent to np.dot(x, w1)
            h_relu = npx.relu(h)  # equivalent to npx.relu(h) but generating np.ndarray
            y_pred = h_relu.dot(self.w2.data(device))  # equivalent to np.dot(h_relu, w2)
            return y_pred
        
        def infer_shape(self, x, *args):
            pre_shape = self.w1.shape
            self.w1.shape = (x.shape[x.ndim-1], pre_shape[1])

    class TotalLoss(gluon.HybridBlock):
        def forward(self, pred, label):
            return ((pred - label) ** 2).sum()  # equivalent to np.sum(np.square(pred - label))

    regressor = LinearRegression()
    regressor.initialize(mx.init.Uniform())
    regressor.hybridize()

    # Create random input and output data
    x = np.random.uniform(size=(64, 1000))  # x is of type mxnet.numpy.ndarray
    regressor(x)
    y = np.random.uniform(size=(64, 10))  # y is of type mxnet.numpy.ndarray

    total_loss = TotalLoss()
    total_loss.hybridize()

    trainer = gluon.Trainer(regressor.collect_params(),
                            'sgd',
                            {'learning_rate': 1e-3, 'momentum': 0.9})

    for _ in range(2):
        with autograd.record():
            output = regressor(x)  # output is a type of np.ndarray because np.dot is the last op in the network
            loss = total_loss(output, y)  # loss is a scalar np.ndarray
        loss.backward()
        trainer.step(1)


@use_np
def test_optimizer_backward_compat():
    optimizer = mx.optimizer.SGD()
    delattr(optimizer, "allow_np_array")
    updater = mx.optimizer.Updater(optimizer)
    updater(0, np.ones((0, 0)), np.zeros((0, 0)))


@use_np
def test_np_loss_ndarray():
    # Ported from test_loss.test_loss_ndarray
    output = np.array([1, 2, 3, 4])
    label = np.array([1, 3, 5, 7])
    weighting = np.array([0.5, 1, 0.5, 1])

    loss = gluon.loss.L1Loss()
    assert float(np.sum(loss(output, label))) == 6.
    loss = gluon.loss.L1Loss(weight=0.5)
    assert float(np.sum(loss(output, label))) == 3.
    loss = gluon.loss.L1Loss()
    assert float(np.sum(loss(output, label, weighting))) == 5.

    loss = gluon.loss.L2Loss()
    assert float(np.sum(loss(output, label))) == 7.
    loss = gluon.loss.L2Loss(weight=0.25)
    assert float(np.sum(loss(output, label))) == 1.75
    loss = gluon.loss.L2Loss()
    assert float(np.sum(loss(output, label, weighting))) == 6

    output = np.array([[0, 2], [1, 4]])
    label = np.array([0, 1])
    weighting = np.array([[0.5], [1.0]])

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    L = loss(output, label).asnumpy()
    assert_almost_equal(L, _np.array([2.12692809,  0.04858733]), use_broadcast=False, rtol=1e-3)

    L = loss(output, label, weighting).asnumpy()
    assert_almost_equal(L, _np.array([1.06346405,  0.04858733]), use_broadcast=False, rtol=1e-3)


@use_np
def test_np_get_constant():
    const_arr = _np.random.uniform(0, 100, size=(10, 10)).astype(_np.float32)

    class Foo(gluon.HybridBlock):
        def __init__(self):
            super(Foo, self).__init__()
            self.weight = gluon.Constant(const_arr)

        def forward(self, x):
            device = x.device
            return x + self.weight.data(device).astype(np.float32)

    x = np.random.uniform(size=const_arr.shape, dtype=const_arr.dtype)
    for hybridize in [False, True]:
        foo = Foo()
        if hybridize:
            foo.hybridize()
        foo.initialize()
        out = foo(x)
        assert_almost_equal(out.asnumpy(), (x.asnumpy() + const_arr), atol=1e-5, rtol=1e-4, use_broadcast=False)


@use_np
def test_parameters_zero_grad():
    for hybridize in [False, True]:
        net = gluon.nn.HybridSequential()
        for _ in range(5):
            net.add(gluon.nn.Dense(10))
        if hybridize:
            net.hybridize()
        net.initialize()
        out = net(mx.np.ones((32, 8)))
        for v in net.collect_params().values():
            v.grad()[()] = 1
        net.zero_grad()
        for v in net.collect_params().values():
            assert_almost_equal(v.grad().asnumpy(), mx.np.zeros_like(v.grad()).asnumpy())


def check_gluon_save_load(net_builder, data_l):
    """Verify the consistency between the loaded network and the original network.

    Known limitations: Currently it only supports loading

    Parameters
    ----------
    net_builder : function
        The builder of the HybridBlock.
    data_l : list of numpy.ndarray
        List of the input data that we will use to verify the correctness of the loaded network.

    """
    net = net_builder()
    net.hybridize()
    net.initialize()
    out = net(*data_l)
    out_np = out.asnumpy()
    prefix = str(uuid4())
    net.export(prefix)
    input_names = 'data' if len(data_l) == 1 else ['data{}'.format(i) for i in range(len(data_l))]
    net_imported = gluon.SymbolBlock.imports('{}-symbol.json'.format(prefix),
                                             input_names, param_file='{}-0000.params'.format(prefix))
    # Clean up the directory
    os.remove('{}-symbol.json'.format(prefix))
    os.remove('{}-0000.params'.format(prefix))
    loaded_out = net_imported(*data_l).asnumpy()
    assert_almost_equal(out_np, loaded_out)


def hashable_index(tuple_idx):
    """Return an hashable representation of a tuple of slice object

    We add this because the slice object in python is not hashable.

    Parameters
    ----------
    tuple_idx : tuple
        A tuple of slice/int objects

    Returns
    -------
    ret : tuple
        A hashable representation of the slice data
    """
    l = []
    for ele in tuple_idx:
        if isinstance(ele, slice):
            l.append(ele.__reduce__())
        else:
            l.append(ele)
    return tuple(l)


@use_np
def test_symbolic_basic_slicing():
    def random_slice_index(shape):
        index = []
        step_switch = random.randint(-1, 1)
        for i in range(len(shape)):
            if shape[i] == 0:
                index.append(slice(None))
                continue
            r = random.randint(0, 5)
            if r < 2:
                index.append(random.randint(1 - shape[i], shape[i] - 1))
                continue
            elif r < 3:
                index.append(slice(None))
                continue
            s = random.randint(0, shape[i] - 1)
            e = random.randint(s + 1, shape[i])
            if step_switch == 1:
                index.append(slice(s, e, 1))
            elif step_switch == -1:
                e -= 1
                s -= 1
                index.append(slice(e, s, -1))
            else:
                index.append(slice(s, e, 2))
        return tuple(index)

    shapes = [
        (4, 6, 8, 5),
        (1, 1, 1, 6),
        (5, 6, 4),
        (5, 6),
        (10,),
        (100, 0, 10, 0, 0),
        (100, 0, 0),
        (0, 0, 0),
        (),
    ]
    for shape in shapes:
        cache = set()
        x = mx.np.random.normal(0, 1, shape)
        y = mx.np.random.normal(0, 1, shape)
        for _ in range(200):
            index1 = random_slice_index(shape)
            index2 = random_slice_index(x.asnumpy()[index1].shape)
            if (hashable_index(index1), hashable_index(index2)) in cache:
                continue
            cache.add((hashable_index(index1), hashable_index(index2)))
            # Test basic slicing on a single symbol
            class TestSlicingSingleSymbol1(gluon.HybridBlock):
                def forward(self, x, y):
                    return x[()][index1] + y[()][index1]

            # Test basic slicing on a single symbol
            class TestSlicingSingleSymbol2(gluon.HybridBlock):
                def forward(self, x, y):
                    return (x[()][index1] + y[()][index1])[index2]

            check_gluon_hybridize_consistency(TestSlicingSingleSymbol1, [x, y],
                                              numpy_func=lambda a, b: a[()][index1] + b[()][index1])
            check_gluon_hybridize_consistency(TestSlicingSingleSymbol2, [x, y],
                                              numpy_func=lambda a, b:
                                              (a[()][index1] + b[()][index1])[index2])
        # Test for split/hsplit/vsplit
        class TestSlicingWithSplit(gluon.HybridBlock):
            def forward(self, x):
                x = mx.np.split(x, shape[2], axis=2)
                x = x[1:-1]
                x = mx.np.concatenate(x, axis=2)
                return x

        class TestSlicingWithSplit2(gluon.HybridBlock):
            def __init__(self):
                super(TestSlicingWithSplit2, self).__init__()
                self.layer = gluon.nn.Dense(16, flatten=False)

            def forward(self, x, y):
                x = mx.np.split(x, 1)
                x = x[0]
                return self.layer(x[:, -1, :] + y[:, -1, :])

        class TestSlicingWithHSplit(gluon.HybridBlock):
            def forward(self, x):
                x = mx.np.hsplit(x, shape[1])
                x = x[1:-1]
                x = mx.np.concatenate(x, axis=1)
                return x

        class TestSlicingWithVSplit(gluon.HybridBlock):
            def forward(self, x):
                x = mx.np.vsplit(x, shape[0])
                x = x[1:-1]
                x = mx.np.concatenate(x, axis=0)
                return x

        if len(shape) > 2 and shape[2] > 2:
            check_gluon_hybridize_consistency(
                TestSlicingWithSplit, [x],
                numpy_func=lambda a: _np.concatenate(_np.split(a, shape[2], axis=2)[1:-1],
                                                     axis=2))
        if len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
            check_gluon_hybridize_consistency(TestSlicingWithSplit2, [x, y])

        if len(shape) > 1 and shape[1] > 2:
            check_gluon_hybridize_consistency(
                TestSlicingWithHSplit, [x],
                numpy_func=lambda a: _np.concatenate(_np.hsplit(a, shape[1])[1:-1], axis=1))
        if len(shape) > 1 and shape[0] > 2:
            check_gluon_hybridize_consistency(
                TestSlicingWithVSplit, [x],
                numpy_func=lambda a: _np.concatenate(_np.vsplit(a, shape[0])[1:-1], axis=0))

    for data_shape, idx in [((4, 3), 2),
                            ((3,), -1),
                            ((3,), 0)]:
        class IntegerIndexing(gluon.HybridBlock):
            def forward(self, x):
                return x[idx]
        check_gluon_hybridize_consistency(IntegerIndexing,
                                          [mx.np.ones(data_shape)],
                                          numpy_func=lambda a: a[idx])


@use_np
def test_net_symbol_save_load():
    class Case1(gluon.HybridBlock):
        def __init__(self):
            super(Case1, self).__init__()
            self.layer = gluon.nn.Dense(64, flatten=False)

        def forward(self, x, y):
            x = mx.np.split(x, 1)
            x = x[0]
            return self.layer(x[:, -1, :] + y[:, -1, :])
    check_gluon_save_load(Case1, [mx.np.random.normal(0, 1, (10, 5, 8, 6)),
                                  mx.np.random.normal(0, 1, (10, 5, 8, 6))])

    class Case2(gluon.HybridBlock):
        def __init__(self):
            super(Case2, self).__init__()
            self.layer1 = gluon.nn.Dense(64, flatten=False)
            self.layer2 = gluon.nn.Dense(64, flatten=False)

        def forward(self, x, y):
            x = mx.np.split(x, 1)
            x = x[0]
            return self.layer1(x[:, -1, :]) + self.layer2(y[:, -1, :])
    check_gluon_save_load(Case2, [mx.np.random.normal(0, 1, (10, 5, 8)),
                                  mx.np.random.normal(0, 1, (10, 5, 8))])

@use_np
def test_hybridize_boolean_dtype():
    class Foo(gluon.HybridBlock):
        def __init__(self):
            super(Foo, self).__init__()

        def forward(self, valid_length):
            mask = ((np.ones((10,)) / 2) < valid_length)
            return mask

    valid_length = mx.np.random.uniform(size=(10,))
    foo = Foo()
    out1 = foo(valid_length)

    foo = Foo()
    foo.hybridize()
    out2 = foo(valid_length)

    assert mx.test_utils.same(out1.asnumpy(), out2.asnumpy())


@use_np
def test_optimize_for():
    class TestBlock(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock, self).__init__()
            self.d = mx.gluon.nn.Dense(1)
        def forward(self, a):
            res = self.d(a)
            return res

    a = mx.np.random.uniform(low=-1, high=1, size=(1,1))

    net = TestBlock()
    net.initialize()
    net.hybridize()

    out = net(a)
    b = net.collect_params().pop('d.weight').data()
    net.optimize_for(a, b, backend="ONEDNN")
    out2 = net(a)


@use_np
def test_activations_leakyrelu():
    # Currently, all the activation tests, we will just test for runnable.
    act_layer = nn.LeakyReLU(0.1)
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@use_np
def test_activations_prelu():
    act_layer = nn.PReLU()
    act_layer.initialize()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@use_np
def test_activations_elu():
    act_layer = nn.ELU(1.0)
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@use_np
def test_activations_selu():
    act_layer = nn.SELU()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@use_np
def test_activations_gelu():
    act_layer = nn.GELU()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()

@use_np
def test_activations_gelu_tanh():
    act_layer = nn.GELU(approximation='tanh')
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()

@use_np
def test_activations_swish():
    act_layer = nn.Swish()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@use_np
def test_activations_silu():
    act_layer = nn.SiLU()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()

@use_np
def test_concatenate():
    model = nn.HybridConcatenate(axis=1)
    model.add(nn.Dense(128, activation='tanh', in_units=10))
    model.add(nn.Dense(64, activation='tanh', in_units=10))
    model.add(nn.Dense(32, in_units=10))
    model2 = nn.Concatenate(axis=1)
    model2.add(nn.Dense(128, activation='tanh', in_units=10))
    model2.add(nn.Dense(64, activation='tanh', in_units=10))
    model2.add(nn.Dense(32, in_units=10))

    # ndarray
    model.initialize(mx.init.Xavier(magnitude=2.24))
    model2.initialize(mx.init.Xavier(magnitude=2.24))
    x = model(mx.np.zeros((32, 10)))
    x2 = model2(mx.np.zeros((32, 10)))
    assert x.shape == (32, 224)
    assert x2.shape == (32, 224)
    x.wait_to_read()
    x2.wait_to_read()

@use_np
def test_identity():
    model = nn.Identity()
    x = mx.np.random.uniform(size=(128, 33, 64))
    assert_almost_equal(model(x), x)

@use_np
def test_pixelshuffle1d():
    nchan = 2
    up_x = 2
    nx = 3
    shape_before = (1, nchan * up_x, nx)
    shape_after = (1, nchan, nx * up_x)
    layer = nn.PixelShuffle1D(up_x)
    x = mx.np.reshape(mx.np.arange(_np.prod(shape_before)), shape_before)
    y = layer(x)
    assert y.shape == shape_after
    assert_allclose(
        y,
        [[[0, 3, 1, 4, 2, 5],
          [6, 9, 7, 10, 8, 11]]]
    )

@use_np
def test_pixelshuffle2d():
    nchan = 2
    up_x = 2
    up_y = 3
    nx = 2
    ny = 3
    shape_before = (1, nchan * up_x * up_y, nx, ny)
    shape_after = (1, nchan, nx * up_x, ny * up_y)
    layer = nn.PixelShuffle2D((up_x, up_y))
    x = mx.np.reshape(mx.np.arange(_np.prod(shape_before)), shape_before)
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

@use_np
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
    x = mx.np.arange(_np.prod(shape_before)).reshape(shape_before)
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

@use_np
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


@use_np
@pytest.mark.parametrize('dshape', [(10, ), (2, 10, 10, 10)])
def test_layernorm(dshape):
    layer = nn.LayerNorm(in_channels=10)
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
