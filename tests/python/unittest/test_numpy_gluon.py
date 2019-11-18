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

import numpy as _np
import mxnet as mx
from mxnet import gluon, autograd, np
from mxnet.test_utils import use_np, assert_almost_equal
from common import with_seed
import random


@with_seed()
def test_create_np_param():
    M, K, N = 10, 9, 20

    def check_block_params(x, TestBlock, hybridize, expected_type, initializer):
        net = TestBlock()
        net.initialize(initializer())
        if hybridize:
            net.hybridize()
        net(x)
        params = net.collect_params()
        for k, v in params.items():
            assert type(v.data()) is expected_type

    class TestBlock1(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock1, self).__init__()
            with self.name_scope():
                self.w = self.params.get('w', shape=(K, N), allow_deferred_init=True)

        def hybrid_forward(self, F, x, w):
            return F.dot(x, w)

    @use_np
    class TestBlock2(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock2, self).__init__()
            with self.name_scope():
                self.w = self.params.get('w', shape=(K, N), allow_deferred_init=True)

        def hybrid_forward(self, F, x, w):
            return F.np.dot(x, w)

    x = mx.nd.random.uniform(shape=(M, K))
    for initializer in [mx.initializer.Uniform, mx.initializer.Normal]:
        check_block_params(x, TestBlock1, False, mx.nd.NDArray, initializer)
        check_block_params(x, TestBlock1, True, mx.nd.NDArray, initializer)
        check_block_params(x.as_np_ndarray(), TestBlock2, False, np.ndarray, initializer)
        check_block_params(x.as_np_ndarray(), TestBlock2, True, np.ndarray, initializer)


@with_seed()
@use_np
def test_optimizer_with_np_ndarrays():
    class LinearRegression(gluon.HybridBlock):
        def __init__(self, num_input_dim=0, num_hidden_dim=100, num_output_dim=10):
            super(LinearRegression, self).__init__()
            with self.name_scope():
                self.w1 = self.params.get('w1', shape=(num_input_dim, num_hidden_dim),
                                          allow_deferred_init=True)
                self.w2 = self.params.get('w2', shape=(num_hidden_dim, num_output_dim),
                                          allow_deferred_init=True)

        def hybrid_forward(self, F, x, w1, w2):
            h = x.dot(w1)  # equivalent to F.np.dot(x, w1)
            h_relu = F.npx.relu(h)  # equivalent to F.relu(h) but generating np.ndarray
            y_pred = h_relu.dot(w2)  # equivalent to F.np.dot(h_relu, w2)
            return y_pred

    class TotalLoss(gluon.HybridBlock):
        def hybrid_forward(self, F, pred, label):
            return ((pred - label) ** 2).sum()  # equivalent to F.np.sum(F.np.square(pred - label))

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

    for t in range(2):
        with autograd.record():
            output = regressor(x)  # output is a type of np.ndarray because np.dot is the last op in the network
            loss = total_loss(output, y)  # loss is a scalar np.ndarray
        loss.backward()
        trainer.step(1)


@with_seed()
@use_np
def test_optimizer_backward_compat():
    optimizer = mx.optimizer.SGD()
    delattr(optimizer, "allow_np_array")
    updater = mx.optimizer.Updater(optimizer)
    updater(0, np.ones((0, 0)), np.zeros((0, 0)))


@with_seed()
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
    assert_almost_equal(L, _np.array([2.12692809,  0.04858733]), use_broadcast=False)

    L = loss(output, label, weighting).asnumpy()
    assert_almost_equal(L, _np.array([1.06346405,  0.04858733]), use_broadcast=False)


@with_seed()
@use_np
def test_np_get_constant():
    const_arr = _np.random.uniform(0, 100, size=(10, 10)).astype(_np.float32)

    class Foo(gluon.HybridBlock):
        def __init__(self, prefix=None, params=None):
            super(Foo, self).__init__(prefix=prefix, params=params)
            self.weight = self.params.get_constant('const', const_arr)

        def hybrid_forward(self, F, x, weight):
            return x + weight.astype(np.float32)

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
        net.collect_params().zero_grad()
        for v in net.collect_params().values():
            assert_almost_equal(v.grad().asnumpy(), mx.np.zeros_like(v.grad()).asnumpy())


@with_seed()
@use_np
def test_symbolic_basic_slicing():
    def get_slice_index(shape):
        index = []
        step_switch = random.randint(0,1)
        step = None if step_switch == 0 else []
        for i in range(len(shape)):
            if shape[i] == 0:
                index.append(slice(0,1))
                continue
            if random.randint(0, 5) > 4:
                index.append(random.randint(0, shape[i]-1))
                continue
            s = random.randint(0, shape[i]-1)
            e = random.randint(s+1, shape[i])
            if step_switch == 1:
                index.append(slice(s, e, 1))
            elif step_switch == -1:
                if e == shape[i]:
                    e -= 1
                    s -= 1
                    if s == -1:
                        s = None
                index.append(slice(e, s, -1))
            else:
                index.append(slice(s, e))
        return tuple(index)

    shapes = [
        (4, 6, 8, 9),
        (1, 1, 1, 6),
        (10, 20, 30),
    ]
    for shape in shapes:
        for i in range(10):
            index = get_slice_index(shape)
            # Test basic slicing on single symbol
            class TestSlicingSingleSymbol(gluon.HybridBlock):
                def __init__(self, **kwargs):
                    super(TestSlicingSingleSymbol, self).__init__(**kwargs)

                def hybrid_forward(self, F, x):
                    x = x[:]
                    x = x[index]
                    return x

            net = TestSlicingSingleSymbol()
            x = mx.nd.random.normal(shape=shape).as_np_ndarray()
            x.attach_grad()
            with autograd.record():
                imperative_out = net(x)
            imperative_out.backward()
            imperative_grad = x.grad.asnumpy()

            y = x
            y.attach_grad()
            net2 = TestSlicingSingleSymbol()
            net2.hybridize()
            with autograd.record():
                symbolic_out = net2(y)
            symbolic_out.backward()
            symbolic_grad = y.grad.asnumpy()
            assert_almost_equal(imperative_out.asnumpy(), symbolic_out.asnumpy(), rtol=1e-3, atol=1e-5)
            assert_almost_equal(imperative_grad, symbolic_grad, rtol=1e-3, atol=1e-5)

            # Test save and load
            net2.export('gluon')
            net2_imported = gluon.SymbolBlock.imports('gluon-symbol.json', 'data', 'gluon-0000.params')
            assert_almost_equal(net2(x).asnumpy(), net2_imported(x).asnumpy())

            #Test slicing on symbol with list of outputs
            slice_on_first_dim = index[0] if isinstance(index[0], slice) else slice(index[0], index[0] + 1)
            class TestSlicingListOutputs(gluon.HybridBlock):
                def __init__(self, **kwargs):
                    super(TestSlicingListOutputs, self).__init__(**kwargs)

                def hybrid_forward(self, F, x):
                    x = F.np.split(x, shape[0])
                    x = x[slice_on_first_dim]
                    x = F.np.concatenate(x)
                    return F.np.sum(x)

            net = TestSlicingListOutputs()
            x = mx.nd.random.normal(shape=shape).as_np_ndarray()
            x.attach_grad()
            with autograd.record():
                imperative_out = net(x)
            imperative_out.backward()
            imperative_grad = x.grad.asnumpy()

            y = x
            y.attach_grad()
            net2 = TestSlicingListOutputs()
            net2.hybridize()
            with autograd.record():
                symbolic_out = net2(y)
            symbolic_out.backward()
            symbolic_grad = y.grad.asnumpy()
            assert_almost_equal(imperative_out.asnumpy(), symbolic_out.asnumpy(), rtol=1e-3, atol=1e-5)
            assert_almost_equal(imperative_grad, symbolic_grad, rtol=1e-3, atol=1e-5)

            # Test slicing on length one list of symbol (flag enabled list)
            class TestSlicingSingletonList(gluon.HybridBlock):
                def __init__(self, **kwargs):
                    super(TestSlicingSingletonList, self).__init__(**kwargs)

                def hybrid_forward(self, F, x):
                    x = F.np.split(x, 1)
                    x = x[0]
                    x = x[index]
                    return F.np.sum(x)

            net = TestSlicingSingletonList()
            x = mx.nd.random.normal(shape=shape).as_np_ndarray()
            x.attach_grad()
            with autograd.record():
                imperative_out = net(x)
            imperative_out.backward()
            imperative_grad = x.grad.asnumpy()

            y = x
            y.attach_grad()
            net2 = TestSlicingSingletonList()
            net2.hybridize()
            with autograd.record():
                symbolic_out = net2(y)
            symbolic_out.backward()
            symbolic_grad = y.grad.asnumpy()
            assert_almost_equal(imperative_out.asnumpy(), symbolic_out.asnumpy(), rtol=1e-3, atol=1e-5)
            assert_almost_equal(imperative_grad, symbolic_grad, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    import nose
    nose.runmodule()
