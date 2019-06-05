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

import mxnet as mx
from mxnet import gluon, autograd, np, npx


def test_create_np_param():
    M, K, N = 10, 9, 20

    def check_block_params(x, TestBlock, hybridize, expected_type):
        net = TestBlock()
        net.initialize()
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

    @npx.use_np
    class TestBlock2(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock2, self).__init__()
            with self.name_scope():
                self.w = self.params.get('w', shape=(K, N), allow_deferred_init=True)

        def hybrid_forward(self, F, x, w):
            return F.np.dot(x, w)

    x = mx.nd.random.uniform(shape=(M, K))
    check_block_params(x, TestBlock1, False, mx.nd.NDArray)
    check_block_params(x, TestBlock1, True, mx.nd.NDArray)
    check_block_params(x.as_np_ndarray(), TestBlock2, False, np.ndarray)
    check_block_params(x.as_np_ndarray(), TestBlock2, True, np.ndarray)


@npx.use_np
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
    regressor.initialize(mx.init.Normal())
    regressor.hybridize()

    # Create random input and output data
    x = mx.nd.random.normal(shape=(64, 1000)).as_np_ndarray()  # x is of type mxnet.numpy.ndarray
    regressor(x)
    y = mx.nd.random.normal(shape=(64, 10)).as_np_ndarray()  # y is of type mxnet.numpy.ndarray

    total_loss = TotalLoss()
    total_loss.hybridize()

    trainer = gluon.Trainer(regressor.collect_params(),
                            'sgd',
                            {'learning_rate': 1e-3, 'momentum': 0.9})

    for t in range(5):
        with autograd.record():
            output = regressor(x)  # output is a type of np.ndarray because np.dot is the last op in the network
            loss = total_loss(output, y)  # loss is a scalar np.ndarray
        loss.backward()
        trainer.step(1)


if __name__ == '__main__':
    import nose
    nose.runmodule()
