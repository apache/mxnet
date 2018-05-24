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
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block
from mxnet.gluon.contrib.parallel import *
from numpy.testing import assert_allclose, assert_array_equal

def test_data_parallel():
    # test gluon.contrib.parallel.DataParallelModel
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Conv2D(in_channels=1, channels=20, kernel_size=5))
        net.add(nn.Activation('relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Conv2D(in_channels=20, channels=50, kernel_size=5))
        net.add(nn.Activation('relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(nn.Flatten())
        net.add(nn.Dense(512,in_units=800))
        net.add(nn.Activation('relu'))
        net.add(nn.Dense(10, in_units=512))

    net.collect_params().initialize()
    criterion = gluon.loss.SoftmaxCELoss(axis=1)

    def test_net_sync(net, criterion, sync, num_devices):
        ctx_list = [mx.cpu(0) for i in range(num_devices)]
        net = DataParallelModel(net, ctx_list, sync=sync)
        criterion = DataParallelLoss(criterion, ctx_list, sync=sync)
        iters = 100
        # train mode
        for i in range(iters):
            x = mx.random.uniform(shape=(8, 1, 28, 28))
            t = nd.ones(shape=(8))
            with autograd.record():
                y = net(x)
                loss = criterion(y, t)
                autograd.backward(loss)
        # evaluation mode
        for i in range(iters):
            x = mx.random.uniform(shape=(8, 1, 28, 28))
            y = net(x)

    test_net_sync(net, criterion, True, 1)
    test_net_sync(net, criterion, True, 2)
    test_net_sync(net, criterion, False, 1)
    test_net_sync(net, criterion, False, 2)


def test_parallel_barrier():
    def my_callable(*inputs):
        return inputs

    class MyLayer(Block):
        def __init__(self, nGPU):
            super(MyLayer, self).__init__()
            self.barrier = Barrier(nGPU, my_callable)

        def forward(self, x):
            idx = self.barrier.push(x)
            y = self.barrier.pull(idx)
            assert_allclose(y.asnumpy(), x.asnumpy(), rtol=1e-2, atol=1e-4)
            return y
    
    num_devices = 2
    ctx_list = [mx.cpu(0) for i in range(num_devices)]
    net = MyLayer(num_devices)
    net = DataParallelModel(net, ctx_list, sync=True)
    iters = 100
    # train mode
    for i in range(iters):
        x = mx.random.uniform(shape=(8, 1, 28, 28))
        with autograd.record():
            y = net(x)
    # evaluation mode
    for i in range(iters):
        x = mx.random.uniform(shape=(8, 1, 28, 28))
        y = net(x)


if __name__ == "__main__":
    import nose
    nose.runmodule()
