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

"""
MKL-DNN related test cases
"""
import sys
import os
import numpy as np
import mxnet as mx
import unittest
from mxnet.test_utils import assert_almost_equal
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.test_utils import *
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../unittest/'))
from common import with_seed

def check_layer_forward_withinput(net, x):
    x_hybrid = x.copy()
    x.attach_grad()
    x_hybrid.attach_grad()
    net.collect_params().initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x_hybrid)
    out2.backward()
    mx.test_utils.assert_almost_equal(x.grad.asnumpy(), x_hybrid.grad.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(out1.asnumpy(), out2.asnumpy(), rtol=1e-5, atol=1e-6)

@with_seed()
def test_conv2d_16c():
    chn_list = [16, 256]
    kernel_list = [1, 3]
    #kernel_list.append(224)
    batch_size = 8
    class Net(gluon.HybridBlock):
        def __init__(self,
                     chn_num,
                     kernel,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = gluon.nn.Conv2D(chn_num, (kernel, kernel))

        def hybrid_forward(self, F, x):
            out = self.conv0(x)
            return out

    x = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size, 3, 224, 224))
    for i in range(len(chn_list)):
        for j in range(len(kernel_list)):
            net = Net(chn_list[i], kernel_list[j])
            check_layer_forward_withinput(net, x)

@with_seed()
def test_group_conv2d_16c():
    grp_list = [16]
    input_size_list = np.random.randint(low=3, high=65, size=10).tolist()
    kernel_list = [1, 3]
    batch_size = 8
    class Net(gluon.HybridBlock):
        def __init__(self,
                     chn_num,
                     kernel,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = gluon.nn.Conv2D(chn_num, (1, 1))
                self.conv1 = gluon.nn.Conv2D(chn_num, (kernel, kernel), groups=chn_num)

        def hybrid_forward(self, F, x):
            y = self.conv0(x)
            out = self.conv1(y)
            return out

    for i in range(len(input_size_list)):
        x = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size, 3, input_size_list[i], input_size_list[i]))
        for j in range(len(grp_list)):
            for k in range(len(kernel_list)):
                net = Net(grp_list[j], kernel_list[k])
                check_layer_forward_withinput(net, x)


@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_deconv2d_16c():
    in_chn_list = [1024, 512, 256, 128, 64, 32, 16]
    out_chn_list = [512, 256, 128, 64, 32, 16, 3]
    kernel_list = [1, 3, 5, 7]
    in_shape = [4, 8, 16, 32, 64, 224]
    batch_size = 8
    class Net(gluon.HybridBlock):
        def __init__(self, chn_num, kernel, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.deconv0 = gluon.nn.Conv2DTranspose(chn_num, (kernel, kernel))

        def hybrid_forward(self, F, x):
            out = self.deconv0(x)
            return out
    for i in range(len(in_shape)):
        x = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size, in_chn_list[i], in_shape[i], in_shape[i]))
        for j in range(len(kernel_list)):
            net = Net(out_chn_list[i], kernel_list[j])
            check_layer_forward_withinput(net, x)


@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_batchnorm_16c():
    chn_list = [16, 1024]
    shape = np.random.randint(low=1, high=300, size=10)
    shape_list = []
    for i in range(len(shape)):
        shape_list.append((shape[i], shape[i]))
    batch_size = 8
    class Net(gluon.HybridBlock):
        def __init__(self,
                     chn_num,
                     kernel,
                     axis,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = gluon.nn.Conv2D(chn_num, (kernel, kernel))
                self.bn0   = gluon.nn.BatchNorm(axis=axis)

        def hybrid_forward(self, F, x):
            conv = self.conv0(x)
            out = self.bn0(conv)
            return out

    for i in range(len(chn_list)):
        for j in range(len(shape_list)):
            shape = (batch_size, ) + (3,) + shape_list[j]
            x = mx.nd.random.uniform(-1.0, 1.0, shape=shape)
            net = Net(chn_list[i], 1, 1)
            check_layer_forward_withinput(net, x)


@with_seed()
def test_concat():
    chn_list = [64, 16]
    shapes = [7, 5, 3]
    input_num = np.random.randint(low=2, high=11)
    shape_list = []
    for i in range(len(shapes)):
        shape_list.append((shapes[i], shapes[i]))
    batch_size = 8
    class Net(gluon.HybridBlock):
        def __init__(self,
                     check_dim,
                     input_num,
                     chn_num,
                     kernel,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                from mxnet.gluon.contrib.nn import HybridConcurrent
                self.concat = HybridConcurrent(axis=check_dim)
                for i in range(input_num):
                    self.concat.add(gluon.nn.Conv2D(chn_num, (kernel, kernel)))

        def hybrid_forward(self, F, x):
            return self.concat(x)

    for s in range(len(shape_list)):
        shape = (batch_size,) + (3,) + shape_list[i]
        x = mx.nd.random.uniform(-1.0, 1.0, shape=shape)
        for i in range(len(chn_list)):
            for axis in range(4):
                net = Net(axis, input_num, chn_list[i], 1)
                check_layer_forward_withinput(net, x)


@with_seed()
def test_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 448, 112))
            out = self.conv0(x_reshape)
            return out
    x = mx.nd.random.uniform(shape=(32, 3, 224, 224))
    net = Net()
    check_layer_forward_withinput(net, x)


@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_conv_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 448, 112))
            y = self.conv0(x_reshape)
            y_reshape = y.reshape((0, 0, 223, 220))
            out = self.conv1(y_reshape)
            return out
    x = mx.nd.random.uniform(shape=(32, 3, 224, 224))
    net = Net()
    check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(16, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 2, 0, 0), end=(4, 5, 32, 32))
            out = self.conv0(x_slice)
            return out
    x = mx.nd.random.uniform(shape=(8, 6, 32, 32))
    net = Net()
    check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_conv_slice_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(16, (1, 1))
                self.conv1 = nn.Conv2D(16, (1, 1))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 0, 0, 0), end=(4, 3, 16, 16))
            y = self.conv0(x_slice)
            y_slice = y.slice(begin=(0, 1, 0, 0), end=(4, 4, 16, 16))
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(8, 6, 32, 32))
    net = Net()
    check_layer_forward_withinput(net, x)


@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_slice_conv_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 0, 1, 1), end=(32, 3, 225, 225))
            y = self.conv0(x_slice)
            y_reshape = y.reshape((0, 0, 444, 111))
            out = self.conv1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(32, 3, 299, 299))
    net = Net()
    check_layer_forward_withinput(net, x)

@with_seed()
def test_reshape_conv_slice_conv():
    """
    This test will test gluon Conv2d computation with ndarray reshape and slice
    """
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(16, (1, 1))
                self.conv1 = nn.Conv2D(16, (1, 1))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 64, 16))
            y = self.conv0(x_reshape)
            y_slice = y.slice(begin=(0, 0, 0, 0), end=(4, 3, 16, 16))
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(8, 3, 32, 32))
    net = Net()
    check_layer_forward_withinput(net, x)


@with_seed()
def test_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = np.random.randint(1, 17)
                self.dense0 = nn.Dense(channel0)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 128, -1))
            out = self.dense0(x_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 32, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = np.random.randint(1, 17)
                self.dense0 = nn.Dense(channel0)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]),
                              end=tuple(self.slice[1]))
            out = self.dense0(x_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 32, 64, 64))
    slice = [[0, 16, 50, 0], [8, 32, 64, 64]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)

@with_seed()
def test_slice_dense_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = 50
                channel1 = np.random.randint(1, 33)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.dense0(x_slice)
            y_slice = y.slice(begin=(4, 0), end=(-1, 10))
            out = self.dense1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 32, 64, 64))
    slice = [[0, 16, 50, 0], [8, 32, 64, 64]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)

@with_seed()
def test_reshape_dense_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = np.random.randint(1, 17)
                channel1 = np.random.randint(1, 65)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 128, -1))
            y = self.dense0(x_reshape)
            y_reshape = y.reshape((1, -1))
            out = self.dense1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 32, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_dense_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = np.random.randint(1, 17)
                channel1 = np.random.randint(1, 17)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.dense0(x_slice)
            y_reshape = y.reshape((1, -1))
            out = self.dense1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 32, 64, 64))
    slice = [[0, 16, 50, 0], [8, 32, 64, 64]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)


@with_seed()

def test_reshape_dense_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = 64
                channel1 = np.random.randint(1, 17)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 128, -1))
            y = self.dense0(x_reshape)
            y_slice = y.slice(begin=(0, 32), end=(8, 64))
            out = self.dense1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 32, 64, 64))
    net = Net()
    check_layer_forward_withinput(net, x)


@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm()
                self.reshape = shape

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape)
            out = self.bn0(x_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    shape = (32, 512, 128, -1)
    net = Net(shape)
    check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(3)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0]),
                              end=tuple(self.slice[1]))
            out = self.bn0(x_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 64, 50, 0], [8, 128, 256, 256]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_batchnorm_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(3)
                self.bn1 = nn.BatchNorm(1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0][0]), end=tuple(self.slice[0][1]))
            y = self.bn0(x_slice)
            y_slice = y.slice(begin=tuple(self.slice[1][0]), end=tuple(self.slice[1][1]))
            out = self.bn1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[[0, 64, 50, 0], [8, 128, 200, 256]], [[4, 50, 0, 128], [7, -1, -1, -1]]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)


@with_seed()
def test_reshape_batchnorm_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(0)
                self.bn1 = nn.BatchNorm(2)
                self.reshape = shape

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape[0])
            y = self.bn0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.bn1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 512))
    shape = [(8, 256, 128, -1), (32, 128, 512, -1)]
    net = Net(shape)
    check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_batchnorm_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(0)
                self.bn1 = nn.BatchNorm(2)
                self.reshape = shape
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.bn0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.bn1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 64, 50, 0], [8, 128, 200, 256]]
    shape = (1, 128, 256, -1)
    net = Net(shape, slice)
    check_layer_forward_withinput(net, x)


@with_seed()
def test_reshape_batchnorm_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(2)
                self.bn1 = nn.BatchNorm(0)
                self.reshape = shape
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape)
            y = self.bn0(x_reshape)
            y_slice = y.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            out = self.bn1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 0, 50, 0], [8, 1, -1, 100]]
    shape = (128, 1, 256, -1)
    net = Net(shape, slice)
    check_layer_forward_withinput(net, x)

@with_seed()
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
            with self.name_scope():
                self.reshape = shape
                self.pool0 = pooling_layer

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape)
            out = self.pool0(x_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    shape = (128, 256, 256, -1)
    for i in range(len(pooling_layers)):
        net = Net(shape, pooling_layers[i])
        check_layer_forward_withinput(net, x)

@with_seed()
def test_slice_pooling2d():
    max_pooling = nn.MaxPool2D(strides=(2, 3), padding=(1, 1))
    avg_pooling = nn.AvgPool2D(strides=(2, 2), padding=(1, 1))
    global_maxpooling = nn.GlobalMaxPool2D()
    global_avgpooling = nn.GlobalAvgPool2D()
    pooling_layers = [max_pooling, avg_pooling, global_maxpooling, global_avgpooling]
    class Net(gluon.HybridBlock):
        def __init__(self,
                     slice,
                     pooling_layer,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.slice = slice
                self.pool0 = pooling_layer

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            out = self.pool0(x_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [(12, 0, 128, 64), (16, 16, 256, 256)]
    for i in range(len(pooling_layers)):
        net = Net(slice, pooling_layers[i])
        check_layer_forward_withinput(net, x)

@with_seed()
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
            with self.name_scope():
                self.reshape = shape
                self.pool0 = pooling_layer1
                self.pool1 = pooling_layer2

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape[0])
            y = self.pool0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.pool1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    shape = [(128, 256, 64, -1), (128, 256, 11, -1)]
    for i in range(len(pooling_layers)):
        for j in range(len(pooling_layers)):
            if isinstance(pooling_layers[i], (nn.GlobalMaxPool2D, nn.GlobalAvgPool2D)):
                shape[1] = (256, 128, 1, 1)
            net = Net(shape, pooling_layers[i], pooling_layers[j])
            check_layer_forward_withinput(net, x)

@with_seed()
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
            with self.name_scope():
                self.slice = slice
                self.pool0 = pooling_layer1
                self.pool1 = pooling_layer2

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0][0], end=self.slice[0][1])
            y = self.pool0(x_slice)
            y_slice = y.slice(begin=self.slice[1][0], end=self.slice[1][1])
            out = self.pool1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[(8, 0, 100, 50), (16, -1, -1, -1)], [(0, 64, 0, 50), (2, -1, -1, -1)]]
    for i in range(len(pooling_layers)):
        for j in range(len(pooling_layers)):
            if isinstance(pooling_layers[i], (nn.GlobalMaxPool2D, nn.GlobalAvgPool2D)):
                slice[1] = [(0, 64, 0, 0), (2, -1, 1, 1)]
            net = Net(slice, pooling_layers[i], pooling_layers[j])
            check_layer_forward_withinput(net, x)

@with_seed()
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
            with self.name_scope():
                self.reshape = shape
                self.slice = slice
                self.pool0 = pooling_layer1
                self.pool1 = pooling_layer2

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            y = self.pool0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.pool1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [(8, 0, 100, 50), (16, 128, 256, 256)]
    shape = (32, -1, 0, 0)
    for i in range(len(pooling_layers)):
        for j in range(len(pooling_layers)):
            net = Net(shape, slice, pooling_layers[i], pooling_layers[j])
            check_layer_forward_withinput(net, x)

@with_seed()
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
            with self.name_scope():
                self.reshape = shape
                self.slice = slice
                self.pool0 = pooling_layer1
                self.pool1 = pooling_layer2

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape)
            y = self.pool0(x_reshape)
            y_slice = y.slice(begin=self.slice[0], end=self.slice[1])
            out = self.pool1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    shape = (0, 512, 64, -1)
    slice = [(8, 256, 10, 20), (-1, -1, -1, 70)]
    for i in range(len(pooling_layers)):
        for j in range(len(pooling_layers)):
            if isinstance(pooling_layers[i], (nn.GlobalMaxPool2D, nn.GlobalAvgPool2D)):
                slice = [(8, 256, 0, 0), (-1, -1, 1, 1)]
            net = Net(shape, slice, pooling_layers[i], pooling_layers[j])
            check_layer_forward_withinput(net, x)

@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.reshape = shape
                self.conv0 = nn.Conv2DTranspose(64, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape)
            out = self.conv0(x_reshape)
            return out
    x = mx.nd.random.uniform(shape=(64, 2, 256, 256))
    shape = (8, 16, 64, -1)
    net = Net(shape)
    check_layer_forward_withinput(net, x)

@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_slice_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.slice = slice
                self.conv0 = nn.Conv2DTranspose(64, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            out = self.conv0(x_slice)
            return out
    x = mx.nd.random.uniform(shape=(128, 32, 500, 500))
    slice = [(0, 16, 0, 0), (1, 32, 256, 256)]
    net = Net(slice)
    check_layer_forward_withinput(net, x)

@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_deconv_reshape_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.reshape = shape
                self.conv0 = nn.Conv2DTranspose(64, (3, 3))
                self.conv1 = nn.Conv2DTranspose(128, (2, 3), strides=(2, 2))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape[0])
            y = self.conv0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.conv1(y_reshape)
            return out
    x = mx.nd.random.uniform(shape=(16, 32, 256, 512))
    shape = [(32, 0, 256, -1), (64, 32, 129, -1)]
    net = Net(shape)
    check_layer_forward_withinput(net, x)

@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_slice_deconv_slice_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.slice = slice
                self.conv0 = nn.Conv2DTranspose(64, (3, 3))
                self.conv1 = nn.Conv2DTranspose(128, (2, 3), strides=(2, 2))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0][0], end=self.slice[0][1])
            y = self.conv0(x_slice)
            y_slice = y.slice(begin=self.slice[1][0], end=self.slice[1][1])
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(128, 32, 500, 500))
    slice = [[(0, 16, 0, 0), (8, 32, 128, 128)], [(4, 0, 2, 0), (8, 32, 130, 128)]]
    net = Net(slice)
    check_layer_forward_withinput(net, x)

@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_reshape_deconv_slice_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.reshape = shape
                self.slice = slice
                self.conv0 = nn.Conv2DTranspose(64, (3, 3))
                self.conv1 = nn.Conv2DTranspose(128, (2, 3), strides=(2, 2))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape)
            y = self.conv0(x_reshape)
            y_slice = y.slice(begin=self.slice[0], end=self.slice[1])
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(16, 4, 500, 500))
    shape = (32, 16, 125, -1)
    slice = [(4, 32, 0, 0), (20, 64, 64, 224)]
    net = Net(shape, slice)
    check_layer_forward_withinput(net, x)

@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/11164')
def test_slice_deconv_reshape_deconv():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.reshape = shape
                self.slice = slice
                self.conv0 = nn.Conv2DTranspose(64, (3, 3))
                self.conv1 = nn.Conv2DTranspose(128, (2, 3), strides=(2, 2))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            y = self.conv0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.conv1(y_reshape)
            return out
    x = mx.nd.random.uniform(shape=(16, 32, 256, 512))
    shape = (24, 16, 452, -1)
    slice = [(4, 0, 0, 0), (16, 32, 224, 224)]
    net = Net(shape, slice)
    check_layer_forward_withinput(net, x)

@with_seed()
def test_reshape_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.reshape = shape
                self.act = nn.Activation(act)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape)
            out = self.act(x_reshape)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu"]
    for act in acts:
        x = mx.nd.random.uniform(-1, 1, shape=(16, 32, 256, 512))
        shape = (64, 8, 128, -1)
        net = Net(act, shape)
        check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.slice = slice
                self.act = nn.Activation(act)

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            out = self.act(x_slice)
            return out

    acts = ["relu", "sigmoid", "tanh", "softrelu"]
    for act in acts:
        x = mx.nd.random.uniform(-1, 1, shape=(16, 32, 256, 512))
        slice = [(8, 16, 0, 0), (16, 32, 100, 100)]
        net = Net(act, slice)
        check_layer_forward_withinput(net, x)


@with_seed()
def test_reshape_activation_reshape_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act0, act1, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.reshape = shape
                self.act0 = nn.Activation(act0)
                self.act1 = nn.Activation(act1)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape[0])
            y = self.act0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.act1(y_reshape)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu"]
    for idx0, act0 in enumerate(acts):
        for idx1, act1 in enumerate(acts):
            if idx1 == idx0:
                continue
            x = mx.nd.random.uniform(-1, 1, shape=(16, 32, 256, 512))
            shape = [(64, 8, 128, -1), (16, 64, 128, -1)]
            net = Net(act0, act1, shape)
            check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_activation_slice_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act0, act1, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.slice = slice
                self.act0 = nn.Activation(act0)
                self.act1 = nn.Activation(act1)

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0][0], end=self.slice[0][1])
            y = self.act0(x_slice)
            y_slice = y.slice(begin=self.slice[1][0], end=self.slice[1][1])
            out = self.act1(y_slice)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu"]
    for idx0, act0 in enumerate(acts):
        for idx1, act1 in enumerate(acts):
            if idx1 == idx0:
                continue
            x = mx.nd.random.uniform(-1, 1, shape=(16, 32, 256, 512))
            slice = [[(0, 0, 100, 100), (8, 16, 256, 512)], [(2, 4, 0, 0), (8, 10, 128, 128)]]
            net = Net(act0, act1, slice)
            check_layer_forward_withinput(net, x)


@with_seed()
def test_reshape_activation_slice_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act0, act1, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.reshape = shape
                self.slice = slice
                self.act0 = nn.Activation(act0)
                self.act1 = nn.Activation(act1)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape(self.reshape)
            y = self.act0(x_reshape)
            y_slice = y.slice(begin=self.slice[0], end=self.slice[1])
            out = self.act1(y_slice)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu"]
    for idx0, act0 in enumerate(acts):
        for idx1, act1 in enumerate(acts):
            if idx1 == idx0:
                continue
            x = mx.nd.random.uniform(-1, 1, shape=(16, 32, 256, 512))
            shape = (64, 16, 128, -1)
            slice = [(0, 0, 0, 100), (8, 16, 64, 228)]
            net = Net(act0, act1, shape, slice)
            check_layer_forward_withinput(net, x)


@with_seed()
def test_slice_activation_reshape_activation():
    class Net(gluon.HybridBlock):
        def __init__(self, act0, act1, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.reshape = shape
                self.slice = slice
                self.act0 = nn.Activation(act0)
                self.act1 = nn.Activation(act1)

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=self.slice[0], end=self.slice[1])
            y = self.act0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.act1(y_reshape)
            return out
    acts = ["relu", "sigmoid", "tanh", "softrelu"]
    for idx0, act0 in enumerate(acts):
        for idx1, act1 in enumerate(acts):
            if idx1 == idx0:
                continue
            x = mx.nd.random.uniform(-1, 1, shape=(16, 32, 256, 512))
            slice = [(0, 0, 0, 100), (8, 16, 64, 228)]
            shape = (64, 16, 64, -1)
            net = Net(act0, act1, shape, slice)
            check_layer_forward_withinput(net, x)

if __name__ == '__main__':
    import nose
    nose.runmodule()
