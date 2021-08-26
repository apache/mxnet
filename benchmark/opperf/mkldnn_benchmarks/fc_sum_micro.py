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

#!/usr/bin/python
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import time
SHAPES = [(1, 224), (16, 1024), (32, 4096), (512, 512)]
num_hidden = [512, 1024, 4096]
rounds = 5000
warmup = 100


class FCWithSum(nn.HybridBlock):
    def __init__(self, num_in, num_hidden, **kwargs):
        super(FCWithSum, self).__init__(**kwargs)
        self.fc0 = nn.Dense(units=num_hidden, in_units=num_in)
        self.fc1 = nn.Dense(units=num_hidden)

    def hybrid_forward(self, F, data0, data1, data2):
        _fc0 = self.fc0(data0)
        _fc1 = self.fc1(data1)
        _sum0 = data2 + _fc0
        _sum1 = _fc1 + _sum0
        return _sum1


def benchmark_float():
    for shape in SHAPES:
        for nhid in num_hidden:
            net = FCWithSum(shape[1], nhid)
            net.initialize()
            net.hybridize(static_alloc=True, static_shape=True)
            data0 = mx.nd.random_uniform(shape=shape, low=-1.0, high=1.0)
            data1 = mx.nd.random_uniform(shape=shape, low=-1.0, high=1.0)
            shape2 = (shape[0], nhid)
            data2 = mx.nd.random_uniform(shape=shape2, low=-1.0, high=1.0)
            tic = 0
            for i in range(rounds + warmup):
                if i == warmup:
                    tic = time.time()
                o = net(data0, data1, data2)
                o.wait_to_read()
            toc = time.time()
            print("Shape: ({:4}, {:4}) | num_hidden: {:4} | Time: {:8.3f} s | Mean: {:8.3f} ms".format(
                shape[0], shape[1], nhid, toc - tic, 1000 * (toc-tic)/rounds))


class CalibIter(mx.io.DataIter):
    def __init__(self, batch, data_shape, batch_size):
        super(CalibIter, self).__init__(batch_size)
        self.label_shape = (batch_size,)
        self.data_shape = data_shape
        if isinstance(data_shape, tuple):
            self.provide_data = [('data', data_shape)]
        else:
            self.provide_data = data_shape
        self.provide_label = []
        self.batch = batch

    def __iter__(self):
        yield self.batch


def benchmark_int8():
    for shape in SHAPES:
        for nhid in num_hidden:
            net = FCWithSum(shape[1], nhid)
            net.initialize()
            net.hybridize(static_alloc=True, static_shape=True)

            data0 = mx.nd.random_uniform(shape=shape, low=-1.0, high=1.0)
            data1 = mx.nd.random_uniform(shape=shape, low=-1.0, high=1.0)
            shape2 = (shape[0], nhid)
            data2 = mx.nd.random_uniform(shape=shape2, low=-1.0, high=1.0)
            batch = mx.io.DataBatch([data0, data1, data2], [])
            calib_data = CalibIter(batch, [mx.io.DataDesc("data0", shape=shape, dtype='float32'),
                                           mx.io.DataDesc("data1", shape=shape, dtype='float32'),
                                           mx.io.DataDesc("data2", shape=shape2, dtype='float32')], 1)
            net_quantized = mx.contrib.quant.quantize_net_v2(net, quantized_dtype='auto',
                                                             exclude_layers=None,
                                                             exclude_layers_match=None,
                                                             calib_data=calib_data,
                                                             calib_mode='naive',
                                                             quantize_mode='smart',
                                                             num_calib_examples=1,
                                                             ctx=mx.current_context())
            tic = 0
            for i in range(rounds + warmup):
                if i == warmup:
                    tic = time.time()
                o = net_quantized(data0, data1, data2)
                o.wait_to_read()
            toc = time.time()
            print("Shape: ({:4}, {:4}) | num_hidden: {:4} | Time: {:8.3f} s | Mean: {:8.3f} ms".format(
                shape[0], shape[1], nhid, toc - tic, 1000 * (toc-tic)/rounds))


benchmark_int8()
print("------- float: ------")
benchmark_float()
