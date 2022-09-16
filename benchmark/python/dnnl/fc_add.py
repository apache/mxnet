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

import time
import gc
import sys
import mxnet as mx
from mxnet.gluon import nn
from mxnet.contrib import quantization

#shape, num_hidden:
sizes = [
    ((  1, 224),   512),
    ((  1, 224),  4096),
    (( 16, 1024), 1024),
    (( 32, 4096), 1024),
    (( 32, 4096), 4096),
    ((512,  512), 4096)]

rounds = 1000
warmup = 10

test_header = "--no_test_header" not in sys.argv
table_header = "--no_table_header" not in sys.argv
table_left_colums = "--no_size_column" not in sys.argv
dump_graph = "--dump_graph" in sys.argv

def dump_graph_fn(net, postfix):
    if dump_graph:
        net.export("/tmp/fc_add_" + postfix)

def operator_string(elemwise_add):
    return 'elemwise_add' if elemwise_add else 'npi_add'

def print_header(header):
    print("\n")
    print(header if test_header else "", "\n")
    if table_header:
        if table_left_colums:
            print("|    Shape    | Hidden | Mean [ms] |" )
            print("|------------:|-------:|----------:|" )
        else:
            print(" Mean [ms] |" )
            print("----------:|" )

def print_value(shape, hidden, mean):
    if table_left_colums:
        print(f"| ({shape[0]:4},{shape[1]:4}) | {hidden:6} | {mean:9.3f} |")
    else:
        print(f" {mean:9.3f} |")


def measure(net, data0, data1, data2, shape, nhid):
    mx.nd.waitall()
    gc.collect()
    gc.disable()
    for i in range(rounds + warmup):
        if i == warmup:
            start_time = time.time()
        o = net(data0, data1, data2)
        o.wait_to_read()
    end_time = time.time()
    run_time = (end_time - start_time)
    print_value(shape, nhid, 1000 * run_time / rounds)
    gc.enable()


class FCWithSum(nn.HybridBlock):
    def __init__(self, num_in, num_hidden, elemwise_add, **kwargs):
        super(FCWithSum, self).__init__(**kwargs)
        self.fc0 = nn.Dense(units=num_hidden, in_units=num_in)
        self.fc1 = nn.Dense(units=num_hidden)
        self.elemwise_add = elemwise_add

    def forward(self, data0, data1, data2):
        _fc0 = self.fc0(data0)
        _fc1 = self.fc1(data1)
        if  self.elemwise_add:
            _sum0 = mx.nd.elemwise_add(data2.as_nd_ndarray(), _fc0.as_nd_ndarray()).as_np_ndarray()
            _sum1 = mx.nd.elemwise_add(_fc1.as_nd_ndarray(), _sum0.as_nd_ndarray()).as_np_ndarray()
        else:
            _sum0 = data2 + _fc0
            _sum1 = _fc1 + _sum0
        return _sum1

def benchmark_float(elemwise_add, broadcast=False):
    header = operator_string(elemwise_add) + ', float' + (' , broadcast' if broadcast else "")
    print_header(header)
    for shape, nhid in sizes:
        net = FCWithSum(shape[1], nhid, elemwise_add)
        net.initialize()
        net.hybridize(static_alloc=True, static_shape=True)
        data0 = mx.np.random.uniform(size=shape, low=-1.0, high=1.0)
        data1 = mx.np.random.uniform(size=shape, low=-1.0, high=1.0)
        shape2 = (shape[0], nhid)
        if broadcast and not elemwise_add:
            # broadcast is allowed only for npi_add version
            shape2 = (1, 1)
        data2 = mx.np.random.uniform(size=shape2, low=-1.0, high=1.0)
        net.optimize_for(data0, data1, data2, backend='ONEDNN')
        measure(net, data0, data1, data2, shape, nhid)
    dump_graph_fn(net, operator_string(elemwise_add) + '_float')

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

def benchmark_int8(quantize_mode, quantize_granularity, elemwise_add, broadcast = False):
    header = operator_string(elemwise_add) + ', mode = ' + quantize_mode + \
             ', granularity = ' + quantize_granularity + (' , broadcast' if broadcast else "")
    print_header(header)
    for shape, nhid in sizes:
        net = FCWithSum(shape[1], nhid, elemwise_add)
        net.initialize()
        net.hybridize(static_alloc=True, static_shape=True)
        data0 = mx.np.random.uniform(size=shape, low=-1.0, high=1.0)
        data1 = mx.np.random.uniform(size=shape, low=-1.0, high=1.0)
        shape2 = (shape[0], nhid)
        if broadcast and not elemwise_add:
            # broadcast is allowed only for npi_add
            shape2 = (shape[0], 1)
        data2 = mx.np.random.uniform(size=shape2, low=-1.0, high=1.0)
        data = mx.gluon.data.ArrayDataset(data0, data1, data2)
        calib_data = mx.gluon.data.DataLoader(data, batch_size=1)
        net = quantization.quantize_net(net,
                                        device=mx.cpu(),
                                        exclude_layers=None,
                                        exclude_operators=None,
                                        calib_mode='naive',
                                        calib_data=calib_data,
                                        num_calib_batches=1,
                                        quantize_mode=quantize_mode,
                                        quantize_granularity=quantize_granularity
                                        )
        net.hybridize(static_alloc=True, static_shape=True)
        measure(net, data0, data1, data2, shape, nhid)
    dump_graph_fn(net, operator_string(elemwise_add) + \
                    '_' + str(quantize_mode) + '_' + str(quantize_granularity))

for elemwise_add in [True, False]:
    benchmark_float(elemwise_add)

for quantize_mode in ['smart', 'full']:
    for quantize_granularity in ['tensor-wise', 'channel-wise']:
        for elemwise_add in [True, False]:
            benchmark_int8(quantize_mode, quantize_granularity, elemwise_add)

# Benchmark FC + npi_add with broadcasted input
benchmark_float(False, True)

# Benchmark quantized FC + npi_add with broadcasted input
for quantize_mode in ['smart', 'full']:
    for quantize_granularity in ['tensor-wise', 'channel-wise']:
        benchmark_int8(quantize_mode, quantize_granularity, False, True)
