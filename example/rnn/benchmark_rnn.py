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

import subprocess
import mxnet as mx
from mxnet import gluon
import time
import copy

def get_gpus():
    """
    return a list of GPUs
    """
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return []
    return range(len([i for i in re.split('\n') if 'GPU' in i]))

class TestRNNLayer(gluon.HybridBlock):
    def __init__(self, cell, prefix=None, params=None):
        super(TestRNNLayer, self).__init__(prefix=prefix, params=params)
        self.cell = cell

    def hybrid_forward(self, F, inputs, states):
        out, states = F.contrib.foreach(self.cell, inputs, states)
        return out

def test_contrib_rnn(cell, rnn_data, states):
    ctx = rnn_data.context
    num_batches = 20
    cell1 = copy.deepcopy(cell)
    cell2 = copy.deepcopy(cell)

    # Imperative
    layer1 = TestRNNLayer(cell1)
    layer1.initialize(ctx=ctx)

    # Hybridize
    layer2 = TestRNNLayer(cell2)
    layer2.initialize(ctx=ctx)
    layer2.hybridize()

    tic = time.time()
    for i in range(num_batches):
        res1 = layer1(rnn_data, states)
        mx.nd.waitall()
    print("Imperative inference takes " + str(time.time() - tic))

    tic = time.time()
    for i in range(num_batches):
        res2 = layer2(rnn_data, states)
        mx.nd.waitall()
    print("Hybrid inference takes " + str(time.time() - tic))

    tic = time.time()
    for i in range(num_batches):
        with mx.autograd.record():
            res1 = layer1(rnn_data, states)
        res1.backward()
    print("Imperative training takes " + str(time.time() - tic))

    tic = time.time()
    for i in range(num_batches):
        with mx.autograd.record():
            res2 = layer2(rnn_data, states)
        res2.backward()
    print("Hybrid training takes " + str(time.time() - tic))

    layer2.export("foreach_rnn")
    symnet = mx.symbol.load('foreach_rnn-symbol.json')
    # Inputs
    args1 = {}
    params = layer2.collect_params()
    for key in params.keys():
        args1[key] = params[key].data()
    args1['data0'] = rnn_data
    for i in range(len(states)):
        args1['data' + str(i + 1)] = states[i]
    # gradients for the backward of the foreach symbol
    args_grad1 = {}
    for key in args1.keys():
        args_grad1[key] = mx.nd.empty(args1[key].shape, ctx=ctx)
    exe = symnet.bind(ctx=ctx, args=args1, args_grad=args_grad1)
    tic = time.time()
    for i in range(num_batches):
        exe.forward(is_train=True)
        exe.backward(res2)
    print("Symbol training takes " + str(time.time() - tic))

if __name__ == '__main__':
    ndim = 500
    seq_len = 100
    batch_sizes = [1, 32]
    cells = [gluon.rnn.RNNCell(ndim, prefix='rnn_'),
             gluon.rnn.LSTMCell(ndim, prefix='rnn_')]
    ctxs = [mx.cpu(0), mx.gpu(0)]
    for ctx in ctxs:
        for batch_size in batch_sizes:
            for cell in cells:
                if len(get_gpus()) == 0 and ctx == mx.gpu(0):
                    continue

                if isinstance(cell, gluon.rnn.RNNCell):
                    rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, ndim),
                                            ctx=mx.cpu(0))
                    states = []
                    states.append(mx.nd.normal(loc=0, scale=1, shape=(batch_size, ndim),
                                               ctx=mx.cpu(0)))
                elif isinstance(cell, gluon.rnn.LSTMCell):
                    rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, ndim),
                                            ctx=mx.cpu(0))
                    states = []
                    states.append(mx.nd.normal(loc=0, scale=1, shape=(batch_size, ndim),
                                               ctx=mx.cpu(0)))
                    states.append(mx.nd.normal(loc=0, scale=1, shape=(batch_size, ndim),
                                               ctx=mx.cpu(0)))
                print("Benchmark {} in CPU (batch size: {})".format(cell._alias(), batch_size))
                test_contrib_rnn(cell, rnn_data, states)
