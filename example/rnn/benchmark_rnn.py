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
    def __init__(self, hidden_size, prefix=None, params=None):
        super(TestRNNLayer, self).__init__(prefix=prefix, params=params)
        self.cell = gluon.rnn.RNNCell(hidden_size, prefix='rnn_')

    def hybrid_forward(self, F, inputs, states):
        states = [states]
        out, states = F.contrib.foreach(self.cell, inputs, states)
        return out

def test_contrib_rnn(batch_size, input_size, hidden_size, seq_len, ctx):
    rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, input_size))
    states = mx.nd.normal(loc=0, scale=1, shape=(batch_size, hidden_size))
    num_batches = 20

    # Imperative
    layer1 = TestRNNLayer(hidden_size)
    layer1.initialize(ctx=ctx)

    # Hybridize
    layer2 = TestRNNLayer(hidden_size)
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

    #trainer = gluon.Trainer(layer1.collect_params(), 'sgd', {'learning_rate' : 0.03})
    tic = time.time()
    for i in range(num_batches):
        with mx.autograd.record():
            res1 = layer1(rnn_data, states)
        res1.backward()
        #trainer.step(batch_size)
    print("Imperative training takes " + str(time.time() - tic))

    #trainer = gluon.Trainer(layer2.collect_params(), 'sgd', {'learning_rate' : 0.03})
    tic = time.time()
    for i in range(num_batches):
        with mx.autograd.record():
            res2 = layer2(rnn_data, states)
        res2.backward()
        #trainer.step(batch_size)
    print("Hybrid training takes " + str(time.time() - tic))

    layer2.export("foreach_rnn")
    symnet = mx.symbol.load('foreach_rnn-symbol.json')
    # Inputs
    args1 = {}
    params = layer2.collect_params()
    for key in params.keys():
        args1[key] = params[key].data()
    args1['data0'] = rnn_data
    args1['data1'] = states
    # gradients for the backward of the foreach symbol
    args_grad1 = {}
    for key in args1.keys():
        args_grad1[key] = mx.nd.empty(args1[key].shape)
    exe = symnet.bind(ctx=ctx, args=args1, args_grad=args_grad1)
    tic = time.time()
    for i in range(num_batches):
        exe.forward(is_train=True)
        exe.backward(res2)
    print("Symbol training takes " + str(time.time() - tic))

if __name__ == '__main__':
    print("Benchmark in CPU (batch size: 1)")
    test_contrib_rnn(1, 100, 100, 100, mx.cpu(0))
    print("Benchmark in CPU (batch size: 32)")
    test_contrib_rnn(32, 100, 100, 100, mx.cpu(0))
    if len(get_gpus()) > 0:
        print("Benchmark in GPU (batch size: 1)")
        test_contrib_rnn(1, 100, 100, 100, mx.gpu(0))
        print("Benchmark in GPU (batch size: 32)")
        test_contrib_rnn(32, 100, 100, 100, mx.gpu(0))
