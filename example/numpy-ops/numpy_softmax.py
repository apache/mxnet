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
import mxnet as mx
from mxnet.test_utils import get_mnist_iterator
import numpy as np
import logging


class NumpySoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(NumpySoftmax, self).__init__(False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        l = l.reshape((l.size,)).astype(np.int)
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = y
        dx[np.arange(l.shape[0]), l] -= 1.0

# define mlp

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
#mlp = mx.symbol.Softmax(data = fc3, name = 'mlp')
mysoftmax = NumpySoftmax()
mlp = mysoftmax(data=fc3, name = 'softmax')

# data

train, val = get_mnist_iterator(batch_size=100, input_shape = (784,))

# train

logging.basicConfig(level=logging.DEBUG)

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
context=mx.cpu()
# Uncomment this line to train on GPU instead of CPU
# context=mx.gpu(0)

mod = mx.mod.Module(mlp, context=context)

mod.fit(train_data=train, eval_data=val, optimizer='sgd',
    optimizer_params={'learning_rate':0.1, 'momentum': 0.9, 'wd': 0.00001},
    num_epoch=10, batch_end_callback=mx.callback.Speedometer(100, 100))
