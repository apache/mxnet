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

class NDArraySoftmax(mx.operator.NDArrayOp):
    def __init__(self):
        super(NDArraySoftmax, self).__init__(False)
        self.fwd_kernel = None
        self.bwd_kernel = None

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
        if self.fwd_kernel is None:
            self.fwd_kernel = mx.rtc('softmax', [('x', x)], [('y', y)], """
int i = threadIdx.x + blockIdx.x*blockDim.x;
float max_x = x[i*x_dims[1]];
for (int j = 1; j < x_dims[1]; ++j) {
    if (max_x < x[i*x_dims[1]+j]) {
        max_x = x[i*x_dims[1]+j];
    }
}
float sum = 0.0f;
for (int j = 0; j < x_dims[1]; ++j) {
    sum += expf(x[i*x_dims[1]+j]-max_x);
}
for (int j = 0; j < x_dims[1]; ++j) {
    y[i*x_dims[1]+j] = expf(x[i*x_dims[1]+j]-max_x)/sum;
}
""")
        self.fwd_kernel.push([x], [y], (1, 1, 1), (x.shape[0], 1, 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        y = out_data[0]
        dx = in_grad[0]
        if self.bwd_kernel is None:
            self.bwd_kernel = mx.rtc('softmax_grad', [('y', y), ('l', l)], [('dx', dx)], """
int i = blockIdx.x;
int j = threadIdx.x;
int k = static_cast<int>(l[i]);
if (j == k) {
    dx[i*dx_dims[1]+j] = y[i*dx_dims[1]+j] - 1.0f;
} else {
    dx[i*dx_dims[1]+j] = y[i*dx_dims[1]+j];
}
""")
        self.bwd_kernel.push([y,l], [dx], (y.shape[0],1,1), (y.shape[1], 1, 1))

# define mlp

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
#mlp = mx.symbol.Softmax(data = fc3, name = 'mlp')
mysoftmax = NDArraySoftmax()
mlp = mysoftmax(data=fc3, name = 'softmax')

# data

train, val = get_mnist_iterator(batch_size=100, input_shape = (784,))

# train

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
    ctx = mx.gpu(0), symbol = mlp, num_epoch = 20,
    learning_rate = 0.1, momentum = 0.9, wd = 0.00001)

model.fit(X=train, eval_data=val)

