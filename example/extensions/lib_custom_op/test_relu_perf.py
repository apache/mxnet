#!/usr/bin/env python3

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

# coding: utf-8
# pylint: disable=arguments-differ

# This performance test benchmarks C++ custom operator implementation
# of noisy relu, a relu activation with noise, against Python implementation
# and MXNet built-in relu operator
#
# Please uncomment lines to activate each type of operator

import os
import time
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import mxnet as mx
import numpy as np

# load library
if (os.name=='posix'):
    path = os.path.abspath('librelu_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('librelu_lib.dll')
    mx.library.load(path)

# Python custom operator we want to benchmark against
class PyRelu(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.relu(in_data[0]) + mx.nd.random.normal(ctx=mx.gpu()))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0] * (mx.nd.relu(in_data[0]) / in_data[0]))

@mx.operator.register("pyrelu")
class PyReluProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(PyReluProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return PyRelu()

# ------------------------- start inference ------------------------- #
# we don't count first inference since MXNet does some housekeeping work
arr = mx.nd.uniform(shape=(100,100,100), ctx=mx.gpu())
mx.nd.waitall()
result = []

c = mx.sym.Variable('c')
#d = mx.sym.relu(c)
d = mx.sym.my_noisy_relu(c)
#d = mx.sym.Custom(c, op_type='pyrelu')
exe = d.bind(ctx=mx.gpu(), args={'c':arr})
exe.forward(c=arr)
for _ in range(100):
    t1 = time.time()
    exe.forward(c=arr)
    out = e.outputs[0].asnumpy()
    t2 = time.time()
    result.append((t2 - t1) * 1000)

print("Average is %s ms" % np.average(result))

# ------------------------- start training ------------------------- #
print("Start training benchmark")
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=512)
#act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
act1 = mx.symbol.my_noisy_relu(fc1)
#act1 = mx.sym.Custom(fc1, op_type='pyrelu')
fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=512)
#act2= mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
act2 = mx.symbol.my_noisy_relu(fc2)
#act2 = mx.sym.Custom(fc2, op_type='pyrelu')
fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
output = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

print("Preparing data...")
mnist_data = mx.test_utils.get_mnist()
X = np.concatenate([mnist_data['train_data'], mnist_data['test_data']])
Y = np.concatenate([mnist_data['train_label'], mnist_data['test_label']])
X = X.reshape((X.shape[0], -1)).astype(np.float32) * 255
X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=200, label_name=output.name + "_label")
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=200, label_name=output.name + "_label")

print("\nTesting with %s \n" % output.name)
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
mod = mx.mod.Module(context = ctx,symbol = output, label_names = [output.name + "_label"])
mod.fit(
    train_data=train_iter,
    eval_data=test_iter,
    batch_end_callback = mx.callback.Speedometer(200, 200),  # Logging module to print out progress
    num_epoch = 10,
    optimizer_params = {'learning_rate': 0.1, 'momentum': 0.9, 'wd': 0.00001}
)

print('Accuracy for %s:'%output.name, mod.score(test_iter, mx.metric.Accuracy())[0][1]*100, '%\n')
