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


# This example demonstrates how to manipulate the learning rate of an optimizer
# in gluon. The example uses linear regression as a case study.

from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet import gluon

# Generate synthetic data.
X = np.random.randn(10000, 2)
Y = 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2 + .01 * np.random.normal(size=10000)

net = gluon.nn.Sequential()
# The output dimension is 1.
net.add(gluon.nn.Dense(1))
net.collect_params().initialize()
loss = gluon.loss.L2Loss()

# Initialize the learning rate as 0.1.
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        optimizer_params={'learning_rate': 0.1})
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24),
                                force_reinit=True)
train_data = mx.io.NDArrayIter(X, Y, batch_size=10, shuffle=True)

for epoch in range(5):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0]
        label = batch.label[0].reshape((-1, 1))
        with autograd.record():
            output = net(data)
            mse = loss(output, label)
        mse.backward()
        trainer.step(data.shape[0])
    # After the second epoch, decay the learning rate of the optimizer every
    # epoch.
    if epoch > 1:
        trainer.set_learning_rate(trainer.learning_rate * 0.9)
    print('Epoch:', epoch, 'Learning rate:', trainer.learning_rate)

for para_name, para_value in net.collect_params().items():
    # Print all the parameter values after training.
    print(para_name, para_value.data().asnumpy()[0])