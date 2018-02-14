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

# network
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
mlp = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

# data
train, val = get_mnist_iterator(batch_size=100, input_shape = (784,))

# monitor
def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)
mon = mx.mon.Monitor(100, norm_stat)

# train with monitor
logging.basicConfig(level=logging.DEBUG)
module = mx.module.Module(context=mx.cpu(), symbol=mlp)
module.fit(train_data=train, eval_data=val, monitor=mon, num_epoch=2,
           batch_end_callback = mx.callback.Speedometer(100, 100),
           optimizer_params=(('learning_rate', 0.1), ('momentum', 0.9), ('wd', 0.00001)))
