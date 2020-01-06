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

import mxnet as mx
from mxnet.gluon import nn


def simple_forward():
    ctx = mx.gpu()
    mx.profiler.set_config(profile_all=True)
    mx.profiler.set_state('run')

    # define simple gluon network with random weights
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    input = mx.nd.zeros((128,), ctx=ctx)
    predictions = net(input)
    print('Ran simple NN forward, results:')
    print(predictions.asnumpy())


if __name__ == '__main__':
    simple_forward()
