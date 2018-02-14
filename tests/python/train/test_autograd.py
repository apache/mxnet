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
from __future__ import print_function

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.test_utils import get_mnist_ubyte
import numpy as np
import logging
from mxnet import autograd
logging.basicConfig(level=logging.DEBUG)

# define network

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(128, activation='relu', prefix='fc1_'))
    net.add(nn.Dense(64, activation='relu', prefix='fc2_'))
    net.add(nn.Dense(10, prefix='fc3_'))
    return net

get_mnist_ubyte()

batch_size = 100
train_data = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_data = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size, shuffle=True, flat=True, silent=False)

def score(net, ctx_list):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        datas = gluon.utils.split_and_load(batch.data[0], ctx_list, batch_axis=0)
        labels = gluon.utils.split_and_load(batch.label[0], ctx_list, batch_axis=0)
        outputs = []
        for x in datas:
            outputs.append(net(x))
        metric.update(labels, outputs)
    return metric.get()[1]

def train(net, epoch, ctx_list):
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx_list)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            datas = gluon.utils.split_and_load(batch.data[0], ctx_list, batch_axis=0)
            labels = gluon.utils.split_and_load(batch.label[0], ctx_list, batch_axis=0)
            outputs = []
            with autograd.record():
                for x, y in zip(datas, labels):
                    z = net(x)
                    L = loss(z, y)
                    L.backward()
                    outputs.append(z)
            trainer.step(batch.data[0].shape[0])
            metric.update(labels, outputs)
        name, acc = metric.get()
        metric.reset()
        print('training acc at epoch %d: %s=%f'%(i, name, acc))


def test_autograd():
    net1 = get_net()
    train(net1, 5, [mx.cpu(0), mx.cpu(1)])
    acc1 = score(net1, [mx.cpu(0)])
    acc2 = score(net1, [mx.cpu(0), mx.cpu(1)])
    assert acc1 > 0.95
    assert abs(acc1 - acc2) < 0.01
    net1.collect_params().save('mnist.params')

    net2 = get_net()
    net2.collect_params().load('mnist.params', ctx=[mx.cpu(0)])
    acc3 = score(net2, [mx.cpu(0)])
    assert abs(acc3 - acc1) < 0.0001


if __name__ == '__main__':
    test_autograd()
