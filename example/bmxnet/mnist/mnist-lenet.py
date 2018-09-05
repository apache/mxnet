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

import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

# Parse CLI arguments

parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Train on GPU with CUDA')
parser.add_argument('--hybridize', action='store_true', default=False,
                    help='Train in symbolic mode')
parser.add_argument('--bits', type=int, default=1,
                    help='Number of bits for binarization/quantization')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
opt = parser.parse_args()

num_channels_conv = 64
act = 'tanh'
num_fc = 1000
num_outputs = 10

# define network
net = nn.HybridSequential()
with net.name_scope():
    if opt.bits == 1:
        net.add(gluon.nn.Conv2D(channels=num_channels_conv, kernel_size=5))
        net.add(gluon.nn.Activation(activation=act))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))

        net.add(gluon.nn.BConv2D(channels=num_channels_conv, kernel_size=5))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())

        net.add(gluon.nn.BDense(num_fc))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.Activation(activation=act))

        net.add(gluon.nn.Dense(num_outputs))
    elif opt.bits < 32:
        raise RuntimeError("Quantization not yet supported")
    else:
        net.add(gluon.nn.Conv2D(channels=num_channels_conv, kernel_size=5))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.Activation(activation=act))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        net.add(gluon.nn.Conv2D(channels=num_channels_conv, kernel_size=5))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.Activation(activation=act))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())

        net.add(gluon.nn.Dense(num_fc))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.Activation(activation=act))

        net.add(gluon.nn.Dense(num_outputs))


# data
def transform(data, label):
    return mx.nd.transpose(data.astype(np.float32), (2, 0, 1))/255, label.astype(np.float32)


train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=True, transform=transform),
    batch_size=opt.batch_size, shuffle=True, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=False, transform=transform),
    batch_size=opt.batch_size, shuffle=False)


def dummy_data(ctx):
    return [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in ([opt.batch_size, 1, 28, 28], [100])]


# train
def test(ctx):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])

    return metric.get()


def train(epochs, ctx):
    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # Trainer is for updating parameters with gradient.
    trainer = gluon.Trainer(net.collect_params(), 'adam')
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # do forward pass with dummy data without backwards pass to initialize binary layers
    with autograd.record():
        data, label = dummy_data(ctx)
        output = net(data)
        L = loss(output, label)

    if opt.hybridize:
        net.hybridize()

    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            # Copy data to ctx if necessary
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])
            # update metric at last.
            metric.update([label], [output])

            if i % opt.log_interval == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f'%(epoch, i, name, acc))

        name, acc = metric.get()
        print('[Epoch %d] Training: %s=%f'%(epoch, name, acc))

        name, val_acc = test(ctx)
        print('[Epoch %d] Validation: %s=%f'%(epoch, name, val_acc))

    if not opt.hybridize:
        net.hybridize()
        with autograd.record():
            data, label = dummy_data(ctx)
            output = net(data)
            L = loss(output, label)
    net.export("mnist-lenet-{}-{}-bit".format("symbolic" if opt.hybridize else "gluon", opt.bits), epoch=1)


if __name__ == '__main__':
    if opt.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()
    train(opt.epochs, ctx)
