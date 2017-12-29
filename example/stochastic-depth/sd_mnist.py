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

################################################################################
# A sanity check mainly for debugging purpose. See sd_cifar10.py for a non-trivial
# example of stochastic depth on cifar10.
################################################################################

import os
import sys
import mxnet as mx
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_data

import sd_module

def get_conv(
    name,
    data,
    num_filter,
    kernel,
    stride,
    pad,
    with_relu,
    bn_momentum
):
    conv = mx.symbol.Convolution(
        name=name,
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        no_bias=True
    )
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv,
        fix_gamma=False,
        momentum=bn_momentum,
        # Same with https://github.com/soumith/cudnn.torch/blob/master/BatchNormalization.lua
        # cuDNN v5 don't allow a small eps of 1e-5
        eps=2e-5
    )
    return (
        # It's better to remove ReLU here
        # https://github.com/gcr/torch-residual-networks
        mx.symbol.Activation(name=name + '_relu', data=bn, act_type='relu')
        if with_relu else bn
    )

death_rates = [0.3]
contexts = [mx.context.cpu()]

data = mx.symbol.Variable('data')
conv = get_conv(
    name='conv0',
    data=data,
    num_filter=16,
    kernel=(3, 3),
    stride=(1, 1),
    pad=(1, 1),
    with_relu=True,
    bn_momentum=0.9
)

base_mod = mx.mod.Module(conv, label_names=None, context=contexts)
mod_seq = mx.mod.SequentialModule()
mod_seq.add(base_mod)

for i in range(len(death_rates)):
    conv = get_conv(
        name='conv0_%d' % i,
        data=mx.sym.Variable('data_%d' % i),
        num_filter=16,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        with_relu=True,
        bn_momentum=0.9
    )
    conv = get_conv(
        name='conv1_%d' % i,
        data=conv,
        num_filter=16,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        with_relu=False,
        bn_momentum=0.9
    )
    mod = sd_module.StochasticDepthModule(conv, data_names=['data_%d' % i],
                                          context=contexts, death_rate=death_rates[i])
    mod_seq.add(mod, auto_wiring=True)

act = mx.sym.Activation(mx.sym.Variable('data_final'), act_type='relu')
flat = mx.sym.Flatten(act)
pred = mx.sym.FullyConnected(flat, num_hidden=10)
softmax = mx.sym.SoftmaxOutput(pred, name='softmax')
mod_seq.add(mx.mod.Module(softmax, context=contexts, data_names=['data_final']),
            auto_wiring=True, take_labels=True)


n_epoch = 2
batch_size = 100

basedir = os.path.dirname(__file__)
get_data.get_mnist(os.path.join(basedir, "data"))

train = mx.io.MNISTIter(
        image=os.path.join(basedir, "data", "train-images-idx3-ubyte"),
        label=os.path.join(basedir, "data", "train-labels-idx1-ubyte"),
        input_shape=(1, 28, 28), flat=False,
        batch_size=batch_size, shuffle=True, silent=False, seed=10)
val = mx.io.MNISTIter(
        image=os.path.join(basedir, "data", "t10k-images-idx3-ubyte"),
        label=os.path.join(basedir, "data", "t10k-labels-idx1-ubyte"),
        input_shape=(1, 28, 28), flat=False,
        batch_size=batch_size, shuffle=True, silent=False)

logging.basicConfig(level=logging.DEBUG)
mod_seq.fit(train, val, optimizer_params={'learning_rate': 0.01, 'momentum': 0.9},
            num_epoch=n_epoch, batch_end_callback=mx.callback.Speedometer(batch_size, 10))
