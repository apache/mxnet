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

from __future__ import print_function
import sys
import os
import time
import multiprocessing as mp
import unittest
import mxnet as mx
import numpy as np
import unittest
from nose.tools import assert_raises
from mxnet.test_utils import check_consistency, set_default_context, assert_almost_equal
from mxnet.base import MXNetError
from mxnet import autograd
from numpy.testing import assert_allclose

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown, assert_raises_cudnn_disabled
from test_gluon import *
from test_loss import *
from test_gluon_rnn import *

set_default_context(mx.gpu(0))

def check_rnn_layer(layer):
    layer.collect_params().initialize(ctx=[mx.cpu(0), mx.gpu(0)])
    with mx.gpu(0):
        x = mx.nd.ones((10, 16, 30))
        states = layer.begin_state(16)
        go, gs = layer(x, states)

    with mx.cpu(0):
        x = mx.nd.ones((10, 16, 30))
        states = layer.begin_state(16)
        co, cs = layer(x, states)

    # atol of 1e-6 required, as exposed by seed 2124685726
    assert_almost_equal(go.asnumpy(), co.asnumpy(), rtol=1e-2, atol=1e-6)
    for g, c in zip(gs, cs):
        assert_almost_equal(g.asnumpy(), c.asnumpy(), rtol=1e-2, atol=1e-6)


def check_rnn_layer_w_rand_inputs(layer):
    layer.collect_params().initialize(ctx=[mx.cpu(0), mx.gpu(0)])
    x = mx.nd.uniform(shape=(10, 16, 30))
    with mx.gpu(0):
        x = x.copyto(mx.gpu(0))
        states = layer.begin_state(16)
        go, gs = layer(x, states)

    with mx.cpu(0):
        x = x.copyto(mx.cpu(0))
        states = layer.begin_state(16)
        co, cs = layer(x, states)

    assert_almost_equal(go.asnumpy(), co.asnumpy(), rtol=1e-2, atol=1e-6)
    for g, c in zip(gs, cs):
        assert_almost_equal(g.asnumpy(), c.asnumpy(), rtol=1e-2, atol=1e-6)


@with_seed()
@assert_raises_cudnn_disabled()
def test_rnn_layer():
    check_rnn_layer(gluon.rnn.RNN(100, num_layers=3))
    check_rnn_layer(gluon.rnn.RNN(100, activation='tanh', num_layers=3))
    check_rnn_layer(gluon.rnn.LSTM(100, num_layers=3))
    check_rnn_layer(gluon.rnn.GRU(100, num_layers=3))

    check_rnn_layer(gluon.rnn.LSTM(100, num_layers=3, bidirectional=True))
    check_rnn_layer_w_rand_inputs(gluon.rnn.LSTM(100, num_layers=3, bidirectional=True))


@with_seed()
def test_gluon_ctc_consistency():
    loss = mx.gluon.loss.CTCLoss()
    data = mx.nd.arange(0, 4, repeat=40, ctx=mx.gpu(0)).reshape((2,20,4)).flip(axis=0)
    cpu_label = mx.nd.array([[2,1,-1,-1],[3,2,2,-1]], ctx=mx.cpu(0))
    gpu_label = mx.nd.array([[2,1,-1,-1],[3,2,2,-1]], ctx=mx.gpu(0))

    cpu_data = data.copy().as_in_context(mx.cpu(0))
    cpu_data.attach_grad()
    with mx.autograd.record():
        l_cpu = loss(cpu_data, cpu_label)
        l_cpu.backward()

    gpu_data = data.copyto(mx.gpu(0))
    gpu_data.attach_grad()
    with mx.autograd.record():
        l_gpu = loss(gpu_data, gpu_label)
        l_gpu.backward()

    assert_almost_equal(cpu_data.grad.asnumpy(), gpu_data.grad.asnumpy(), atol=1e-3, rtol=1e-3)


@with_seed()
def test_global_norm_clip_multi_device():
    x1 = mx.nd.ones((3,3), ctx=mx.gpu(0))
    x2 = mx.nd.ones((4,4), ctx=mx.cpu(0))
    norm = gluon.utils.clip_global_norm([x1, x2], 1.0)
    assert norm == 5.0
    assert_almost_equal(x1.asnumpy(), np.ones((3,3))/5)
    assert_almost_equal(x2.asnumpy(), np.ones((4,4))/5)


def _check_batchnorm_result(input, num_devices=1, cuda=False):
    from mxnet.gluon.utils import split_and_load
    def _find_bn(module):
        if isinstance(module, (mx.gluon.nn.BatchNorm, mx.gluon.contrib.nn.SyncBatchNorm)):
            return module
        elif isinstance(module.module, (mx.gluon.nn.BatchNorm, mx.gluon.contrib.nn.SyncBatchNorm)):
            return module.module

        raise RuntimeError('BN not found')

    def _syncParameters(bn1, bn2, ctx):
        ctx = input.context
        bn2.gamma.set_data(bn1.gamma.data(ctx))
        bn2.beta.set_data(bn1.beta.data(ctx))
        bn2.running_mean.set_data(bn1.running_mean.data(ctx))
        bn2.running_var.set_data(bn1.running_var.data(ctx))

    input1 = input.copy()
    input2 = input.copy()

    if cuda:
        input1 = input.as_in_context(mx.gpu(0))
        ctx_list = [mx.gpu(i) for i in range(num_devices)]
    else:
        ctx_list = [mx.cpu(0) for _ in range(num_devices)]

    nch = input.shape[1]
    bn1 = mx.gluon.nn.BatchNorm(in_channels=nch)
    bn2 = mx.gluon.contrib.nn.SyncBatchNorm(in_channels=nch, num_devices=num_devices)

    bn1.initialize(ctx=ctx_list[0])
    bn2.initialize(ctx=ctx_list)

    # using the same values for gamma and beta
    #_syncParameters(_find_bn(bn1), _find_bn(bn2), ctx_list[0])

    input1.attach_grad()
    inputs2 = split_and_load(input2, ctx_list, batch_axis=0)
    for xi in inputs2:
        xi.attach_grad()

    with mx.autograd.record():
        output1 = bn1(input1)
        output2  = [bn2(xi) for xi in inputs2]
        loss1 = (output1 ** 2).sum()
        loss2 = [(output ** 2).sum() for output in output2]
        mx.autograd.backward(loss1)
        mx.autograd.backward(loss2)

    output2 = mx.nd.concat(*[output.as_in_context(input.context) for output in output2], dim=0)
    # assert forwarding
    assert_almost_equal(input1.asnumpy(), input2.asnumpy(), atol=1e-3, rtol=1e-3)
    assert_almost_equal(output1.asnumpy(), output2.asnumpy(), atol=1e-3, rtol=1e-3)
    assert_almost_equal(_find_bn(bn1).running_mean.data(ctx_list[0]).asnumpy(),
                        _find_bn(bn2).running_mean.data(ctx_list[0]).asnumpy(),
                        atol=1e-3, rtol=1e-3)
    assert_almost_equal(_find_bn(bn1).running_var.data(ctx_list[0]).asnumpy(),
                        _find_bn(bn2).running_var.data(ctx_list[0]).asnumpy(),
                        atol=1e-3, rtol=1e-3)
    input2grad = mx.nd.concat(*[output.grad.as_in_context(input.context) for output in inputs2], dim=0)
    assert_almost_equal(input1.grad.asnumpy(), input2grad.asnumpy(), atol=1e-3, rtol=1e-3)


def test_sync_batchnorm():
    def get_num_devices():
        for i in range(100):
            try:
                mx.nd.zeros((1,), ctx=mx.gpu(i))
            except:
                return i
    # no need to use SyncBN with 1 gpu
    if get_num_devices() < 2:
        return
    ndev = 2
    # check with unsync version
    for i in range(10):
        _check_batchnorm_result(mx.nd.random.uniform(shape=(4, 1, 4, 4)),
                                num_devices=ndev, cuda=True)

if __name__ == '__main__':
    import nose
    nose.runmodule()
