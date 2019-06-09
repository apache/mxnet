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
import tempfile
import time
import multiprocessing as mp
import unittest
import random
import mxnet as mx
import numpy as np
import unittest
import math
from nose.tools import assert_raises
from mxnet.test_utils import check_consistency, set_default_context, assert_almost_equal
from mxnet.base import MXNetError
from mxnet import autograd
from numpy.testing import assert_allclose
from mxnet.test_utils import rand_ndarray


curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown, assert_raises_cudnn_not_satisfied
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

@with_seed()
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
@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_lstmp():
    hidden_size, projection_size = 3, 2
    rtol, atol = 1e-2, 1e-2
    batch_size, seq_len = 7, 11
    input_size = 5
    lstm_input = mx.nd.uniform(shape=(seq_len, batch_size, input_size), ctx=mx.gpu(0))
    shapes = {'i2h_weight': (hidden_size*4, input_size),
              'h2h_weight': (hidden_size*4, projection_size),
              'i2h_bias': (hidden_size*4,),
              'h2h_bias': (hidden_size*4,),
              'h2r_weight': (projection_size, hidden_size)}
    weights = {k: rand_ndarray(v) for k, v in shapes.items()}
    lstm_layer = gluon.rnn.LSTM(hidden_size, projection_size=projection_size,
                                input_size=input_size, prefix='lstm0_')
    lstm_cell = gluon.contrib.rnn.LSTMPCell(hidden_size=hidden_size,
                                            projection_size=projection_size,
                                            input_size=input_size,
                                            prefix='lstm0_l0_')
    lstm_layer.initialize(ctx=mx.gpu(0))
    lstm_cell.initialize(ctx=mx.gpu(0))
    layer_params = lstm_layer.collect_params()
    cell_params = lstm_cell.collect_params()
    for k, v in weights.items():
        layer_params['lstm0_l0_'+k].set_data(v.copy())
        cell_params['lstm0_l0_'+k].set_data(v.copy())
    with autograd.record():
        layer_output = lstm_layer(lstm_input.copy())
        cell_output = lstm_cell.unroll(seq_len, lstm_input.copy(), layout='TNC',
                                       merge_outputs=True)[0]
    assert_almost_equal(layer_output.asnumpy(), cell_output.asnumpy(), rtol=rtol, atol=atol)
    layer_output.backward()
    cell_output.backward()
    for k, v in weights.items():
        layer_grad = layer_params['lstm0_l0_'+k].grad()
        cell_grad = cell_params['lstm0_l0_'+k].grad()
        print('checking gradient for {}'.format('lstm0_l0_'+k))
        assert_almost_equal(layer_grad.asnumpy(), cell_grad.asnumpy(),
                            rtol=rtol, atol=atol)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, projection_size=5), mx.nd.ones((8, 3, 20)))
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, projection_size=5, bidirectional=True), mx.nd.ones((8, 3, 20)), [mx.nd.ones((4, 3, 5)), mx.nd.ones((4, 3, 10))])

    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, dropout=0.5, projection_size=5), mx.nd.ones((8, 3, 20)),
                            run_only=True)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, bidirectional=True, dropout=0.5, projection_size=5),
                            mx.nd.ones((8, 3, 20)),
                            [mx.nd.ones((4, 3, 5)), mx.nd.ones((4, 3, 10))], run_only=True)


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_lstm_clip():
    hidden_size, projection_size = 4096, 2048
    batch_size, seq_len = 32, 80
    input_size = 50
    clip_min, clip_max, clip_nan = -5, 5, True
    lstm_input = mx.nd.uniform(shape=(seq_len, batch_size, input_size), ctx=mx.gpu(0))
    lstm_states = [mx.nd.uniform(shape=(2, batch_size, projection_size), ctx=mx.gpu(0)),
                   mx.nd.uniform(shape=(2, batch_size, hidden_size), ctx=mx.gpu(0))]
    lstm_layer = gluon.rnn.LSTM(hidden_size, projection_size=projection_size,
                                input_size=input_size, prefix='lstm0_',
                                bidirectional=True,
                                state_clip_min=clip_min,
                                state_clip_max=clip_max,
                                state_clip_nan=clip_nan)
    lstm_layer.initialize(ctx=mx.gpu(0))
    with autograd.record():
        _, layer_output_states = lstm_layer(lstm_input, lstm_states)
    cell_states = layer_output_states[0].asnumpy()
    assert (cell_states >= clip_min).all() and (cell_states <= clip_max).all()
    assert not np.isnan(cell_states).any()


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnn_layer():
    check_rnn_layer(gluon.rnn.RNN(100, num_layers=3))
    check_rnn_layer(gluon.rnn.RNN(100, activation='tanh', num_layers=3))
    check_rnn_layer(gluon.rnn.LSTM(100, num_layers=3))
    check_rnn_layer(gluon.rnn.GRU(100, num_layers=3))

    check_rnn_layer(gluon.rnn.LSTM(100, num_layers=3, bidirectional=True))
    check_rnn_layer_w_rand_inputs(gluon.rnn.LSTM(100, num_layers=3, bidirectional=True))


def check_layer_bidirectional(size, in_size, proj_size):
    class RefBiLSTM(gluon.Block):
        def __init__(self, size, proj_size, **kwargs):
            super(RefBiLSTM, self).__init__(**kwargs)
            with self.name_scope():
                self._lstm_fwd = gluon.rnn.LSTM(size, projection_size=proj_size, bidirectional=False, prefix='l0')
                self._lstm_bwd = gluon.rnn.LSTM(size, projection_size=proj_size, bidirectional=False, prefix='r0')

        def forward(self, inpt):
            fwd = self._lstm_fwd(inpt)
            bwd_inpt = nd.flip(inpt, 0)
            bwd = self._lstm_bwd(bwd_inpt)
            bwd = nd.flip(bwd, 0)
            return nd.concat(fwd, bwd, dim=2)
    weights = {}
    for d in ['l', 'r']:
        weights['lstm_{}0_i2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, in_size))
        if proj_size:
            weights['lstm_{}0_h2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, proj_size))
            weights['lstm_{}0_h2r_weight'.format(d)] = mx.random.uniform(shape=(proj_size, size))
        else:
            weights['lstm_{}0_h2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, size))
        weights['lstm_{}0_i2h_bias'.format(d)] = mx.random.uniform(shape=(size*4,))
        weights['lstm_{}0_h2h_bias'.format(d)] = mx.random.uniform(shape=(size*4,))

    net = gluon.rnn.LSTM(size, projection_size=proj_size, bidirectional=True, prefix='lstm_')
    ref_net = RefBiLSTM(size, proj_size, prefix='lstm_')
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    for k in weights:
        net_params[k].set_data(weights[k])
        ref_net_params[k.replace('l0', 'l0l0').replace('r0', 'r0l0')].set_data(weights[k])

    data = mx.random.uniform(shape=(11, 10, in_size))
    assert_allclose(net(data).asnumpy(), ref_net(data).asnumpy())

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_layer_bidirectional():
    check_layer_bidirectional(7, 5, 0)

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_layer_bidirectional_proj():
    check_layer_bidirectional(7, 5, 3)


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnn_layer_begin_state_type():
    fake_data = nd.random.uniform(shape=(3, 5, 7), dtype='float16')
    modeling_layer = gluon.rnn.LSTM(hidden_size=11, num_layers=2, dropout=0.2, bidirectional=True)
    modeling_layer.cast('float16')
    modeling_layer.initialize()
    modeling_layer(fake_data)


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
    for check_isfinite in [True, False]:
        x1 = mx.nd.ones((3,3), ctx=mx.gpu(0))
        x2 = mx.nd.ones((4,4), ctx=mx.cpu(0))
        norm = gluon.utils.clip_global_norm([x1, x2], 1.0, check_isfinite=check_isfinite)
        if check_isfinite:
            assert norm == 5.0
        else:
            assert norm.asscalar() == 5.0
        assert_almost_equal(x1.asnumpy(), np.ones((3, 3)) / 5)
        assert_almost_equal(x2.asnumpy(), np.ones((4, 4)) / 5)


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

@with_seed()
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


@with_seed()
def test_symbol_block_fp16():
    # Test case to verify if initializing the SymbolBlock from a model with params
    # other than fp32 param dtype.

    # 1. Load a resnet model, cast it to fp16 and export
    tmp = tempfile.mkdtemp()
    tmpfile = os.path.join(tmp, 'resnet34_fp16')
    ctx = mx.gpu(0)

    net_fp32 = mx.gluon.model_zoo.vision.resnet34_v2(pretrained=True, ctx=ctx, root=tmp)
    net_fp32.cast('float16')
    net_fp32.hybridize()
    data = mx.nd.zeros((1,3,224,224), dtype='float16', ctx=ctx)
    net_fp32.forward(data)
    net_fp32.export(tmpfile, 0)

    # 2. Load the saved model and verify if all the params are loaded correctly.
    # and choose one of the param to verify the type if fp16.
    sm = mx.sym.load(tmpfile + '-symbol.json')
    inputs = mx.sym.var('data', dtype='float16')
    net_fp16 = mx.gluon.SymbolBlock(sm, inputs)
    net_fp16.collect_params().load(tmpfile + '-0000.params', ctx=ctx)
    # 3. Get a conv layer's weight parameter name. Conv layer's weight param is
    # expected to be of dtype casted, fp16.
    for param_name in net_fp16.params.keys():
        if 'conv' in param_name and 'weight' in param_name:
            break
    assert np.dtype(net_fp16.params[param_name].dtype) == np.dtype(np.float16)


@with_seed()
def test_large_models():
    ctx = default_context()
    # Create model
    net = gluon.nn.HybridSequential()

    largest_num_features = 256
    with net.name_scope():
        net.add(nn.Conv2D(largest_num_features, 3))

    net.hybridize()
    net.initialize(mx.init.Normal(sigma=0.01), ctx=ctx)

    # Compute the height (=width) of the square tensor of the given size in bytes
    def tensor_size(big_tensor_bytes):
        bytes_per_float = 4
        sz = int(math.sqrt(big_tensor_bytes / largest_num_features / bytes_per_float))
        return (sz // 100) * 100

    # The idea is to create models with large tensors of (say) 20% of the total memory.
    # This in the past has given cudnnFind() trouble when it needed to allocate similar I/O's
    # from the area carved out by the MXNET_GPU_MEM_POOL_RESERVE setting (by default 5%).
    (free_mem_bytes, total_mem_bytes) = mx.context.gpu_memory_info(ctx.device_id)
    start_size = tensor_size(0.20 * total_mem_bytes)
    num_trials = 10
    sys.stderr.write(' testing global memory of size {} ... '.format(total_mem_bytes))
    sys.stderr.flush()
    for i in range(num_trials):
        sz = start_size - 10 * i
        (height, width) = (sz,sz)
        sys.stderr.write(" {}x{} ".format(height,width))
        sys.stderr.flush()
        data_in = nd.random_uniform(low=0, high=255, shape=(1, 3, height, width),
                                    ctx=ctx, dtype="float32")
        # Evaluate model
        net(data_in).asnumpy()


if __name__ == '__main__':
    import nose
    nose.runmodule()
