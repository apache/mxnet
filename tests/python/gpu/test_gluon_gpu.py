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

import sys
import os
import time
import random
import mxnet as mx
import multiprocessing as mp
from mxnet.test_utils import check_consistency, set_default_device, assert_almost_equal, rand_ndarray, environment
import numpy as _np
import math
from mxnet import autograd
import pytest

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import assert_raises_cudnn_not_satisfied, run_in_spawned_process, random_seed
from test_gluon import *
from test_loss import *
from test_numpy_loss import *
from test_gluon_rnn import *

set_default_device(mx.gpu(0))


def check_rnn_layer(layer):
    layer.initialize(device=[mx.cpu(0), mx.gpu(0)])
    with mx.gpu(0):
        x = mx.np.ones((10, 16, 30))
        states = layer.begin_state(16)
        go, gs = layer(x, states)

    with mx.cpu(0):
        x = mx.np.ones((10, 16, 30))
        states = layer.begin_state(16)
        co, cs = layer(x, states)

    assert_almost_equal(go, co)
    for g, c in zip(gs, cs):
        assert_almost_equal(g, c)


def check_rnn_layer_w_rand_inputs(layer):
    layer.initialize(device=[mx.cpu(0), mx.gpu(0)])
    x = mx.np.random.uniform(size=(10, 16, 30))
    with mx.gpu(0):
        x = x.copyto(mx.gpu(0))
        states = layer.begin_state(16)
        go, gs = layer(x, states)

    with mx.cpu(0):
        x = x.copyto(mx.cpu(0))
        states = layer.begin_state(16)
        co, cs = layer(x, states)

    assert_almost_equal(go, co)
    for g, c in zip(gs, cs):
        assert_almost_equal(g, c)


@mx.util.use_np
@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_lstmp():
    hidden_size, projection_size = 3, 2
    rtol, atol = 1e-2, 1e-2
    batch_size, seq_len = 7, 11
    input_size = 5
    device = mx.gpu(0)
    lstm_input = mx.np.random.uniform(
        size=(seq_len, batch_size, input_size), device=device)
    shapes = {'i2h_weight': (hidden_size * 4, input_size),
              'h2h_weight': (hidden_size * 4, projection_size),
              'i2h_bias': (hidden_size * 4,),
              'h2h_bias': (hidden_size * 4,),
              'h2r_weight': (projection_size, hidden_size)}
    weights = {k: rand_ndarray(v).as_np_ndarray() for k, v in shapes.items()}
    lstm_layer = gluon.rnn.LSTM(hidden_size, projection_size=projection_size,
                                input_size=input_size)
    lstm_cell = gluon.rnn.LSTMPCell(hidden_size=hidden_size,
                                    projection_size=projection_size,
                                    input_size=input_size)
    lstm_layer.initialize(device=device)
    lstm_cell.initialize(device=device)
    layer_params = lstm_layer.collect_params()
    cell_params = lstm_cell.collect_params()
    params = (weights['{}_{}'.format(g, t)].reshape(-1)
              for t in ['weight', 'bias']
              for g in ['i2h', 'h2h', 'h2r']
              if g != 'h2r' or t != 'bias')

    net_params_concat = mx.np.concatenate(params)
    layer_params['rnn_param'].set_data(net_params_concat)
    for k, v in weights.items():
        cell_params[k].set_data(v)
    with autograd.record():
        layer_output = lstm_layer(lstm_input.copy())
        cell_output = lstm_cell.unroll(seq_len, lstm_input.copy(), layout='TNC',
                                       merge_outputs=True)[0]

    assert_almost_equal(layer_output, cell_output, rtol=rtol, atol=atol)
    layer_output.backward()
    cell_output.backward()
    layer_params_split = split_rnn_params(layer_params['rnn_param'].grad(),\
        'lstm', 1, input_size, hidden_size, False, projection_size=projection_size)
    for k, _ in weights.items():
        layer_grad = layer_params_split['l0_' + k]
        cell_grad = cell_params[k].grad()
        print('checking gradient for {}'.format('lstm0_l0_' + k))
        assert_almost_equal(layer_grad, cell_grad, rtol=rtol, atol=atol)
    check_rnn_layer_forward(gluon.rnn.LSTM(
        10, 2, projection_size=5), mx.np.ones((8, 3, 20)), device=device)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, projection_size=5, bidirectional=True), mx.np.ones(
        (8, 3, 20)), [mx.np.ones((4, 3, 5)), mx.np.ones((4, 3, 10))], device=device)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, dropout=0.5, projection_size=5), mx.np.ones((8, 3, 20)),
                            run_only=True, device=device)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, bidirectional=True, dropout=0.5, projection_size=5),
                            mx.np.ones((8, 3, 20)),
                            [mx.np.ones((4, 3, 5)), mx.np.ones((4, 3, 10))], run_only=True, device=device)
    lstm_layer.save_parameters('gpu_tmp.params')
    lstm_layer.load_parameters('gpu_tmp.params')


@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
@pytest.mark.flaky
def test_lstm_clip():
    hidden_size, projection_size = 4096, 2048
    batch_size, seq_len = 32, 80
    input_size = 50
    clip_min, clip_max, clip_nan = -5, 5, True
    lstm_input = mx.np.random.uniform(
        size=(seq_len, batch_size, input_size), device=mx.gpu(0))
    lstm_states = [mx.np.random.uniform(size=(2, batch_size, projection_size), device=mx.gpu(0)),
                   mx.np.random.uniform(size=(2, batch_size, hidden_size), device=mx.gpu(0))]
    lstm_layer = gluon.rnn.LSTM(hidden_size, projection_size=projection_size,
                                input_size=input_size,
                                bidirectional=True,
                                state_clip_min=clip_min,
                                state_clip_max=clip_max,
                                state_clip_nan=clip_nan)
    lstm_layer.initialize(device=mx.gpu(0))
    with autograd.record():
        _, layer_output_states = lstm_layer(lstm_input, lstm_states)
    cell_states = layer_output_states[0]
    assert (cell_states >= clip_min).all() and (cell_states <= clip_max).all()
    assert not _np.isnan(cell_states).any()


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnn_layer():
    check_rnn_layer(gluon.rnn.RNN(100, num_layers=3))
    check_rnn_layer(gluon.rnn.RNN(100, activation='tanh', num_layers=3))
    check_rnn_layer(gluon.rnn.LSTM(100, num_layers=3))
    check_rnn_layer(gluon.rnn.GRU(100, num_layers=3))

    check_rnn_layer(gluon.rnn.LSTM(100, num_layers=3, bidirectional=True))
    check_rnn_layer_w_rand_inputs(gluon.rnn.LSTM(
        100, num_layers=3, bidirectional=True))


@mx.util.use_np
def check_layer_bidirectional(size, in_size, proj_size):
    class RefBiLSTM(gluon.Block):
        def __init__(self, size, proj_size, **kwargs):
            super(RefBiLSTM, self).__init__(**kwargs)
            self._lstm_fwd = gluon.rnn.LSTM(
                size, projection_size=proj_size, bidirectional=False)
            self._lstm_bwd = gluon.rnn.LSTM(
                size, projection_size=proj_size, bidirectional=False)

        def forward(self, inpt):
            fwd = self._lstm_fwd(inpt)
            bwd_inpt = mx.np.flip(inpt, 0)
            bwd = self._lstm_bwd(bwd_inpt)
            bwd = mx.np.flip(bwd, 0)
            return mx.np.concatenate([fwd, bwd], axis=2)
    weights = {}
    for d in ['l', 'r']:
        weights['{}0_i2h_weight'.format(d)] = mx.np.random.uniform(
            size=(size * 4, in_size))
        if proj_size:
            weights['{}0_h2h_weight'.format(d)] = mx.np.random.uniform(
                size=(size * 4, proj_size))
            weights['{}0_h2r_weight'.format(d)] = mx.np.random.uniform(
                size=(proj_size, size))
        else:
            weights['{}0_h2h_weight'.format(
                d)] = mx.np.random.uniform(size=(size * 4, size))
        weights['{}0_i2h_bias'.format(
            d)] = mx.np.random.uniform(size=(size * 4,))
        weights['{}0_h2h_bias'.format(
            d)] = mx.np.random.uniform(size=(size * 4,))

    if proj_size:
        params = (weights['{}0_{}_{}'.format(d, g, t)].reshape(-1)
                    for t in ['weight', 'bias']
                    for d in ['l', 'r']
                    for g in ['i2h', 'h2h', 'h2r']
                    if g != 'h2r' or t != 'bias')
    else:
        params = (weights['{}0_{}_{}'.format(d, g, t)].reshape(-1)
                  for t in ['weight', 'bias']
                  for d in ['l', 'r']
                  for g in ['i2h', 'h2h'])

    net_params_concat = mx.np.concatenate(params)
    if proj_size:
        params_left = (weights['l0_{}_{}'.format(g, t)].reshape(-1)
                       for t in ['weight', 'bias']
                       for g in ['i2h', 'h2h', 'h2r']
                       if g != 'h2r' or t != 'bias')
    else:
        params_left = (weights['l0_{}_{}'.format(g, t)].reshape(-1)
                       for t in ['weight', 'bias']
                       for g in ['i2h', 'h2h'])
    if proj_size:
        params_right = (weights['r0_{}_{}'.format(g, t)].reshape(-1)
                        for t in ['weight', 'bias']
                        for g in ['i2h', 'h2h', 'h2r']
                        if g != 'h2r' or t != 'bias')
    else:
        params_right = (weights['r0_{}_{}'.format(g, t)].reshape(-1)
                        for t in ['weight', 'bias']
                        for g in ['i2h', 'h2h'])
    net_ref_left_params = mx.np.concatenate(params_left)
    net_ref_right_params = mx.np.concatenate(params_right)
    net = gluon.rnn.LSTM(size, projection_size=proj_size,
                         bidirectional=True)
    ref_net = RefBiLSTM(size, proj_size)
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    net_params['rnn_param'].set_data(net_params_concat)
    ref_net_params['_lstm_fwd.rnn_param'].set_data(net_ref_left_params)
    ref_net_params['_lstm_bwd.rnn_param'].set_data(net_ref_right_params)

    data = mx.np.random.uniform(size=(11, 10, in_size))
    mx.test_utils.assert_allclose(net(data), ref_net(data), rtol=1e-6)



def check_layer_bidirectional_varseqlen(size, in_size):
    weight = mx.np.random.uniform(size=(784,))

    net = gluon.rnn.LSTM(size, bidirectional=True, use_sequence_length=True)
    ref_net  = gluon.rnn.LSTM(size, bidirectional=True, use_sequence_length=False)
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    net_params['rnn_param'].set_data(weight)
    ref_net_params['rnn_param'].set_data(weight)

    batch_size = 10
    num_timesteps = 11
    data = mx.np.random.uniform(size=(num_timesteps, batch_size, in_size))
    data_np = data.asnumpy()

    sequence_length = mx.np.random.randint(1, num_timesteps+1, size=(batch_size)).astype("int32")
    sequence_length_np = sequence_length.asnumpy().astype("int32")

    # Reference net is processing batch elements one at a time, so that it is "perfectly sized"
    # Because of that, we need to accumulate gradients in reference net.
    for p in ref_net.collect_params().values():
        p.grad_req = 'add'

    ref_net_output = []
    with autograd.record():
        net_output = net(data.copy(), sequence_length=sequence_length.copy())

        for b in range(batch_size):
            data_slice = mx.np.array(data_np[:sequence_length_np[b], b, :]).reshape(sequence_length_np[b], 1, in_size)
            ref_output_slice = ref_net(data_slice)
            ref_net_output.append(ref_output_slice)

    net_output_np = net_output.asnumpy()

    # TODO: test state return value as well output
    # Only compare the valid sections for each batch entry
    for b in range(batch_size):
        assert_allclose(net_output_np[:sequence_length_np[b], b], ref_net_output[b].asnumpy().squeeze(1),
                        rtol=1e-2, atol=1e-6)

    # Now test backward
    net_output.backward()

    for ref_output_slice in ref_net_output:
        ref_output_slice.backward()

    ref_net_params = ref_net.collect_params()

    net_grad = net_params['rnn_param'].grad()
    ref_net_grad = ref_net_params['rnn_param'].grad()
    assert_almost_equal(net_grad.asnumpy(), ref_net_grad.asnumpy(),
                        rtol=1e-2, atol=1e-6)


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_layer_bidirectional():
    check_layer_bidirectional(7, 5, 0)


@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_layer_bidirectional_proj():
    check_layer_bidirectional(7, 5, 3)

@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_layer_bidirectional_varseqlength():
    check_layer_bidirectional_varseqlen(7, 5)


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnn_layer_begin_state_type():
    fake_data = mx.np.random.uniform(size=(3, 5, 7), dtype='float16')
    modeling_layer = gluon.rnn.LSTM(
        hidden_size=11, num_layers=2, dropout=0.2, bidirectional=True)
    modeling_layer.cast('float16')
    modeling_layer.initialize()
    modeling_layer(fake_data)


def test_gluon_ctc_consistency():
    loss = mx.gluon.loss.CTCLoss()
    data = mx.np.flip(mx.np.repeat(mx.np.arange(0, 4, device=mx.gpu(0)), 40).reshape((2, 20, 4)), axis=0)
    cpu_label = mx.np.array([[2, 1, -1, -1], [3, 2, 2, -1]], device=mx.cpu(0))
    gpu_label = mx.np.array([[2, 1, -1, -1], [3, 2, 2, -1]], device=mx.gpu(0))

    cpu_data = data.copy().to_device(mx.cpu(0))
    cpu_data.attach_grad()
    with mx.autograd.record():
        l_cpu = loss(cpu_data, cpu_label)
        l_cpu.backward()

    gpu_data = data.copyto(mx.gpu(0))
    gpu_data.attach_grad()
    with mx.autograd.record():
        l_gpu = loss(gpu_data, gpu_label)
        l_gpu.backward()

    assert_almost_equal(cpu_data.grad, gpu_data.grad, atol=1e-3, rtol=1e-3)


def test_global_norm_clip_multi_device():
    for check_isfinite in [True, False]:
        x1 = mx.np.ones((3, 3), device=mx.gpu(0))
        x2 = mx.np.ones((4, 4), device=mx.cpu(0))
        x3 = mx.np.ones((7, 4), device=mx.gpu(0))
        x4 = mx.np.ones((7, 4), device=mx.cpu(0))
        norm = gluon.utils.clip_global_norm(
            [x1, x2, x3, x4], 1.0, check_isfinite=check_isfinite)
        if check_isfinite:
            assert norm == 9.0
        else:
            assert norm.item() == 9.0
        assert_almost_equal(x1, _np.ones((3, 3)) / 9)
        assert_almost_equal(x2, _np.ones((4, 4)) / 9)
        assert_almost_equal(x3, _np.ones((7, 4)) / 9)
        assert_almost_equal(x4, _np.ones((7, 4)) / 9)


def _check_batchnorm_result(input, num_devices=1, cuda=False):
    from mxnet.gluon.utils import split_and_load
    def _find_bn(module):
        if isinstance(module, (mx.gluon.nn.BatchNorm, mx.gluon.nn.SyncBatchNorm)):
            return module
        elif isinstance(module.module, (mx.gluon.nn.BatchNorm, mx.gluon.nn.SyncBatchNorm)):
            return module.module

        raise RuntimeError('BN not found')

    def _syncParameters(bn1, bn2, device):
        device = input.context
        bn2.gamma.set_data(bn1.gamma.data(device))
        bn2.beta.set_data(bn1.beta.data(device))
        bn2.running_mean.set_data(bn1.running_mean.data(device))
        bn2.running_var.set_data(bn1.running_var.data(device))

    input1 = input.copy()
    input2 = input.copy()

    if cuda:
        input1 = input.to_device(mx.gpu(0))
        device_list = [mx.gpu(i) for i in range(num_devices)]
    else:
        device_list = [mx.cpu(0) for _ in range(num_devices)]

    nch = input.shape[1]
    bn1 = mx.gluon.nn.BatchNorm(in_channels=nch)
    bn2 = mx.gluon.nn.SyncBatchNorm(in_channels=nch, num_devices=num_devices)

    bn1.initialize(device=device_list[0])
    bn2.initialize(device=device_list)

    # using the same values for gamma and beta
    #_syncParameters(_find_bn(bn1), _find_bn(bn2), device_list[0])

    input1.attach_grad()
    inputs2 = split_and_load(input2, device_list, batch_axis=0)
    for xi in inputs2:
        xi.attach_grad()

    with mx.autograd.record():
        output1 = bn1(input1)
        output2  = [bn2(xi) for xi in inputs2]
        loss1 = (output1 ** 2).sum()
        loss2 = [(output ** 2).sum() for output in output2]
        mx.autograd.backward(loss1)
        mx.autograd.backward(loss2)

    output2 = mx.np.concatenate([output.to_device(input.context) for output in output2], axis=0)
    # assert forwarding
    assert_almost_equal(input1, input2, atol=1e-3, rtol=1e-3)
    assert_almost_equal(output1, output2, atol=1e-3, rtol=1e-3)
    assert_almost_equal(_find_bn(bn1).running_mean.data(device_list[0]),
                        _find_bn(bn2).running_mean.data(device_list[0]),
                        atol=1e-3, rtol=1e-3)
    assert_almost_equal(_find_bn(bn1).running_var.data(device_list[0]),
                        _find_bn(bn2).running_var.data(device_list[0]),
                        atol=1e-3, rtol=1e-3)
    input2grad = mx.np.concatenate([output.grad.to_device(input.context) for output in inputs2], axis=0)
    assert_almost_equal(input1.grad, input2grad, atol=1e-3, rtol=1e-3)

@mx.util.use_np
def test_sync_batchnorm():
    def get_num_devices():
        for i in range(100):
            try:
                mx.np.zeros((1,), device=mx.gpu(i))
            except:
                return i
    # no need to use SyncBN with 1 gpu
    if get_num_devices() < 2:
        return
    ndev = 2
    # check with unsync version
    for _ in range(10):
        _check_batchnorm_result(mx.np.random.uniform(size=(4, 1, 4, 4)),
                                num_devices=ndev, cuda=True)

def test_symbol_block_fp16(tmpdir):
    # Test case to verify if initializing the SymbolBlock from a model with params
    # other than fp32 param dtype.

    # 1. Load a resnet model, cast it to fp16 and export
    tmp = str(tmpdir)
    tmpfile = os.path.join(tmp, 'resnet34_fp16')
    device = mx.gpu(0)

    net_fp32 = mx.gluon.model_zoo.vision.resnet34_v2(
        pretrained=True, device=device, root=tmp)
    net_fp32.cast('float16')
    net_fp32.hybridize()
    data = mx.np.zeros((1, 3, 224, 224), dtype='float16', device=device)
    net_fp32(data)
    symbol_file, param_file = net_fp32.export(tmpfile, 0)

    # 2. Load the saved model and verify if all the params are loaded correctly.
    # Choose one of the parameters to verify the type is fp16.
    sm = mx.sym.load(symbol_file)
    inputs = mx.sym.var('data', dtype='float16')
    net_fp16 = mx.gluon.SymbolBlock(sm, inputs)
    net_fp16.load_parameters(param_file, device=device)
    # 3. Get a conv layer's weight parameter name. Conv layer's weight param is
    # expected to be of dtype casted, fp16.
    name = None
    for param_name in net_fp32.collect_params().keys():
        if 'conv' in param_name and 'weight' in param_name:
            name = param_name
            break
    assert _np.dtype(net_fp16.params[name].dtype) == _np.dtype(_np.float16)


@pytest.mark.serial
def test_large_models():
    device = default_device()
    # Create model
    net = gluon.nn.HybridSequential()

    largest_num_features = 256
    net.add(nn.Conv2D(largest_num_features, 3))

    net.hybridize()
    net.initialize(mx.init.Normal(sigma=0.01), device=device)

    # Compute the height (=width) of the square tensor of the given size in bytes
    def tensor_size(big_tensor_bytes):
        bytes_per_float = 4
        sz = int(math.sqrt(big_tensor_bytes /
                           largest_num_features / bytes_per_float))
        return (sz // 100) * 100

    # The idea is to create models with large tensors of (say) 20% of the total memory.
    # This in the past has given cudnnFind() trouble when it needed to allocate similar I/O's
    # from the area carved out by the MXNET_GPU_MEM_POOL_RESERVE setting (by default 5%).
    (free_mem_bytes, total_mem_bytes) = mx.device.gpu_memory_info(device.device_id)
    # This test needs to be 'qualified' for use with each new larger memory size
    largest_supported_total_mem_GB = 32
    if (total_mem_bytes > largest_supported_total_mem_GB * 1024 * 1024 * 1024):
        sys.stderr.write(
        ' bypassing test due to too-large global memory of size {} ... '.format(total_mem_bytes))
        return

    start_size = tensor_size(0.20 * total_mem_bytes)
    num_trials = 10
    sys.stderr.write(
        ' testing global memory of size {} ... '.format(total_mem_bytes))
    sys.stderr.flush()
    for i in range(num_trials):
        sz = start_size - 10 * i
        (height, width) = (sz, sz)
        sys.stderr.write(" {}x{} ".format(height, width))
        sys.stderr.flush()
        data_in = mx.np.random.uniform(low=0, high=255, size=(1, 3, height, width),
                                       device=device, dtype="float32")
        # Evaluate model
        net(data_in).asnumpy()


@mx.util.use_np
def test_hybridblock_mix_device_raise():
    class FooHybrid(gluon.HybridBlock):
        def forward(self, a, b):
            if isinstance(a, (list, tuple)):
                a = sum(a)
            if isinstance(b, (list, tuple)):
                b = sum(b)
            return a + b
    foo_hybrid = FooHybrid()
    foo_hybrid.hybridize()
    pytest.raises(ValueError, lambda: foo_hybrid(mx.np.ones((10,), device=mx.gpu()),
                                                 mx.np.ones((10,), device=mx.cpu())))


@mx.util.use_np
def test_gemms_true_fp16():
    device = mx.gpu(0)
    input = mx.np.random.uniform(size=(1, 512), dtype='float16', device=device)
    weights = mx.np.random.uniform(size=(128, 512), device=device)

    net = nn.Dense(128, in_units=512, use_bias=False)
    net.cast('float16')
    net.initialize(device=device)
    net.weight.set_data(weights)

    with environment('MXNET_FC_TRUE_FP16', '0'):
      ref_results = net(input)

    with environment('MXNET_FC_TRUE_FP16', '1'):
      results_trueFP16 = net(input)

    atol = 1e-2
    rtol = 1e-2
    assert_almost_equal(ref_results.asnumpy(), results_trueFP16.asnumpy(),
                        atol=atol, rtol=rtol)

@mx.util.use_np
def test_cudnn_dropout_reproducibility():
    d = nn.Dropout(0.5)
    d.initialize()
    a = mx.np.random.uniform(size=(100,100))
    b = a.copy()
    a.attach_grad()
    b.attach_grad()
    seed = mx.np.random.randint(0, 100000).item()
    N = 10
    mx.np.random.seed(seed)
    out1 = []
    for _ in range(N):
        with autograd.record():
            out1.append(d(a))
    out1[0].backward()
    mx.np.random.seed(seed)
    out2 = []
    for _ in range(N):
        with autograd.record():
            out2.append(d(b))
    out2[0].backward()

    for first, second in zip(out1, out2):
        assert_almost_equal(first, second)

    assert_almost_equal(a.grad, b.grad)

@mx.util.use_np
def test_cuda_graphs():
    class GraphTester(gluon.HybridBlock):
        def __init__(self, function_to_test, **kwargs):
            super(GraphTester, self).__init__(**kwargs)
            self.f = function_to_test()

        def forward(self, *args):
            # We need to isolate the operation to be fully inside the graph
            # in order for graphs usage to be possible
            copied_args = [mx.np.copy(a) for a in args]
            outputs = self.f(*copied_args)
            if isinstance(outputs, (list, tuple)):
                return [mx.np.copy(o) for o in outputs]
            else:
                return mx.np.copy(outputs)

    class TestDesc:
        def __init__(self, name, f, num_inputs=1, input_dim=4):
            self.name = name
            self.f = f
            self.num_inputs = num_inputs
            self.input_dim = input_dim

        def generate_inputs(self):
            shape = tuple(_np.random.randint(4, 11, size=self.input_dim))
            ret = [mx.np.random.uniform(size=shape) for _ in range(self.num_inputs)]
            for r in ret:
                r.attach_grad()
            return ret

    tested_ops = [
            TestDesc('add', lambda: (lambda x, y: x + y), num_inputs = 2),
            TestDesc('add_scalar', lambda: (lambda x: x + 0.5)),
            TestDesc('Conv', lambda: mx.gluon.nn.Conv2D(channels=32, kernel_size=(1,1))),
            TestDesc('ConvTranspose', lambda: mx.gluon.nn.Conv2DTranspose(channels=32, kernel_size=(1,1))),
            TestDesc('Dense', lambda: mx.gluon.nn.Dense(units=128)),
            TestDesc('Activation', lambda: mx.gluon.nn.Activation('tanh')),
            TestDesc('Dropout', lambda: mx.gluon.nn.Dropout(0.5)),
            TestDesc('Flatten', lambda: mx.gluon.nn.Flatten()),
            TestDesc('MaxPool', lambda: mx.gluon.nn.MaxPool2D()),
            TestDesc('AvgPool', lambda: mx.gluon.nn.AvgPool2D()),
            TestDesc('GlobalMaxPool', lambda: mx.gluon.nn.GlobalMaxPool2D()),
            TestDesc('GlobalAvgPool', lambda: mx.gluon.nn.GlobalAvgPool2D()),
            TestDesc('ReflectionPad2D', lambda: mx.gluon.nn.ReflectionPad2D()),
            TestDesc('BatchNorm', lambda: mx.gluon.nn.BatchNorm()),
            TestDesc('InstanceNorm', lambda: mx.gluon.nn.InstanceNorm()),
            TestDesc('LayerNorm', lambda: mx.gluon.nn.LayerNorm()),
            TestDesc('LeakyReLU', lambda: mx.gluon.nn.LeakyReLU(0.1)),
            TestDesc('PReLU', lambda: mx.gluon.nn.PReLU()),
            TestDesc('ELU', lambda: mx.gluon.nn.ELU()),
            TestDesc('SELU', lambda: mx.gluon.nn.SELU()),
            TestDesc('Swish', lambda: mx.gluon.nn.Swish()),
        ]

    N = 10

    with environment({'MXNET_ENABLE_CUDA_GRAPHS': '1',
                      'MXNET_USE_FUSION': '0'}):
        device = mx.gpu(0)
        for test_desc in tested_ops:
            print("Testing ", test_desc.name)
            inputs = test_desc.generate_inputs()
            inputsg = [i.copy() for i in inputs]
            for i in inputsg:
                i.attach_grad()
            seed = random.randint(0, 10000)
            net = GraphTester(test_desc.f)
            netg = GraphTester(test_desc.f)

            # initialize parameters
            net.initialize(device=device)
            netg.initialize(device=device)

            net(*inputs)

            for p1, p2 in zip(net.collect_params().values(), netg.collect_params().values()):
                p2.set_data(p1.data())

            netg.hybridize(static_alloc=True, static_shape=True)

            print("Testing inference mode")
            with random_seed(seed):
                for _ in range(N):
                    assert_almost_equal(net(*inputs), netg(*inputsg))

            mx.npx.waitall()
            print("Testing training mode")
            for _ in range(N):
                with random_seed(seed):
                    with mx.autograd.record():
                        out = net(*inputs)
                    out.backward()

                with random_seed(seed):
                    with mx.autograd.record():
                        outg = netg(*inputsg)
                    outg.backward()

                assert_almost_equal(out, outg)
                for i, ig in zip(inputs, inputsg):
                    assert_almost_equal(i.grad, ig.grad)

                for p1, p2 in zip(net.collect_params().values(), netg.collect_params().values()):
                    assert_almost_equal(p1.data(), p2.data())
                    if p1.grad_req != 'null':
                        assert_almost_equal(p1.grad(), p2.grad())
            mx.npx.waitall()
