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
import mxnet as mx
import multiprocessing as mp
from mxnet.test_utils import check_consistency, set_default_context, assert_almost_equal, rand_ndarray, environment
import numpy as _np
import math
from mxnet import autograd
import pytest

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import assert_raises_cudnn_not_satisfied, run_in_spawned_process
from test_gluon import *
from test_loss import *
from test_numpy_loss import *
from test_gluon_rnn import *

set_default_context(mx.gpu(0))


def check_rnn_layer(layer):
    layer.initialize(ctx=[mx.cpu(0), mx.gpu(0)])
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
    layer.initialize(ctx=[mx.cpu(0), mx.gpu(0)])
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
    ctx = mx.gpu(0)
    lstm_input = mx.np.random.uniform(
        size=(seq_len, batch_size, input_size), ctx=ctx)
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
    lstm_layer.initialize(ctx=ctx)
    lstm_cell.initialize(ctx=ctx)
    layer_params = lstm_layer.collect_params()
    cell_params = lstm_cell.collect_params()
    for k, v in weights.items():
        layer_params['l0_' + k].set_data(v.copy())
        cell_params[k].set_data(v.copy())
    with autograd.record():
        layer_output = lstm_layer(lstm_input.copy())
        cell_output = lstm_cell.unroll(seq_len, lstm_input.copy(), layout='TNC',
                                       merge_outputs=True)[0]

    assert_almost_equal(layer_output, cell_output, rtol=rtol, atol=atol)
    layer_output.backward()
    cell_output.backward()
    for k, v in weights.items():
        layer_grad = layer_params['l0_' + k].grad()
        cell_grad = cell_params[k].grad()
        print('checking gradient for {}'.format('lstm0_l0_' + k))
        assert_almost_equal(layer_grad, cell_grad, rtol=rtol, atol=atol)
    check_rnn_layer_forward(gluon.rnn.LSTM(
        10, 2, projection_size=5), mx.np.ones((8, 3, 20)), ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, projection_size=5, bidirectional=True), mx.np.ones(
        (8, 3, 20)), [mx.np.ones((4, 3, 5)), mx.np.ones((4, 3, 10))], ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, dropout=0.5, projection_size=5), mx.np.ones((8, 3, 20)),
                            run_only=True, ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, bidirectional=True, dropout=0.5, projection_size=5),
                            mx.np.ones((8, 3, 20)),
                            [mx.np.ones((4, 3, 5)), mx.np.ones((4, 3, 10))], run_only=True, ctx=ctx)
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
        size=(seq_len, batch_size, input_size), ctx=mx.gpu(0))
    lstm_states = [mx.np.random.uniform(size=(2, batch_size, projection_size), ctx=mx.gpu(0)),
                   mx.np.random.uniform(size=(2, batch_size, hidden_size), ctx=mx.gpu(0))]
    lstm_layer = gluon.rnn.LSTM(hidden_size, projection_size=projection_size,
                                input_size=input_size,
                                bidirectional=True,
                                state_clip_min=clip_min,
                                state_clip_max=clip_max,
                                state_clip_nan=clip_nan)
    lstm_layer.initialize(ctx=mx.gpu(0))
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

    net = gluon.rnn.LSTM(size, projection_size=proj_size,
                         bidirectional=True)
    ref_net = RefBiLSTM(size, proj_size)
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    for k in weights:
        net_params[k].set_data(weights[k])
        ref_net_params[k.replace('l0', '_lstm_fwd.l0').replace(
            'r0', '_lstm_bwd.l0')].set_data(weights[k])

    data = mx.np.random.uniform(size=(11, 10, in_size))
    mx.test_utils.assert_allclose(net(data), ref_net(data), rtol=1e-6)



def check_layer_bidirectional_varseqlen(size, in_size):
    weights = {}
    for d in ['l', 'r']:
        weights['{}0_i2h_weight'.format(d)] = mx.np.random.uniform(size=(size*4, in_size))
        weights['{}0_h2h_weight'.format(d)] = mx.np.random.uniform(size=(size*4, size))
        weights['{}0_i2h_bias'.format(d)] = mx.np.random.uniform(size=(size*4,))
        weights['{}0_h2h_bias'.format(d)] = mx.np.random.uniform(size=(size*4,))

    net = gluon.rnn.LSTM(size, bidirectional=True, use_sequence_length=True)
    ref_net  = gluon.rnn.LSTM(size, bidirectional=True, use_sequence_length=False)
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    for k in weights:
        net_params[k].set_data(weights[k])
        ref_net_params[k].set_data(weights[k])

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

    for k in weights:
        net_grad = net_params[k].grad()
        ref_net_grad = ref_net_params[k].grad()
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
    data = mx.np.flip(mx.np.repeat(mx.np.arange(0, 4, ctx=mx.gpu(0)), 40).reshape((2, 20, 4)), axis=0)
    cpu_label = mx.np.array([[2, 1, -1, -1], [3, 2, 2, -1]], ctx=mx.cpu(0))
    gpu_label = mx.np.array([[2, 1, -1, -1], [3, 2, 2, -1]], ctx=mx.gpu(0))

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

    assert_almost_equal(cpu_data.grad, gpu_data.grad, atol=1e-3, rtol=1e-3)


def test_global_norm_clip_multi_device():
    for check_isfinite in [True, False]:
        x1 = mx.np.ones((3, 3), ctx=mx.gpu(0))
        x2 = mx.np.ones((4, 4), ctx=mx.cpu(0))
        x3 = mx.np.ones((7, 4), ctx=mx.gpu(0))
        x4 = mx.np.ones((7, 4), ctx=mx.cpu(0))
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
    bn2 = mx.gluon.nn.SyncBatchNorm(in_channels=nch, num_devices=num_devices)

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

    output2 = mx.np.concatenate([output.as_in_context(input.context) for output in output2], axis=0)
    # assert forwarding
    assert_almost_equal(input1, input2, atol=1e-3, rtol=1e-3)
    assert_almost_equal(output1, output2, atol=1e-3, rtol=1e-3)
    assert_almost_equal(_find_bn(bn1).running_mean.data(ctx_list[0]),
                        _find_bn(bn2).running_mean.data(ctx_list[0]),
                        atol=1e-3, rtol=1e-3)
    assert_almost_equal(_find_bn(bn1).running_var.data(ctx_list[0]),
                        _find_bn(bn2).running_var.data(ctx_list[0]),
                        atol=1e-3, rtol=1e-3)
    input2grad = mx.np.concatenate([output.grad.as_in_context(input.context) for output in inputs2], axis=0)
    assert_almost_equal(input1.grad, input2grad, atol=1e-3, rtol=1e-3)

@mx.util.use_np
def test_sync_batchnorm():
    def get_num_devices():
        for i in range(100):
            try:
                mx.np.zeros((1,), ctx=mx.gpu(i))
            except:
                return i
    # no need to use SyncBN with 1 gpu
    if get_num_devices() < 2:
        return
    ndev = 2
    # check with unsync version
    for i in range(10):
        _check_batchnorm_result(mx.np.random.uniform(size=(4, 1, 4, 4)),
                                num_devices=ndev, cuda=True)

def test_symbol_block_fp16(tmpdir):
    # Test case to verify if initializing the SymbolBlock from a model with params
    # other than fp32 param dtype.

    # 1. Load a resnet model, cast it to fp16 and export
    tmp = str(tmpdir)
    tmpfile = os.path.join(tmp, 'resnet34_fp16')
    ctx = mx.gpu(0)

    net_fp32 = mx.gluon.model_zoo.vision.resnet34_v2(
        pretrained=True, ctx=ctx, root=tmp)
    net_fp32.cast('float16')
    net_fp32.hybridize()
    data = mx.np.zeros((1, 3, 224, 224), dtype='float16', ctx=ctx)
    net_fp32(data)
    symbol_file, param_file = net_fp32.export(tmpfile, 0)

    # 2. Load the saved model and verify if all the params are loaded correctly.
    # Choose one of the parameters to verify the type is fp16.
    sm = mx.sym.load(symbol_file)
    inputs = mx.sym.var('data', dtype='float16')
    net_fp16 = mx.gluon.SymbolBlock(sm, inputs)
    net_fp16.load_parameters(param_file, ctx=ctx)
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
    ctx = default_context()
    # Create model
    net = gluon.nn.HybridSequential()

    largest_num_features = 256
    net.add(nn.Conv2D(largest_num_features, 3))

    net.hybridize()
    net.initialize(mx.init.Normal(sigma=0.01), ctx=ctx)

    # Compute the height (=width) of the square tensor of the given size in bytes
    def tensor_size(big_tensor_bytes):
        bytes_per_float = 4
        sz = int(math.sqrt(big_tensor_bytes /
                           largest_num_features / bytes_per_float))
        return (sz // 100) * 100

    # The idea is to create models with large tensors of (say) 20% of the total memory.
    # This in the past has given cudnnFind() trouble when it needed to allocate similar I/O's
    # from the area carved out by the MXNET_GPU_MEM_POOL_RESERVE setting (by default 5%).
    (free_mem_bytes, total_mem_bytes) = mx.context.gpu_memory_info(ctx.device_id)
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
                                       ctx=ctx, dtype="float32")
        # Evaluate model
        net(data_in).asnumpy()

# isolated execution bulking test function to be invoked with different env var settings


@mx.util.use_np
def _test_bulking_in_process(seed, time_per_iteration):
    # Use flip since it's a simple function with same-sized I/O unlikely to ever be fused.
    class Flip(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Flip, self).__init__(**kwargs)

        def forward(self, x):
            return mx.np.flip(x, axis=0)

    def get_net(num_ops):
        net = nn.HybridSequential()
        for _ in range(num_ops):
            net.add(Flip())
        return net

    data_shape = (10,)
    num_ops = 1000
    num_iterations = 20

    # build model
    x = mx.np.zeros(data_shape)
    x.attach_grad()
    dy = mx.np.ones(data_shape)
    net = get_net(num_ops)
    net.hybridize(static_alloc=True, static_shape=True)

    # time a number of forward() and backward() executions after some warm-up iterations
    warmups = 1
    for i in range(num_iterations + warmups):
        with autograd.record():
            if i == warmups:
                start = time.time()
            y = net(x)
            y.backward(dy)
            x.grad.wait_to_read()

    time_per_iteration.value = (time.time() - start) / num_iterations

def _test_bulking(test_bulking_func):
    # test case format: (max_fwd_segment_size, max_bwd_segment_size, enable_bulking_in_training)
    test_cases = [(0, 0, True), (1, 1, True), (15, 15, False),
                  (15, 0, True), (0, 15, True), (15, 15, True)]
    times = {}
    times_str = ''
    for seg_sizes in test_cases:
        # Create shared variable to return measured time from test process
        time_per_iteration = mp.Manager().Value('d', 0.0)

        if not run_in_spawned_process(test_bulking_func,
                                      {'MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD': str(seg_sizes[0]),
                                       'MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD': str(seg_sizes[1]),
                                       'MXNET_EXEC_BULK_EXEC_TRAIN': str(seg_sizes[2])},
                                      time_per_iteration):
            # skip test since the python version can't run it properly.  Warning msg was logged.
            return
        times[seg_sizes] = time_per_iteration.value
        times_str += \
            '\n    runtime of (fwd,bwd,enable) op seg setting ({},{},{}) =\t{:.1f} msec'.format(
                seg_sizes[0], seg_sizes[1], seg_sizes[2], 1000.0 * times[seg_sizes])

    fastest_non_bulked_time = min(times[(0, 0, True)], times[(1, 1, True)], times[(15, 15, False)])
    slowest_half_bulked_time = max(times[(0, 15, True)], times[(15, 0, True)])
    fastest_half_bulked_time = min(times[(0, 15, True)], times[(15, 0, True)])
    fully_bulked_time = times[(15, 15, True)]

    print(times_str)
    # Non-bulked times[0,0,True], times[1,1,True] and times[15,15,False] should be about the same,
    # slower than both half-bulked times[0,15,True] and times[15,0,True]
    assert slowest_half_bulked_time < fastest_non_bulked_time, \
        'A half-bulked exec time is slower than the non-bulked time by {} secs! {}' \
        .format(slowest_half_bulked_time - fastest_non_bulked_time, times_str)
    # The fully bulked times[15,15,True] should be faster than both half-bulked runs
    assert fully_bulked_time < fastest_half_bulked_time, \
        'The fully-bulked exec time is slower than a half-bulked time by {} secs! {}' \
        .format(fully_bulked_time - fastest_half_bulked_time, times_str)

@pytest.mark.skip(reason='skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/14970')
def test_bulking_gluon_gpu():
    _test_bulking(_test_bulking_in_process)


@mx.util.use_np
def test_hybridblock_mix_ctx_raise():
    class FooHybrid(gluon.HybridBlock):
        def forward(self, a, b):
            if isinstance(a, (list, tuple)):
                a = sum(a)
            if isinstance(b, (list, tuple)):
                b = sum(b)
            return a + b
    foo_hybrid = FooHybrid()
    foo_hybrid.hybridize()
    pytest.raises(ValueError, lambda: foo_hybrid(mx.np.ones((10,), ctx=mx.gpu()),
                                                 mx.np.ones((10,), ctx=mx.cpu())))


@mx.util.use_np
def test_gemms_true_fp16():
    ctx = mx.gpu(0)
    input = mx.np.random.uniform(size=(1, 512), dtype='float16', ctx=ctx)
    weights = mx.np.random.uniform(size=(128, 512), ctx=ctx)

    net = nn.Dense(128, in_units=512, use_bias=False)
    net.cast('float16')
    net.initialize(ctx=ctx)
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

