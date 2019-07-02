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
import mxnet.ndarray as nd
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
from common import run_in_spawned_process
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
    ctx = mx.gpu(0)
    lstm_input = mx.nd.uniform(
        shape=(seq_len, batch_size, input_size), ctx=ctx)
    shapes = {'i2h_weight': (hidden_size * 4, input_size),
              'h2h_weight': (hidden_size * 4, projection_size),
              'i2h_bias': (hidden_size * 4,),
              'h2h_bias': (hidden_size * 4,),
              'h2r_weight': (projection_size, hidden_size)}
    weights = {k: rand_ndarray(v) for k, v in shapes.items()}
    lstm_layer = gluon.rnn.LSTM(hidden_size, projection_size=projection_size,
                                input_size=input_size, prefix='lstm0_')
    lstm_cell = gluon.contrib.rnn.LSTMPCell(hidden_size=hidden_size,
                                            projection_size=projection_size,
                                            input_size=input_size,
                                            prefix='lstm0_l0_')
    lstm_layer.initialize(ctx=ctx)
    lstm_cell.initialize(ctx=ctx)
    layer_params = lstm_layer.collect_params()
    cell_params = lstm_cell.collect_params()
    for k, v in weights.items():
        layer_params['lstm0_l0_' + k].set_data(v.copy())
        cell_params['lstm0_l0_' + k].set_data(v.copy())
    with autograd.record():
        layer_output = lstm_layer(lstm_input.copy())
        cell_output = lstm_cell.unroll(seq_len, lstm_input.copy(), layout='TNC',
                                       merge_outputs=True)[0]
    assert_almost_equal(layer_output.asnumpy(),
                        cell_output.asnumpy(), rtol=rtol, atol=atol)
    layer_output.backward()
    cell_output.backward()
    for k, v in weights.items():
        layer_grad = layer_params['lstm0_l0_' + k].grad()
        cell_grad = cell_params['lstm0_l0_' + k].grad()
        print('checking gradient for {}'.format('lstm0_l0_' + k))
        assert_almost_equal(layer_grad.asnumpy(), cell_grad.asnumpy(),
                            rtol=rtol, atol=atol)
    check_rnn_layer_forward(gluon.rnn.LSTM(
        10, 2, projection_size=5), mx.nd.ones((8, 3, 20)), ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, projection_size=5, bidirectional=True), mx.nd.ones(
        (8, 3, 20)), [mx.nd.ones((4, 3, 5)), mx.nd.ones((4, 3, 10))], ctx=ctx)

    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, dropout=0.5, projection_size=5), mx.nd.ones((8, 3, 20)),
                            run_only=True, ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, bidirectional=True, dropout=0.5, projection_size=5),
                            mx.nd.ones((8, 3, 20)),
                            [mx.nd.ones((4, 3, 5)), mx.nd.ones((4, 3, 10))], run_only=True, ctx=ctx)


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_lstm_clip():
    hidden_size, projection_size = 4096, 2048
    batch_size, seq_len = 32, 80
    input_size = 50
    clip_min, clip_max, clip_nan = -5, 5, True
    lstm_input = mx.nd.uniform(
        shape=(seq_len, batch_size, input_size), ctx=mx.gpu(0))
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
    check_rnn_layer_w_rand_inputs(gluon.rnn.LSTM(
        100, num_layers=3, bidirectional=True))


def check_layer_bidirectional(size, in_size, proj_size):
    class RefBiLSTM(gluon.Block):
        def __init__(self, size, proj_size, **kwargs):
            super(RefBiLSTM, self).__init__(**kwargs)
            with self.name_scope():
                self._lstm_fwd = gluon.rnn.LSTM(
                    size, projection_size=proj_size, bidirectional=False, prefix='l0')
                self._lstm_bwd = gluon.rnn.LSTM(
                    size, projection_size=proj_size, bidirectional=False, prefix='r0')

        def forward(self, inpt):
            fwd = self._lstm_fwd(inpt)
            bwd_inpt = nd.flip(inpt, 0)
            bwd = self._lstm_bwd(bwd_inpt)
            bwd = nd.flip(bwd, 0)
            return nd.concat(fwd, bwd, dim=2)
    weights = {}
    for d in ['l', 'r']:
        weights['lstm_{}0_i2h_weight'.format(d)] = mx.random.uniform(
            shape=(size * 4, in_size))
        if proj_size:
            weights['lstm_{}0_h2h_weight'.format(d)] = mx.random.uniform(
                shape=(size * 4, proj_size))
            weights['lstm_{}0_h2r_weight'.format(d)] = mx.random.uniform(
                shape=(proj_size, size))
        else:
            weights['lstm_{}0_h2h_weight'.format(
                d)] = mx.random.uniform(shape=(size * 4, size))
        weights['lstm_{}0_i2h_bias'.format(
            d)] = mx.random.uniform(shape=(size * 4,))
        weights['lstm_{}0_h2h_bias'.format(
            d)] = mx.random.uniform(shape=(size * 4,))

    net = gluon.rnn.LSTM(size, projection_size=proj_size,
                         bidirectional=True, prefix='lstm_')
    ref_net = RefBiLSTM(size, proj_size, prefix='lstm_')
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    for k in weights:
        net_params[k].set_data(weights[k])
        ref_net_params[k.replace('l0', 'l0l0').replace(
            'r0', 'r0l0')].set_data(weights[k])

    data = mx.random.uniform(shape=(11, 10, in_size))
    assert_allclose(net(data).asnumpy(), ref_net(data).asnumpy())


def check_layer_bidirectional_varseqlen(size, in_size):
    weights = {}
    for d in ['l', 'r']:
        weights['lstm_{}0_i2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, in_size))
        weights['lstm_{}0_h2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, size))
        weights['lstm_{}0_i2h_bias'.format(d)] = mx.random.uniform(shape=(size*4,))
        weights['lstm_{}0_h2h_bias'.format(d)] = mx.random.uniform(shape=(size*4,))

    net = gluon.rnn.LSTM(size, bidirectional=True, use_sequence_length=True, prefix='lstm_')
    ref_net  = gluon.rnn.LSTM(size, bidirectional=True, use_sequence_length=False, prefix='lstm_ref_')
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    for k in weights:
        net_params[k].set_data(weights[k])
        ref_net_params[k.replace("lstm_", "lstm_ref_")].set_data(weights[k])

    batch_size = 10
    num_timesteps = 11
    data = mx.random.uniform(shape=(num_timesteps, batch_size, in_size))
    data_np = data.asnumpy()

    sequence_length = nd.random.randint(1, num_timesteps+1, shape=(batch_size)).astype("int32")
    sequence_length_np = sequence_length.asnumpy().astype("int32")

    # Reference net is processing batch elements one at a time, so that it is "perfectly sized"
    # Because of that, we need to accumulate gradients in reference net.
    for p in ref_net.collect_params().values():
        p.grad_req = 'add'

    ref_net_output = []
    with autograd.record():
        net_output = net(data.copy(), sequence_length=sequence_length.copy())

        for b in range(batch_size):
            data_slice = mx.nd.array(data_np[:sequence_length_np[b], b, :]).reshape(sequence_length_np[b], 1, in_size)
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
        ref_net_grad = ref_net_params[k.replace('lstm_', 'lstm_ref_')].grad()
        assert_almost_equal(net_grad.asnumpy(), ref_net_grad.asnumpy(),
                            rtol=1e-2, atol=1e-6)


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_layer_bidirectional():
    check_layer_bidirectional(7, 5, 0)


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_layer_bidirectional_proj():
    check_layer_bidirectional(7, 5, 3)

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
def test_layer_bidirectional_varseqlength():
    check_layer_bidirectional_varseqlen(7, 5)


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnn_layer_begin_state_type():
    fake_data = nd.random.uniform(shape=(3, 5, 7), dtype='float16')
    modeling_layer = gluon.rnn.LSTM(
        hidden_size=11, num_layers=2, dropout=0.2, bidirectional=True)
    modeling_layer.cast('float16')
    modeling_layer.initialize()
    modeling_layer(fake_data)


def test_gluon_ctc_consistency():
    loss = mx.gluon.loss.CTCLoss()
    data = mx.nd.arange(0, 4, repeat=40, ctx=mx.gpu(0)
                        ).reshape((2, 20, 4)).flip(axis=0)
    cpu_label = mx.nd.array([[2, 1, -1, -1], [3, 2, 2, -1]], ctx=mx.cpu(0))
    gpu_label = mx.nd.array([[2, 1, -1, -1], [3, 2, 2, -1]], ctx=mx.gpu(0))

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

    assert_almost_equal(cpu_data.grad.asnumpy(),
                        gpu_data.grad.asnumpy(), atol=1e-3, rtol=1e-3)


@with_seed()
def test_global_norm_clip_multi_device():
    for check_isfinite in [True, False]:
        x1 = mx.nd.ones((3, 3), ctx=mx.gpu(0))
        x2 = mx.nd.ones((4, 4), ctx=mx.cpu(0))
        norm = gluon.utils.clip_global_norm(
            [x1, x2], 1.0, check_isfinite=check_isfinite)
        if check_isfinite:
            assert norm == 5.0
        else:
            assert norm.asscalar() == 5.0
        assert_almost_equal(x1.asnumpy(), np.ones((3, 3)) / 5)
        assert_almost_equal(x2.asnumpy(), np.ones((4, 4)) / 5)


@with_seed()
def test_symbol_block_fp16():
    # Test case to verify if initializing the SymbolBlock from a model with params
    # other than fp32 param dtype.

    # 1. Load a resnet model, cast it to fp16 and export
    tmp = tempfile.mkdtemp()
    tmpfile = os.path.join(tmp, 'resnet34_fp16')
    ctx = mx.gpu(0)

    net_fp32 = mx.gluon.model_zoo.vision.resnet34_v2(
        pretrained=True, ctx=ctx, root=tmp)
    net_fp32.cast('float16')
    net_fp32.hybridize()
    data = mx.nd.zeros((1, 3, 224, 224), dtype='float16', ctx=ctx)
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
        sz = int(math.sqrt(big_tensor_bytes /
                           largest_num_features / bytes_per_float))
        return (sz // 100) * 100

    # The idea is to create models with large tensors of (say) 20% of the total memory.
    # This in the past has given cudnnFind() trouble when it needed to allocate similar I/O's
    # from the area carved out by the MXNET_GPU_MEM_POOL_RESERVE setting (by default 5%).
    (free_mem_bytes, total_mem_bytes) = mx.context.gpu_memory_info(ctx.device_id)
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
        data_in = nd.random_uniform(low=0, high=255, shape=(1, 3, height, width),
                                    ctx=ctx, dtype="float32")
        # Evaluate model
        net(data_in).asnumpy()

# isolated execution bulking test function to be invoked with different env var settings


def _test_bulking_in_process(seed, time_per_iteration):
    # Use flip since it's a simple function with same-sized I/O unlikely to ever be fused.
    class Flip(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Flip, self).__init__(**kwargs)

        def hybrid_forward(self, F, x):
            return F.flip(x, axis=0)

    def get_net(num_ops):
        net = nn.HybridSequential()
        with net.name_scope():
            for _ in range(num_ops):
                net.add(Flip())
        return net

    data_shape = (10,)
    num_ops = 1000
    num_iterations = 20

    # build model
    x = mx.ndarray.zeros(data_shape)
    x.attach_grad()
    dy = mx.ndarray.ones(data_shape)
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


@with_seed()
@unittest.skip('skippping temporarily, tracked by https://github.com/apache/incubator-mxnet/issues/14970')
def test_bulking():
    # test case format: (max_fwd_segment_size, max_bwd_segment_size, enable_bulking_in_training)
    test_cases = [(0, 0, True), (1, 1, True), (15, 15, False),
                  (15, 0, True), (0, 15, True), (15, 15, True)]
    times = {}
    times_str = ''
    for seg_sizes in test_cases:
        # Create shared variable to return measured time from test process
        time_per_iteration = mp.Manager().Value('d', 0.0)
        if not run_in_spawned_process(_test_bulking_in_process,
                                      {'MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD': seg_sizes[0],
                                       'MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD': seg_sizes[1],
                                       'MXNET_EXEC_BULK_EXEC_TRAIN': seg_sizes[2]},
                                      time_per_iteration):
            # skip test since the python version can't run it properly.  Warning msg was logged.
            return
        times[seg_sizes] = time_per_iteration.value
        times_str += \
            '\n    runtime of (fwd,bwd,enable) op seg setting ({},{},{}) =\t{:.1f} msec'.format(
                seg_sizes[0], seg_sizes[1], seg_sizes[2], 1000.0 * times[seg_sizes])

    fastest_non_bulked_time = min(
        times[(0, 0, True)], times[(1, 1, True)], times[(15, 15, False)])
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


if __name__ == '__main__':
    import nose
    nose.runmodule()
