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
from mxnet import gluon, nd
import numpy as np
import copy
from itertools import product
from functools import partial
from numpy.testing import assert_allclose
import pytest
from mxnet.test_utils import almost_equal, assert_almost_equal, default_context
from common import assert_raises_cudnn_not_satisfied, retry


def check_rnn_states(fused_states, stack_states, num_layers, bidirectional=False, is_lstm=True):
    directions = 2 if bidirectional else 1
    assert len(stack_states) / len(fused_states) == num_layers * directions

    fused_states = [state.asnumpy() for state in fused_states]
    stack_states = [np.expand_dims(state.asnumpy(), axis=0) for state in stack_states]
    if is_lstm:
        stack_states_h = stack_states[0::2]
        stack_states_c = stack_states[1::2]
        stack_states = [np.concatenate(stack_states_h, axis=0), np.concatenate(stack_states_c, axis=0)]
    else:
        stack_states = [np.concatenate(stack_states, axis=0)]

    for f, s in zip(fused_states, stack_states):
        assert f.shape == s.shape
        assert_almost_equal(f, s, atol=1e-4, rtol=1e-4)


def test_rnn():
    cell = gluon.rnn.RNNCell(100)
    inputs = [mx.sym.Variable('t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == ['h2h_bias', 'h2h_weight',
                                                    'i2h_bias', 'i2h_weight']
    assert outputs.list_outputs() == [
        'rnncell_t0_out_output', 'rnncell_t1_out_output',
        'rnncell_t2_out_output'
    ]

    args, outs, auxs = outputs.infer_shape(t0_data=(10,50), t1_data=(10,50), t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_lstm():
    cell = gluon.rnn.LSTMCell(100)
    inputs = [mx.sym.Variable('t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == ['h2h_bias', 'h2h_weight', 'i2h_bias', 'i2h_weight']
    assert outputs.list_outputs() == [
        'lstmcell_t0_out_output', 'lstmcell_t1_out_output',
        'lstmcell_t2_out_output'
    ]

    args, outs, auxs = outputs.infer_shape(t0_data=(10,50), t1_data=(10,50), t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


@assert_raises_cudnn_not_satisfied(min_version='7.2.1')
@pytest.mark.serial
def test_lstmp():
    hidden_size, projection_size = 512, 256
    rtol, atol = 1e-4, 1e-4
    batch_size, seq_len = 5, 3
    input_size = 128
    lstm_input = mx.nd.uniform(shape=(seq_len, batch_size, input_size))

    # ==== Unidirectional Layer ====
    for num_layers in [1, 3]:
        fused_layer = gluon.rnn.LSTM(hidden_size, projection_size=projection_size,
                                     num_layers=num_layers, layout='TNC', bidirectional=False)

        stack_layer = mx.gluon.rnn.HybridSequentialRNNCell()
        for i in range(num_layers):
            stack_layer.add(gluon.rnn.LSTMPCell(hidden_size,
                                                projection_size=projection_size))
        fused_layer.initialize()
        stack_layer.initialize()

        fused_begin_state = fused_layer.begin_state(batch_size)
        stack_begin_state = stack_layer.begin_state(batch_size=batch_size)
        fused_layer.infer_shape(lstm_input, fused_begin_state)
        fused_layer_params = fused_layer.collect_params()
        stack_layer_params = stack_layer.collect_params()

        for name, value in fused_layer_params.items():
            w = mx.nd.random.uniform(shape=value.shape)
            value.set_data(w.copy())
            stack_layer_params[name[1:].replace('_', '.', 1)].set_data(w.copy())

        fused_output, fused_states = fused_layer(lstm_input.copy(), fused_begin_state)
        stack_output, stack_states = stack_layer.unroll(seq_len, lstm_input.copy(), begin_state=stack_begin_state,
                                                        layout='TNC',
                                                        merge_outputs=True)

        assert_almost_equal(fused_output.asnumpy(), stack_output.asnumpy(), rtol=rtol, atol=atol)
        check_rnn_states(fused_states, stack_states, num_layers, False)

    # ==== Bidirectional Layer ====
    for num_layers in [1, 3]:
        fused_layer = gluon.rnn.LSTM(hidden_size, projection_size=projection_size,
                                     num_layers=num_layers, layout='TNC', bidirectional=True)

        stack_layer = mx.gluon.rnn.HybridSequentialRNNCell()
        for i in range(num_layers):
            stack_layer.add(
                gluon.rnn.BidirectionalCell(gluon.rnn.LSTMPCell(hidden_size,
                                                                projection_size=projection_size),
                                            gluon.rnn.LSTMPCell(hidden_size,
                                                                projection_size=projection_size)))
        fused_layer.initialize()
        stack_layer.initialize()

        fused_begin_state = fused_layer.begin_state(batch_size)
        stack_begin_state = stack_layer.begin_state(batch_size=batch_size)
        fused_layer.infer_shape(lstm_input, fused_begin_state)
        fused_layer_params = fused_layer.collect_params()
        stack_layer_params = stack_layer.collect_params()

        for name, value in fused_layer_params.items():
            w = mx.nd.random.uniform(shape=value.shape)
            value.set_data(w.copy())
            cur = name.split("_")[0]
            stack_layer_params["{}.{}_cell.{}".format(cur[1:], name[0], name[len(cur)+1:])].set_data(w.copy())

        fused_output, fused_states = fused_layer(lstm_input.copy(), fused_begin_state)
        stack_output, stack_states = stack_layer.unroll(seq_len, lstm_input.copy(), begin_state=stack_begin_state,
                                                        layout='TNC',
                                                        merge_outputs=True)

        assert_almost_equal(fused_output.asnumpy(), stack_output.asnumpy(), rtol=rtol, atol=atol)
        check_rnn_states(fused_states, stack_states, num_layers, True)


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_lstm_cpu_inference():
    # should behave the same as lstm cell
    EXPECTED_LSTM_OUTPUT = np.array([[[0.72045636, 0.72045636, 0.95215213, 0.95215213],
                                      [0.72045636, 0.72045636, 0.95215213, 0.95215213]],
                                     [[0.95215213, 0.95215213, 0.72045636, 0.72045636],
                                      [0.95215213, 0.95215213, 0.72045636, 0.72045636]]])
    x = mx.nd.ones(shape=(2, 2, 2))
    model = mx.gluon.rnn.LSTM(2, num_layers=6, bidirectional=True)
    model.initialize(mx.init.One())

    y = model(x).asnumpy()
    mx.test_utils.assert_almost_equal(y, EXPECTED_LSTM_OUTPUT,
                                      rtol=1e-3, atol=1e-5)


def test_gru():
    cell = gluon.rnn.GRUCell(100, activation='relu', recurrent_activation='tanh')
    inputs = [mx.sym.Variable('t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == ['h2h_bias', 'h2h_weight', 'i2h_bias', 'i2h_weight']
    assert outputs.list_outputs() == [
        'grucell_t0_out_output', 'grucell_t1_out_output',
        'grucell_t2_out_output'
    ]

    args, outs, auxs = outputs.infer_shape(t0_data=(10,50), t1_data=(10,50), t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


@pytest.mark.serial
def test_residual():
    cell = gluon.rnn.ResidualCell(gluon.rnn.GRUCell(50))
    inputs = [mx.sym.Variable('t%d_data'%i) for i in range(2)]
    outputs, _ = cell.unroll(2, inputs)
    outputs = mx.sym.Group(outputs)
    params = cell.collect_params()
    assert sorted(params.keys()) == \
           ['base_cell.h2h_bias', 'base_cell.h2h_weight', 'base_cell.i2h_bias', 'base_cell.i2h_weight']

    args, outs, auxs = outputs.infer_shape(t0_data=(10, 50), t1_data=(10, 50))
    assert outs == [(10, 50), (10, 50)]
    outputs = outputs.eval(**{'t0_data': mx.nd.ones((10, 50)),
                              't1_data': mx.nd.ones((10, 50)),
                              cell.base_cell.i2h_weight.var().name: mx.nd.zeros((150, 50)),
                              cell.base_cell.i2h_bias.var().name: mx.nd.zeros((150, )),
                              cell.base_cell.h2h_weight.var().name: mx.nd.zeros((150, 50)),
                              cell.base_cell.h2h_bias.var().name: mx.nd.zeros((150, ))})
    expected_outputs = np.ones((10, 50))
    assert np.array_equal(outputs[0].asnumpy(), expected_outputs)
    assert np.array_equal(outputs[1].asnumpy(), expected_outputs)


@pytest.mark.serial
def test_residual_bidirectional():
    cell = gluon.rnn.ResidualCell(
            gluon.rnn.BidirectionalCell(
                gluon.rnn.GRUCell(25),
                gluon.rnn.GRUCell(25)))
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(2)]
    outputs, _ = cell.unroll(2, inputs, merge_outputs=False)
    outputs = mx.sym.Group(outputs)
    params = cell.collect_params()
    assert sorted(params.keys()) == \
           ['base_cell.l_cell.h2h_bias', 'base_cell.l_cell.h2h_weight',
            'base_cell.l_cell.i2h_bias', 'base_cell.l_cell.i2h_weight',
            'base_cell.r_cell.h2h_bias', 'base_cell.r_cell.h2h_weight',
            'base_cell.r_cell.i2h_bias', 'base_cell.r_cell.i2h_weight']
    # assert outputs.list_outputs() == \
    #        ['bi_t0_plus_residual_output', 'bi_t1_plus_residual_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10, 50), rnn_t1_data=(10, 50))
    assert outs == [(10, 50), (10, 50)]
    outputs = outputs.eval(**{'rnn_t0_data':mx.nd.ones((10, 50))+5,
                              'rnn_t1_data':mx.nd.ones((10, 50))+5,
                              cell.base_cell.l_cell.i2h_weight.var().name:mx.nd.zeros((75, 50)),
                              cell.base_cell.l_cell.i2h_bias.var().name:mx.nd.zeros((75,)),
                              cell.base_cell.l_cell.h2h_weight.var().name:mx.nd.zeros((75, 25)),
                              cell.base_cell.l_cell.h2h_bias.var().name:mx.nd.zeros((75,)),
                              cell.base_cell.r_cell.i2h_weight.var().name:mx.nd.zeros((75, 50)),
                              cell.base_cell.r_cell.i2h_bias.var().name:mx.nd.zeros((75,)),
                              cell.base_cell.r_cell.h2h_weight.var().name:mx.nd.zeros((75, 25)),
                              cell.base_cell.r_cell.h2h_bias.var().name:mx.nd.zeros((75,))})
    expected_outputs = np.ones((10, 50))+5
    assert np.array_equal(outputs[0].asnumpy(), expected_outputs)
    assert np.array_equal(outputs[1].asnumpy(), expected_outputs)


def test_stack():
    cell = gluon.rnn.SequentialRNNCell()
    for i in range(5):
        if i == 1:
            cell.add(gluon.rnn.ResidualCell(gluon.rnn.LSTMCell(100)))
        else:
            cell.add(gluon.rnn.LSTMCell(100))
    inputs = [mx.sym.Variable('t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    keys = sorted(cell.collect_params().keys())
    for i in range(5):
        if i==1:
            continue
        assert '%d.h2h_weight'%i in keys
        assert '%d.h2h_bias'%i in keys
        assert '%d.i2h_weight'%i in keys
        assert '%d.i2h_bias'%i in keys
    assert '1.base_cell.h2h_weight' in keys
    assert '1.base_cell.h2h_bias' in keys
    assert '1.base_cell.i2h_weight' in keys
    assert '1.base_cell.i2h_bias' in keys
    assert outputs.list_outputs() == ['lstmcell_t0_out_output', 'lstmcell_t1_out_output', 'lstmcell_t2_out_output']

    args, outs, auxs = outputs.infer_shape(t0_data=(10,50), t1_data=(10,50), t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


@pytest.mark.serial
def test_hybridstack():
    cell = gluon.rnn.HybridSequentialRNNCell()
    for i in range(5):
        if i == 1:
            cell.add(gluon.rnn.ResidualCell(gluon.rnn.LSTMCell(100)))
        else:
            cell.add(gluon.rnn.LSTMCell(100))
    inputs = [mx.sym.Variable('t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    keys = sorted(cell.collect_params().keys())
    for i in range(5):
        if i==1:
            continue
        assert '%d.h2h_weight'%i in keys
        assert '%d.h2h_bias'%i in keys
        assert '%d.i2h_weight'%i in keys
        assert '%d.i2h_bias'%i in keys
    assert '1.base_cell.h2h_weight' in keys
    assert '1.base_cell.h2h_bias' in keys
    assert '1.base_cell.i2h_weight' in keys
    assert '1.base_cell.i2h_bias' in keys
    assert outputs.list_outputs() == ['lstmcell_t0_out_output', 'lstmcell_t1_out_output', 'lstmcell_t2_out_output']

    args, outs, auxs = outputs.infer_shape(t0_data=(10,50), t1_data=(10,50), t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

    # Test HybridSequentialRNNCell nested in nn.HybridBlock, SequentialRNNCell will fail in this case
    class BidirectionalOfSequential(gluon.HybridBlock):
        def __init__(self):
            super(BidirectionalOfSequential, self).__init__()

            cell0 = gluon.rnn.HybridSequentialRNNCell()
            cell0.add(gluon.rnn.LSTMCell(100))
            cell0.add(gluon.rnn.LSTMCell(100))

            cell1 = gluon.rnn.HybridSequentialRNNCell()
            cell1.add(gluon.rnn.LSTMCell(100))
            cell1.add(gluon.rnn.LSTMCell(100))

            self.rnncell = gluon.rnn.BidirectionalCell(cell0, cell1)

        def hybrid_forward(self, F, x):
            return self.rnncell.unroll(3, x, layout="NTC", merge_outputs=True)

    x = mx.nd.random.uniform(shape=(10, 3, 100))
    net = BidirectionalOfSequential()
    net.initialize()
    outs, _ = net(x)

    assert outs.shape == (10, 3, 200)


def test_bidirectional():
    cell = gluon.rnn.BidirectionalCell(
            gluon.rnn.LSTMCell(100),
            gluon.rnn.LSTMCell(100))
    inputs = [mx.sym.Variable('t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert outputs.list_outputs() == ['t0_output', 't1_output', 't2_output']

    args, outs, auxs = outputs.infer_shape(t0_data=(10,50), t1_data=(10,50), t2_data=(10,50))
    assert outs == [(10, 200), (10, 200), (10, 200)]


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
@pytest.mark.serial
def test_layer_bidirectional():
    class RefBiLSTM(gluon.Block):
        def __init__(self, size, **kwargs):
            super(RefBiLSTM, self).__init__(**kwargs)
            self._lstm_fwd = gluon.rnn.LSTM(size, bidirectional=False)
            self._lstm_bwd = gluon.rnn.LSTM(size, bidirectional=False)

        def forward(self, inpt):
            fwd = self._lstm_fwd(inpt)
            bwd_inpt = nd.flip(inpt, 0)
            bwd = self._lstm_bwd(bwd_inpt)
            bwd = nd.flip(bwd, 0)
            return nd.concat(fwd, bwd, dim=2)

    size = 7
    in_size = 5
    weights = {}
    for d in ['l', 'r']:
        weights['{}0_i2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, in_size))
        weights['{}0_h2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, size))
        weights['{}0_i2h_bias'.format(d)] = mx.random.uniform(shape=(size*4,))
        weights['{}0_h2h_bias'.format(d)] = mx.random.uniform(shape=(size*4,))

    net = gluon.rnn.LSTM(size, bidirectional=True)
    ref_net = RefBiLSTM(size)
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    for k in weights:
        net_params[k].set_data(weights[k])
        ref_net_params[k.replace('l0', '_lstm_fwd.l0').replace('r0', '_lstm_bwd.l0')].set_data(weights[k])

    data = mx.random.uniform(shape=(11, 10, in_size))
    assert_allclose(net(data).asnumpy(), ref_net(data).asnumpy(), rtol=1e-04, atol=1e-02)



def test_zoneout():
    cell = gluon.rnn.ZoneoutCell(gluon.rnn.RNNCell(100), zoneout_outputs=0.5,
                                 zoneout_states=0.5)
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


@pytest.mark.serial
def test_unroll_layout():
    cell = gluon.rnn.HybridSequentialRNNCell()
    for i in range(5):
        if i == 1:
            cell.add(gluon.rnn.ResidualCell(gluon.rnn.LSTMCell(100)))
        else:
            cell.add(gluon.rnn.LSTMCell(100))
    cell.initialize()
    inputs = [mx.nd.random.uniform(shape=(10,50)) for _ in range(3)]
    outputs, _ = cell.unroll(3, inputs, layout='TNC')
    assert outputs[0].shape == (10, 100)
    assert outputs[1].shape == (10, 100)
    assert outputs[2].shape == (10, 100)

    outputs, _ = cell.unroll(3, inputs, layout='NTC')
    assert outputs[0].shape == (10, 100)
    assert outputs[1].shape == (10, 100)
    assert outputs[2].shape == (10, 100)


def check_rnn_forward_backward(layer, merged_inputs, hybridize, merge_outputs, deterministic):
    input_size = 5
    if merged_inputs:
        inputs = mx.nd.ones((8, 3, 5))
        inputs.attach_grad()
    else:
        inputs = [mx.nd.ones((8, 5)) for _ in range(3)]
        for x in inputs:
            x.attach_grad()

    if hybridize:
        layer.hybridize()
    layer.initialize()

    with mx.autograd.record():
        out = layer.unroll(3, inputs, merge_outputs=merge_outputs)[0]
        mx.autograd.backward(out)

    if hasattr(layer, 'i2h_weight'):
        assert layer.i2h_weight.shape[1] == input_size, (layer.i2h_weight.shape[1], input_size)

    if merge_outputs:
        np_out = out.asnumpy()
    else:
        np_out = np.stack([x.asnumpy() for x in out], axis=1)

    if merged_inputs:
        np_dx = inputs.grad.asnumpy()
    else:
        np_dx = np.stack([x.grad.asnumpy() for x in inputs], axis=1)

    with mx.autograd.record():
        out = layer.unroll(3, inputs, merge_outputs=not merge_outputs)[0]
        mx.autograd.backward(out)

    if merged_inputs:
        input_grads = inputs.grad.asnumpy()
    else:
        input_grads = np.stack([x.grad.asnumpy() for x in inputs], axis=1)

    if deterministic:
        if not merge_outputs:
            ref_np_out = out.asnumpy()
        else:
            ref_np_out = np.stack([x.asnumpy() for x in out], axis=1)
        mx.test_utils.assert_almost_equal(np_out, ref_np_out, rtol=1e-3, atol=1e-5)
        mx.test_utils.assert_almost_equal(np_dx, input_grads, rtol=1e-3, atol=1e-5)


@retry(3)
@pytest.mark.parametrize('layer,determinism', [
    (gluon.rnn.LSTMCell(10, input_size=5), True),
    (gluon.rnn.RNNCell(10, input_size=5), True),
    (gluon.rnn.GRUCell(10, input_size=5), True),
    (gluon.rnn.BidirectionalCell(
        gluon.rnn.LSTMCell(10, input_size=5),
        gluon.rnn.LSTMCell(10, input_size=5)
     ), True),
    (gluon.rnn.DropoutCell(0.5), False),
])
@pytest.mark.parametrize('merged_inputs', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('merge_outputs', [True, False, None])
@pytest.mark.skip(reason='https://github.com/apache/incubator-mxnet/issues/18225')
def test_rnn_forward_backward(layer, merged_inputs, hybridize, merge_outputs, determinism):
    check_rnn_forward_backward(layer, merged_inputs, hybridize, merge_outputs, determinism)


@pytest.mark.parametrize('seq_rnn_type', [
    gluon.rnn.SequentialRNNCell,
    gluon.rnn.HybridSequentialRNNCell
])
@pytest.mark.parametrize('determinism', [True, False])
@pytest.mark.parametrize('merged_inputs', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('merge_outputs', [True, False, None])
@pytest.mark.skip(reason='https://github.com/apache/incubator-mxnet/issues/18291')
def test_sequential_rnn_cells(seq_rnn_type, determinism, merged_inputs, hybridize, merge_outputs):
    net = gluon.rnn.SequentialRNNCell()
    net.add(gluon.rnn.LSTMCell(10, input_size=5))
    net.add(gluon.rnn.RNNCell(10, input_size=10))
    net.add(gluon.rnn.GRUCell(10, input_size=10))
    if not determinism:
        net.add(gluon.rnn.DropoutCell(0.5))
    check_rnn_forward_backward(net, merged_inputs, hybridize, merge_outputs, determinism)


def test_rnn_cells_export_import():
    class RNNLayer(gluon.HybridBlock):
        def __init__(self):
            super(RNNLayer, self).__init__()
            self.cell = gluon.rnn.RNNCell(hidden_size=1)

        def hybrid_forward(self, F, seq):
            outputs, state = self.cell.unroll(inputs=seq, length=2, merge_outputs=True)
            return outputs

    class LSTMLayer(gluon.HybridBlock):
        def __init__(self):
            super(LSTMLayer, self).__init__()
            self.cell = gluon.rnn.LSTMCell(hidden_size=1)

        def hybrid_forward(self, F, seq):
            outputs, state = self.cell.unroll(inputs=seq, length=2, merge_outputs=True)
            return outputs

    class GRULayer(gluon.HybridBlock):
        def __init__(self):
            super(GRULayer, self).__init__()
            self.cell = gluon.rnn.GRUCell(hidden_size=1)

        def hybrid_forward(self, F, seq):
            outputs, state = self.cell.unroll(inputs=seq, length=2, merge_outputs=True)
            return outputs

    for hybrid in [RNNLayer(), LSTMLayer(), GRULayer()]:
        hybrid.initialize()
        hybrid.hybridize()
        input = mx.nd.ones(shape=(1, 2, 1))
        output1 = hybrid(input)
        hybrid.export(path="./model", epoch=0)
        symbol = mx.gluon.SymbolBlock.imports(
            symbol_file="./model-symbol.json",
            input_names=["data"],
            param_file="./model-0000.params",
            ctx=mx.context.current_context()
        )
        output2 = symbol(input)
        assert_almost_equal(output1.asnumpy(), output2.asnumpy())


def check_rnn_layer_forward(layer, inputs, states=None, run_only=False, ctx=mx.cpu()):
    layer.initialize(ctx=ctx)
    inputs = inputs.as_in_context(ctx)
    inputs.attach_grad()
    if states is not None:
        if isinstance(states, (list, tuple)):
            states = [s.as_in_context(ctx) for s in states]
        else:
            states = states.as_in_context(ctx)
    with mx.autograd.record():
        if states is None:
            out = layer(inputs)
        else:
            out = layer(inputs, states)
        if states is not None:
            assert isinstance(out, (list, tuple)) and len(out) == 2
            out = out[0]
        else:
            assert isinstance(out, mx.nd.NDArray)
        out.backward()

    np_out = out.asnumpy()
    np_dx = inputs.grad.asnumpy()

    layer.hybridize()

    with mx.autograd.record():
        if states is not None:
            out = layer(inputs, states)
            assert isinstance(out, (list, tuple)) and len(out) == 2
            out = out[0]
        else:
            out = layer(inputs)
            assert isinstance(out, mx.nd.NDArray)
        out.backward()

    if states is not None:
        layer(inputs, states) # test is_training = false
    else:
        layer(inputs)

    if not run_only:
        mx.test_utils.assert_almost_equal(np_out, out.asnumpy(), rtol=1e-3, atol=1e-5)
        mx.test_utils.assert_almost_equal(np_dx, inputs.grad.asnumpy(), rtol=1e-3, atol=1e-5)



def run_rnn_layers(dtype, dtype2, ctx=mx.cpu()):

    check_rnn_layer_forward(gluon.rnn.RNN(10, 2, dtype=dtype), mx.nd.ones((8, 3, 20), dtype=dtype), ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.RNN(10, 2, dtype=dtype, bidirectional=True), mx.nd.ones((8, 3, 20),  dtype=dtype), mx.nd.ones((4, 3, 10),  dtype=dtype), ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2,dtype=dtype), mx.nd.ones((8, 3, 20),  dtype=dtype), ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2,dtype=dtype,  bidirectional=True), mx.nd.ones((8, 3, 20),  dtype=dtype), [mx.nd.ones((4, 3, 10),  dtype=dtype), mx.nd.ones((4, 3, 10),  dtype=dtype)],ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.GRU(10, 2, dtype=dtype, ), mx.nd.ones((8, 3, 20), dtype=dtype),ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.GRU(10, 2, dtype=dtype, bidirectional=True), mx.nd.ones((8, 3, 20),  dtype=dtype), mx.nd.ones((4, 3, 10),  dtype=dtype),ctx=ctx)


    check_rnn_layer_forward(gluon.rnn.RNN(10, 2, dtype=dtype, dropout=0.5), mx.nd.ones((8, 3, 20), dtype=dtype),
                            run_only=True, ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.RNN(10, 2, bidirectional=True, dropout=0.5, dtype=dtype),
                            mx.nd.ones((8, 3, 20), dtype=dtype), mx.nd.ones((4, 3, 10), dtype=dtype), run_only=True, ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, dropout=0.5, dtype=dtype), mx.nd.ones((8, 3, 20), dtype=dtype),
                            run_only=True, ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.LSTM(10, 2, bidirectional=True, dropout=0.5, dtype=dtype),
                            mx.nd.ones((8, 3, 20), dtype=dtype),
                            [mx.nd.ones((4, 3, 10), dtype=dtype), mx.nd.ones((4, 3, 10), dtype=dtype)], run_only=True, ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.GRU(10, 2, dropout=0.5, dtype=dtype), mx.nd.ones((8, 3, 20), dtype=dtype),
                            run_only=True, ctx=ctx)
    check_rnn_layer_forward(gluon.rnn.GRU(10, 2, bidirectional=True, dropout=0.5, dtype=dtype),
                            mx.nd.ones((8, 3, 20), dtype=dtype), mx.nd.ones((4, 3, 10), dtype=dtype), run_only=True, ctx=ctx)

    net = gluon.nn.Sequential()
    net.add(gluon.rnn.LSTM(10, bidirectional=True, dtype=dtype2))
    net.add(gluon.nn.BatchNorm(axis=2))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(3, activation='relu'))
    net.initialize(ctx=ctx)
    net.cast(dtype)
    with mx.autograd.record():
        out = net(mx.nd.ones((2, 3, 10), dtype=dtype, ctx=ctx))
        out.backward()
        out = out.asnumpy()

    net2 = gluon.nn.HybridSequential()
    net2.add(gluon.rnn.LSTM(10, bidirectional=True, dtype=dtype2))
    net2.add(gluon.nn.BatchNorm(axis=2))
    net2.add(gluon.nn.Flatten())
    net2.add(gluon.nn.Dense(3, activation='relu'))
    net2.hybridize()
    net2.initialize(ctx=ctx)
    net2.cast(dtype)
    with mx.autograd.record():
        out = net2(mx.nd.ones((2, 3, 10), dtype=dtype, ctx=ctx))
        out.backward()
        out = out.asnumpy()

    net3 = gluon.nn.HybridSequential()
    net3.add(gluon.rnn.LSTM(10, bidirectional=True, dtype=dtype))
    net3.add(gluon.nn.BatchNorm(axis=2))
    net3.add(gluon.nn.Flatten())
    net3.add(gluon.nn.Dense(3, activation='relu'))
    net3.hybridize()
    net3.initialize(ctx=ctx)
    net3.cast(dtype2)
    with mx.autograd.record():
        out = net3(mx.nd.ones((2, 3, 10), dtype=dtype2, ctx=ctx))
        out.backward()
        out = out.asnumpy()

@pytest.mark.serial
def test_rnn_layers_fp32():
    run_rnn_layers('float32', 'float32')

@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
@pytest.mark.skipif(mx.context.num_gpus() == 0, reason="RNN FP16 only implemented for GPU for now")
@pytest.mark.serial
def test_rnn_layers_fp16():
    run_rnn_layers('float16', 'float32', mx.gpu())


def check_rnn_consistency(fused_layer, stack_layer, loss, input_size, hidden_size, bidirectional=False, rtol=1e-2, atol=1e-4):
    x = nd.random.normal(shape=(1, 5, input_size))
    fused_begin_state = fused_layer.begin_state(1)
    stack_states = stack_layer.begin_state(batch_size=1)
    fused_layer.infer_shape(x, fused_begin_state)
    fused_layer_params = fused_layer.collect_params()
    stack_layer_params = stack_layer.collect_params()

    for name, value in fused_layer_params.items():
        if 'weight' in name:
            w = mx.nd.zeros(shape=value.shape)
        else:
            w = mx.nd.random.normal(shape=value.shape)
        value.set_data(w.copy())
        cur = name.split('_')[0]
        num = cur[1:]
        stack_name = ('{}.{}_cell.'.format(num, name[0]) if bidirectional else num + '.' ) + name[len(cur)+1:]
        stack_layer_params[stack_name].set_data(w.copy())

    fx = x.copy()
    sx = x.copy()
    y = nd.random.uniform(shape=(1, 5, hidden_size * 2 if bidirectional else hidden_size))

    fx.attach_grad()
    with mx.autograd.record():
        fused_out, fused_states = fused_layer(fx, fused_begin_state)
        l = loss(fused_out, y).mean()
    l.backward()
    fused_grads = dict([(name, p.grad()) for name, p in fused_layer.collect_params().items()])
    fused_input_grad = fx.grad.asnumpy()

    sx.attach_grad()
    with mx.autograd.record():
        stack_out, stack_states = stack_layer.unroll(5, sx, begin_state=stack_states, merge_outputs=True)
        l = loss(stack_out, y).mean()
    l.backward()
    stack_grads = dict([(name, p.grad()) for name, p in stack_layer.collect_params().items()])
    stack_input_grad = sx.grad.asnumpy()

    assert_allclose(fused_out.asnumpy(), stack_out.asnumpy(), rtol=rtol, atol=atol)
    assert_allclose(fused_input_grad, stack_input_grad, rtol=rtol, atol=atol)
    for name, value in fused_grads.items():
        cur = name.split('_')[0]
        num = cur[1:]
        stack_name = ('{}.{}_cell.'.format(num, name[0]) if bidirectional else num + '.' ) + name[len(cur)+1:]
        assert_allclose(value.asnumpy(), stack_grads[stack_name].asnumpy(), rtol=rtol, atol=atol)

    num_layers = fused_begin_state[0].shape[0] // (2 if bidirectional else 1)
    check_rnn_states(fused_states, stack_states, num_layers, bidirectional, len(fused_begin_state) == 2)


def create_op_by_mode(mode):
    if mode == 'lstm':
        fused_op = gluon.rnn.LSTM
        stack_op = gluon.rnn.LSTMCell
        recurrent_block_prefix = 'lstm0_'
    elif mode == 'gru':
        fused_op = gluon.rnn.GRU
        stack_op = gluon.rnn.GRUCell
        recurrent_block_prefix = 'gru0_'
    elif mode == 'rnn_relu':
        fused_op = partial(gluon.rnn.RNN, activation='relu')
        stack_op = partial(gluon.rnn.RNNCell, activation='relu')
        recurrent_block_prefix = 'rnn0_'
    elif mode == 'rnn_tanh':
        fused_op = partial(gluon.rnn.RNN, activation='tanh')
        stack_op = partial(gluon.rnn.RNNCell, activation='tanh')
        recurrent_block_prefix = 'rnn0_'

    return fused_op, stack_op, recurrent_block_prefix


def check_rnn_unidir_layer_gradients(mode, input_size, hidden_size, num_layers, loss):
    fused_op, stack_op, recurrent_block_prefix = create_op_by_mode(mode)

    fused_layer = fused_op(hidden_size, num_layers=num_layers, layout='NTC', bidirectional=False)
    fused_layer.initialize()

    stack_layer = mx.gluon.rnn.HybridSequentialRNNCell()
    for n in range(num_layers):
        stack_layer.add(stack_op(hidden_size))
    stack_layer.initialize()
    check_rnn_consistency(fused_layer, stack_layer, loss, input_size, hidden_size)


def check_rnn_bidir_layer_gradients(mode, input_size, hidden_size, num_layers, loss):
    fused_op, stack_op, recurrent_block_prefix = create_op_by_mode(mode)

    fused_layer = fused_op(hidden_size, num_layers=num_layers, layout='NTC', bidirectional=True)
    fused_layer.initialize()

    stack_layer = mx.gluon.rnn.HybridSequentialRNNCell()
    for n in range(num_layers):
        stack_layer.add(gluon.rnn.BidirectionalCell(stack_op(hidden_size),
                                                    stack_op(hidden_size)))
    stack_layer.initialize()
    check_rnn_consistency(fused_layer, stack_layer, loss, input_size, hidden_size, bidirectional=True)


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_fused_lstm_layer():
    input_sizes = [8]
    hidden_sizes = [8, 16]
    num_layers = [1, 2, 3, 4]
    for input_size, hidden_size, num_layers in product(input_sizes, hidden_sizes, num_layers):
        loss = mx.gluon.loss.L2Loss()
        check_rnn_unidir_layer_gradients('lstm', input_size, hidden_size, num_layers, loss)
        check_rnn_bidir_layer_gradients('lstm', input_size, hidden_size, num_layers, loss)


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_fused_gru_layer():
    input_sizes = [8]
    hidden_sizes = [8, 16]
    num_layers = [1, 2, 3, 4]
    for input_size, hidden_size, num_layers in product(input_sizes, hidden_sizes, num_layers):
        loss = mx.gluon.loss.L2Loss()
        check_rnn_unidir_layer_gradients('gru', input_size, hidden_size, num_layers, loss)
        check_rnn_bidir_layer_gradients('gru', input_size, hidden_size, num_layers, loss)


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_fused_rnnrelu_layer():
    input_sizes = [8]
    hidden_sizes = [8, 16]
    num_layers = [1, 2, 3, 4]
    for input_size, hidden_size, num_layers in product(input_sizes, hidden_sizes, num_layers):
        loss = mx.gluon.loss.L2Loss()
        check_rnn_unidir_layer_gradients('rnn_relu', input_size, hidden_size, num_layers, loss)
        check_rnn_bidir_layer_gradients('rnn_relu', input_size, hidden_size, num_layers, loss)


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_fused_rnntanh_layer():
    input_sizes = [8]
    hidden_sizes = [8, 16]
    num_layers = [1, 2, 3, 4]
    for input_size, hidden_size, num_layers in product(input_sizes, hidden_sizes, num_layers):
        loss = mx.gluon.loss.L2Loss()
        check_rnn_unidir_layer_gradients('rnn_tanh', input_size, hidden_size, num_layers, loss)
        check_rnn_bidir_layer_gradients('rnn_tanh', input_size, hidden_size, num_layers, loss)


@pytest.mark.serial
def test_rnn_unroll_variant_length():
    # Test for imperative usage
    cell_list = []
    for base_cell_class in [gluon.rnn.RNNCell, gluon.rnn.LSTMCell, gluon.rnn.GRUCell]:
        cell_list.append(base_cell_class(20))
        cell_list.append(gluon.rnn.BidirectionalCell(
                         l_cell=base_cell_class(20),
                         r_cell=base_cell_class(20)))
        cell_list.append(gluon.rnn.VariationalDropoutCell(base_cell=base_cell_class(20)))
    stack_res_rnn_cell = gluon.rnn.SequentialRNNCell()
    stack_res_rnn_cell.add(gluon.rnn.ResidualCell(base_cell=gluon.rnn.RNNCell(20)))
    stack_res_rnn_cell.add(gluon.rnn.ResidualCell(base_cell=gluon.rnn.RNNCell(20)))
    cell_list.append(stack_res_rnn_cell)
    batch_size = 4
    max_length = 10
    valid_length = [3, 10, 5, 6]
    valid_length_nd = mx.nd.array(valid_length)
    for cell in cell_list:
        cell.initialize()
        cell.hybridize()
        print(cell.collect_params())
        # Test for NTC layout
        data_nd = mx.nd.random.normal(0, 1, shape=(batch_size, max_length, 20))
        outs, states = cell.unroll(length=max_length, inputs=data_nd,
                                   valid_length=valid_length_nd,
                                   merge_outputs=True,
                                   layout='NTC')
        for i, ele_length in enumerate(valid_length):
            # Explicitly unroll each sequence and compare the final states and output
            ele_out, ele_states = cell.unroll(length=ele_length,
                                              inputs=data_nd[i:(i+1), :ele_length, :],
                                              merge_outputs=True,
                                              layout='NTC')
            assert_allclose(ele_out.asnumpy(), outs[i:(i+1), :ele_length, :].asnumpy(),
                            atol=1E-4, rtol=1E-4)
            if ele_length < max_length:
                # Check the padded outputs are all zero
                assert_allclose(outs[i:(i+1), ele_length:max_length, :].asnumpy(), 0)
            for valid_out_state, gt_state in zip(states, ele_states):
                assert_allclose(valid_out_state[i:(i+1)].asnumpy(), gt_state.asnumpy(),
                                atol=1E-4, rtol=1E-4)

        # Test for TNC layout
        data_nd = mx.nd.random.normal(0, 1, shape=(max_length, batch_size, 20))
        outs, states = cell.unroll(length=max_length, inputs=data_nd,
                                   valid_length=valid_length_nd,
                                   layout='TNC')
        for i, ele_length in enumerate(valid_length):
            # Explicitly unroll each sequence and compare the final states and output
            ele_out, ele_states = cell.unroll(length=ele_length,
                                              inputs=data_nd[:ele_length, i:(i+1), :],
                                              merge_outputs=True,
                                              layout='TNC')
            assert_allclose(ele_out.asnumpy(), outs[:ele_length, i:(i + 1), :].asnumpy(),
                            atol=1E-4, rtol=1E-4)
            if ele_length < max_length:
                # Check the padded outputs are all zero
                assert_allclose(outs[ele_length:max_length, i:(i+1), :].asnumpy(), 0)
            for valid_out_state, gt_state in zip(states, ele_states):
                assert_allclose(valid_out_state[i:(i+1)].asnumpy(), gt_state.asnumpy(),
                                atol=1E-4, rtol=1E-4)


def test_cell_fill_shape():
    cell = gluon.rnn.LSTMCell(10, input_size=7)
    cell.hybridize()
    assert cell.i2h_weight.shape[1] == 7, cell.i2h_weight.shape[1]

def test_layer_fill_shape():
    layer = gluon.rnn.LSTM(10)
    layer.hybridize()
    check_rnn_layer_forward(layer, mx.nd.ones((3, 2, 7)))
    print(layer)
    assert layer.l0_i2h_weight.shape[1] == 7, layer.l0_i2h_weight.shape[1]


@pytest.mark.serial
def test_bidirectional_unroll_valid_length():
    def _check_bidirectional_unroll_valid_length(length):
        class BiLSTM(gluon.nn.HybridBlock):
            def __init__(self, rnn_size, time_step, **kwargs):
                super(BiLSTM, self).__init__(**kwargs)
                self.time_step = time_step
                self.bi_lstm = gluon.rnn.BidirectionalCell(
                    gluon.rnn.LSTMCell(rnn_size),
                    gluon.rnn.LSTMCell(rnn_size))

            def hybrid_forward(self, F, inputs, valid_len):
                outputs, states = self.bi_lstm.unroll(self.time_step, inputs, valid_length=valid_len,
                                                      layout='NTC', merge_outputs=True)
                return outputs, states

        rnn_size = 100
        net = BiLSTM(rnn_size, length)
        net.initialize()
        net.hybridize()
        inputs_data = mx.nd.random.uniform(shape=(10, length, 50))
        valid_len = mx.nd.array([length]*10)
        outputs, _ = net(inputs_data, valid_len)
        assert outputs.shape == (10, length, 200)

    _check_bidirectional_unroll_valid_length(1)
    _check_bidirectional_unroll_valid_length(3)


def check_rnn_cell(cell, in_shape=(10, 50), out_shape=(10, 100), begin_state=None):
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs, begin_state=begin_state)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == ['h2h_bias', 'h2h_weight',
                                                    'i2h_bias', 'i2h_weight']
    assert outputs.list_outputs() == [type(cell).__name__.lower() + name for name in ['_t0_out_output', '_t1_out_output', '_t2_out_output']]

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=in_shape,
                                           rnn_t1_data=in_shape,
                                           rnn_t2_data=in_shape)
    assert outs == [out_shape] * 3


def check_rnn_forward(layer, inputs):
    inputs.attach_grad()
    layer.initialize()
    with mx.autograd.record():
        layer.unroll(3, inputs, merge_outputs=True)[0].backward()
        mx.autograd.backward(layer.unroll(3, inputs, merge_outputs=False)[0])
    mx.nd.waitall()


def test_rnn_cells():
    check_rnn_forward(gluon.rnn.Conv1DLSTMCell((5, 7), 10, (3,), (3,)),
                      mx.nd.ones((8, 3, 5, 7)))
    check_rnn_forward(gluon.rnn.Conv1DRNNCell((5, 7), 10, (3,), (3,)),
                      mx.nd.ones((8, 3, 5, 7)))
    check_rnn_forward(gluon.rnn.Conv1DGRUCell((5, 7), 10, (3,), (3,)),
                      mx.nd.ones((8, 3, 5, 7)))

    net = mx.gluon.rnn.SequentialRNNCell()
    net.add(gluon.rnn.Conv1DLSTMCell((5, 7), 10, (3,), (3,)))
    net.add(gluon.rnn.Conv1DRNNCell((10, 5), 11, (3,), (3,)))
    net.add(gluon.rnn.Conv1DGRUCell((11, 3), 12, (3,), (3,)))
    check_rnn_forward(net, mx.nd.ones((8, 3, 5, 7)))


def test_convrnn():
    cell = gluon.rnn.Conv1DRNNCell((10, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 50), out_shape=(1, 100, 48))

    cell = gluon.rnn.Conv2DRNNCell((10, 20, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 20, 50), out_shape=(1, 100, 18, 48))

    cell = gluon.rnn.Conv3DRNNCell((10, 20, 30, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 20, 30, 50), out_shape=(1, 100, 18, 28, 48))


def test_convlstm():
    cell = gluon.rnn.Conv1DLSTMCell((10, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 50), out_shape=(1, 100, 48))

    cell = gluon.rnn.Conv2DLSTMCell((10, 20, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 20, 50), out_shape=(1, 100, 18, 48))

    cell = gluon.rnn.Conv3DLSTMCell((10, 20, 30, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 20, 30, 50), out_shape=(1, 100, 18, 28, 48))


def test_convgru():
    cell = gluon.rnn.Conv1DGRUCell((10, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 50), out_shape=(1, 100, 48))

    cell = gluon.rnn.Conv2DGRUCell((10, 20, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 20, 50), out_shape=(1, 100, 18, 48))

    cell = gluon.rnn.Conv3DGRUCell((10, 20, 30, 50), 100, 3, 3)
    check_rnn_cell(cell, in_shape=(1, 10, 20, 30, 50), out_shape=(1, 100, 18, 28, 48))


def test_conv_fill_shape():
    cell = gluon.rnn.Conv1DLSTMCell((0, 7), 10, (3,), (3,))
    cell.hybridize()
    check_rnn_forward(cell, mx.nd.ones((8, 3, 5, 7)))
    assert cell.i2h_weight.shape[1] == 5, cell.i2h_weight.shape[1]


def test_lstmp():
    nhid = 100
    nproj = 64
    cell = gluon.rnn.LSTMPCell(nhid, nproj)
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    expected_params = ['h2h_bias', 'h2h_weight', 'h2r_weight', 'i2h_bias', 'i2h_weight']
    expected_outputs = [type(cell).__name__.lower() + name for name in ['_t0_out_output', '_t1_out_output', '_t2_out_output']]
    assert sorted(cell.collect_params().keys()) == expected_params
    assert outputs.list_outputs() == expected_outputs, outputs.list_outputs()

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, nproj), (10, nproj), (10, nproj)]


def test_vardrop():
    def check_vardrop(drop_inputs, drop_states, drop_outputs):
        cell = gluon.rnn.VariationalDropoutCell(mx.gluon.rnn.RNNCell(100),
                                                drop_outputs=drop_outputs,
                                                drop_states=drop_states,
                                                drop_inputs=drop_inputs)
        cell.initialize(init='xavier')
        input_data = mx.nd.random_uniform(shape=(10, 3, 50), ctx=mx.context.current_context())
        with mx.autograd.record():
            outputs1, _ = cell.unroll(3, input_data, merge_outputs=True)
            mx.nd.waitall()
            outputs2, _ = cell.unroll(3, input_data, merge_outputs=True)
        assert not almost_equal(outputs1.asnumpy(), outputs2.asnumpy())

        inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
        outputs, _ = cell.unroll(3, inputs, merge_outputs=False)
        outputs = mx.sym.Group(outputs)

        args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
        assert outs == [(10, 100), (10, 100), (10, 100)]

        cell.reset()
        cell.hybridize()
        with mx.autograd.record():
            outputs3, _ = cell.unroll(3, input_data, merge_outputs=True)
            mx.nd.waitall()
            outputs4, _ = cell.unroll(3, input_data, merge_outputs=True)
        assert not almost_equal(outputs3.asnumpy(), outputs4.asnumpy())
        assert not almost_equal(outputs1.asnumpy(), outputs3.asnumpy())

    check_vardrop(0.5, 0.5, 0.5)
    check_vardrop(0.5, 0, 0.5)


@pytest.mark.parametrize('cell_type,num_states', [
    (gluon.rnn.RNNCell, 1),
    (gluon.rnn.LSTMCell, 2),
    (gluon.rnn.GRUCell, 1)
])
@pytest.mark.parametrize('layout', ['NTC', 'TNC'])
def test_unroll(cell_type, num_states, layout):
    class RNNLayer(gluon.HybridBlock):
        def __init__(self, cell_type, hidden_size, layout):
            super(RNNLayer, self).__init__()
            self.cell = cell_type(hidden_size)
            self.layout = layout

        def hybrid_forward(self, F, inputs, states, valid_length):
            if isinstance(valid_length, list) and len(valid_length) == 0:
                valid_length = None
            return gluon.rnn.rnn_cell.dynamic_unroll(self.cell, inputs, states,
                                                     valid_length=valid_length,
                                                     layout=self.layout)
    batch_size = 20
    input_size = 50
    hidden_size = 30
    seq_len = 10
    ctx = default_context()
    if layout == 'TNC':
        rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, input_size), ctx=ctx)
    elif layout == 'NTC':
        rnn_data = mx.nd.normal(loc=0, scale=1, shape=(batch_size, seq_len, input_size), ctx=ctx)
    else:
        print("Wrong layout")
        return
    valid_length = mx.nd.round(mx.nd.random.uniform(low=1, high=10, shape=(batch_size), ctx=ctx))
    state_shape = (batch_size, hidden_size)
    states = [mx.nd.normal(loc=0, scale=1, shape=state_shape, ctx=ctx) for i in range(num_states)]

    cell = cell_type(hidden_size)
    cell.initialize(ctx=default_context())
    if layout == 'TNC':
        cell(rnn_data[0], states)
    else:
        cell(rnn_data[:,0,:], states)
    params1 = cell.collect_params()
    orig_params1 = copy.deepcopy(params1)

    trainer = gluon.Trainer(params1, 'sgd', {'learning_rate' : 0.03})
    with mx.autograd.record():
        res1, states1 = cell.unroll(seq_len, rnn_data, states, valid_length=valid_length,
                                    layout=layout, merge_outputs=True)
    res1.backward()
    trainer.step(batch_size)

    configs = [
            lambda layer: None,
            lambda layer: layer.hybridize(),
            lambda layer: layer.hybridize({'inline_limit': 0}),
            lambda layer: layer.hybridize({'static_alloc': True}),
            lambda layer: layer.hybridize({'static_alloc': True, 'static_shape': True}) ]
    # We can't pass None to a hybrid block, but it accepts an empty list.
    # so we use an empty list to represent valid_length if it's None.
    if valid_length is None:
        valid_length = []
    for config in configs:
        layer = RNNLayer(cell_type, hidden_size, layout)
        layer.initialize(ctx=default_context())
        config(layer)
        res2, states2 = layer(rnn_data, states, valid_length)
        params2 = layer.collect_params()
        for key, val in orig_params1.items():
            params2['cell.' + key].set_data(copy.deepcopy(val.data()))

        trainer = gluon.Trainer(params2, 'sgd', {'learning_rate' : 0.03})
        with mx.autograd.record():
            res2, states2 = layer(rnn_data, states, valid_length)
        assert_almost_equal(res1, res2, rtol=0.001, atol=0.0001)
        assert len(states1) == len(states2)
        for i in range(len(states1)):
            assert_almost_equal(states1[i], states2[i], rtol=0.001, atol=0.0001)
        res2.backward()
        trainer.step(batch_size)

        for key, val in params1.items():
            weight1 = val.data()
            weight2 = params2['cell.' + key].data()
            assert_almost_equal(weight1, weight2, rtol=0.001, atol=0.0001)
