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
from numpy.testing import assert_allclose
import unittest
from mxnet.test_utils import almost_equal, assert_almost_equal
from common import assert_raises_cudnn_not_satisfied, with_seed

def test_rnn():
    cell = gluon.rnn.RNNCell(100, prefix='rnn_')
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight',
                                                    'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_lstm():
    cell = gluon.rnn.LSTMCell(100, prefix='rnn_')
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_lstm_forget_bias():
    forget_bias = 2.0
    stack = gluon.rnn.SequentialRNNCell()
    stack.add(gluon.rnn.LSTMCell(100, i2h_bias_initializer=mx.init.LSTMBias(forget_bias), prefix='l0_'))
    stack.add(gluon.rnn.LSTMCell(100, i2h_bias_initializer=mx.init.LSTMBias(forget_bias), prefix='l1_'))

    dshape = (32, 1, 200)
    data = mx.sym.Variable('data')

    sym, _ = stack.unroll(1, data, merge_outputs=True)
    mod = mx.mod.Module(sym, label_names=None, context=mx.cpu(0))
    mod.bind(data_shapes=[('data', dshape)], label_shapes=None)

    mod.init_params()

    bias_argument = next(x for x in sym.list_arguments() if x.endswith('i2h_bias'))
    expected_bias = np.hstack([np.zeros((100,)),
                               forget_bias * np.ones(100, ), np.zeros((2 * 100,))])
    assert_allclose(mod.get_params()[0][bias_argument].asnumpy(), expected_bias)


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_lstm_cpu_inference():
    # should behave the same as lstm cell
    EXPECTED_LSTM_OUTPUT = np.array([[[0.72045636, 0.72045636, 0.95215213, 0.95215213],
                                      [0.72045636, 0.72045636, 0.95215213, 0.95215213]],
                                     [[0.95215213, 0.95215213, 0.72045636, 0.72045636],
                                      [0.95215213, 0.95215213, 0.72045636, 0.72045636]]])
    x = mx.nd.ones(shape=(2, 2, 2))
    model = mx.gluon.rnn.LSTM(2, num_layers=6, bidirectional=True)
    model_cell = model._unfuse()
    model.initialize(mx.init.One())

    y = model(x).asnumpy()
    y_cell = model_cell.unroll(2, x, layout='TNC', merge_outputs=True)[0].asnumpy()

    mx.test_utils.assert_almost_equal(y_cell, EXPECTED_LSTM_OUTPUT,
                                      rtol=1e-3, atol=1e-5)
    mx.test_utils.assert_almost_equal(y, EXPECTED_LSTM_OUTPUT,
                                      rtol=1e-3, atol=1e-5)


def test_gru():
    cell = gluon.rnn.GRUCell(100, prefix='rnn_')
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_residual():
    cell = gluon.rnn.ResidualCell(gluon.rnn.GRUCell(50, prefix='rnn_'))
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(2)]
    outputs, _ = cell.unroll(2, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == \
           ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    # assert outputs.list_outputs() == \
    #        ['rnn_t0_out_plus_residual_output', 'rnn_t1_out_plus_residual_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10, 50), rnn_t1_data=(10, 50))
    assert outs == [(10, 50), (10, 50)]
    outputs = outputs.eval(rnn_t0_data=mx.nd.ones((10, 50)),
                           rnn_t1_data=mx.nd.ones((10, 50)),
                           rnn_i2h_weight=mx.nd.zeros((150, 50)),
                           rnn_i2h_bias=mx.nd.zeros((150,)),
                           rnn_h2h_weight=mx.nd.zeros((150, 50)),
                           rnn_h2h_bias=mx.nd.zeros((150,)))
    expected_outputs = np.ones((10, 50))
    assert np.array_equal(outputs[0].asnumpy(), expected_outputs)
    assert np.array_equal(outputs[1].asnumpy(), expected_outputs)


def test_residual_bidirectional():
    cell = gluon.rnn.ResidualCell(
            gluon.rnn.BidirectionalCell(
                gluon.rnn.GRUCell(25, prefix='rnn_l_'),
                gluon.rnn.GRUCell(25, prefix='rnn_r_')))

    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(2)]
    outputs, _ = cell.unroll(2, inputs, merge_outputs=False)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == \
           ['rnn_l_h2h_bias', 'rnn_l_h2h_weight', 'rnn_l_i2h_bias', 'rnn_l_i2h_weight',
            'rnn_r_h2h_bias', 'rnn_r_h2h_weight', 'rnn_r_i2h_bias', 'rnn_r_i2h_weight']
    # assert outputs.list_outputs() == \
    #        ['bi_t0_plus_residual_output', 'bi_t1_plus_residual_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10, 50), rnn_t1_data=(10, 50))
    assert outs == [(10, 50), (10, 50)]
    outputs = outputs.eval(rnn_t0_data=mx.nd.ones((10, 50))+5,
                           rnn_t1_data=mx.nd.ones((10, 50))+5,
                           rnn_l_i2h_weight=mx.nd.zeros((75, 50)),
                           rnn_l_i2h_bias=mx.nd.zeros((75,)),
                           rnn_l_h2h_weight=mx.nd.zeros((75, 25)),
                           rnn_l_h2h_bias=mx.nd.zeros((75,)),
                           rnn_r_i2h_weight=mx.nd.zeros((75, 50)),
                           rnn_r_i2h_bias=mx.nd.zeros((75,)),
                           rnn_r_h2h_weight=mx.nd.zeros((75, 25)),
                           rnn_r_h2h_bias=mx.nd.zeros((75,)))
    expected_outputs = np.ones((10, 50))+5
    assert np.array_equal(outputs[0].asnumpy(), expected_outputs)
    assert np.array_equal(outputs[1].asnumpy(), expected_outputs)


def test_stack():
    cell = gluon.rnn.SequentialRNNCell()
    for i in range(5):
        if i == 1:
            cell.add(gluon.rnn.ResidualCell(gluon.rnn.LSTMCell(100, prefix='rnn_stack%d_' % i)))
        else:
            cell.add(gluon.rnn.LSTMCell(100, prefix='rnn_stack%d_'%i))
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    keys = sorted(cell.collect_params().keys())
    for i in range(5):
        assert 'rnn_stack%d_h2h_weight'%i in keys
        assert 'rnn_stack%d_h2h_bias'%i in keys
        assert 'rnn_stack%d_i2h_weight'%i in keys
        assert 'rnn_stack%d_i2h_bias'%i in keys
    assert outputs.list_outputs() == ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_hybridstack():
    cell = gluon.rnn.HybridSequentialRNNCell()
    for i in range(5):
        if i == 1:
            cell.add(gluon.rnn.ResidualCell(gluon.rnn.LSTMCell(100, prefix='rnn_stack%d_' % i)))
        else:
            cell.add(gluon.rnn.LSTMCell(100, prefix='rnn_stack%d_'%i))
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    keys = sorted(cell.collect_params().keys())
    for i in range(5):
        assert 'rnn_stack%d_h2h_weight'%i in keys
        assert 'rnn_stack%d_h2h_bias'%i in keys
        assert 'rnn_stack%d_i2h_weight'%i in keys
        assert 'rnn_stack%d_i2h_bias'%i in keys
    assert outputs.list_outputs() == ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

    # Test HybridSequentialRNNCell nested in nn.HybridBlock, SequentialRNNCell will fail in this case
    class BidirectionalOfSequential(gluon.HybridBlock):
        def __init__(self):
            super(BidirectionalOfSequential, self).__init__()

            with self.name_scope():
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
    net.collect_params().initialize()
    outs, _ = net(x)

    assert outs.shape == (10, 3, 200)


def test_bidirectional():
    cell = gluon.rnn.BidirectionalCell(
            gluon.rnn.LSTMCell(100, prefix='rnn_l0_'),
            gluon.rnn.LSTMCell(100, prefix='rnn_r0_'),
            output_prefix='rnn_bi_')
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert outputs.list_outputs() == ['rnn_bi_t0_output', 'rnn_bi_t1_output', 'rnn_bi_t2_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 200), (10, 200), (10, 200)]


@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
@with_seed()
def test_layer_bidirectional():
    class RefBiLSTM(gluon.Block):
        def __init__(self, size, **kwargs):
            super(RefBiLSTM, self).__init__(**kwargs)
            with self.name_scope():
                self._lstm_fwd = gluon.rnn.LSTM(size, bidirectional=False, prefix='l0')
                self._lstm_bwd = gluon.rnn.LSTM(size, bidirectional=False, prefix='r0')

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
        weights['lstm_{}0_i2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, in_size))
        weights['lstm_{}0_h2h_weight'.format(d)] = mx.random.uniform(shape=(size*4, size))
        weights['lstm_{}0_i2h_bias'.format(d)] = mx.random.uniform(shape=(size*4,))
        weights['lstm_{}0_h2h_bias'.format(d)] = mx.random.uniform(shape=(size*4,))

    net = gluon.rnn.LSTM(size, bidirectional=True, prefix='lstm_')
    ref_net = RefBiLSTM(size, prefix='lstm_')
    net.initialize()
    ref_net.initialize()
    net_params = net.collect_params()
    ref_net_params = ref_net.collect_params()
    for k in weights:
        net_params[k].set_data(weights[k])
        ref_net_params[k.replace('l0', 'l0l0').replace('r0', 'r0l0')].set_data(weights[k])

    data = mx.random.uniform(shape=(11, 10, in_size))
    assert_allclose(net(data).asnumpy(), ref_net(data).asnumpy(), rtol=1e-04, atol=1e-02)



def test_zoneout():
    cell = gluon.rnn.ZoneoutCell(gluon.rnn.RNNCell(100, prefix='rnn_'), zoneout_outputs=0.5,
                              zoneout_states=0.5)
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_unroll_layout():
    cell = gluon.rnn.HybridSequentialRNNCell()
    for i in range(5):
        if i == 1:
            cell.add(gluon.rnn.ResidualCell(gluon.rnn.LSTMCell(100, prefix='rnn_stack%d_' % i)))
        else:
            cell.add(gluon.rnn.LSTMCell(100, prefix='rnn_stack%d_'%i))
    cell.collect_params().initialize()
    inputs = [mx.nd.random.uniform(shape=(10,50)) for _ in range(3)]
    outputs, _ = cell.unroll(3, inputs, layout='TNC')
    assert outputs[0].shape == (10, 100)
    assert outputs[1].shape == (10, 100)
    assert outputs[2].shape == (10, 100)

    outputs, _ = cell.unroll(3, inputs, layout='NTC')
    assert outputs[0].shape == (10, 100)
    assert outputs[1].shape == (10, 100)
    assert outputs[2].shape == (10, 100)


def check_rnn_forward(layer, inputs, deterministic=True):
    if isinstance(inputs, mx.nd.NDArray):
        inputs.attach_grad()
    else:
        for x in inputs:
            x.attach_grad()
    layer.collect_params().initialize()
    with mx.autograd.record():
        out = layer.unroll(3, inputs, merge_outputs=False)[0]
        mx.autograd.backward(out)
        out = layer.unroll(3, inputs, merge_outputs=True)[0]
        out.backward()

    np_out = out.asnumpy()
    if isinstance(inputs, mx.nd.NDArray):
        np_dx = inputs.grad.asnumpy()
    else:
        np_dx = np.stack([x.grad.asnumpy() for x in inputs], axis=1)

    layer.hybridize()

    with mx.autograd.record():
        out = layer.unroll(3, inputs, merge_outputs=False)[0]
        mx.autograd.backward(out)
        out = layer.unroll(3, inputs, merge_outputs=True)[0]
        out.backward()

    if isinstance(inputs, mx.nd.NDArray):
        input_grads = inputs.grad.asnumpy()
    else:
        input_grads = np.stack([x.grad.asnumpy() for x in inputs], axis=1)

    if deterministic:
        mx.test_utils.assert_almost_equal(np_out, out.asnumpy(), rtol=1e-3, atol=1e-5)
        mx.test_utils.assert_almost_equal(np_dx, input_grads, rtol=1e-3, atol=1e-5)


def test_rnn_cells():
    check_rnn_forward(gluon.rnn.LSTMCell(100, input_size=200), mx.nd.ones((8, 3, 200)))
    check_rnn_forward(gluon.rnn.RNNCell(100, input_size=200), mx.nd.ones((8, 3, 200)))
    check_rnn_forward(gluon.rnn.GRUCell(100, input_size=200), mx.nd.ones((8, 3, 200)))

    check_rnn_forward(gluon.rnn.LSTMCell(100, input_size=200),
                      [mx.nd.ones((8, 200)), mx.nd.ones((8, 200)), mx.nd.ones((8, 200))])
    check_rnn_forward(gluon.rnn.RNNCell(100, input_size=200),
                      [mx.nd.ones((8, 200)), mx.nd.ones((8, 200)), mx.nd.ones((8, 200))])
    check_rnn_forward(gluon.rnn.GRUCell(100, input_size=200),
                      [mx.nd.ones((8, 200)), mx.nd.ones((8, 200)), mx.nd.ones((8, 200))])

    bilayer = gluon.rnn.BidirectionalCell(gluon.rnn.LSTMCell(100, input_size=200),
                                       gluon.rnn.LSTMCell(100, input_size=200))
    check_rnn_forward(bilayer, mx.nd.ones((8, 3, 200)))

    check_rnn_forward(gluon.rnn.DropoutCell(0.5), mx.nd.ones((8, 3, 200)), False)

    check_rnn_forward(gluon.rnn.ZoneoutCell(gluon.rnn.LSTMCell(100, input_size=200),
                                         0.5, 0.2),
                      mx.nd.ones((8, 3, 200)), False)

    net = gluon.rnn.SequentialRNNCell()
    net.add(gluon.rnn.LSTMCell(100, input_size=200))
    net.add(gluon.rnn.RNNCell(100, input_size=100))
    net.add(gluon.rnn.GRUCell(100, input_size=100))
    check_rnn_forward(net, mx.nd.ones((8, 3, 200)))


def test_rnn_cells_export_import():
    class RNNLayer(gluon.HybridBlock):
        def __init__(self):
            super(RNNLayer, self).__init__()
            with self.name_scope():
                self.cell = gluon.rnn.RNNCell(hidden_size=1)

        def hybrid_forward(self, F, seq):
            outputs, state = self.cell.unroll(inputs=seq, length=2, merge_outputs=True)
            return outputs

    class LSTMLayer(gluon.HybridBlock):
        def __init__(self):
            super(LSTMLayer, self).__init__()
            with self.name_scope():
                self.cell = gluon.rnn.LSTMCell(hidden_size=1)

        def hybrid_forward(self, F, seq):
            outputs, state = self.cell.unroll(inputs=seq, length=2, merge_outputs=True)
            return outputs

    class GRULayer(gluon.HybridBlock):
        def __init__(self):
            super(GRULayer, self).__init__()
            with self.name_scope():
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
            ctx=mx.Context.default_ctx
        )
        output2 = symbol(input)
        assert_almost_equal(output1.asnumpy(), output2.asnumpy())


def check_rnn_layer_forward(layer, inputs, states=None, run_only=False, ctx=mx.cpu()):
    layer.collect_params().initialize(ctx=ctx)
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
    net.collect_params().initialize(ctx=ctx)
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
    net2.collect_params().initialize(ctx=ctx)
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
    net3.collect_params().initialize(ctx=ctx)
    net3.cast(dtype2)
    with mx.autograd.record():
        out = net3(mx.nd.ones((2, 3, 10), dtype=dtype2, ctx=ctx))
        out.backward()
        out = out.asnumpy()

def test_rnn_layers_fp32():
    run_rnn_layers('float32', 'float32')

@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
@unittest.skipIf(mx.context.num_gpus() == 0, "RNN FP16 only implemented for GPU for now")
def test_rnn_layers_fp16():
    run_rnn_layers('float16', 'float32', mx.gpu())


def test_rnn_unroll_variant_length():
    # Test for imperative usage
    cell_list = []
    for base_cell_class in [gluon.rnn.RNNCell, gluon.rnn.LSTMCell, gluon.rnn.GRUCell]:
        cell_list.append(base_cell_class(20))
        cell_list.append(gluon.rnn.BidirectionalCell(
                         l_cell=base_cell_class(20),
                         r_cell=base_cell_class(20)))
        cell_list.append(gluon.contrib.rnn.VariationalDropoutCell(base_cell=base_cell_class(20)))
    stack_res_rnn_cell = gluon.rnn.SequentialRNNCell()
    stack_res_rnn_cell.add(gluon.rnn.ResidualCell(base_cell=gluon.rnn.RNNCell(20)))
    stack_res_rnn_cell.add(gluon.rnn.ResidualCell(base_cell=gluon.rnn.RNNCell(20)))
    cell_list.append(stack_res_rnn_cell)
    batch_size = 4
    max_length = 10
    valid_length = [3, 10, 5, 6]
    valid_length_nd = mx.nd.array(valid_length)
    for cell in cell_list:
        cell.collect_params().initialize()
        cell.hybridize()
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
    # For symbolic test, we need to make sure that it can be binded and run
    data = mx.sym.var('data', shape=(4, 10, 2))
    cell = gluon.rnn.RNNCell(100)
    valid_length = mx.sym.var('valid_length', shape=(4,))
    outs, states = cell.unroll(length=10, inputs=data, valid_length=valid_length,
                               merge_outputs=True, layout='NTC')
    mod = mx.mod.Module(states[0], data_names=('data', 'valid_length'), label_names=None,
                        context=mx.cpu())
    mod.bind(data_shapes=[('data', (4, 10, 2)), ('valid_length', (4,))], label_shapes=None)
    mod.init_params()
    mod.forward(mx.io.DataBatch([mx.random.normal(0, 1, (4, 10, 2)), mx.nd.array([3, 6, 10, 2])]))
    mod.get_outputs()[0].asnumpy()


def test_cell_fill_shape():
    cell = gluon.rnn.LSTMCell(10)
    cell.hybridize()
    check_rnn_forward(cell, mx.nd.ones((2, 3, 7)))
    assert cell.i2h_weight.shape[1] == 7, cell.i2h_weight.shape[1]

def test_layer_fill_shape():
    layer = gluon.rnn.LSTM(10)
    layer.hybridize()
    check_rnn_layer_forward(layer, mx.nd.ones((3, 2, 7)))
    print(layer)
    assert layer.l0_i2h_weight.shape[1] == 7, layer.l0_i2h_weight.shape[1]


def test_bidirectional_unroll_valid_length():
    def _check_bidirectional_unroll_valid_length(length):
        class BiLSTM(gluon.nn.HybridBlock):
            def __init__(self, rnn_size, time_step, **kwargs):
                super(BiLSTM, self).__init__(**kwargs)
                self.time_step = time_step
                with self.name_scope():
                    self.bi_lstm = gluon.rnn.BidirectionalCell(
                        gluon.rnn.LSTMCell(rnn_size, prefix='rnn_l0_'),
                        gluon.rnn.LSTMCell(rnn_size, prefix='rnn_r0_'),
                        output_prefix='lstm_bi_')

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


if __name__ == '__main__':
    import nose
    nose.runmodule()
