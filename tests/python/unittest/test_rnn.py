import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose


def test_rnn():
    cell = mx.rnn.RNNCell(100, prefix='rnn_')
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_lstm():
    cell = mx.rnn.LSTMCell(100, prefix='rnn_', forget_bias=1.0)
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_lstm_forget_bias():
    forget_bias = 2.0
    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.LSTMCell(100, forget_bias=forget_bias, prefix='l0_'))
    stack.add(mx.rnn.LSTMCell(100, forget_bias=forget_bias, prefix='l1_'))

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


def test_gru():
    cell = mx.rnn.GRUCell(100, prefix='rnn_')
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_residual():
    cell = mx.rnn.ResidualCell(mx.rnn.GRUCell(50, prefix='rnn_'))
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(2)]
    outputs, _ = cell.unroll(2, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == \
           ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == \
           ['rnn_t0_out_plus_residual_output', 'rnn_t1_out_plus_residual_output']

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
    cell = mx.rnn.ResidualCell(
            mx.rnn.BidirectionalCell(
                mx.rnn.GRUCell(25, prefix='rnn_l_'),
                mx.rnn.GRUCell(25, prefix='rnn_r_')))

    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(2)]
    outputs, _ = cell.unroll(2, inputs, merge_outputs=False)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == \
           ['rnn_l_h2h_bias', 'rnn_l_h2h_weight', 'rnn_l_i2h_bias', 'rnn_l_i2h_weight',
            'rnn_r_h2h_bias', 'rnn_r_h2h_weight', 'rnn_r_i2h_bias', 'rnn_r_i2h_weight']
    assert outputs.list_outputs() == \
           ['bi_t0_plus_residual_output', 'bi_t1_plus_residual_output']

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
    cell = mx.rnn.SequentialRNNCell()
    for i in range(5):
        if i == 1:
            cell.add(mx.rnn.ResidualCell(mx.rnn.LSTMCell(100, prefix='rnn_stack%d_' % i)))
        else:
            cell.add(mx.rnn.LSTMCell(100, prefix='rnn_stack%d_'%i))
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    keys = sorted(cell.params._params.keys())
    for i in range(5):
        assert 'rnn_stack%d_h2h_weight'%i in keys
        assert 'rnn_stack%d_h2h_bias'%i in keys
        assert 'rnn_stack%d_i2h_weight'%i in keys
        assert 'rnn_stack%d_i2h_bias'%i in keys
    assert outputs.list_outputs() == ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_bidirectional():
    cell = mx.rnn.BidirectionalCell(
            mx.rnn.LSTMCell(100, prefix='rnn_l0_'),
            mx.rnn.LSTMCell(100, prefix='rnn_r0_'),
            output_prefix='rnn_bi_')
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert outputs.list_outputs() == ['rnn_bi_t0_output', 'rnn_bi_t1_output', 'rnn_bi_t2_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 200), (10, 200), (10, 200)]


def test_zoneout():
    cell = mx.rnn.ZoneoutCell(mx.rnn.RNNCell(100, prefix='rnn_'), zoneout_outputs=0.5,
                              zoneout_states=0.5)
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_unfuse():
    cell = mx.rnn.FusedRNNCell(100, num_layers=3, mode='lstm',
                               prefix='test_', bidirectional=True,
                               dropout=0.5)
    cell = cell.unfuse()
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert outputs.list_outputs() == ['test_bi_l2_t0_output', 'test_bi_l2_t1_output', 'test_bi_l2_t2_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 200), (10, 200), (10, 200)]


if __name__ == '__main__':
    test_rnn()
    test_lstm()
    test_lstm_forget_bias()
    test_gru()
    test_stack()
    test_bidirectional()
    test_unfuse()
