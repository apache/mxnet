import mxnet as mx
import numpy as np

def test_rnn():
    cell = mx.rnn.RNNCell(100, prefix='rnn_')
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

def test_lstm():
    cell = mx.rnn.LSTMCell(100, prefix='rnn_')
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_gru():
    cell = mx.rnn.GRUCell(100, prefix='rnn_')
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


def test_stack():
    cell = mx.rnn.SequentialRNNCell()
    for i in range(5):
        cell.add(mx.rnn.LSTMCell(100, prefix='rnn_stack%d_'%i))
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
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
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert outputs.list_outputs() == ['rnn_bi_t0_output', 'rnn_bi_t1_output', 'rnn_bi_t2_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 200), (10, 200), (10, 200)]

def test_unfuse():
    cell = mx.rnn.FusedRNNCell(100, num_layers=1, mode='lstm',
            prefix='test_', bidirectional=True).unfuse()
    outputs, _ = cell.unroll(3, input_prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert outputs.list_outputs() == ['test_bi_lstm_0t0_output', 'test_bi_lstm_0t1_output', 'test_bi_lstm_0t2_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 200), (10, 200), (10, 200)]

if __name__ == '__main__':
    test_rnn()
    test_lstm()
    test_gru()
    test_stack()
    test_bidirectional()
    test_unfuse()
