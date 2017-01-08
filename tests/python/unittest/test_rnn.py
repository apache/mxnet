import mxnet as mx
import numpy as np

def test_rnn():
    cell = mx.rnn.RNNCell(100)
    outputs, _, params = mx.rnn.rnn_unroll(cell, 3, prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert sorted(params._params.keys()) == ['rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

def test_lstm():
    cell = mx.rnn.LSTMCell(100)
    outputs, _, params = mx.rnn.rnn_unroll(cell, 3, prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    assert sorted(params._params.keys()) == ['rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

def test_stack():
    lstm = mx.rnn.LSTMCell(100)
    cell = mx.rnn.StackedRNNCell([lstm]*5)
    outputs, _, params = mx.rnn.rnn_unroll(cell, 3, prefix='rnn_')
    outputs = mx.sym.Group(outputs)
    keys = sorted(params._params.keys())
    for i in range(5):
        assert 'rnn_stack%d_h2h_weight'%i in keys
        assert 'rnn_stack%d_i2h_bias'%i in keys
        assert 'rnn_stack%d_i2h_weight'%i in keys
    assert outputs.list_outputs() == ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


if __name__ == '__main__':
    test_rnn()
    test_lstm()
    test_stack()
