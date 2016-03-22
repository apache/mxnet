# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import LSTMState, LSTMParam, lstm
from rnn import RNNState, RNNParam, RNNModel, rnn

# we define a new unrolling function here because the original
# one in lstm.py concats all the labels at the last layer together,
# making the mini-batch size of the label different from the data.
# I think the existing data-parallelization code need some modification
# to allow this situation to work properly
def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    loss_all = []
    for seqidx in range(seq_len):
        # embeding layer
        data = mx.sym.Variable("data/%d" % seqidx)

        hidden = mx.sym.Embedding(data=data, weight=embed_weight,
                                  input_dim=input_size,
                                  output_dim=num_embed,
                                  name="t%d_embed" % seqidx)
        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        fc = mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias,
                                   num_hidden=num_label)
        sm = mx.sym.SoftmaxOutput(data=fc, label=mx.sym.Variable('label/%d' % seqidx),
                                  name='t%d_sm' % seqidx)
        loss_all.append(sm)

    return mx.sym.Group(loss_all)


def rnn_unroll(num_rnn_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0., batch_norm=False):

    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_rnn_layer):
        param_cells.append(RNNParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                    i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                    h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                    h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = RNNState(h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_rnn_layer)

    loss_all = []
    for seqidx in range(seq_len):
        # embeding layer
        data = mx.sym.Variable("data/%d" % seqidx)

        hidden = mx.sym.Embedding(data=data, weight=embed_weight,
                                  input_dim=input_size,
                                  output_dim=num_embed,
                                  name="t%d_embed" % seqidx)
        # stack RNN
        for i in range(num_rnn_layer):
            if i==0:
                dp=0.
            else:
                dp = dropout
            next_state = rnn(num_hidden, in_data=hidden,
                             prev_state=last_states[i],
                             param=param_cells[i],
                             seqidx=seqidx, layeridx=i, dropout=dp, batch_norm=batch_norm)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        fc = mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias,
                                   num_hidden=num_label)
        sm = mx.sym.SoftmaxOutput(data=fc, label=mx.sym.Variable('label/%d' % seqidx),
                                  name='t%d_sm' % seqidx)
        loss_all.append(sm)
    return mx.sym.Group(loss_all)
