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
"""Contain helpers for creating LSTM symbolic graph for training and inference """

from __future__ import print_function

from collections import namedtuple

import mxnet as mx


__all__ = ["lstm_unroll", "init_states"]


LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])


def _lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def _lstm_unroll_base(num_lstm_layer, seq_len, num_hidden):
    """ Returns symbol for LSTM model up to loss/softmax"""
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
    assert len(last_states) == num_lstm_layer

    # embedding layer
    data = mx.sym.Variable('data')
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for i in range(num_lstm_layer):
            next_state = _lstm(
                num_hidden=num_hidden,
                indata=hidden,
                prev_state=last_states[i],
                param=param_cells[i],
                seqidx=seqidx,
                layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred_fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=11, name="pred_fc")
    return pred_fc


def _add_warp_ctc_loss(pred, seq_len, num_label, label):
    """ Adds Symbol.contrib.ctc_loss on top of pred symbol and returns the resulting symbol """
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    return mx.sym.WarpCTC(data=pred, label=label, label_length=num_label, input_length=seq_len)


def _add_mxnet_ctc_loss(pred, seq_len, label):
    """ Adds Symbol.WapCTC on top of pred symbol and returns the resulting symbol """
    pred_ctc = mx.sym.Reshape(data=pred, shape=(-4, seq_len, -1, 0))

    loss = mx.sym.contrib.ctc_loss(data=pred_ctc, label=label)
    ctc_loss = mx.sym.MakeLoss(loss)

    softmax_class = mx.symbol.SoftmaxActivation(data=pred)
    softmax_loss = mx.sym.MakeLoss(softmax_class)
    softmax_loss = mx.sym.BlockGrad(softmax_loss)
    return mx.sym.Group([softmax_loss, ctc_loss])


def _add_ctc_loss(pred, seq_len, num_label, loss_type):
    """ Adds CTC loss on top of pred symbol and returns the resulting symbol """
    label = mx.sym.Variable('label')
    if loss_type == 'warpctc':
        print("Using WarpCTC Loss")
        sm = _add_warp_ctc_loss(pred, seq_len, num_label, label)
    else:
        print("Using MXNet CTC Loss")
        assert loss_type == 'ctc'
        sm = _add_mxnet_ctc_loss(pred, seq_len, label)
    return sm


def lstm_unroll(num_lstm_layer, seq_len, num_hidden, num_label, loss_type=None):
    """
    Creates an unrolled LSTM symbol for inference if loss_type is not specified, and for training
    if loss_type is specified. loss_type must be one of 'ctc' or 'warpctc'

    Parameters
    ----------
    num_lstm_layer: int
    seq_len: int
    num_hidden: int
    num_label: int
    loss_type: str
        'ctc' or 'warpctc'

    Returns
    -------
    mxnet.symbol.symbol.Symbol
    """
    # Create the base (shared between training and inference) and add loss to the end
    pred = _lstm_unroll_base(num_lstm_layer, seq_len, num_hidden)

    if loss_type:
        # Training mode, add loss
        return _add_ctc_loss(pred, seq_len, num_label, loss_type)
    else:
        # Inference mode, add softmax
        return mx.sym.softmax(data=pred, name='softmax')


def init_states(batch_size, num_lstm_layer, num_hidden):
    """
    Returns name and shape of init states of LSTM network

    Parameters
    ----------
    batch_size: list of tuple of str and tuple of int and int
    num_lstm_layer: int
    num_hidden: int

    Returns
    -------
    list of tuple of str and tuple of int and int
    """
    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    return init_c + init_h
