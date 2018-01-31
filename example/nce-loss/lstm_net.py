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

# pylint: disable=missing-docstring
from __future__ import print_function

from collections import namedtuple

import mxnet as mx
from nce import nce_loss

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])


def _lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
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


def get_lstm_net(vocab_size, seq_len, num_lstm_layer, num_hidden):
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

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    label_embed_weight = mx.sym.Variable('label_embed_weight')
    data_embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                  weight=embed_weight,
                                  output_dim=100, name='data_embed')
    datavec = mx.sym.SliceChannel(data=data_embed,
                                  num_outputs=seq_len,
                                  squeeze_axis=True, name='data_slice')
    labelvec = mx.sym.SliceChannel(data=label,
                                   num_outputs=seq_len,
                                   squeeze_axis=True, name='label_slice')
    labelweightvec = mx.sym.SliceChannel(data=label_weight,
                                         num_outputs=seq_len,
                                         squeeze_axis=True, name='label_weight_slice')
    probs = []
    for seqidx in range(seq_len):
        hidden = datavec[seqidx]

        for i in range(num_lstm_layer):
            next_state = _lstm(num_hidden, indata=hidden,
                               prev_state=last_states[i],
                               param=param_cells[i],
                               seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state

        probs.append(nce_loss(data=hidden,
                              label=labelvec[seqidx],
                              label_weight=labelweightvec[seqidx],
                              embed_weight=label_embed_weight,
                              vocab_size=vocab_size,
                              num_hidden=100))
    return mx.sym.Group(probs)
