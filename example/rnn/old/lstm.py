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

# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
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

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]

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
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    ################################################################################
    # Make label the same shape as our produced data path
    # I did not observe big speed difference between the following two ways

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))

    #label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
    #label = [label_slice[t] for t in range(seq_len)]
    #label = mx.sym.Concat(*label, dim=0)
    #label = mx.sym.Reshape(data=label, target_shape=(0,))
    ################################################################################

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm

def lstm_inference_symbol(num_lstm_layer, input_size,
                          num_hidden, num_embed, num_label, dropout=0.):
    seqidx = 0
    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    data = mx.sym.Variable("data")

    hidden = mx.sym.Embedding(data=data,
                              input_dim=input_size,
                              output_dim=num_embed,
                              weight=embed_weight,
                              name="embed")
    # stack LSTM
    for i in range(num_lstm_layer):
        if i==0:
            dp=0.
        else:
            dp = dropout
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[i],
                          param=param_cells[i],
                          seqidx=seqidx, layeridx=i, dropout=dp)
        hidden = next_state.h
        last_states[i] = next_state
    # decoder
    if dropout > 0.:
        hidden = mx.sym.Dropout(data=hidden, p=dropout)
    fc = mx.sym.FullyConnected(data=hidden, num_hidden=num_label,
                               weight=cls_weight, bias=cls_bias, name='pred')
    sm = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)
