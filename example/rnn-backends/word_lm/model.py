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


def rnn(bptt, vocab_size, num_embed, nhid,
        num_layers, dropout, batch_size, tied, backend):
    # ==================================================================================================================
    # Encoder
    # ==================================================================================================================

    data = mx.sym.Variable('data')
    weight = mx.sym.Variable("encoder_weight", init=mx.init.Uniform(0.1))
    embed = mx.sym.Embedding(data=data, weight=weight, input_dim=vocab_size,
                             output_dim=num_embed, name='embed')

    outputs = mx.sym.Dropout(embed, p=dropout)

    # ==================================================================================================================
    # RNN
    # ==================================================================================================================

    # stacked rnn layers

    # Given below is the original source code, which we do argue makes bad use of CuDNN-RNN
    # as there is no need to slice each layer apart, please refer to the open issue on github @
    #     https://github.com/apache/incubator-mxnet/issues/10304
    # The throughput measurements reported by MXNet speedometer showed that
    # there is around 10~20% increase in throughput after all the layers have been fused.
    # The implementation also messed up with the order of cell and hidden state,
    # because according to the state order specified in rnn_cell.py (+L682),
    # cell state should appear AFTER hidden state in FusedRNNCell. However, this has zero effect
    # on the final training results because they are both zero-initialized.

    states = []
    state_names = []

    # for i in range(num_layers):
    #     prefix = 'lstm_l%d_' % i
    #     cell = mx.rnn.FusedRNNCell(num_hidden=nhid, prefix=prefix, get_next_state=True,
    #                                forget_bias=0.0, dropout=dropout)
    #     state_shape = (1, batch_size, nhid)
    #     begin_cell_state_name = prefix + 'cell'
    #     begin_hidden_state_name = prefix + 'hidden'
    #     begin_cell_state = mx.sym.var(begin_cell_state_name, shape=state_shape)
    #     begin_hidden_state = mx.sym.var(begin_hidden_state_name, shape=state_shape)
    #     state_names += [begin_cell_state_name, begin_hidden_state_name]
    #     outputs, next_states = cell.unroll(bptt, inputs=outputs,
    #                                        begin_state=[begin_cell_state, begin_hidden_state],
    #                                        merge_outputs=True, layout='TNC')
    #     outputs = mx.sym.Dropout(outputs, p=dropout)
    #     states += next_states

    prefix = 'lstm_'

    if backend == 'default':
        cell = mx.rnn.FusedRNNCell(num_hidden=nhid, num_layers=num_layers,
                                   get_next_state=True, forget_bias=0.0,
                                   dropout=dropout, prefix=prefix).unfuse()
        begin_state = []

        for layer_idx in range(num_layers):
            layer_prefix = '%sl%d_' % (prefix, layer_idx)

            hidden_state_name, cell_state_name = layer_prefix + 'hidden', layer_prefix + 'cell'
            state_names += [hidden_state_name, cell_state_name]
            begin_hidden_state = mx.sym.Variable(name=hidden_state_name, shape=(batch_size, nhid))
            begin_cell_state = mx.sym.Variable(name=cell_state_name, shape=(batch_size, nhid))
            begin_state += [begin_hidden_state, begin_cell_state]

        outputs, states = cell.unroll(length=bptt, inputs=outputs,
                                      begin_state=begin_state,
                                      merge_outputs=True, layout='NTC')

    elif backend == 'cudnn':
        cell = mx.rnn.FusedRNNCell(num_hidden=nhid, num_layers=num_layers,
                                   get_next_state=True, forget_bias=0.0,
                                   dropout=dropout, prefix=prefix)
        hidden_state_name, cell_state_name = prefix + 'hidden', prefix + 'cell'
        state_names = [hidden_state_name, cell_state_name]
        begin_hidden_state = mx.sym.Variable(name=hidden_state_name, shape=(num_layers, batch_size, nhid))
        begin_cell_state = mx.sym.Variable(name=cell_state_name, shape=(num_layers, batch_size, nhid))
        outputs, states = cell.unroll(length=bptt, inputs=outputs,
                                      begin_state=[begin_hidden_state,
                                                   begin_cell_state],
                                      merge_outputs=True, layout='TNC')
    elif backend == 'open':
        cell = mx.rnn.OpenLSTMRNNCell(num_hidden=nhid, num_layers=num_layers,
                                      get_next_state=True, forget_bias=0.0,
                                      dropout=dropout, prefix=prefix)
        hidden_state_name, cell_state_name = prefix + 'hidden', prefix + 'cell'
        state_names = [hidden_state_name, cell_state_name]
        begin_hidden_state = mx.sym.Variable(name=hidden_state_name, shape=(num_layers, batch_size, nhid))
        begin_cell_state = mx.sym.Variable(name=cell_state_name, shape=(num_layers, batch_size, nhid))
        outputs, states = cell.unroll(length=bptt, inputs=outputs,
                                      begin_state=[begin_hidden_state,
                                                   begin_cell_state])
    else:
        assert 0, "Invalid backend argument. " \
                  "Valid ones are default/cudnn/open."

    outputs = mx.sym.Dropout(outputs, p=dropout)

    # ==================================================================================================================
    # Decoder
    # ==================================================================================================================

    pred = mx.sym.Reshape(outputs, shape=(-1, nhid))
    if tied:
        assert(nhid == num_embed), \
               "the number of hidden units and the embedding size must batch for weight tying"
        pred = mx.sym.FullyConnected(data=pred, weight=weight,
                                     num_hidden=vocab_size, name='pred')
    else:
        pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_size, name='pred')
    pred = mx.sym.Reshape(pred, shape=(-1, vocab_size))

    return pred, [mx.sym.stop_gradient(s) for s in states], state_names


def softmax_ce_loss(pred):
    # softmax cross-entropy loss
    label = mx.sym.Variable('label')
    label = mx.sym.Reshape(label, shape=(-1,))
    logits = mx.sym.log_softmax(pred, axis=-1)
    loss = -mx.sym.pick(logits, label, axis=-1, keepdims=True)
    loss = mx.sym.mean(loss, axis=0, exclude=True)
    return mx.sym.make_loss(loss, name='nll')
