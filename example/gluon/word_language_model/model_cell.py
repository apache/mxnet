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
from mxnet import gluon
from mxnet.gluon import contrib, nn, rnn

class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size,
                 num_embed, num_hidden, num_layers,
                 dropout=0., var_drop_in=0., var_drop_state=0., var_drop_out=0.,
                 tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer=mx.init.Uniform(0.1))
            assert mode in ['rnn_relu', 'rnn_tanh', 'lstm', 'gru'], \
                   'Invalid mode %s. Options are rnn_relu, rnn_tanh, lstm, and gru'%mode
            self.rnn = self._get_rnn_cell(mode, num_layers, num_hidden, dropout,
                                          var_drop_in, var_drop_state, var_drop_out)

            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units=num_hidden,
                                        params=self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units=num_hidden)

            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn.unroll(emb.shape[0], emb, hidden, layout='TNC',
                                         merge_outputs=True)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

    def _get_rnn_cell(self, mode, num_layers, num_hidden, dropout,
                      var_drop_in, var_drop_state, var_drop_out):
        rnn_net = rnn.SequentialRNNCell()
        for i in range(num_layers):
            if mode == 'rnn_relu':
                cell = rnn.RNNCell(num_hidden, 'relu')
            elif mode == 'rnn_tanh':
                cell = rnn.RNNCell(num_hidden, 'tanh')
            elif mode == 'lstm':
                cell = rnn.LSTMCell(num_hidden)
            elif mode == 'gru':
                cell = rnn.GRUCell(num_hidden)
            if var_drop_in + var_drop_state + var_drop_out != 0:
                cell = contrib.rnn.VariationalDropoutCell(cell,
                                                          var_drop_in,
                                                          var_drop_state,
                                                          var_drop_out)

            rnn_net.add(cell)
            if i != num_layers - 1 and dropout != 0:
                rnn_net.add(rnn.DropoutCell(dropout))
        return rnn_net
