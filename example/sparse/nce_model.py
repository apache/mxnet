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

def nce_loss():
    return None

#model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
#                       args.nlayers, args.dropout, args.tied)
def rnn_model(bptt, mode, vocab_size, num_embed, num_hidden,
              num_layers, use_dense_embedding):
    data = mx.sym.Variable('data')
    # TODO rename the label
    label = mx.sym.Variable('label')
    if use_dense_embedding:
        embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                 output_dim=num_embed, name='embed')
    else:
        weight = mx.sym.var("embedding_weight", stype='row_sparse')
        embed = mx.sym.contrib.SparseEmbedding(data=data, weight=weight, input_dim=vocab_size,
                                               output_dim=num_embed, name='embed')
    #stack.reset()
    assert(mode == 'lstm')
    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_'%i))

    outputs, states = stack.unroll(bptt, inputs=embed, merge_outputs=True)

    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    # TODO adjust the last layer
    pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_size, name='pred')

    label = mx.sym.Reshape(label, shape=(-1,))
    pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    return pred
