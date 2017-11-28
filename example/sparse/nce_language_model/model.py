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

def nce_loss(pred, vocab_size, num_hidden, k, decoder_weight=None):
    # shape=(k,)
    sample = mx.sym.Variable('sample', shape=(k,), dtype='float32')
    label = mx.sym.Variable('label')
    # shape=(batch_size*bptt)
    label = mx.sym.Reshape(label, shape=(-1,), name="label_reshape")

    # weight and bias
    if decoder_weight is None:
        decoder_weight = mx.sym.var("decoder_weight", stype='row_sparse')
    decoder_bias = mx.sym.var("decoder_bias") #, stype='row_sparse')

    # lookup weights
    # shape=(k, num_hidden)
    sample_weight = mx.sym.contrib.SparseEmbedding(data=sample, weight=decoder_weight,
                                                   input_dim=vocab_size, output_dim=num_hidden)
    true_weight = mx.sym.contrib.SparseEmbedding(data=label, weight=decoder_weight,
                                                 input_dim=vocab_size, output_dim=num_hidden)
    # shape=(batch_size*bptt, num_hidden, 1)
    true_weight = mx.sym.Reshape(true_weight, (-1, num_hidden, 1))

    # lookup bias
    # shape=(k, 1)
    #sample_bias = mx.sym.contrib.SparseEmbedding(data=sample, weight=decoder_bias,
    sample_bias = mx.sym.Embedding(data=sample, weight=decoder_bias,
                                                 input_dim=vocab_size, output_dim=1)
    # shape=(batch_size*bptt, 1)
    #true_bias = mx.sym.contrib.SparseEmbedding(data=label, weight=decoder_bias,
    true_bias = mx.sym.Embedding(data=label, weight=decoder_bias,
                                               input_dim=vocab_size, output_dim=1)
    # pred.shape=(batch_size*bptt, num_hidden)
    # shape=(batch_size*bptt, k)
    sample_pred = mx.sym.FullyConnected(data=pred, weight=sample_weight,
                                        num_hidden=k, bias=mx.sym.Reshape(sample_bias, shape=(k,)))
    # shape=(batch_size*bptt, 1, num_hidden)
    pred_3d = mx.sym.Reshape(pred, (-1, 1, num_hidden), name='pred_reshape')
    # shape=(batch_size*bptt, 1, 1)
    true_pred = mx.sym.batch_dot(pred_3d, true_weight)
    true_pred = mx.sym.Reshape(true_pred, (-1, 1), name='true_pred_reshape')
    true_pred = mx.sym.broadcast_add(true_pred, true_bias)
    '''
    check = true_pred
    arg = check.list_arguments()
    arg_shape, out_shape, aux_shape = check.infer_shape(data=(32, 35), label=(32, 35)) #, sample=(32,)))

    for a, s in zip(arg, arg_shape):
        print(a, s)
    for a, s in zip(true_pred.list_auxiliary_states(), aux_shape):
        print(a, s)
    for a, s in zip(true_pred.list_outputs(), out_shape):
        print(a, s)
    #raise NotImplementedError()
    '''
    # concat preds and labels
    # shape=(batch_size*bptt, k+1)
    preds = mx.sym.concat(sample_pred, true_pred, dim=1, name='concat_debug0')
    # shape=(batch_size*bptt, k+1)
    sample_labels = mx.sym.zeros_like(sample_pred)
    true_labels = mx.sym.ones_like(true_pred)
    labels = mx.sym.concat(sample_labels, true_labels, dim=1, name='concat_debug')

    # loss
    out = mx.sym.LinearRegressionOutput(data=preds, label=labels)
    return out

def ce_loss(pred, vocab_size):
    pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_size, name='pred')
    label = mx.sym.Variable('label')
    label = mx.sym.Reshape(label, shape=(-1,))
    pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    return pred

# TODO add tied arg
def rnn_model(bptt, mode, vocab_size, num_embed, num_hidden,
              num_layers, dropout, use_dense_embedding):
    assert(mode == 'lstm')
    # embedding
    data = mx.sym.Variable('data')
    if use_dense_embedding:
        embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                 output_dim=num_embed, name='embed')
    else:
        weight = mx.sym.var("embedding_weight", stype='row_sparse')
        embed = mx.sym.contrib.SparseEmbedding(data=data, weight=weight, input_dim=vocab_size,
                                               output_dim=num_embed, name='embed')
    if dropout > 0:
        embed = mx.sym.Dropout(embed, p=dropout)
    # stacked rnn layers
    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_'%i))
        if dropout > 0:
            stack.add(mx.rnn.rnn_cell.DropoutCell(dropout))
    # unroll the model
    outputs, _ = stack.unroll(bptt, inputs=embed, merge_outputs=True)
    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    return pred
