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

def nce_decoder_weights_block(vocab_size, nhid, k, decoder_w=None):
    # shape=(k,)
    sample = mx.sym.Variable('sample', shape=(k,), dtype='float32')
    label = mx.sym.Variable('label')
    # shape=(batch_size*bptt, )
    label = mx.sym.Reshape(label, shape=(-1,), name="label_reshape")

    # weight and bias
    if decoder_w is None:
        decoder_w = mx.sym.var("decoder_weight", stype='row_sparse')
    decoder_b = mx.sym.var("decoder_bias") #, stype='row_sparse')

    # lookup weights
    # shape=(k, nhid)
    sample_w = mx.sym.contrib.SparseEmbedding(data=sample, weight=decoder_w,
                                              input_dim=vocab_size, output_dim=nhid)
    true_w = mx.sym.contrib.SparseEmbedding(data=label, weight=decoder_w,
                                            input_dim=vocab_size, output_dim=nhid)

    # lookup bias
    # shape=(k, 1)
    # TODO use mx.sym.contrib.SparseEmbedding
    sample_b = mx.sym.Embedding(data=sample, weight=decoder_b,
                                input_dim=vocab_size, output_dim=1)
    # shape=(batch_size*bptt, 1)
    # TODO use mx.sym.contrib.SparseEmbedding
    true_b = mx.sym.Embedding(data=label, weight=decoder_b,
                              input_dim=vocab_size, output_dim=1)
    return true_w, true_b, sample_w, sample_b

def nce_decoder_block(true_w, true_b, sample_w, sample_b, pred, nhid, k):
    # shape=(batch_size*bptt, nhid, 1)
    true_w = mx.sym.Reshape(true_w, (-1, nhid, 1))
    sample_b = mx.sym.Reshape(sample_b, shape=(k,))
    # shape=(batch_size*bptt, 1, nhid)
    pred_3d = mx.sym.Reshape(pred, (-1, 1, nhid), name='pred_reshape')

    # pred.shape=(batch_size*bptt, nhid)
    # shape=(batch_size*bptt, k)
    sample_pred = mx.sym.FullyConnected(data=pred, weight=sample_w,
                                        num_hidden=k, bias=sample_b)
    # shape=(batch_size*bptt, 1, 1)
    true_pred = mx.sym.batch_dot(pred_3d, true_w)
    true_pred = mx.sym.Reshape(true_pred, (-1, 1), name='true_pred_reshape')
    true_pred = mx.sym.broadcast_add(true_pred, true_b)
    # concat preds and labels
    # shape=(batch_size*bptt, k+1)
    preds = mx.sym.concat(sample_pred, true_pred, dim=1, name='concat_debug0')
    # shape=(batch_size*bptt, k+1)
    sample_labels = mx.sym.zeros_like(sample_pred)
    true_labels = mx.sym.ones_like(true_pred)
    labels = mx.sym.concat(sample_labels, true_labels, dim=1, name='concat_debug')

    # loss
    out = mx.sym.LogisticRegressionOutput(data=preds, label=labels)
    return out

def ce_decoder_block(pred, vocab_size):
    pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_size, name='pred')
    label = mx.sym.Variable('label')
    label = mx.sym.Reshape(label, shape=(-1,))
    output = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    return output

def nce_loss(pred, vocab_size, nhid, k, on_cpu, decoder_w=None):
    if on_cpu:
        with mx.AttrScope(ctx_group='cpu_dev'):
            true_w, true_b, sample_w, sample_b = nce_decoder_weights_block(vocab_size, nhid, k, decoder_w=decoder_w)
        with mx.AttrScope(ctx_group='gpu_dev'):
            output = nce_decoder_block(true_w, true_b, sample_w, sample_b, pred, nhid, k)
    else:
        true_w, true_b, sample_w, sample_b = nce_decoder_weights_block(vocab_size, nhid, k, decoder_w=decoder_w)
        output = nce_decoder_block(true_w, true_b, sample_w, sample_b, pred, nhid, k)
    return output

def encoder_block(bptt, vocab_size, num_embed, nhid,
            num_layers, dropout, use_dense_embedding):
    data = mx.sym.Variable('data')
    if use_dense_embedding:
        embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                 output_dim=num_embed, name='embed')
    else:
        weight = mx.sym.var("encoder_weight", stype='row_sparse')
        embed = mx.sym.contrib.SparseEmbedding(data=data, weight=weight, input_dim=vocab_size,
                                               output_dim=num_embed, name='embed')
    return embed

def rnn_block(embed, bptt, vocab_size, num_embed, nhid,
        num_layers, dropout, use_dense_embedding):
    if dropout > 0:
        embed = mx.sym.Dropout(embed, p=dropout)
    # stacked rnn layers
    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=nhid, prefix='lstm_l%d_'%i))
        if dropout > 0:
            stack.add(mx.rnn.rnn_cell.DropoutCell(dropout))
    # unroll the rnn
    outputs, _ = stack.unroll(bptt, inputs=embed, merge_outputs=True)
    pred = mx.sym.Reshape(outputs, shape=(-1, nhid))
    return pred

# TODO put ce on cpu, too?
def ce_loss(pred, vocab_size):
    with mx.AttrScope(ctx_group='gpu_dev'):
        output = ce_decoder_block(pred, vocab_size)
    return output

# TODO add tied arg
def rnn(bptt, vocab_size, num_embed, nhid,
        num_layers, dropout, use_dense_embedding, on_cpu):
    # embedding consumes lots of memory, stored on cpus
    if on_cpu:
        with mx.AttrScope(ctx_group='cpu_dev'):
            embed = encoder_block(bptt, vocab_size, num_embed, nhid,
                                  num_layers, dropout, use_dense_embedding)
        with mx.AttrScope(ctx_group='gpu_dev'):
            output = rnn_block(embed, bptt, vocab_size, num_embed, nhid,
                               num_layers, dropout, use_dense_embedding)
    else:
        embed = encoder_block(bptt, vocab_size, num_embed, nhid,
                              num_layers, dropout, use_dense_embedding)
        output = rnn_block(embed, bptt, vocab_size, num_embed, nhid,
                           num_layers, dropout, use_dense_embedding)
    return output
