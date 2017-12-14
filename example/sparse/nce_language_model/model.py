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
from op import *

def nce_decoder_weights_block(vocab_size, nhid, k, batch_size, bptt, decoder_w=None):
    '''
    inputs:
    decoder_w: (vocab_size, nhid)
    returns:
    true_w: (bptt*batch_size, nhid)
    true_b: (bptt*batch_size, 1)
    sample_w: (bptt*batch_size, k, nhid)
    sample_b: (bptt*batch_size, k, 1)
    '''
    # (bptt*batch_size, k)
    sample = mx.sym.Variable('sample', shape=(bptt*batch_size, k), dtype='float32')
    # (bptt*batch_size, 1)
    label = mx.sym.Variable('label')
    # (bptt*batch_size, )
    label = mx.sym.Reshape(label, shape=(-1,), name="label_reshape")

    # weight and bias
    if decoder_w is None:
        assert(False)
        decoder_w = mx.sym.var("decoder_weight", stype='row_sparse')
    decoder_b = mx.sym.var("decoder_bias", shape=(vocab_size, )) #, stype='row_sparse')
    decoder_b = mx.sym.Reshape(decoder_b, shape=(vocab_size, 1))

    # lookup weights
    # (bptt*batch_size, k, nhid)
    sample_w = mx.sym.Embedding(data=sample, weight=decoder_w,
                                              input_dim=vocab_size, output_dim=nhid)
    true_w = mx.sym.Embedding(data=label, weight=decoder_w,
                                            input_dim=vocab_size, output_dim=nhid)
    # lookup bias
    # (bptt*batch_size, k, 1)
    # TODO use mx.sym.contrib.SparseEmbedding
    sample_b = mx.sym.Embedding(data=sample, weight=decoder_b,
                                input_dim=vocab_size, output_dim=1)
    # (bptt*batch_size, 1)
    # TODO use mx.sym.contrib.SparseEmbedding
    true_b = mx.sym.Embedding(data=label, weight=decoder_b,
                              input_dim=vocab_size, output_dim=1)
    return true_w, true_b, sample_w, sample_b

def nce_decoder_block(true_w, true_b, sample_w, sample_b, pred, nhid, k):
    '''
    inputs:
    true_w: (bptt*batch_size, nhid)
    true_b: (bptt*batch_size, 1)
    sample_w: (bptt*batch_size, k, nhid)
    sample_b: (bptt*batch_size, k, 1)
    pred: (bptt*batch_size, nhid)
    outputs: 
    '''
    # true labels
    # (bptt*batch_size, nhid, 1)
    true_w = mx.sym.Reshape(true_w, (-1, nhid, 1))
    # (bptt*batch_size, 1, nhid)
    pred_3d_t = mx.sym.Reshape(pred, (-1, 1, nhid), name='pred_reshape_t')
    # (bptt*batch_size, 1, 1)
    true_pred = mx.sym.batch_dot(pred_3d_t, true_w)
    # (bptt*batch_size, 1)
    true_pred = mx.sym.Reshape(true_pred, (-1, 1), name='true_pred_reshape')
    true_pred = true_pred + true_b

    # noise samples
    # (bptt*batch_size, nhid, 1)
    pred_3d_s = mx.sym.Reshape(pred, (-1, nhid, 1), name='pred_reshape_s')
    # (bptt*batch_size, k, 1)
    sample_pred = mx.sym.batch_dot(sample_w, pred_3d_s)
    # (bptt*batch_size, k)
    sample_pred = mx.sym.Reshape(sample_pred, (-1, k), name='sample_pred_reshape')
    # (bptt*batch_size, k)
    sample_b = mx.sym.Reshape(sample_b, (-1, k), name='sample_b_reshape')
    sample_pred = sample_pred + sample_b

    # targets
    sample_labels = mx.sym.zeros_like(sample_pred)
    true_labels = mx.sym.ones_like(true_pred)

    # concat preds and targets
    # (bptt*batch_size, k+1)
    preds = mx.sym.concat(sample_pred, true_pred, dim=1, name='concat_pred')
    # (bptt*batch_size, k+1)
    labels = mx.sym.concat(sample_labels, true_labels, dim=1, name='concat_targets')
    # loss
    #out = mx.sym.LogisticRegressionOutput(data=preds, label=labels)
    preds = mx.sym.sigmoid(preds)
    logits = mx.sym.Custom(data=preds, label=labels, name='ce', op_type='CrossEntropyLoss')
    out = mx.sym.make_loss(logits)
    return out

def nce_loss(pred, vocab_size, nhid, k, on_cpu, batch_size, bptt, decoder_w=None):
    if on_cpu:
        assert(False)
        with mx.AttrScope(ctx_group='cpu_dev'):
            true_w, true_b, sample_w, sample_b = nce_decoder_weights_block(vocab_size, nhid, k, batch_size, bptt, decoder_w=decoder_w)
        with mx.AttrScope(ctx_group='gpu_dev'):
            output = nce_decoder_block(true_w, true_b, sample_w, sample_b, pred, nhid, k)
    else:
        true_w, true_b, sample_w, sample_b = nce_decoder_weights_block(vocab_size, nhid, k, batch_size, bptt, decoder_w=decoder_w)
        output = nce_decoder_block(true_w, true_b, sample_w, sample_b, pred, nhid, k)
    return output

########## COMMON BLOCKS ##########

def ce_decoder_block(pred, vocab_size, tied, weight=None):
    if tied:
        decoder_b = mx.sym.var("decoder_bias")
        pred = mx.sym.FullyConnected(data=pred, weight=weight, bias=decoder_b, num_hidden=vocab_size, name='pred')
    else:
        assert(False)
        pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_size, name='pred')
    label = mx.sym.Variable('label')
    pred = mx.sym.Reshape(pred, shape=(-1, vocab_size))
    label = mx.sym.Reshape(label, shape=(-1,))
    #output = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    logits = pred
    target = label
    # 700 X 10K
    pred = mx.sym.log_softmax(logits, axis=-1)
    # 700 X 1
    loss = -mx.sym.pick(pred, target, axis=-1, keepdims=True)
    loss = mx.sym.make_loss(mx.sym.mean(loss, axis=0, exclude=True), name="nll")
    return loss

def encoder_block(bptt, vocab_size, num_embed, nhid,
                  num_layers, dropout, use_dense_embedding):
    data = mx.sym.Variable('data')
    if use_dense_embedding:
        weight = mx.sym.var("encoder_weight", init=mx.init.Uniform(0.1))
        embed = mx.sym.Embedding(data=data, weight=weight, input_dim=vocab_size,
                                 output_dim=num_embed, name='embed')
    else:
        weight = mx.sym.var("encoder_weight", stype='row_sparse')
        embed = mx.sym.contrib.SparseEmbedding(data=data, weight=weight, input_dim=vocab_size,
                                               output_dim=num_embed, name='embed')
    return embed, weight

# TODO put ce on cpu, too?
def ce_loss(pred, vocab_size, tied, weight):
    #with mx.AttrScope(ctx_group='gpu_dev'):
    output = ce_decoder_block(pred, vocab_size, tied, weight)
    return output


def rnn_block(embed, bptt, vocab_size, num_embed, nhid,
              num_layers, dropout, use_dense_embedding, batch_size):
    embed = mx.sym.Dropout(embed, p=dropout)
    # stack the rnn layers
    outputs = embed
    states_list = []
    for i in range(num_layers):
        prefix = 'lstm_l%d_' % i
        cell = mx.rnn.FusedRNNCell(num_hidden=nhid, prefix=prefix, num_layers=1, get_next_state=True, forget_bias=0.0, dropout=dropout)
        state_shape = (1, batch_size, nhid)
        state0 = mx.sym.var(prefix + str(0), shape=state_shape)
        state1 = mx.sym.var(prefix + str(1), shape=state_shape)
        outputs, next_states = cell.unroll(bptt, inputs=outputs, begin_state=[state0, state1], merge_outputs=True, layout='TNC')
        outputs = mx.sym.Dropout(outputs, p=dropout)
        states_list += next_states

    # TODO block graph is not required
    states = [mx.sym.BlockGrad(s) for s in states_list]
    pred = mx.sym.Reshape(outputs, shape=(-1, nhid))
    return pred, states

# TODO check nhid & num_embed for weight tying
def rnn(bptt, vocab_size, num_embed, nhid,
        num_layers, dropout, use_dense_embedding, on_cpu, batch_size):
    # embedding consumes lots of memory, stored on cpus
    assert(not on_cpu)
    embed, weight = encoder_block(bptt, vocab_size, num_embed, nhid,
                          num_layers, dropout, use_dense_embedding)
    output, states = rnn_block(embed, bptt, vocab_size, num_embed, nhid,
                       num_layers, dropout, use_dense_embedding, batch_size)
    return output, weight, states
