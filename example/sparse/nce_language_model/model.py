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

def nce_criterion(p_target, p_sample, bptt, batch_size, k):
    p_noise_sample = mx.sym.var("p_noise_sample", shape=(bptt*batch_size, k))
    p_noise_target = mx.sym.var("p_noise_target", shape=(bptt*batch_size, 1))
    mask = mx.sym.var("mask", shape=(bptt, batch_size))
    mask = mx.sym.reshape(mask, shape=(bptt*batch_size, 1))
    eps = 1e-7
    # equation 5 in ref. A
    # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
    rnn_loss = mx.sym.log(p_target / (p_target + k * p_noise_target + eps)) * mask
    noise_loss = mx.sym.log(k * p_noise_sample / (p_sample + k * p_noise_sample + eps))
    loss = mx.sym.sum(rnn_loss) + mx.sym.sum(noise_loss)
    return mx.sym.make_loss(-loss / (bptt * batch_size))

def nce_decoder_weights_block(vocab_size, nhid, k, batch_size, bptt, use_dense, decoder_w=None):
    '''
    inputs:
    decoder_w: (vocab_size, nhid)
    returns:
    target_w: (bptt*batch_size, nhid)
    target_b: (bptt*batch_size, 1)
    sample_w: (bptt*batch_size, k, nhid)
    sample_b: (bptt*batch_size, k, 1)
    '''
    # (bptt*batch_size, k)
    sample = mx.sym.Variable('sample', shape=(bptt*batch_size, k), dtype='float32')
    # (bptt*batch_size, 1)
    label = mx.sym.Variable('label')
    label = mx.sym.reshape(label, shape=(-1, 1), name="label_reshape")
    sample_label = mx.sym.Concat(sample, label, dim=1)
    # (bptt*batch_size, )
    assert(decoder_w is not None)
    # weight and bias
    decoder_w = decoder_w if decoder_w is not None else mx.sym.var("decoder_weight")
    decoder_b = mx.sym.var("decoder_bias", shape=(vocab_size, 1), stype='row_sparse' if not use_dense else 'default')
    #decoder_b = mx.sym.reshape(decoder_b, shape=(vocab_size, 1))
    #if not use_dense:
    #    decoder_b = mx.sym.cast_storage(decoder_b, 'row_sparse')

    # lookup weights
    # (bptt*batch_size, k+1, nhid)
    if use_dense:
        sample_target_w = mx.sym.Embedding(data=sample_label, weight=decoder_w,
                                       input_dim=vocab_size, output_dim=nhid)
    else:
        sample_target_w = mx.sym.contrib.SparseEmbedding(data=sample_label, weight=decoder_w,
                                       input_dim=vocab_size, output_dim=nhid)

    # lookup bias
    # (bptt*batch_size, k, 1)
    # TODO use mx.sym.contrib.SparseEmbedding
    if use_dense:
        sample_target_b = mx.sym.Embedding(data=sample_label, weight=decoder_b,
                                       input_dim=vocab_size, output_dim=1)
    else:
        sample_target_b = mx.sym.contrib.SparseEmbedding(data=sample_label, weight=decoder_b,
                                       input_dim=vocab_size, output_dim=1)
    return sample_target_w, sample_target_b

def nce_decoder_block(sample_target_w, sample_target_b, pred, nhid, k, bptt, batch_size):
    '''
    inputs:
    target_w: (bptt*batch_size, nhid)
    target_b: (bptt*batch_size, 1)
    sample_w: (bptt*batch_size, k, nhid)
    sample_b: (bptt*batch_size, k, 1)
    pred: (bptt*batch_size, nhid)
    outputs: 
    '''
    sample_w = mx.sym.slice(sample_target_w, begin=(0, 0, 0), end=(bptt*batch_size, k, nhid))
    target_w = mx.sym.slice(sample_target_w, begin=(0, k, 0), end=(bptt*batch_size, k+1, nhid))
    sample_b = mx.sym.slice(sample_target_b, begin=(0, 0, 0), end=(bptt*batch_size, k, 1))
    target_b = mx.sym.slice(sample_target_b, begin=(0, k, 0), end=(bptt*batch_size, k+1, 1))

    target_w = mx.sym.reshape(target_w, shape=(bptt*batch_size, nhid))
    target_b = mx.sym.reshape(target_b, shape=(bptt*batch_size, 1))
    true_pred = mx.sym.sum(target_w * pred, axis=1, keepdims=True) + target_b

    # noise samples
    # (bptt*batch_size, nhid, 1)
    pred_3d_s = mx.sym.reshape(pred, (-1, nhid, 1), name='pred_reshape_s')
    # (bptt*batch_size, k, 1)
    sample_pred = mx.sym.batch_dot(sample_w, pred_3d_s)
    # (bptt*batch_size, k)
    sample_pred = mx.sym.reshape(sample_pred, (-1, k), name='sample_pred_reshape')
    # (bptt*batch_size, k)
    sample_b = mx.sym.reshape(sample_b, (-1, k), name='sample_b_reshape')
    sample_pred = sample_pred + sample_b
    p_target = mx.sym.exp(true_pred - 9)
    p_sample = mx.sym.exp(sample_pred - 9)
    return nce_criterion(p_target, p_sample, bptt, batch_size, k)

def nce_loss(pred, vocab_size, nhid, k, on_cpu, batch_size, bptt, use_dense, decoder_w=None):
    if on_cpu:
        assert(False)
        with mx.AttrScope(ctx_group='cpu_dev'):
            sample_target_w, sample_target_b = nce_decoder_weights_block(vocab_size, nhid, k, batch_size, bptt, use_dense, decoder_w=decoder_w)
        with mx.AttrScope(ctx_group='gpu_dev'):
            output = nce_decoder_block(sample_target_w, sample_target_b, pred, nhid, k, bptt, batch_size)
    else:
        sample_target_w, sample_target_b = nce_decoder_weights_block(vocab_size, nhid, k, batch_size, bptt, use_dense, decoder_w=decoder_w)
        output = nce_decoder_block(sample_target_w, sample_target_b, pred, nhid, k, bptt, batch_size)
    return output

########## COMMON BLOCKS ##########


def ce_decoder_block(pred, vocab_size, tied, use_dense, weight=None):
    decoder_b = mx.sym.var("decoder_bias", shape=(vocab_size, 1), stype='default' if use_dense else 'row_sparse')
    decoder_b = mx.sym.reshape(decoder_b, shape=(vocab_size,))
    if tied:
        pred = mx.sym.FullyConnected(data=pred, weight=weight, bias=decoder_b, num_hidden=vocab_size, name='pred')
    else:
        assert(use_dense)
        decoder_w = mx.sym.var('decoder_weight')
        pred = mx.sym.FullyConnected(data=pred, weight=decoder_w, num_hidden=vocab_size, name='pred', bias=decoder_b)
    label = mx.sym.Variable('label')
    pred = mx.sym.reshape(pred, shape=(-1, vocab_size))
    label = mx.sym.reshape(label, shape=(-1,))
    #output = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    logits = pred
    target = label
    # 700 X 10K
    pred = mx.sym.log_softmax(logits, axis=-1)
    # 700 X 1
    loss = -mx.sym.pick(pred, target, axis=-1, keepdims=True)
    mask = mx.sym.var("mask")
    mask = mx.sym.reshape(mask, shape=(-1, 1))
    loss = loss * mask
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
        weight = mx.sym.var("encoder_weight", init=mx.init.Uniform(0.1), stype='row_sparse')
        embed = mx.sym.contrib.SparseEmbedding(data=data, weight=weight, input_dim=vocab_size,
                                               output_dim=num_embed, name='embed')
    return embed, weight

# TODO put ce on cpu, too?
def ce_loss(pred, vocab_size, tied, use_dense, weight):
    #with mx.AttrScope(ctx_group='gpu_dev'):
    output = ce_decoder_block(pred, vocab_size, tied, use_dense, weight)
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
    pred = mx.sym.reshape(outputs, shape=(-1, nhid))
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
