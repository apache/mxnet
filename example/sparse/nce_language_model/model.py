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

def nce_criterion(p_target, p_sample, n, k):
    # p_target = (n, 1)
    # p_sample = (n, k)
    p_noise_sample = mx.sym.var("p_noise_sample", shape=(1, k))
    p_noise_sample = mx.sym.repeat(p_noise_sample, repeats=n, axis=0)
    p_noise_target = mx.sym.var("p_noise_target", shape=(n, 1))
    mask = mx.sym.var("mask")
    mask = mx.sym.reshape(mask, shape=(n, 1))
    eps = 1e-7
    # equation 5 in ref. A
    # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
    rnn_loss = mx.sym.log(p_target / (p_target + k * p_noise_target + eps)) * mask

    noise_loss = mx.sym.log(k * p_noise_sample / (p_sample + k * p_noise_sample + eps))
    loss = mx.sym.sum(rnn_loss) + mx.sym.sum(noise_loss)
    return mx.sym.make_loss(-loss / n)

def nce_decoder_weights(vocab_size, nhid, num_samples, n, dense, bptt):
    EMBEDDING = mx.sym.Embedding if dense else mx.sym.contrib.SparseEmbedding
    # (num_samples, )
    sample = mx.sym.Variable('sample', shape=(num_samples,), dtype='float32')
    # (n, )
    label = mx.sym.Variable('label')
    label = mx.sym.reshape(label, shape=(-1,), name="label_reshape")
    # (num_samples+n, )
    sample_label = mx.sym.Concat(sample, label, dim=0)
    # weight and bias
    stype = 'row_sparse' if not dense else 'default'
    decoder_w = mx.sym.var("decoder_weight", stype=stype)
    decoder_b = mx.sym.var("decoder_bias", shape=(vocab_size, 1), stype=stype)
    # lookup weights and biases
    # (num_samples+n, nhid)
    sample_target_w = EMBEDDING(data=sample_label, weight=decoder_w,
                                   input_dim=vocab_size, output_dim=nhid)
    # (num_samples+n, 1)
    sample_target_b = EMBEDDING(data=sample_label, weight=decoder_b,
                                input_dim=vocab_size, output_dim=1)
    return sample_target_w, sample_target_b

def nce_decoder(sample_target_w, sample_target_b, pred, nhid, num_samples, n):
    # pred = (n, nhid)
    # (num_samples, nhid)
    sample_w = mx.sym.slice(sample_target_w, begin=(0, 0), end=(num_samples, nhid))
    target_w = mx.sym.slice(sample_target_w, begin=(num_samples, 0), end=(num_samples+n, nhid))
    sample_b = mx.sym.slice(sample_target_b, begin=(0, 0), end=(num_samples, 1))
    target_b = mx.sym.slice(sample_target_b, begin=(num_samples, 0), end=(num_samples+n, 1))

    # target
    # (n, 1)
    true_pred = mx.sym.sum(target_w * pred, axis=1, keepdims=True) + target_b
    # samples
    # (n, num_samples)
    sample_b = mx.sym.reshape(sample_b, (-1,))
    sample_pred = mx.sym.FullyConnected(pred, weight=sample_w, bias=sample_b, num_hidden=num_samples)
    p_target = mx.sym.exp(true_pred - 9)
    p_sample = mx.sym.exp(sample_pred - 9)
    return nce_criterion(p_target, p_sample, n, num_samples)

def nce_loss(pred, vocab_size, nhid, num_samples, batch_size, bptt, dense):
    sample_target_w, sample_target_b = nce_decoder_weights(vocab_size, nhid, num_samples, batch_size * bptt, dense, bptt)
    output = nce_decoder(sample_target_w, sample_target_b, pred, nhid, num_samples, batch_size * bptt)
    return output

########## COMMON BLOCKS ##########

def ce_loss(pred, vocab_size, dense):
    stype = 'row_sparse' if not dense else 'default'
    decoder_b = mx.sym.var("decoder_bias", shape=(vocab_size, 1), stype=stype)
    decoder_w = mx.sym.var('decoder_weight', stype=stype)
    pred = mx.sym.FullyConnected(data=pred, weight=decoder_w, num_hidden=vocab_size, name='pred', bias=decoder_b)
    label = mx.sym.Variable('label')
    pred = mx.sym.reshape(pred, shape=(-1, vocab_size))
    label = mx.sym.reshape(label, shape=(-1,))
    pred = mx.sym.log_softmax(pred, axis=-1)
    loss = -mx.sym.pick(pred, label, axis=-1, keepdims=True)
    mask = mx.sym.var("mask")
    mask = mx.sym.reshape(mask, shape=(-1, 1))
    loss = loss * mask
    loss = mx.sym.make_loss(mx.sym.mean(loss, axis=0, exclude=True), name="nll")
    return loss

def rnn_block(embed, bptt, vocab_size, num_embed, nhid,
              num_layers, dropout, dense_embedding, batch_size):
    embed = mx.sym.Dropout(embed, p=dropout)
    # stack the rnn layers
    outputs = embed
    states_list = []
    for i in range(num_layers):
        prefix = 'lstm_l%d_' % i
        cell = mx.rnn.LSTMCell(num_hidden=nhid, prefix=prefix, forget_bias=0.0)
        state_shape = (batch_size, nhid)
        state0 = mx.sym.var(prefix + str(0), shape=state_shape)
        state1 = mx.sym.var(prefix + str(1), shape=state_shape)
        outputs, next_states = cell.unroll(bptt, inputs=outputs, begin_state=[state0, state1], merge_outputs=True, layout='NTC')
        outputs = mx.sym.Dropout(outputs, p=dropout)
        states_list += next_states
    states = [mx.sym.stop_gradient(s) for s in states_list]
    pred = mx.sym.reshape(outputs, shape=(-1, nhid))
    return pred, states

# TODO check nhid & num_embed for weight tying
def rnn(bptt, vocab_size, num_embed, nhid,
        num_layers, dropout, dense, batch_size):
    data = mx.sym.Variable('data')
    EMBEDDING = mx.sym.Embedding if dense else mx.sym.contrib.SparseEmbedding
    stype = 'default' if dense else 'row_sparse'
    weight = mx.sym.var("encoder_weight", init=mx.init.Uniform(0.1), stype=stype)
    embed = EMBEDDING(data=data, weight=weight, input_dim=vocab_size,
                      output_dim=num_embed, name='embed')
    output, states = rnn_block(embed, bptt, vocab_size, num_embed, nhid,
                               num_layers, dropout, dense, batch_size)
    return output, states
