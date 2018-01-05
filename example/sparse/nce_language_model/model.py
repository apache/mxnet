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

def nce_criterion(p_target, p_sample, n, num_samples):
    # p_target = (n, 1)
    # p_sample = (n, num_samples)
    p_noise_sample = mx.sym.var("p_noise_sample", shape=(1, num_samples))
    p_noise_sample = mx.sym.repeat(p_noise_sample, repeats=n, axis=0)
    p_noise_target = mx.sym.var("p_noise_target", shape=(n, 1))
    mask = mx.sym.var("mask")
    mask = mx.sym.reshape(mask, shape=(n, 1))
    eps = 1e-7
    # equation 5 in ref. A
    # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
    rnn_loss = mx.sym.log(p_target / (p_target + num_samples * p_noise_target + eps)) * mask

    noise_loss = mx.sym.log(num_samples * p_noise_sample / (p_sample + num_samples * p_noise_sample + eps))
    loss = mx.sym.sum(rnn_loss) + mx.sym.sum(noise_loss)
    return mx.sym.make_loss(-loss / n)

def nce_loss(pred, vocab_size, nhid, num_samples, batch_size, bptt, dense, init, num_proj):
    dim = num_proj if num_proj > 0 else nhid
    n = batch_size * bptt
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
    decoder_w = mx.sym.var("decoder_weight", stype=stype, init=init)
    decoder_b = mx.sym.var("decoder_bias", shape=(vocab_size, 1), stype=stype)
    # lookup weights and biases
    # (num_samples+n, nhid)
    sample_target_w = EMBEDDING(data=sample_label, weight=decoder_w,
                                   input_dim=vocab_size, output_dim=dim)
    # (num_samples+n, 1)
    sample_target_b = EMBEDDING(data=sample_label, weight=decoder_b,
                                input_dim=vocab_size, output_dim=1)
    # pred = (n, nhid)
    # (num_samples, nhid)
    sample_w = mx.sym.slice(sample_target_w, begin=(0, 0), end=(num_samples, dim))
    target_w = mx.sym.slice(sample_target_w, begin=(num_samples, 0), end=(num_samples+n, dim))
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

def nce_loss_tf(pred, vocab_size, nhid, num_samples, batch_size, bptt, dense, init, num_proj):
    dim = num_proj if num_proj > 0 else nhid
    n = batch_size * bptt
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
    decoder_w = mx.sym.var("decoder_weight", stype=stype, init=init)
    decoder_b = mx.sym.var("decoder_bias", shape=(vocab_size, 1), stype=stype)
    # lookup weights and biases
    # (num_samples+n, nhid)
    sample_target_w = EMBEDDING(data=sample_label, weight=decoder_w,
                                   input_dim=vocab_size, output_dim=dim)
    # (num_samples+n, 1)
    sample_target_b = EMBEDDING(data=sample_label, weight=decoder_b,
                                input_dim=vocab_size, output_dim=1)
    # pred = (n, nhid)
    # (num_samples, nhid)
    sample_w = mx.sym.slice(sample_target_w, begin=(0, 0), end=(num_samples, dim))
    target_w = mx.sym.slice(sample_target_w, begin=(num_samples, 0), end=(num_samples+n, dim))
    sample_b = mx.sym.slice(sample_target_b, begin=(0, 0), end=(num_samples, 1))
    target_b = mx.sym.slice(sample_target_b, begin=(num_samples, 0), end=(num_samples+n, 1))

    # target
    # (n, 1)
    true_pred = mx.sym.sum(target_w * pred, axis=1, keepdims=True) + target_b
    # samples
    # (n, num_samples)
    sample_b = mx.sym.reshape(sample_b, (-1,))
    sample_pred = mx.sym.FullyConnected(pred, weight=sample_w, bias=sample_b, num_hidden=num_samples)

    # TF Implementation #
    p_noise_sample = mx.sym.var("p_noise_sample", shape=(1, num_samples))
    p_noise_target = mx.sym.var("p_noise_target", shape=(n, 1))

    p_target = true_pred - mx.sym.log(p_noise_target)
    p_sample = mx.sym.broadcast_sub(sample_pred, mx.sym.log(p_noise_sample))

    p_target = mx.sym.reshape(p_target, shape=(n, 1))
    p_sample = mx.sym.reshape(p_sample, shape=(n, num_samples))
    # (n, 1+num_samples)
    p_concat = mx.sym.Concat(p_target, p_sample, dim=1)

    l_target = mx.sym.ones_like(p_target)
    l_sample = mx.sym.zeros_like(p_sample)
    # (n, 1+num_samples)
    l_concat = mx.sym.Concat(l_target, l_sample, dim=1)

    mask = mx.sym.var("mask")
    mask = mx.sym.reshape(mask, shape=(n,))

    relu_logits = mx.sym.relu(p_concat)
    left = relu_logits - p_concat * l_concat
    right = mx.sym.log1p(mx.sym.exp(-mx.sym.abs(p_concat)))
    loss = left + right

    mean_loss = mx.sym.sum(loss, axis=1)
    masked_loss = mean_loss * mask
    return mx.sym.make_loss(mx.sym.mean(masked_loss))

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
              num_layers, dropout, dense_embedding, batch_size, num_proj):
    embed = mx.sym.Dropout(embed, p=dropout)
    # stack the rnn layers
    outputs = embed
    states_list = []
    dim = num_proj if num_proj > 0 else nhid
    for i in range(num_layers):
        prefix = 'lstm_l%d_' % i
        cell = mx.rnn.LSTMCell(num_hidden=nhid, prefix=prefix, forget_bias=0.0, num_proj=num_proj)
        state_shape = (batch_size, nhid)
        state0 = mx.sym.var(prefix + str(0), shape=(batch_size, dim))
        state1 = mx.sym.var(prefix + str(1), shape=state_shape)
        outputs, next_states = cell.unroll(bptt, inputs=outputs, begin_state=[state0, state1],
                                           merge_outputs=True, layout='NTC')
        outputs = mx.sym.Dropout(outputs, p=dropout)
        states_list += next_states
    states = [mx.sym.stop_gradient(s) for s in states_list]
    pred = mx.sym.reshape(outputs, shape=(-1, dim))
    return pred, states

# TODO check nhid & num_embed for weight tying
def rnn(bptt, vocab_size, num_embed, nhid,
        num_layers, dropout, dense, batch_size, init, num_proj):
    data = mx.sym.Variable('data')
    EMBEDDING = mx.sym.Embedding if dense else mx.sym.contrib.SparseEmbedding
    stype = 'default' if dense else 'row_sparse'
    weight = mx.sym.var("encoder_weight", init=init, stype=stype)
    embed = EMBEDDING(data=data, weight=weight, input_dim=vocab_size,
                      output_dim=num_embed, name='embed')
    output, states = rnn_block(embed, bptt, vocab_size, num_embed, nhid,
                               num_layers, dropout, dense, batch_size, num_proj)
    return output, states
