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
    return mx.sym.make_loss(mx.sym.mean(masked_loss) * bptt)

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
        cell = mx.rnn.LSTMCell(num_hidden=nhid, prefix=prefix, forget_bias=1.0, num_proj=num_proj)
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

class RNNModel():

    def __init__(self, bptt, vocab_size, num_embed, nhid, num_layers,
                 dropout, num_proj):
        self.bptt = bptt
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.nhid = nhid
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_proj = num_proj
        self.state_names = []
        self.embed = mx.sym.contrib.SparseEmbedding
        self.dim = self.num_proj if self.num_proj > 0 else self.nhid

    def forward(self, batch_size):
        F = mx.symbol
        data = F.var('data')
        weight = F.var("encoder_weight", stype='row_sparse')
        embed = self.embed(data=data, weight=weight, input_dim=self.vocab_size,
                           output_dim=self.num_embed, name='embed')
        states = []
        outputs = F.Dropout(embed, p=self.dropout)
        for i in range(self.num_layers):
            prefix = 'lstm_l%d_' % i
            # TODO(haibin) what's the forget bias?
            lstm = mx.rnn.LSTMCell(num_hidden=self.nhid, prefix=prefix, forget_bias=1.0,
                                   num_proj=self.num_proj)
            init_h = F.var(prefix + '_init_h', shape=(batch_size, self.dim))
            init_c = F.var(prefix + '_init_c', shape=(batch_size, self.nhid))
            self.state_names += [prefix + '_init_h', prefix + '_init_c']
            # TODO(haibin) better layout?
            outputs, next_states = lstm.unroll(self.bptt, inputs=outputs, begin_state=[init_h, init_c],
                                               merge_outputs=True, layout='NTC')
            outputs = F.Dropout(outputs, p=self.dropout)
            states += [F.stop_gradient(s) for s in next_states]
        outputs = F.reshape(outputs, shape=(-1, self.dim))
        return outputs, states

class SampledModule():

    def __init__(self, vocab_size, nhid, num_samples, bptt, num_proj, is_nce=False):
        self.vocab_size = vocab_size
        self.nhid = nhid
        self.num_samples = num_samples
        self.bptt = bptt
        self.num_proj = num_proj
        self.dim = num_proj if num_proj > 0 else nhid
        self.embed = mx.sym.contrib.SparseEmbedding
        self.is_nce = is_nce

    def forward(self, inputs, batch_size):
        # inputs = (n, nhid)
        n = batch_size * self.bptt
        F = mx.symbol
        # (num_samples, )
        sample = F.var('sample', shape=(self.num_samples,), dtype='float32')
        # (n, )
        label = F.var('label')
        label = F.reshape(label, shape=(-1,), name="label_reshape")
        # (num_samples+n, )
        sample_label = F.concat(sample, label, dim=0)
        # weight and bias
        decoder_w = F.var("decoder_weight", stype='row_sparse')
        decoder_b = F.var("decoder_bias", shape=(self.vocab_size, 1), stype='row_sparse')
        # lookup weights and biases
        # (num_samples+n, nhid)
        sample_target_w = self.embed(data=sample_label, weight=decoder_w,
                                     input_dim=self.vocab_size, output_dim=self.dim)
        # (num_samples+n, 1)
        sample_target_b = self.embed(data=sample_label, weight=decoder_b,
                                     input_dim=self.vocab_size, output_dim=1)
        # (num_samples, nhid)
        sample_w = F.slice(sample_target_w, begin=(0, 0), end=(self.num_samples, self.dim))
        target_w = F.slice(sample_target_w, begin=(self.num_samples, 0), end=(self.num_samples+n, self.dim))
        sample_b = F.slice(sample_target_b, begin=(0, 0), end=(self.num_samples, 1))
        target_b = F.slice(sample_target_b, begin=(self.num_samples, 0), end=(self.num_samples+n, 1))
    
        # target
        # (n, 1)
        true_pred = F.sum(target_w * inputs, axis=1, keepdims=True) + target_b
        # samples
        # (n, num_samples)
        sample_b = F.reshape(sample_b, (-1,))
        sample_pred = F.FullyConnected(inputs, weight=sample_w, bias=sample_b, num_hidden=self.num_samples)

        if self.is_nce:
            p_target = F.exp(true_pred - 9)
            p_sample = F.exp(sample_pred - 9)
            return p_target, p_sample
        else:
            # TODO(haibin) remove accidental hits
            p_noise_sample = F.var("p_noise_sample", shape=(1, self.num_samples))
            p_noise_target = F.var("p_noise_target", shape=(n, 1))
            p_target = true_pred - F.log(p_noise_target)
            p_sample = F.broadcast_sub(sample_pred, F.log(p_noise_sample))

            # return logits and new_labels
            # (n, 1+num_samples)
            logits = F.concat(p_target, p_sample, dim=1)
            new_targets = F.zeros(shape=(n))
            return logits, new_targets

class CrossEntropyLoss():

    def __init__(self):
        self.criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    def forward(self, inputs, labels):
        loss = self.criterion.hybrid_forward(mx.symbol, inputs, labels)
        F = mx.symbol
        mask = F.var('mask')
        loss = loss * F.reshape(mask, shape=(-1,))
        return F.make_loss(loss.mean())
