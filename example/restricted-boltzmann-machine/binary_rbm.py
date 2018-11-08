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

import ast
import numpy as np
import mxnet as mx

class BinaryRBM(mx.operator.CustomOp):

    def __init__(self, k):
        self.k = k # Persistent contrastive divergence k

    def forward(self, is_train, req, in_data, out_data, aux):
        visible_layer_data = in_data[0] # (num_batch, num_visible)
        visible_layer_bias = in_data[1] # (num_visible,)
        hidden_layer_bias = in_data[2]  # (num_hidden,)
        interaction_weight= in_data[3]        # (num_visible, num_hidden)

        if is_train:
            _, hidden_layer_prob_1 = self.sample_hidden_layer(visible_layer_data, hidden_layer_bias, interaction_weight)
            hidden_layer_sample = aux[1] # The initial state of the Gibbs sampling for persistent CD
        else:
            hidden_layer_sample, hidden_layer_prob_1 = self.sample_hidden_layer(visible_layer_data, hidden_layer_bias, interaction_weight)

        # k-step Gibbs sampling
        for _ in range(self.k):
            visible_layer_sample, visible_layer_prob_1 = self.sample_visible_layer(hidden_layer_sample, visible_layer_bias, interaction_weight)
            hidden_layer_sample, _ = self.sample_hidden_layer(visible_layer_sample, hidden_layer_bias, interaction_weight)

        if is_train:
            # Used in backward and next forward
            aux[0][:] = visible_layer_sample
            aux[1][:] = hidden_layer_sample

        self.assign(out_data[0], req[0], visible_layer_prob_1)
        self.assign(out_data[1], req[1], hidden_layer_prob_1)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        visible_layer_data = in_data[0]    # (num_batch, num_visible)
        visible_layer_sample = aux[0]      # (num_batch, num_visible)
        hidden_layer_prob_1 = out_data[1]  # (num_batch, num_hidden)
        hidden_layer_sample = aux[1]       # (num_batch, num_hidden)

        grad_visible_layer_bias = (visible_layer_sample - visible_layer_data).mean(axis=0)
        grad_hidden_layer_bias = (hidden_layer_sample - hidden_layer_prob_1).mean(axis=0)
        grad_interaction_weight= (mx.nd.linalg.gemm2(visible_layer_sample.expand_dims(2), hidden_layer_sample.expand_dims(1)) -
                            mx.nd.linalg.gemm2(visible_layer_data.expand_dims(2), hidden_layer_prob_1.expand_dims(1))
                           ).mean(axis=0)

        # We don't need the gradient on the visible layer input
        self.assign(in_grad[1], req[1], grad_visible_layer_bias)
        self.assign(in_grad[2], req[2], grad_hidden_layer_bias)
        self.assign(in_grad[3], req[3], grad_interaction_weight)

    def sample_hidden_layer(self, visible_layer_batch, hidden_layer_bias, interaction_weight):
        return self.sample_layer(visible_layer_batch, hidden_layer_bias, interaction_weight, False)

    def sample_visible_layer(self, hidden_layer_batch, visible_layer_bias, interaction_weight):
        return self.sample_layer(hidden_layer_batch, visible_layer_bias, interaction_weight, True)

    def sample_layer(self, other_layer_sample, layer_bias, interaction_weight, interaction_transpose):
        prob_1 = mx.nd.linalg.gemm(
            other_layer_sample,
            interaction_weight,
            layer_bias.tile(reps=(other_layer_sample.shape[0], 1)),
            transpose_b=interaction_transpose) # (num_batch, num_units_in_layer)
        prob_1.sigmoid(out=prob_1)
        return mx.nd.random.uniform(shape=prob_1.shape) < prob_1, prob_1

@mx.operator.register('BinaryRBM')
class BinaryRBMProp(mx.operator.CustomOpProp):

    # Auxiliary states are requested only if `for_training` is true.
    def __init__(self, num_hidden, k, for_training):
        super(BinaryRBMProp, self).__init__(False)
        self.num_hidden = int(num_hidden)
        self.k = int(k)
        self.for_training = ast.literal_eval(for_training)

    def list_arguments(self):
        # 0: (batch size, the number of visible units)
        # 1: (the number of visible units,)
        # 2: (the number of hidden units,)
        # 3: (the number of visible units, the number of hidden units)
        return ['data', 'visible_layer_bias', 'hidden_layer_bias', 'interaction_weight']

    def list_outputs(self):
        # 0: The probabilities that each visible unit is 1 after `k` steps of Gibbs sampling starting from the given `data`.
        #    (batch size, the number of visible units)
        # 1: The probabilities that each hidden unit is 1 conditional on the given `data`.
        #    (batch size, the number of hidden units)
        return ['visible_layer_prob_1', 'hidden_layer_prob_1']

    def list_auxiliary_states(self):
        # Used only if `self.for_trainig is true.
        # 0: Store the visible layer samples obtained in the forward pass, used in the backward pass.
        #    (batch size, the number of visible units)
        # 1: Store the hidden layer samples obtained in the forward pass, used in the backward and next forward pass.
        #    (batch size, the number of hidden units)
        return ['aux_visible_layer_sample', 'aux_hidden_layer_sample'] if self.for_training else []

    def infer_shape(self, in_shapes):
        visible_layer_data_shape = in_shapes[0] # The input data
        visible_layer_bias_shape = (visible_layer_data_shape[1],)
        hidden_layer_bias_shape = (self.num_hidden,)
        interaction_shape = (visible_layer_data_shape[1], self.num_hidden)
        visible_layer_sample_shape = visible_layer_data_shape
        visible_layer_prob_1_shape = visible_layer_sample_shape
        hidden_layer_sample_shape = (visible_layer_data_shape[0], self.num_hidden)
        hidden_layer_prob_1_shape = hidden_layer_sample_shape
        return [visible_layer_data_shape, visible_layer_bias_shape, hidden_layer_bias_shape, interaction_shape], \
               [visible_layer_prob_1_shape, hidden_layer_prob_1_shape], \
               [visible_layer_sample_shape, hidden_layer_sample_shape] if self.for_training else []

    def infer_type(self, in_type):
        return [in_type[0], in_type[0], in_type[0], in_type[0]], \
               [in_type[0], in_type[0]], \
               [in_type[0], in_type[0]] if self.for_training else []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return BinaryRBM(self.k)

# For gluon API
class BinaryRBMBlock(mx.gluon.HybridBlock):

    def __init__(self, num_hidden, k, for_training, **kwargs):
        super(BinaryRBMBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.num_hidden = num_hidden
            self.k = k
            self.for_training = for_training
            self.visible_layer_bias = self.params.get('visible_layer_bias', shape=(0,), allow_deferred_init=True)
            self.hidden_layer_bias = self.params.get('hidden_layer_bias', shape=(0,), allow_deferred_init=True)
            self.interaction_weight= self.params.get('interaction_weight', shape=(0, 0), allow_deferred_init=True)
            if for_training:
                self.aux_visible_layer_sample = self.params.get('aux_visible_layer_sample', shape=(0, 0), allow_deferred_init=True)
                self.aux_hidden_layer_sample = self.params.get('aux_hidden_layer_sample', shape=(0, 0), allow_deferred_init=True)

    def hybrid_forward(self, F, data, visible_layer_bias, hidden_layer_bias, interaction_weight, aux_visible_layer_sample=None, aux_hidden_layer_sample=None):
        # As long as `for_training` is kept constant, this conditional statement does not prevent hybridization.
        if self.for_training:
            return F.Custom(
                data,
                visible_layer_bias,
                hidden_layer_bias,
                interaction_weight,
                aux_visible_layer_sample,
                aux_hidden_layer_sample,
                num_hidden=self.num_hidden,
                k=self.k,
                for_training=self.for_training,
                op_type='BinaryRBM')
        else:
            return F.Custom(
                data,
                visible_layer_bias,
                hidden_layer_bias,
                interaction_weight,
                num_hidden=self.num_hidden,
                k=self.k,
                for_training=self.for_training,
                op_type='BinaryRBM')

def estimate_log_likelihood(visible_layer_bias, hidden_layer_bias, interaction_weight, ais_batch_size, ais_num_batch, ais_intermediate_steps, ais_burn_in_steps, data, ctx):
    # The base-rate RBM with no hidden layer. The visible layer bias is set to the same with the given RBM.
    # This is not the only possible choice but simple and works well.
    base_rate_visible_layer_bias = visible_layer_bias
    base_rate_visible_prob_1 = base_rate_visible_layer_bias.sigmoid()
    log_base_rate_z = base_rate_visible_layer_bias.exp().log1p().sum()

    def log_intermediate_unnormalized_prob(visible_layer_sample, beta):
        p = mx.nd.dot(
                visible_layer_sample, 
                (1 - beta) * base_rate_visible_layer_bias + beta * visible_layer_bias)
        if beta != 0:
            p += mx.nd.linalg.gemm(
                    visible_layer_sample,
                    interaction_weight,
                    hidden_layer_bias.tile(reps=(visible_layer_sample.shape[0], 1)),
                    transpose_b=False,
                    alpha=beta,
                    beta=beta).exp().log1p().sum(axis=1)
        return p

    def sample_base_rbm():
        rands = mx.nd.random.uniform(shape=(ais_batch_size, base_rate_visible_prob_1.shape[0]), ctx=ctx)
        return rands < base_rate_visible_prob_1.tile(reps=(ais_batch_size, 1))

    def sample_intermediate_visible_layer(visible_layer_sample, beta):
        for _ in range(ais_burn_in_steps):
            hidden_prob_1 = mx.nd.linalg.gemm(
                visible_layer_sample,
                interaction_weight,
                hidden_layer_bias.tile(reps=(visible_layer_sample.shape[0], 1)),
                transpose_b=False,
                alpha=beta,
                beta=beta)
            hidden_prob_1.sigmoid(out=hidden_prob_1)
            hidden_layer_sample = mx.nd.random.uniform(shape=hidden_prob_1.shape, ctx=ctx) < hidden_prob_1
            visible_prob_1 = mx.nd.linalg.gemm(
                hidden_layer_sample,
                interaction_weight,
                visible_layer_bias.tile(reps=(hidden_layer_sample.shape[0], 1)),
                transpose_b=True,
                alpha=beta,
                beta=beta) + (1 - beta) * base_rate_visible_layer_bias
            visible_prob_1.sigmoid(out=visible_prob_1)
            visible_layer_sample = mx.nd.random.uniform(shape=visible_prob_1.shape, ctx=ctx) < visible_prob_1
        return visible_layer_sample

    def array_from_batch(batch):
        if isinstance(batch, mx.io.DataBatch):
            return batch.data[0].as_in_context(ctx).flatten()
        else: # batch is an instance of list in the case of gluon DataLoader
            return batch[0].as_in_context(ctx).flatten()

    importance_weight_sum = 0
    num_ais_samples = ais_num_batch * ais_batch_size
    for _ in range(ais_num_batch):
        log_importance_weight = 0
        visible_layer_sample = sample_base_rbm()
        for n in range(1, ais_intermediate_steps + 1):
            beta = 1. * n / ais_intermediate_steps
            log_importance_weight += \
                log_intermediate_unnormalized_prob(visible_layer_sample, beta) - \
                log_intermediate_unnormalized_prob(visible_layer_sample, (n - 1.) / ais_intermediate_steps)
            visible_layer_sample = sample_intermediate_visible_layer(visible_layer_sample, beta)
        importance_weight_sum += log_importance_weight.exp().sum()
    log_z = (importance_weight_sum / num_ais_samples).log() + log_base_rate_z

    log_likelihood = 0
    num_data = 0
    for batch in data:
        batch_array = array_from_batch(batch)
        log_likelihood += log_intermediate_unnormalized_prob(batch_array, 1) - log_z
        num_data += batch_array.shape[0]
    log_likelihood = log_likelihood.sum() / num_data

    return log_likelihood.asscalar(), log_z.asscalar()
