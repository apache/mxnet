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


def wide_deep_model(num_linear_features, num_embed_features, num_cont_features, 
                    input_dims, hidden_units):
    # wide model
    csr_data = mx.symbol.Variable("csr_data", stype='csr')
    label = mx.symbol.Variable("softmax_label")

    norm_init = mx.initializer.Normal(sigma=0.01)
    # weight with row_sparse storage type to enable sparse gradient updates
    weight = mx.symbol.Variable("linear_weight", shape=(num_linear_features, 2),
                                init=norm_init, stype='row_sparse')
    bias = mx.symbol.Variable("linear_bias", shape=(2,))
    dot = mx.symbol.sparse.dot(csr_data, weight)
    linear_out = mx.symbol.broadcast_add(dot, bias)
    # deep model
    dns_data = mx.symbol.Variable("dns_data")
    # embedding features
    x = mx.symbol.slice(data=dns_data, begin=(0, 0),
                        end=(None, num_embed_features))
    embeds = mx.symbol.split(data=x, num_outputs=num_embed_features, squeeze_axis=1)
    # continuous features
    x = mx.symbol.slice(data=dns_data, begin=(0, num_embed_features),
                        end=(None, num_embed_features + num_cont_features))
    features = [x]

    for i, embed in enumerate(embeds):
        embed_weight = mx.symbol.Variable('embed_%d_weight' % i, stype='row_sparse')
        features.append(mx.symbol.contrib.SparseEmbedding(data=embed, weight=embed_weight,
                        input_dim=input_dims[i], output_dim=hidden_units[0]))

    hidden = mx.symbol.concat(*features, dim=1)
    hidden = mx.symbol.FullyConnected(data=hidden, num_hidden=hidden_units[1])
    hidden = mx.symbol.Activation(data=hidden, act_type='relu')
    hidden = mx.symbol.FullyConnected(data=hidden, num_hidden=hidden_units[2])
    hidden = mx.symbol.Activation(data=hidden, act_type='relu')
    deep_out = mx.symbol.FullyConnected(data=hidden, num_hidden=2)

    out = mx.symbol.SoftmaxOutput(linear_out + deep_out, label, name='model')
    return out
