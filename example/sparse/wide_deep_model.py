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
from weighted_softmax_ce import *


def wide_deep_model(num_linear_features, num_embed_features, num_cont_features, 
                    input_dims, hidden_units, positive_cls_weight):
    data = mx.symbol.Variable("data", stype='csr')
    label = mx.symbol.Variable("softmax_label")

    x = mx.symbol.slice_axis(data=data, axis=1, begin=0, end=num_linear_features)
    norm_init = mx.initializer.Normal(sigma=0.01)
    # weight with row_sparse storage type to enable sparse gradient updates
    weight = mx.symbol.Variable("weight", shape=(num_linear_features, 2),
                                init=norm_init, stype='row_sparse')
    bias = mx.symbol.Variable("bias", shape=(2,))
    dot = mx.symbol.sparse.dot(x, weight)
    linear_out = mx.symbol.broadcast_add(dot, bias)

    x = mx.symbol.slice_axis(data=data, axis=1, begin=num_linear_features, end=(num_linear_features+num_embed_features))
    embeds = mx.symbol.split(data=x, num_outputs=num_embed_features, squeeze_axis=1)

    x = mx.symbol.slice_axis(data=data, axis=1, begin=(num_linear_features+num_embed_features),
                             end=(num_linear_features+num_embed_features+num_cont_features))
    features = [x]

    for i, embed in enumerate(embeds):
        features.append(mx.symbol.Embedding(data=embed, input_dim=input_dims[i], output_dim=hidden_units[0]))

    hidden = mx.symbol.concat(*features, dim=1)
    hidden = mx.symbol.BatchNorm(data=hidden)
    hidden = mx.symbol.FullyConnected(data=hidden, num_hidden=hidden_units[1])
    hideen = mx.symbol.Activation(data=hidden, act_type='relu')
    hidden = mx.symbol.FullyConnected(data=hidden, num_hidden=hidden_units[2])
    hideen = mx.symbol.Activation(data=hidden, act_type='relu')
    deep_out = mx.symbol.FullyConnected(data=hidden, num_hidden=2)

    out = mx.symbol.Custom(linear_out+deep_out, label, op_type='weighted_softmax_ce_loss',
                           positive_cls_weight=positive_cls_weight, name='model')
    return mx.symbol.MakeLoss(out)
