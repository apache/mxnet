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

def matrix_fact_model_parallel_net(factor_size, num_hidden, max_user, max_item):
    # set ctx_group attribute to 'dev1' for the symbols created in this scope,
    # the symbols will be bound to the context that 'dev1' map to in group2ctxs
    with mx.AttrScope(ctx_group='dev1'):
        # input
        user = mx.symbol.Variable('user')
        item = mx.symbol.Variable('item')
        # user feature lookup
        user_weight = mx.symbol.Variable('user_weight')
        user = mx.symbol.Embedding(data=user, weight=user_weight,
                                   input_dim=max_user, output_dim=factor_size)
        # item feature lookup
        item_weight = mx.symbol.Variable('item_weight')
        item = mx.symbol.Embedding(data=item, weight=item_weight,
                                   input_dim=max_item, output_dim=factor_size)

    # set ctx_group attribute to 'dev2' for the symbols created in this scope,
    # the symbols will be bound to the context that 'dev2' map to in group2ctxs
    with mx.AttrScope(ctx_group='dev2'):
        # non-linear transformation of user features
        user = mx.symbol.Activation(data=user, act_type='relu')
        fc_user_weight = mx.symbol.Variable('fc_user_weight')
        fc_user_bias = mx.symbol.Variable('fc_user_bias')
        user = mx.symbol.FullyConnected(data=user, weight=fc_user_weight, bias=fc_user_bias, num_hidden=num_hidden)
        # non-linear transformation of user features
        item = mx.symbol.Activation(data=item, act_type='relu')
        fc_item_weight = mx.symbol.Variable('fc_item_weight')
        fc_item_bias = mx.symbol.Variable('fc_item_bias')
        item = mx.symbol.FullyConnected(data=item, weight=fc_item_weight, bias=fc_item_bias, num_hidden=num_hidden)
        # predict by the inner product, which is element-wise product and then sum
        pred = user * item
        pred = mx.symbol.sum(data=pred, axis=1)
        pred = mx.symbol.Flatten(data=pred)
        # label
        score = mx.symbol.Variable('score')
        # loss layer
        pred = mx.symbol.LinearRegressionOutput(data=pred, label=score)
    return pred
