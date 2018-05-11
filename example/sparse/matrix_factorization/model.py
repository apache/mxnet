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

def matrix_fact_net(factor_size, num_hidden, max_user, max_item, dense):
    # input
    user = mx.sym.Variable('user')
    item = mx.sym.Variable('item')
    score = mx.sym.Variable('score')
    stype = 'default' if dense else 'row_sparse'
    sparse_grad = not dense
    user_weight = mx.sym.Variable('user_weight', stype=stype)
    item_weight = mx.sym.Variable('item_weight', stype=stype)
    # user feature lookup
    user = mx.sym.Embedding(data=user, weight=user_weight, sparse_grad=sparse_grad,
                            input_dim=max_user, output_dim=factor_size)
    # item feature lookup
    item = mx.sym.Embedding(data=item, weight=item_weight, sparse_grad=sparse_grad,
                            input_dim=max_item, output_dim=factor_size)

    # non-linear transformation of user features
    user = mx.sym.Activation(data=user, act_type='relu')
    user_act = mx.sym.FullyConnected(data=user, num_hidden=num_hidden)
    # non-linear transformation of item features
    item = mx.sym.Activation(data=item, act_type='relu')
    item_act = mx.sym.FullyConnected(data=item, num_hidden=num_hidden)
    # predict by the inner product, which is elementwise product and then sum
    pred = user_act * item_act
    pred = mx.sym.sum(data=pred, axis=1)
    pred = mx.sym.Flatten(data=pred)
    # loss layer
    pred = mx.sym.LinearRegressionOutput(data=pred, label=score)
    return pred
