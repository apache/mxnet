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

def factorization_machine_model(factor_size, num_features,
                                lr_mult_config, wd_mult_config, init_config):
    """ builds factorization machine network with proper formulation:
    y = w_0 \sum(x_i w_i) + 0.5(\sum\sum<v_i,v_j>x_ix_j - \sum<v_iv_i>x_i^2)
    """
    x = mx.symbol.Variable("data", stype='csr')
    # factor, linear and bias terms
    v = mx.symbol.Variable("v", shape=(num_features, factor_size), stype='row_sparse',
                           init=init_config['v'], lr_mult=lr_mult_config['v'],
                           wd_mult=wd_mult_config['v'])
    w = mx.symbol.var('w', shape=(num_features, 1), stype='row_sparse',
                      init=init_config['w'], lr_mult=lr_mult_config['w'],
                      wd_mult=wd_mult_config['w'])
    w0 = mx.symbol.var('w0', shape=(1,), init=init_config['w0'],
                       lr_mult=lr_mult_config['w0'], wd_mult=wd_mult_config['w0'])
    w1 = mx.symbol.broadcast_add(mx.symbol.dot(x, w), w0)

    # squared terms for subtracting self interactions
    v_s = mx.symbol._internal._square_sum(data=v, axis=1, keepdims=True)
    x_s = x.square()
    bd_sum = mx.sym.dot(x_s, v_s)

    # interactions
    w2 = mx.symbol.dot(x, v)
    w2_squared = 0.5 * mx.symbol.square(data=w2)

    # putting everything together
    w_all = mx.symbol.Concat(w1, w2_squared, dim=1)
    sum1 = w_all.sum(axis=1, keepdims=True)
    sum2 = -0.5 * bd_sum
    model = sum1 + sum2

    y = mx.symbol.Variable("softmax_label")
    model = mx.symbol.LogisticRegressionOutput(data=model, label=y)
    return model
