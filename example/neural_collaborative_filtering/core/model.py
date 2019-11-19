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
# 
import mxnet as mx
import numpy as np

@mx.init.register
class golorot_uniform(mx.init.Initializer):
    def __init__(self, fan_in, fan_out):
        super(golorot_uniform, self).__init__(fan_in=fan_in, fan_out=fan_out)
        self._fan_in = fan_in
        self._fan_out = fan_out
    def _init_weight(self, _, arr):
        limit = np.sqrt(6. / (self._fan_in + self._fan_out))
        mx.random.uniform(-limit, limit, out=arr)

@mx.init.register
class lecunn_uniform(mx.init.Initializer):
    def __init__(self, fan_in):
        super(lecunn_uniform, self).__init__(fan_in=fan_in)
        self._fan_in = fan_in
    def _init_weight(self, _, arr):
        limit = np.sqrt(3. / self._fan_in)
        mx.random.uniform(-limit, limit, out=arr)


def mlp(user, item, factor_size, model_layers, max_user, max_item):
    user_weight = mx.sym.Variable('mlp_user_weight', init=mx.init.Normal(0.01))
    item_weight = mx.sym.Variable('mlp_item_weight', init=mx.init.Normal(0.01))
    embed_user = mx.sym.Embedding(data=user, weight=user_weight, input_dim=max_user,
                                  output_dim=factor_size, name='embed_user'+str(factor_size))
    embed_item = mx.sym.Embedding(data=item, weight=item_weight, input_dim=max_item,
                                  output_dim=factor_size, name='embed_item'+str(factor_size))
    pre_gemm_concat = mx.sym.concat(embed_user, embed_item, dim=1, name='pre_gemm_concat')

    for i, layer in enumerate(model_layers):
        if i==0:
            mlp_weight_init = golorot_uniform(2 * factor_size, model_layers[i])
        else:
            mlp_weight_init = golorot_uniform(model_layers[i-1], model_layers[i])
        mlp_weight = mx.sym.Variable('fc_{}_weight'.format(i), init=mlp_weight_init)
        pre_gemm_concat = mx.sym.FullyConnected(data=pre_gemm_concat, weight=mlp_weight, num_hidden=layer, name='fc_'+str(i))
        pre_gemm_concat = mx.sym.Activation(data=pre_gemm_concat, act_type='relu', name='act_'+str(i))

    return pre_gemm_concat

def gmf(user, item, factor_size, max_user, max_item):
    user_weight = mx.sym.Variable('gmf_user_weight', init=mx.init.Normal(0.01))
    item_weight = mx.sym.Variable('gmf_item_weight', init=mx.init.Normal(0.01))
    embed_user = mx.sym.Embedding(data=user, weight=user_weight, input_dim=max_user,
                                  output_dim=factor_size, name='embed_user'+str(factor_size))
    embed_item = mx.sym.Embedding(data=item, weight=item_weight, input_dim=max_item,
                                  output_dim=factor_size, name='embed_item'+str(factor_size))
    pred = embed_user * embed_item

    return pred

def get_model(model_type='neumf', factor_size_mlp=128, factor_size_gmf=64,
              model_layers=[256, 128, 64], num_hidden=1, 
              max_user=138493, max_item=26744):
    # input
    user = mx.sym.Variable('user')
    item = mx.sym.Variable('item')

    if model_type == 'mlp':
        net = mlp(user=user, item=item,
                  factor_size=factor_size_mlp, model_layers=model_layers,
                  max_user=max_user, max_item=max_item)
    elif model_type == 'gmf':
        net = gmf(user=user, item=item,
                  factor_size=factor_size_gmf,
                  max_user=max_user, max_item=max_item)
    elif model_type == 'neumf':
        net_mlp = mlp(user=user, item=item,
                      factor_size=factor_size_mlp, model_layers=model_layers,
                      max_user=max_user, max_item=max_item)
        net_gmf = gmf(user=user, item=item,
                      factor_size=factor_size_gmf,
                      max_user=max_user, max_item=max_item)

        net = mx.sym.concat(net_gmf, net_mlp, dim=1, name='post_gemm_concat')

    else:
        raise ValueError('Unsupported ncf model %s.' % model_type)

    final_weight = mx.sym.Variable('fc_final_weight', init=lecunn_uniform(factor_size_gmf + model_layers[-1]))
    net = mx.sym.FullyConnected(data=net, weight=final_weight, num_hidden=num_hidden, name='fc_final') 
   
    y_label = mx.sym.Variable('softmax_label')
    net = mx.symbol.LogisticRegressionOutput(data=net, label=y_label, name='sigmoid_final')

    return net

