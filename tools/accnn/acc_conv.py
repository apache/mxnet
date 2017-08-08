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

import numpy as np
from scipy import linalg as LA
import mxnet as mx
import argparse
import utils

def conv_vh_decomposition(model, args):
  W = model.arg_params[args.layer+'_weight'].asnumpy()
  N, C, y, x = W.shape
  b = model.arg_params[args.layer+'_bias'].asnumpy()
  W = W.transpose((1,2,0,3)).reshape((C*y, -1))

  U, D, Q = np.linalg.svd(W, full_matrices=False)
  sqrt_D = LA.sqrtm(np.diag(D))
  K = args.K
  V = U[:,:K].dot(sqrt_D[:K, :K])
  H = Q.T[:,:K].dot(sqrt_D[:K, :K])
  V = V.T.reshape(K, C, y, 1)
  b_1 = np.zeros((K, ))
  H = H.reshape(N, x, 1, K).transpose((0,3,2,1))
  b_2 = b

  W1, b1, W2, b2 = V, b_1, H, b_2
  def sym_handle(data, node):
    kernel = eval(node['param']['kernel'])
    pad = eval(node['param']['pad'])
    name = node['name']

    name1 = name + '_v'
    kernel1 = tuple((kernel[0], 1))
    pad1 = tuple((pad[0], 0))
    num_filter = W1.shape[0]
    sym1 = mx.symbol.Convolution(data=data, kernel=kernel1, pad=pad1, num_filter=num_filter, name=name1)

    name2 = name + '_h'
    kernel2 = tuple((1, kernel[1]))
    pad2 = tuple((0, pad[1]))
    num_filter = W2.shape[0]
    sym2 = mx.symbol.Convolution(data=sym1, kernel=kernel2, pad=pad2, num_filter=num_filter, name=name2)
    return sym2

  def arg_handle(arg_shape_dic, arg_params):
    name1 = args.layer + '_v'
    name2 = args.layer + '_h'
    weight1 = mx.ndarray.array(W1)
    bias1 = mx.ndarray.array(b1)
    weight2 = mx.ndarray.array(W2)
    bias2 = mx.ndarray.array(b2)
    assert weight1.shape == arg_shape_dic[name1+'_weight'], 'weight1'
    assert weight2.shape == arg_shape_dic[name2+'_weight'], 'weight2'
    assert bias1.shape == arg_shape_dic[name1+'_bias'], 'bias1'
    assert bias2.shape == arg_shape_dic[name2+'_bias'], 'bias2'

    arg_params[name1 + '_weight'] = weight1
    arg_params[name1 + '_bias'] = bias1
    arg_params[name2 + '_weight'] = weight2
    arg_params[name2 + '_bias'] = bias2

  new_model = utils.replace_conv_layer(args.layer, model, sym_handle, arg_handle)
  return new_model

def main():
  model = utils.load_model(args)
  new_model = conv_vh_decomposition(model, args)
  new_model.save(args.save_model)

if __name__ == '__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('-m', '--model', help='the model to speed up')
  parser.add_argument('-g', '--gpus', default='0', help='the gpus to be used in ctx')
  parser.add_argument('--load-epoch',type=int,default=1)
  parser.add_argument('--layer')
  parser.add_argument('--K', type=int)
  parser.add_argument('--save-model')
  args = parser.parse_args()
  main()

