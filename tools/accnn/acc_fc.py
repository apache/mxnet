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
import pdb

def fc_decomposition(model, args):
  W = model.arg_params[args.layer+'_weight'].asnumpy()
  b = model.arg_params[args.layer+'_bias'].asnumpy()
  W = W.reshape((W.shape[0],-1))
  b = b.reshape((b.shape[0],-1))
  u, s, v = LA.svd(W, full_matrices=False)
  s = np.diag(s)
  t = u.dot(s.dot(v))
  rk = args.K
  P = u[:,:rk]
  Q = s[:rk,:rk].dot(v[:rk,:])

  name1 = args.layer + '_red'
  name2 = args.layer + '_rec'
  def sym_handle(data, node):
    W1, W2 = Q, P
    sym1 = mx.symbol.FullyConnected(data=data, num_hidden=W1.shape[0], no_bias=True,  name=name1)
    sym2 = mx.symbol.FullyConnected(data=sym1, num_hidden=W2.shape[0], no_bias=False, name=name2)
    return sym2

  def arg_handle(arg_shape_dic, arg_params):
    W1, W2 = Q, P
    W1 = W1.reshape(arg_shape_dic[name1+'_weight'])
    weight1 = mx.ndarray.array(W1)
    W2 = W2.reshape(arg_shape_dic[name2+'_weight'])
    b2 = b.reshape(arg_shape_dic[name2+'_bias'])
    weight2 = mx.ndarray.array(W2)
    bias2 = mx.ndarray.array(b2)
    arg_params[name1 + '_weight'] = weight1
    arg_params[name2 + '_weight'] = weight2
    arg_params[name2 + '_bias'] = bias2

  new_model = utils.replace_conv_layer(args.layer, model, sym_handle, arg_handle)
  return new_model

def main():
  model = utils.load_model(args)
  new_model = fc_decomposition(model, args)
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
