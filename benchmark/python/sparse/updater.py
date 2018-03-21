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

import time
import mxnet as mx
from mxnet.ndarray.sparse import adam_update
import numpy as np
import argparse

mx.random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Benchmark adam updater')
parser.add_argument('--dim-in', type=int, default=240000, help='weight.shape[0]')
parser.add_argument('--dim-out', type=int, default=512, help='weight.shape[1]')
parser.add_argument('--nnr', type=int, default=5000, help='grad.indices.shape[0]')
parser.add_argument('--repeat', type=int, default=1000, help='num repeat')
parser.add_argument('--dense-grad', action='store_true',
                    help='if set to true, both gradient and weight are dense.')
parser.add_argument('--dense-state', action='store_true',
                    help='if set to true, states are dense, indicating standard update')
parser.add_argument('--cpu', action='store_true')


args = parser.parse_args()
dim_in = args.dim_in
dim_out = args.dim_out
nnr = args.nnr
ctx = mx.cpu() if args.cpu else mx.gpu()

ones = mx.nd.ones((dim_in, dim_out), ctx=ctx)

if not args.dense_grad:
    weight = ones.tostype('row_sparse')
    indices = np.arange(dim_in)
    np.random.shuffle(indices)
    indices = np.unique(indices[:nnr])
    indices = mx.nd.array(indices, ctx=ctx)
    grad = mx.nd.sparse.retain(weight, indices)
else:
    weight = ones.copy()
    grad = ones.copy()

if args.dense_state:
    mean = ones.copy()
else:
    mean = ones.tostype('row_sparse')

var = mean.copy()

# warmup 
for i in range(10):
    adam_update(weight, grad, mean, var, out=weight, lr=1, wd=0, beta1=0.9,
                beta2=0.99, rescale_grad=0.5, epsilon=1e-8)
weight.wait_to_read()

# measure speed
a = time.time()
for i in range(args.repeat):
    adam_update(weight, grad, mean, var, out=weight, lr=1, wd=0, beta1=0.9,
                beta2=0.99, rescale_grad=0.5, epsilon=1e-8)
weight.wait_to_read()
b = time.time()
print(b - a)
