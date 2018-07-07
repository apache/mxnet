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

import random as pyrnd
import argparse
import numpy as np
import mxnet as mx
from matplotlib import pyplot as plt
import binary_rbm

mx.random.seed(pyrnd.getrandbits(32))
ctx = mx.gpu()

### Set hyperparameters

parser = argparse.ArgumentParser(description='Restricted Boltzmann machine learning MNIST')
parser.add_argument('--num_hidden', type=int, default=500, help='number of hidden units')
parser.add_argument('--k', type=int, default=20, help='number of Gibbs sampling steps used in the PCD algorithm')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate for stochastic gradient descent') # The optimizer rescales this with `1 / batch_size`
parser.add_argument('--momentum', type=float, default=0, help='momentum for the stochastic gradient descent')
parser.add_argument('--ais_batch_size', type=int, default=100, help='batch size for AIS to estimate the log-likelihood')
parser.add_argument('--ais_num_batch', type=int, default=10, help='number of batches for AIS to estimate the log-likelihood')
parser.add_argument('--ais_intermediate_steps', type=int, default=10, help='number of intermediate distributions for AIS to estimate the log-likelihood')
parser.add_argument('--ais_burn_in_steps', type=int, default=10, help='number of burn in steps for each intermediate distributions of AIS to estimate the log-likelihood')

args = parser.parse_args()


### Prepare data

mnist = mx.test_utils.get_mnist() # Each pixel has a value in [0, 1].
mnist_train_data = mnist['train_data']
mnist_test_data = mnist['test_data']
img_height = mnist_train_data.shape[2]
img_width = mnist_train_data.shape[3]
num_visible = img_width * img_height

# The iterators generate arrays with shape (batch_size, num_channel = 1, height = 28, width = 28)
train_iter = mx.io.NDArrayIter(
    data={'data': mnist_train_data},
    batch_size=args.batch_size,
    shuffle=True)
test_iter = mx.io.NDArrayIter(
    data={'data': mnist_test_data},
    batch_size=args.batch_size,
    shuffle=True)


### Define symbols

data = mx.sym.Variable('data') # (batch_size, num_channel = 1, height, width)
flattened_data = mx.sym.flatten(data=data) # (batch_size, num_channel * height * width)
visible_layer_bias = mx.sym.Variable('visible_layer_bias', init=mx.init.Normal(sigma=.01))
hidden_layer_bias = mx.sym.Variable('hidden_layer_bias', init=mx.init.Normal(sigma=.01))
interaction_weight = mx.sym.Variable('interaction_weight', init=mx.init.Normal(sigma=.01))
aux_hidden_layer_sample = mx.sym.Variable('aux_hidden_layer_sample', init=mx.init.Normal(sigma=.01))
aux_hidden_layer_prob_1 = mx.sym.Variable('aux_hidden_layer_prob_1', init=mx.init.Constant(0))


### Train

rbm = mx.sym.Custom(
    flattened_data,
    visible_layer_bias,
    hidden_layer_bias,
    interaction_weight,
    aux_hidden_layer_sample,
    aux_hidden_layer_prob_1,
    num_hidden=args.num_hidden,
    k=args.k,
    for_training=True,
    op_type='BinaryRBM',
    name='rbm')
model = mx.mod.Module(symbol=rbm, context=ctx, data_names=['data'], label_names=None)
model.bind(data_shapes=train_iter.provide_data)
model.init_params()
model.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': args.learning_rate, 'momentum': args.momentum})

for epoch in range(args.num_epoch):
    train_iter.reset()
    for batch in train_iter:
        model.forward(batch)
        model.backward()
        model.update()
    mx.nd.waitall()
    test_iter.reset()
    params = model.get_params()[0]
    l = binary_rbm.estimate_log_likelihood(
        params['visible_layer_bias'].as_in_context(ctx), 
        params['hidden_layer_bias'].as_in_context(ctx), 
        params['interaction_weight'].as_in_context(ctx),
        args.ais_batch_size, args.ais_num_batch, args.ais_intermediate_steps, args.ais_burn_in_steps, test_iter, ctx)
    print("Epoch %d completed with test log-likelihood %f and partition function %f" % (epoch, l[0], l[1]))

### Show some samples. Each sample is obtained by 1000 steps of Gibbs sampling starting from a real data.

print("Preparing showcase")

showcase_gibbs_sampling_steps = 1000
showcase_num_samples_w = 15
showcase_num_samples_h = 15
showcase_num_samples = showcase_num_samples_w * showcase_num_samples_h
showcase_img_shape = (showcase_num_samples_h * img_height, showcase_num_samples_w * img_width)
showcase_img_column_shape = (showcase_num_samples_h * img_height, img_width)

params = model.get_params()[0] # We don't need aux states here
showcase_rbm = mx.sym.Custom(
    flattened_data,
    visible_layer_bias,
    hidden_layer_bias,
    interaction_weight,
    num_hidden=args.num_hidden,
    k=showcase_gibbs_sampling_steps,
    for_training=False,
    op_type='BinaryRBM',
    name='showcase_rbm')
showcase_iter = mx.io.NDArrayIter(
    data={'data': mnist['train_data']},
    batch_size=showcase_num_samples_h,
    shuffle=True)
showcase_model = mx.mod.Module(symbol=showcase_rbm, context=ctx, data_names=['data'], label_names=None)
showcase_model.bind(data_shapes=showcase_iter.provide_data, for_training=False)
showcase_model.set_params(params, aux_params=None)
showcase_img = np.zeros(showcase_img_shape)
for sample_batch, i, _ in showcase_model.iter_predict(eval_data=showcase_iter, num_batch=showcase_num_samples_w):
    # Each pixel is the probability that the unit is 1.
    showcase_img[:, i * img_width : (i + 1) * img_width] = sample_batch[0].reshape(showcase_img_column_shape).asnumpy()
s = plt.imshow(showcase_img, cmap='gray')
plt.axis('off')
plt.show(s)

print("Done")