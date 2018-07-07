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
from binary_rbm import BinaryRBMBlock
from binary_rbm import estimate_log_likelihood

mx.random.seed(pyrnd.getrandbits(32))
ctx = mx.gpu()


### Helper function

def get_non_auxiliary_params(rbm):
    return rbm.collect_params('^(?!.*_aux_.*).*$')

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

def data_transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)

mnist_train_dataset = mx.gluon.data.vision.MNIST(train=True, transform=data_transform)
mnist_test_dataset = mx.gluon.data.vision.MNIST(train=False, transform=data_transform)
img_height = mnist_train_dataset[0][0].shape[0]
img_width = mnist_train_dataset[0][0].shape[1]
num_visible = img_width * img_height

# This generates arrays with shape (batch_size, height = 28, width = 28, num_channel = 1)
train_data = mx.gluon.data.DataLoader(mnist_train_dataset, args.batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mnist_test_dataset, args.batch_size, shuffle=True)

### Train

rbm = BinaryRBMBlock(num_hidden=args.num_hidden, k=args.k, for_training=True, prefix='rbm_')
rbm.initialize(mx.init.Normal(sigma=.01), ctx=ctx)
rbm.hybridize()
trainer = mx.gluon.Trainer(
    get_non_auxiliary_params(rbm),
    'sgd', {'learning_rate': args.learning_rate, 'momentum': args.momentum})
for epoch in range(args.num_epoch):
    for i, (batch, _) in enumerate(train_data):
        batch = batch.as_in_context(ctx).reshape((args.batch_size, num_visible))
        with mx.autograd.record():
            out = rbm(batch)
        out[0].backward()
        trainer.step(batch.shape[0])
    mx.nd.waitall() # To restrict memory usage
    params = get_non_auxiliary_params(rbm)
    l = estimate_log_likelihood(
            params['rbm_visible_layer_bias'].data().as_in_context(ctx), 
            params['rbm_hidden_layer_bias'].data().as_in_context(ctx), 
            params['rbm_interaction_weight'].data().as_in_context(ctx),
            args.ais_batch_size, args.ais_num_batch, args.ais_intermediate_steps, args.ais_burn_in_steps, test_data, ctx)
    print("Epoch %d completed with test log-likelihood %f and partition function %f" % (epoch, l[0], l[1]))


### Show some samples. Each sample is obtained by 1000 steps of Gibbs sampling starting from a real data.

print("Preparing showcase")

showcase_gibbs_sampling_steps = 1000
showcase_num_samples_w = 15
showcase_num_samples_h = 15
showcase_num_samples = showcase_num_samples_w * showcase_num_samples_h
showcase_img_shape = (showcase_num_samples_h * img_height, showcase_num_samples_w * img_width)
showcase_img_column_shape = (showcase_num_samples_h * img_height, img_width)

showcase_rbm = BinaryRBMBlock(
    num_hidden=args.num_hidden,
    k=showcase_gibbs_sampling_steps,
    for_training=False,
    params=get_non_auxiliary_params(rbm))
showcase_iter = iter(mx.gluon.data.DataLoader(mnist_train_dataset, showcase_num_samples_h, shuffle=True))
showcase_img = np.zeros(showcase_img_shape)
for i in range(showcase_num_samples_w):
    data_batch = next(showcase_iter)[0].as_in_context(ctx).reshape((showcase_num_samples_h, num_visible))
    sample_batch = showcase_rbm(data_batch)
    # Each pixel is the probability that the unit is 1.
    showcase_img[:, i * img_width : (i + 1) * img_width] = sample_batch[0].reshape(showcase_img_column_shape).asnumpy()
s = plt.imshow(showcase_img, cmap='gray')
plt.axis('off')
plt.show(s)

print("Done")