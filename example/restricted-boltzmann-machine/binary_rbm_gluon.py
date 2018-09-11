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


### Helper function

def get_non_auxiliary_params(rbm):
    return rbm.collect_params('^(?!.*_aux_.*).*$')

### Command line arguments

parser = argparse.ArgumentParser(description='Restricted Boltzmann machine learning MNIST')
parser.add_argument('--num-hidden', type=int, default=500, help='number of hidden units')
parser.add_argument('--k', type=int, default=30, help='number of Gibbs sampling steps used in the PCD algorithm')
parser.add_argument('--batch-size', type=int, default=80, help='batch size')
parser.add_argument('--num-epoch', type=int, default=130, help='number of epochs')
parser.add_argument('--learning-rate', type=float, default=0.1, help='learning rate for stochastic gradient descent') # The optimizer rescales this with `1 / batch_size`
parser.add_argument('--momentum', type=float, default=0.3, help='momentum for the stochastic gradient descent')
parser.add_argument('--ais-batch-size', type=int, default=100, help='batch size for AIS to estimate the log-likelihood')
parser.add_argument('--ais-num-batch', type=int, default=10, help='number of batches for AIS to estimate the log-likelihood')
parser.add_argument('--ais-intermediate-steps', type=int, default=10, help='number of intermediate distributions for AIS to estimate the log-likelihood')
parser.add_argument('--ais-burn-in-steps', type=int, default=10, help='number of burn in steps for each intermediate distributions of AIS to estimate the log-likelihood')
parser.add_argument('--cuda', action='store_true', dest='cuda', help='train on GPU with CUDA')
parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='train on CPU')
parser.add_argument('--device-id', type=int, default=0, help='GPU device id')
parser.add_argument('--data-loader-num-worker', type=int, default=4, help='number of multithreading workers for the data loader')
parser.set_defaults(cuda=True)

args = parser.parse_args()
print(args)

### Global environment

mx.random.seed(pyrnd.getrandbits(32))
ctx = mx.gpu(args.device_id) if args.cuda else mx.cpu()


### Prepare data

def data_transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)

mnist_train_dataset = mx.gluon.data.vision.MNIST(train=True, transform=data_transform)
mnist_test_dataset = mx.gluon.data.vision.MNIST(train=False, transform=data_transform)
img_height = mnist_train_dataset[0][0].shape[0]
img_width = mnist_train_dataset[0][0].shape[1]
num_visible = img_width * img_height

# This generates arrays with shape (batch_size, height = 28, width = 28, num_channel = 1)
train_data = mx.gluon.data.DataLoader(mnist_train_dataset, args.batch_size, shuffle=True, num_workers=args.data_loader_num_worker)
test_data = mx.gluon.data.DataLoader(mnist_test_dataset, args.batch_size, shuffle=True, num_workers=args.data_loader_num_worker)

### Train

rbm = BinaryRBMBlock(num_hidden=args.num_hidden, k=args.k, for_training=True, prefix='rbm_')
rbm.initialize(mx.init.Normal(sigma=.01), ctx=ctx)
rbm.hybridize()
trainer = mx.gluon.Trainer(
    get_non_auxiliary_params(rbm),
    'sgd', {'learning_rate': args.learning_rate, 'momentum': args.momentum})
for epoch in range(args.num_epoch):
    # Update parameters
    for batch, _ in train_data:
        batch = batch.as_in_context(ctx).flatten()
        with mx.autograd.record():
            out = rbm(batch)
        out[0].backward()
        trainer.step(batch.shape[0])
    mx.nd.waitall() # To restrict memory usage

    # Monitor the performace of the model
    params = get_non_auxiliary_params(rbm)
    param_visible_layer_bias = params['rbm_visible_layer_bias'].data(ctx=ctx)
    param_hidden_layer_bias = params['rbm_hidden_layer_bias'].data(ctx=ctx)
    param_interaction_weight = params['rbm_interaction_weight'].data(ctx=ctx)
    test_log_likelihood, _ = estimate_log_likelihood(
            param_visible_layer_bias, param_hidden_layer_bias, param_interaction_weight,
            args.ais_batch_size, args.ais_num_batch, args.ais_intermediate_steps, args.ais_burn_in_steps, test_data, ctx)
    train_log_likelihood, _ = estimate_log_likelihood(
            param_visible_layer_bias, param_hidden_layer_bias, param_interaction_weight,
            args.ais_batch_size, args.ais_num_batch, args.ais_intermediate_steps, args.ais_burn_in_steps, train_data, ctx)
    print("Epoch %d completed with test log-likelihood %f and train log-likelihood %f" % (epoch, test_log_likelihood, train_log_likelihood))


### Show some samples.

# Each sample is obtained by 3000 steps of Gibbs sampling starting from a real sample.
# Starting from the real data is just for convenience of implmentation.
# There must be no correlation between the initial states and the resulting samples.
# You can start from random states and run the Gibbs chain for sufficiently long time.

print("Preparing showcase")

showcase_gibbs_sampling_steps = 3000
showcase_num_samples_w = 15
showcase_num_samples_h = 15
showcase_num_samples = showcase_num_samples_w * showcase_num_samples_h
showcase_img_shape = (showcase_num_samples_h * img_height, 2 * showcase_num_samples_w * img_width)
showcase_img_column_shape = (showcase_num_samples_h * img_height, img_width)

showcase_rbm = BinaryRBMBlock(
    num_hidden=args.num_hidden,
    k=showcase_gibbs_sampling_steps,
    for_training=False,
    params=get_non_auxiliary_params(rbm))
showcase_iter = iter(mx.gluon.data.DataLoader(mnist_train_dataset, showcase_num_samples_h, shuffle=True))
showcase_img = np.zeros(showcase_img_shape)
for i in range(showcase_num_samples_w):
    data_batch = next(showcase_iter)[0].as_in_context(ctx).flatten()
    sample_batch = showcase_rbm(data_batch)
    # Each pixel is the probability that the unit is 1.
    showcase_img[:, i * img_width : (i + 1) * img_width] = data_batch.reshape(showcase_img_column_shape).asnumpy()
    showcase_img[:, (showcase_num_samples_w + i) * img_width : (showcase_num_samples_w + i + 1) * img_width
                ] = sample_batch[0].reshape(showcase_img_column_shape).asnumpy()
s = plt.imshow(showcase_img, cmap='gray')
plt.axis('off')
plt.axvline(showcase_num_samples_w * img_width, color='y')
plt.show(s)

print("Done")