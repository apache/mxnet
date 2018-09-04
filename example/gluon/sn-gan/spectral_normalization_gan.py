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

# This example is inspired by https://github.com/jason71995/Keras-GAN-Library,
# https://github.com/kazizzad/DCGAN-Gluon-MxNet/blob/master/MxnetDCGAN.ipynb
# https://github.com/apache/incubator-mxnet/blob/master/example/gluon/dcgan.py


import math
import os
import random
import logging

import imageio
import numpy as np

import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon.data.vision import CIFAR10
from mxnet.gluon import Block


BATCH_SIZE = 64  # input batch size
IMAGE_SIZE = 64  # image size
Z_DIM = 100  # dimension of the latent z vector
NUM_ITER = 1000  # number of epochs to train for
LEARNING_RATE = 0.0002  # learning rate
BETA = 0.5  # beta1 for adam
OUTPUT_DIR = './data'  # output directory
MANUAL_SEED = random.randint(1, 10000)  # manual seed
CTX = mx.gpu()  # change to gpu if you have gpu
POWER_ITERATION = 1
CLIP_GRADIENT = 10

class SNConv2D(Block):
    """ Customized Conv2D to feed the conv with the weight that we apply spectral normalization """
    def __init__(self, num_filter, kernel_size,
                 strides, padding, in_channels=0,
                 iterations=1):

        super(SNConv2D, self).__init__()

        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.in_channels = in_channels
        self.iterations = iterations

        with self.name_scope():
            # init the weight
            self.weight = self.params.get('weight', shape=(
                num_filter, in_channels, kernel_size, kernel_size))
            self.u = self.params.get(
                'u', init=mx.init.Normal(), shape=(1, num_filter))

    def spectral_norm(self):
        """ spectral normalization """
        w = self.params.get('weight').data(CTX)
        w_mat = w
        w_mat = nd.reshape(w_mat, [w_mat.shape[0], -1])

        _u = self.u.data(CTX)
        _v = None

        for _ in range(POWER_ITERATION):
            _v = nd.L2Normalization(nd.dot(_u, w_mat))
            _u = nd.L2Normalization(nd.dot(_v, w_mat.T))

        sigma = nd.sum(nd.dot(_u, w_mat) * _v)
        if sigma == 0.:
            sigma = 0.00000001

        self.params.setattr('u', _u)

        return w / sigma

    def forward(self, x):
        # x shape is batch_size x in_channels x height x width
        return nd.Convolution(
            data=x,
            weight=self.spectral_norm(),
            kernel=(self.kernel_size, self.kernel_size),
            pad=(self.padding, self.padding),
            stride=(self.strides, self.strides),
            num_filter=self.num_filter,
            no_bias=True
        )


def transformer(data, label):
    """ data preparation """
    data = mx.image.imresize(data, IMAGE_SIZE, IMAGE_SIZE)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 128.0 - 1
    return data, label


def save_image(data, epoch, padding=2):
    """ save image """
    data = data.asnumpy().transpose((0, 2, 3, 1))
    datanp = np.clip(
        (data - np.min(data))*(255.0/(np.max(data) - np.min(data))), 0, 255).astype(np.uint8)
    x_dim = min(8, BATCH_SIZE)
    y_dim = int(math.ceil(float(BATCH_SIZE) / x_dim))
    height, width = int(IMAGE_SIZE + padding), int(IMAGE_SIZE + padding)
    grid = np.zeros((height * y_dim + 1 + padding // 2, width *
                     x_dim + 1 + padding // 2, 3), dtype=np.uint8)
    k = 0
    for y in range(y_dim):
        for x in range(x_dim):
            if k >= BATCH_SIZE:
                break
            start_y = y * height + 1 + padding // 2
            end_y = start_y + height - padding
            start_x = x * width + 1 + padding // 2
            end_x = start_x + width - padding
            np.copyto(grid[start_y:end_y, start_x:end_x, :], datanp[k])
            k += 1
    imageio.imwrite(
        '{}/fake_samples_epoch_{}.png'.format(OUTPUT_DIR, epoch), grid)


def facc(label, pred):
    """ evaluate accuracy """
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


# create output directory
try:
    os.makedirs(OUTPUT_DIR)
except OSError:
    pass
mx.random.seed(MANUAL_SEED)

train_data = gluon.data.DataLoader(
    CIFAR10(train=True, transform=transformer),
    batch_size=BATCH_SIZE, shuffle=True, last_batch='discard')

# define the network structure
g_net = gluon.nn.Sequential()
with g_net.name_scope():

    g_net.add(gluon.nn.Conv2DTranspose(512, 4, 1, 0, use_bias=False))
    g_net.add(gluon.nn.BatchNorm())
    g_net.add(gluon.nn.LeakyReLU(0.2))

    g_net.add(gluon.nn.Conv2DTranspose(256, 4, 2, 1, use_bias=False))
    g_net.add(gluon.nn.BatchNorm())
    g_net.add(gluon.nn.LeakyReLU(0.2))

    g_net.add(gluon.nn.Conv2DTranspose(128, 4, 2, 1, use_bias=False))
    g_net.add(gluon.nn.BatchNorm())
    g_net.add(gluon.nn.LeakyReLU(0.2))

    g_net.add(gluon.nn.Conv2DTranspose(64, 4, 2, 1, use_bias=False))
    g_net.add(gluon.nn.BatchNorm())
    g_net.add(gluon.nn.LeakyReLU(0.2))

    g_net.add(gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False))
    g_net.add(gluon.nn.Activation('tanh'))


d_net = gluon.nn.Sequential()
with d_net.name_scope():

    d_net.add(SNConv2D(64, 4, 2, 1, in_channels=3))
    d_net.add(gluon.nn.LeakyReLU(0.2))

    d_net.add(SNConv2D(128, 4, 2, 1, in_channels=64))
    d_net.add(gluon.nn.LeakyReLU(0.2))

    d_net.add(SNConv2D(256, 4, 2, 1, in_channels=128))
    d_net.add(gluon.nn.LeakyReLU(0.2))

    d_net.add(SNConv2D(512, 4, 2, 1, in_channels=256))
    d_net.add(gluon.nn.LeakyReLU(0.2))

    d_net.add(SNConv2D(1, 4, 1, 0, in_channels=512))

# define loss function
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# initialization
g_net.collect_params().initialize(mx.init.Normal(0.02), ctx=CTX)
d_net.collect_params().initialize(mx.init.Normal(0.02), ctx=CTX)
g_trainer = gluon.Trainer(
    g_net.collect_params(), 'Adam', {'learning_rate': LEARNING_RATE, 'beta1': BETA, 'clip_gradient': CLIP_GRADIENT})
d_trainer = gluon.Trainer(
    d_net.collect_params(), 'Adam', {'learning_rate': LEARNING_RATE, 'beta1': BETA, 'clip_gradient': CLIP_GRADIENT})

g_net.collect_params().zero_grad()
d_net.collect_params().zero_grad()

metric = mx.metric.CustomMetric(facc)

real_label = nd.ones(BATCH_SIZE, CTX)
fake_label = nd.zeros(BATCH_SIZE, CTX)

logging.basicConfig(level=logging.DEBUG)

for epoch in range(NUM_ITER):
    for i, (d, _) in enumerate(train_data):
        # update D
        data = d.as_in_context(CTX)
        noise = nd.normal(loc=0, scale=1, shape=(
            BATCH_SIZE, Z_DIM, 1, 1), ctx=CTX)
        with autograd.record():
            # train with real image
            output = d_net(data).reshape((-1, 1))
            errD_real = loss(output, real_label)
            metric.update([real_label, ], [output, ])

            # train with fake image
            fake_image = g_net(noise)
            output = d_net(fake_image.detach()).reshape((-1, 1))
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label, ], [output, ])

        d_trainer.step(BATCH_SIZE)
        # update G
        with autograd.record():
            fake_image = g_net(noise)
            output = d_net(fake_image).reshape(-1, 1)
            errG = loss(output, real_label)
            errG.backward()

        g_trainer.step(BATCH_SIZE)

        # print log infomation every 100 batches
        if i % 100 == 0:
            name, acc = metric.get()
            logging.info('discriminator loss = {}, generator loss = {}, \
                binary training acc = {} at iter {} epoch {}'.format(
                    nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc, i, epoch))
        if i == 0:
            save_image(fake_image, epoch)

    metric.reset()
