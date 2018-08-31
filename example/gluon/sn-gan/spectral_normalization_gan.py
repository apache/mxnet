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

# This example is inspired by https://github.com/jason71995/Keras-GAN-Library

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


class Options:
    """ Hyperparameter """
    def __init__(self):
        self.batch_size = 64  # input batch size
        self.image_size = 64  # the height / width of the input image to network'
        self.z_dim = 100  # size of the latent z vector
        self.niter = 1000  # number of epochs to train for
        self.learning_rate = 0.0002  # learning rate, default=0.0002
        self.beta1 = 0.5  # beta1 for adam
        self.outf = './data'  # help='folder to output images and model checkpoints')
        self.manual_seed = random.randint(1, 10000)  # manual seed
        self.clip_gradient = 10.0
        self.ctx = mx.cpu()


class SNConv2D(Block):
    """ Customized Conv2D """
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
            self.weight = self.params.get('weight', shape=(
                num_filter, in_channels, kernel_size, kernel_size))
            self.u = self.params.get(
                'u', init=mx.init.Normal(), shape=(1, num_filter))

    def spectral_norm(self):
        """ spectral normalization """
        w = self.params.get('weight').data(opt.ctx)
        w_mat = w
        w_mat = nd.reshape(w_mat, [w_mat.shape[0], -1])

        _u = self.u.data(opt.ctx)
        _v = None

        for _ in range(1):
            _v = nd.L2Normalization(nd.dot(_u, w_mat))
            _u = nd.L2Normalization(nd.dot(_v, w_mat.T))

        sigma = nd.sum(nd.dot(_u, w_mat) * _v)

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
            no_bias=True)


def transformer(data, label):
    """ data preparation """
    data = mx.image.imresize(data, opt.image_size, opt.image_size)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 128.0 - 1
    return data, label


def save_image(data, padding=2):
    """ save image """
    data = data.asnumpy().transpose((0, 2, 3, 1))
    datanp = np.clip(
        (data - np.min(data))*(255.0/(np.max(data) - np.min(data))), 0, 255).astype(np.uint8)
    x_dim = min(8, opt.batch_size)
    y_dim = int(math.ceil(float(opt.batch_size) / x_dim))
    height, width = int(opt.image_size + padding), int(opt.image_size + padding)
    grid = np.zeros((height * y_dim + 1 + padding // 2, width *
                     x_dim + 1 + padding // 2, 3), dtype=np.uint8)
    k = 0
    for y in range(y_dim):
        for x in range(x_dim):
            if k >= opt.batch_size:
                break
            start_y = y * height + 1 + padding // 2
            end_y = start_y + height - padding
            start_x = x * width + 1 + padding // 2
            end_x = start_x + width - padding
            np.copyto(grid[start_y:end_y, start_x:end_x, :], datanp[k])
            k += 1
    imageio.imwrite(
        '{}/fake_samples_epoch_{}.png'.format(opt.outf, epoch), grid)


def facc(label, pred):
    """ evaluate accuracy """
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


opt = Options()
try:
    os.makedirs(opt.outf)
except OSError:
    pass
mx.random.seed(opt.manual_seed)

train_data = gluon.data.DataLoader(
    CIFAR10(train=True, transform=transformer),
    batch_size=opt.batch_size, shuffle=True, last_batch='discard')

g_net = gluon.nn.Sequential()
with g_net.name_scope():
    #first layer
    g_net.add(gluon.nn.Conv2DTranspose(512, 4, 1, 0, use_bias=False))
    g_net.add(gluon.nn.BatchNorm())
    g_net.add(gluon.nn.LeakyReLU(0.2))
    #second layer
    g_net.add(gluon.nn.Conv2DTranspose(256, 4, 2, 1, use_bias=False))
    g_net.add(gluon.nn.BatchNorm())
    g_net.add(gluon.nn.LeakyReLU(0.2))
    #tird layer
    g_net.add(gluon.nn.Conv2DTranspose(128, 4, 2, 1, use_bias=False))
    g_net.add(gluon.nn.BatchNorm())
    g_net.add(gluon.nn.LeakyReLU(0.2))
    #fourth layer
    g_net.add(gluon.nn.Conv2DTranspose(64, 4, 2, 1, use_bias=False))
    g_net.add(gluon.nn.BatchNorm())
    g_net.add(gluon.nn.LeakyReLU(0.2))
    #fifth layer
    g_net.add(gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False))
    g_net.add(gluon.nn.Activation('tanh'))


d_net = gluon.nn.Sequential()
with d_net.name_scope():
    #first layer
    d_net.add(SNConv2D(64, 4, 2, 1, in_channels=3))
    d_net.add(gluon.nn.LeakyReLU(0.2))
    #second layer
    d_net.add(SNConv2D(128, 4, 2, 1, in_channels=64))
    d_net.add(gluon.nn.LeakyReLU(0.2))
    #tird layer
    d_net.add(SNConv2D(256, 4, 2, 1, in_channels=128))
    d_net.add(gluon.nn.LeakyReLU(0.2))
    #fourth layer
    d_net.add(SNConv2D(512, 4, 2, 1, in_channels=256))
    d_net.add(gluon.nn.LeakyReLU(0.2))
    #fifth layer
    d_net.add(SNConv2D(1, 4, 1, 0, in_channels=512))


loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

#Initialization
g_net.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
d_net.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
g_trainer = gluon.Trainer(
    g_net.collect_params(), 'Adam', {'learning_rate': opt.learning_rate, 'beta1': opt.beta1})
d_trainer = gluon.Trainer(
    d_net.collect_params(), 'Adam', {'learning_rate': opt.learning_rate, 'beta1': opt.beta1})


g_net.collect_params().zero_grad()
d_net.collect_params().zero_grad()

metric = mx.metric.CustomMetric(facc)

real_label = nd.ones(opt.batch_size, opt.ctx)
fake_label = nd.zeros(opt.batch_size, opt.ctx)
logging.basicConfig(level=logging.DEBUG)

for epoch in range(opt.niter):
    for i, (d, _) in enumerate(train_data):
        # update D
        data = d.as_in_context(opt.ctx)
        noise = nd.normal(loc=0, scale=1, shape=(
            opt.batch_size, opt.z_dim, 1, 1), ctx=opt.ctx)
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

        d_trainer.step(opt.batch_size)
        # update G
        with autograd.record():
            fake_image = g_net(noise)
            output = d_net(fake_image).reshape(-1, 1)
            errG = loss(output, real_label)
            errG.backward()

        g_trainer.step(opt.batch_size)

        # print log infomation every ten batches
        if i % 100 == 0:
            name, acc = metric.get()
            logging.info('discriminator loss = {}, generator loss = {}, \
                binary training acc = {} at iter {} epoch {}'.format(
                    nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc, iter, epoch))
        if i % 1000 == 0:
            save_image(fake_image)

    metric.reset()
