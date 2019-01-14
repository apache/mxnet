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
# https://github.com/apache/incubator-mxnet/blob/master/example/gluon/dc_gan/dcgan.py

import mxnet as mx
from mxnet import nd
from mxnet import gluon, autograd
from mxnet.gluon import Block


EPSILON = 1e-08
POWER_ITERATION = 1

class SNConv2D(Block):
    """ Customized Conv2D to feed the conv with the weight that we apply spectral normalization """

    def __init__(self, num_filter, kernel_size,
                 strides, padding, in_channels,
                 ctx=mx.cpu(), iterations=1):

        super(SNConv2D, self).__init__()

        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.in_channels = in_channels
        self.iterations = iterations
        self.ctx = ctx

        with self.name_scope():
            # init the weight
            self.weight = self.params.get('weight', shape=(
                num_filter, in_channels, kernel_size, kernel_size))
            self.u = self.params.get(
                'u', init=mx.init.Normal(), shape=(1, num_filter))

    def _spectral_norm(self):
        """ spectral normalization """
        w = self.params.get('weight').data(self.ctx)
        w_mat = nd.reshape(w, [w.shape[0], -1])

        _u = self.u.data(self.ctx)
        _v = None

        for _ in range(POWER_ITERATION):
            _v = nd.L2Normalization(nd.dot(_u, w_mat))
            _u = nd.L2Normalization(nd.dot(_v, w_mat.T))

        sigma = nd.sum(nd.dot(_u, w_mat) * _v)
        if sigma == 0.:
            sigma = EPSILON

        with autograd.pause():
            self.u.set_data(_u)

        return w / sigma

    def forward(self, x):
        # x shape is batch_size x in_channels x height x width
        return nd.Convolution(
            data=x,
            weight=self._spectral_norm(),
            kernel=(self.kernel_size, self.kernel_size),
            pad=(self.padding, self.padding),
            stride=(self.strides, self.strides),
            num_filter=self.num_filter,
            no_bias=True
        )


def get_generator():
    """ construct and return generator """
    g_net = gluon.nn.Sequential()
    with g_net.name_scope():

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=512, kernel_size=4, strides=1, padding=0, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=256, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=128, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=64, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(channels=3, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.Activation('tanh'))

    return g_net


def get_descriptor(ctx):
    """ construct and return descriptor """
    d_net = gluon.nn.Sequential()
    with d_net.name_scope():

        d_net.add(SNConv2D(num_filter=64, kernel_size=4, strides=2, padding=1, in_channels=3, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=128, kernel_size=4, strides=2, padding=1, in_channels=64, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=256, kernel_size=4, strides=2, padding=1, in_channels=128, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=512, kernel_size=4, strides=2, padding=1, in_channels=256, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=1, kernel_size=4, strides=1, padding=0, in_channels=512, ctx=ctx))

    return d_net
