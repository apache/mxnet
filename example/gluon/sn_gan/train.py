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


import os
import random
import logging
import argparse

from data import get_training_data
from model import get_generator, get_descriptor
from utils import save_image

import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon

# CLI
parser = argparse.ArgumentParser(
    description='train a model for Spectral Normalization GAN.')
parser.add_argument('--data-path', type=str, default='./data',
                    help='path of data.')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size. default is 64.')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs. default is 100.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate. default is 0.0001.')
parser.add_argument('--lr-beta', type=float, default=0.5,
                    help='learning rate for the beta in margin based loss. default is 0.5.')
parser.add_argument('--use-gpu', action='store_true',
                    help='use gpu for training.')
parser.add_argument('--clip_gr', type=float, default=10.0,
                    help='Clip the gradient by projecting onto the box. default is 10.0.')
parser.add_argument('--z-dim', type=int, default=100,
                    help='dimension of the latent z vector. default is 100.')
opt = parser.parse_args()

BATCH_SIZE = opt.batch_size
Z_DIM = opt.z_dim
NUM_EPOCHS = opt.epochs
LEARNING_RATE = opt.lr
BETA = opt.lr_beta
OUTPUT_DIR = opt.data_path
CTX = mx.gpu() if opt.use_gpu else mx.cpu()
CLIP_GRADIENT = opt.clip_gr
IMAGE_SIZE = 64


def facc(label, pred):
    """ evaluate accuracy """
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


# setting
mx.random.seed(random.randint(1, 10000))
logging.basicConfig(level=logging.DEBUG)

# create output dir
try:
    os.makedirs(opt.data_path)
except OSError:
    pass

# get training data
train_data = get_training_data(opt.batch_size)

# get model
g_net = get_generator()
d_net = get_descriptor(CTX)

# define loss function
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# initialization
g_net.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
d_net.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
g_trainer = gluon.Trainer(
    g_net.collect_params(), 'Adam', {'learning_rate': LEARNING_RATE, 'beta1': BETA, 'clip_gradient': CLIP_GRADIENT})
d_trainer = gluon.Trainer(
    d_net.collect_params(), 'Adam', {'learning_rate': LEARNING_RATE, 'beta1': BETA, 'clip_gradient': CLIP_GRADIENT})
g_net.collect_params().zero_grad()
d_net.collect_params().zero_grad()
# define evaluation metric
metric = mx.metric.CustomMetric(facc)
# initialize labels
real_label = nd.ones(BATCH_SIZE, CTX)
fake_label = nd.zeros(BATCH_SIZE, CTX)

for epoch in range(NUM_EPOCHS):
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
            logging.info('discriminator loss = %f, generator loss = %f, \
                          binary training acc = %f at iter %d epoch %d',
                         nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc, i, epoch)
        if i == 0:
            save_image(fake_image, epoch, IMAGE_SIZE, BATCH_SIZE, OUTPUT_DIR)

    metric.reset()
