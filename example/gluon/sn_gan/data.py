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

import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import CIFAR10

IMAGE_SIZE = 64

def transformer(data, label):
    """ data preparation """
    data = mx.image.imresize(data, IMAGE_SIZE, IMAGE_SIZE)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 128.0 - 1
    return data, label


def get_training_data(batch_size):
    """ helper function to get dataloader"""
    return gluon.data.DataLoader(
        CIFAR10(train=True, transform=transformer),
        batch_size=batch_size, shuffle=True, last_batch='discard')
