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

# coding: utf-8
# pylint: disable=wildcard-import
"""Gluon Estimator example on MNIST dataset with simple CNN"""

import os
import sys

from mxnet import metric
from mxnet import gluon
from mxnet.gluon import nn, data
from mxnet.gluon.estimator import estimator

net = nn.Sequential()

net.add(nn.Conv2D(32, kernel_size=3, activation='relu'),
        nn.Conv2D(64, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Dense(128, activation="relu"), nn.Dropout(0.5),
        nn.Dropout(0.5),
        nn.Dense(10))


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
    '~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)  # Expand the user path '~'.
    transformer = []
    if resize:
        transformer += [data.vision.transforms.Resize(resize)]
    transformer += [data.vision.transforms.ToTensor()]
    transformer = data.vision.transforms.Compose(transformer)
    mnist_train = data.vision.MNIST(root=root, train=True)
    mnist_test = data.vision.MNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = data.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = data.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter


batch_size = 128
train_data, test_data = load_data_fashion_mnist(batch_size, resize=28)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
acc = metric.Accuracy()
est = estimator.Estimator(net=net, loss=loss, metrics=acc)
est.fit(train_data=train_data, epochs=5)