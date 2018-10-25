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

import logging
import math
import random

import mxnet as mx
from mxnet import gluon, autograd, nd
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def evaluate_network(net, data_iterator, ctx):
    loss_acc = 0.
    l2 = gluon.loss.L2Loss()
    for i, (user, item, score) in enumerate(data_iterator):
        user = user.as_in_context(ctx)
        item = item.as_in_context(ctx)
        score = score.as_in_context(ctx)
        pred = net(user, item)
        loss = l2(pred, score)            
        loss_acc += loss.mean()
    return loss_acc.asscalar()/(i+1)
    

def train(network, train_data, test_data, epochs, learning_rate=0.01, optimizer='sgd', ctx=mx.gpu(0), num_epoch_lr=5, factor=0.2):

    np.random.seed(123)  # Fix random seed for consistent demos
    mx.random.seed(123)  # Fix random seed for consistent demos
    random.seed(123)  # Fix random seed for consistent demos

    schedule = mx.lr_scheduler.FactorScheduler(step=len(train_data)*num_epoch_lr, factor=factor)

    trainer = gluon.Trainer(network.collect_params(), optimizer, 
                            {'learning_rate':learning_rate, 'wd':0.0001, 'lr_scheduler':schedule})

    l2 = gluon.loss.L2Loss()

    network.hybridize()
    
    losses = []
    for e in range(epochs):
        loss_acc = 0.
        for i, (user, item, score) in enumerate(train_data):
            user = user.as_in_context(ctx)
            item = item.as_in_context(ctx)
            score = score.as_in_context(ctx)

            with autograd.record():
                pred = network(user, item)
                loss = l2(pred, score)

            loss.backward()
            loss_acc += loss.mean()
            trainer.update(user.shape[0])

        test_loss = evaluate_network(network, test_data, ctx)
        train_loss = loss_acc.asscalar()/(i+1)
        print("Epoch [{}], Training RMSE {:.4f}, Test RMSE {:.4f}".format(e, train_loss, test_loss))
        losses.append((train_loss, test_loss))
    return losses

