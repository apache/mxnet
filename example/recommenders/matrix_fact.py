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

def evaluate_network(network, data_iterator, ctx):
    loss_acc = 0.
    l2 = gluon.loss.L2Loss()
    for idx, (users, items, scores) in enumerate(data_iterator):
        users_ = gluon.utils.split_and_load(users, ctx)
        items_ = gluon.utils.split_and_load(items, ctx)
        scores_ =gluon.utils.split_and_load(scores, ctx)
        preds = [network(u, i) for u, i in zip(users_, items_)]
        losses = [l2(p, s).asnumpy() for p, s in zip(preds, scores_)]         
        loss_acc += sum(losses).mean()/len(ctx)
    return loss_acc/(idx+1)

def train(network, train_data, test_data, epochs, learning_rate=0.01, optimizer='sgd', ctx=mx.gpu(0), num_epoch_lr=5, factor=0.2):

    np.random.seed(123)  # Fix random seed for consistent demos
    mx.random.seed(123)  # Fix random seed for consistent demos
    random.seed(123)  # Fix random seed for consistent demos

    schedule = mx.lr_scheduler.FactorScheduler(step=len(train_data)*len(ctx)*num_epoch_lr, factor=factor)

    trainer = gluon.Trainer(network.collect_params(), optimizer,
                            {'learning_rate':learning_rate, 'wd':0.0001, 'lr_scheduler':schedule})  
                            #update_on_kvstore=False)

    l2 = gluon.loss.L2Loss()

    network.hybridize()
    
    losses_output = []
    for e in range(epochs):
        loss_acc = 0.
        for idx, (users, items, scores) in enumerate(train_data):
            
            users_ = gluon.utils.split_and_load(users, ctx)
            items_ = gluon.utils.split_and_load(items, ctx)
            scores_ =gluon.utils.split_and_load(scores, ctx)

            with autograd.record():
                preds = [network(u, i) for u, i in zip(users_, items_)]
                losses = [l2(p, s) for p, s in zip(preds, scores_)]

            [l.backward() for l in losses]
            loss_acc += sum([l.asnumpy() for l in losses]).mean()/len(ctx)
            trainer.update(users.shape[0])

        test_loss = evaluate_network(network, test_data, ctx)
        train_loss = loss_acc/(idx+1)
        print("Epoch [{}], Training RMSE {:.4f}, Test RMSE {:.4f}".format(e, train_loss, test_loss))
        losses_output.append((train_loss, test_loss))
    return losses_output