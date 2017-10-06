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

from __future__ import print_function
import argparse
import mxnet as mx
import numpy as np
import logging
import data
from autoencoder import AutoEncoderModel

parser = argparse.ArgumentParser(description='Train an auto-encoder model for mnist dataset.')
parser.add_argument('--print-every', type=int, default=1000,
                    help='the interval of printing during training.')
parser.add_argument('--batch-size', type=int, default=256,
                    help='the batch size used for training.')
parser.add_argument('--pretrain-iter', type=int, default=50000,
                    help='the number of iterations for pretraining.')
parser.add_argument('--finetune-iter', type=int, default=100000,
                    help='the number of iterations for fine-tuning.')
parser.add_argument('--visualize', action='store_true',
                    help='whether to visualize the original image and the reconstructed one.')

# set to INFO to see less information during training
logging.basicConfig(level=logging.DEBUG)
opt = parser.parse_args()
logging.info(opt)
print_every = opt.print_every
batch_size = opt.batch_size
pretrain_iter = opt.pretrain_iter
finetune_iter = opt.finetune_iter
visualize = opt.visualize

if __name__ == '__main__':
    ae_model = AutoEncoderModel(mx.cpu(0), [784,500,500,2000,10], pt_dropout=0.2,
        internal_act='relu', output_act='relu')

    X, _ = data.get_mnist()
    train_X = X[:60000]
    val_X = X[60000:]

    ae_model.layerwise_pretrain(train_X, batch_size, pretrain_iter, 'sgd', l_rate=0.1, decay=0.0,
                                lr_scheduler=mx.misc.FactorScheduler(20000,0.1),
                                print_every=print_every)
    ae_model.finetune(train_X, batch_size, finetune_iter, 'sgd', l_rate=0.1, decay=0.0,
                      lr_scheduler=mx.misc.FactorScheduler(20000,0.1), print_every=print_every)
    ae_model.save('mnist_pt.arg')
    ae_model.load('mnist_pt.arg')
    print("Training error:", ae_model.eval(train_X))
    print("Validation error:", ae_model.eval(val_X))
