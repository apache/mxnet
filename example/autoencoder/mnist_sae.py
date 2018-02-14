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

# pylint: disable=missing-docstring
from __future__ import print_function

import argparse
import logging

import mxnet as mx
import numpy as np
import data
from autoencoder import AutoEncoderModel

parser = argparse.ArgumentParser(description='Train an auto-encoder model for mnist dataset.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--print-every', type=int, default=1000,
                    help='interval of printing during training.')
parser.add_argument('--batch-size', type=int, default=256,
                    help='batch size used for training.')
parser.add_argument('--pretrain-num-iter', type=int, default=50000,
                    help='number of iterations for pretraining.')
parser.add_argument('--finetune-num-iter', type=int, default=100000,
                    help='number of iterations for fine-tuning.')
parser.add_argument('--visualize', action='store_true',
                    help='whether to visualize the original image and the reconstructed one.')
parser.add_argument('--num-units', type=str, default="784,500,500,2000,10",
                    help='number of hidden units for the layers of the encoder.'
                         'The decoder layers are created in the reverse order. First dimension '
                         'must be 784 (28x28) to match mnist image dimension.')
parser.add_argument('--gpu', action='store_true',
                    help='whether to start training on GPU.')

# set to INFO to see less information during training
logging.basicConfig(level=logging.INFO)
opt = parser.parse_args()
logging.info(opt)
print_every = opt.print_every
batch_size = opt.batch_size
pretrain_num_iter = opt.pretrain_num_iter
finetune_num_iter = opt.finetune_num_iter
visualize = opt.visualize
gpu = opt.gpu
layers = [int(i) for i in opt.num_units.split(',')]


if __name__ == '__main__':
    xpu = mx.gpu() if gpu else mx.cpu()
    print("Training on {}".format("GPU" if gpu else "CPU"))

    ae_model = AutoEncoderModel(xpu, layers, pt_dropout=0.2, internal_act='relu',
                                output_act='relu')

    X, _ = data.get_mnist()
    train_X = X[:60000]
    val_X = X[60000:]

    ae_model.layerwise_pretrain(train_X, batch_size, pretrain_num_iter, 'sgd', l_rate=0.1,
                                decay=0.0, lr_scheduler=mx.lr_scheduler.FactorScheduler(20000, 0.1),
                                print_every=print_every)
    ae_model.finetune(train_X, batch_size, finetune_num_iter, 'sgd', l_rate=0.1, decay=0.0,
                      lr_scheduler=mx.lr_scheduler.FactorScheduler(20000, 0.1), print_every=print_every)
    ae_model.save('mnist_pt.arg')
    ae_model.load('mnist_pt.arg')
    print("Training error:", ae_model.eval(train_X))
    print("Validation error:", ae_model.eval(val_X))
    if visualize:
        try:
            from matplotlib import pyplot as plt
            from model import extract_feature
            # sample a random image
            original_image = X[np.random.choice(X.shape[0]), :].reshape(1, 784)
            data_iter = mx.io.NDArrayIter({'data': original_image}, batch_size=1, shuffle=False,
                                          last_batch_handle='pad')
            # reconstruct the image
            reconstructed_image = extract_feature(ae_model.decoder, ae_model.args,
                                                  ae_model.auxs, data_iter, 1,
                                                  ae_model.xpu).values()[0]
            print("original image")
            plt.imshow(original_image.reshape((28, 28)))
            plt.show()
            print("reconstructed image")
            plt.imshow(reconstructed_image.reshape((28, 28)))
            plt.show()
        except ImportError:
            logging.info("matplotlib is required for visualization")
