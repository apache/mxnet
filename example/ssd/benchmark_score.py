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
import os
import sys
import argparse
import importlib
import mxnet as mx
import time
import logging

from symbol.symbol_factory import get_symbol
from symbol.symbol_factory import get_symbol_train
from symbol import symbol_builder


parser = argparse.ArgumentParser(description='MXNet SSD benchmark')
parser.add_argument('--network', '-n', type=str, default='vgg16_reduced')
parser.add_argument('--batch_size', '-b', type=int, default=0)
parser.add_argument('--shape', '-w', type=int, default=300)
parser.add_argument('--class_num', '-class', type=int, default=20)
parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                    default=os.path.join(os.getcwd(), 'model', 'ssd_'), type=str)
parser.add_argument('--deploy', dest='deploy', help='Load network from model',
                    action='store_true', default=False)


def get_data_shapes(batch_size):
    image_shape = (3, 300, 300)
    return [('data', (batch_size,)+image_shape)]

def get_label_shapes(batch_size):
    return [('label', (batch_size,) + (42, 6))]

def get_data(batch_size):
    data_shapes = get_data_shapes(batch_size)
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=mx.cpu()) for _, shape in data_shapes]
    batch = mx.io.DataBatch(data, [])
    return batch


if __name__ == '__main__':
    args = parser.parse_args()
    network = args.network
    image_shape = args.shape
    num_classes = args.class_num
    b = args.batch_size
    prefix = args.prefix
    supported_image_shapes = [300, 512]
    supported_networks = ['vgg16_reduced', 'inceptionv3', 'resnet50']

    if network not in supported_networks:
        raise Exception(network + " is not supported")

    if image_shape not in supported_image_shapes:
       raise Exception("Image shape should be either 300*300 or 512*512!")

    if b == 0:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    else:
        batch_sizes = [b]

    data_shape = (3, image_shape, image_shape)

    if args.deploy == True:
        prefix += network + '_' + str(data_shape[1]) + '-symbol.json'
        net = mx.sym.load(prefix)
    else:
        net = get_symbol(network, data_shape[1], num_classes=num_classes,
                         nms_thresh=0.4, force_suppress=True)
    if not 'label' in net.list_arguments():
        label = mx.sym.Variable(name='label')
        net = mx.sym.Group([net, label])
    
    num_batches = 100
    dry_run = 5   # use 5 iterations to warm up
    
    for bs in batch_sizes:
        batch = get_data(bs)
        mod = mx.mod.Module(net, label_names=('label',), context=mx.cpu())
        mod.bind(for_training = False,
                 inputs_need_grad = False,
                 data_shapes = get_data_shapes(bs),
                 label_shapes = get_label_shapes(bs))
        mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

        # get data
        data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=mx.cpu()) for _, shape in mod.data_shapes]
        batch = mx.io.DataBatch(data, [])

        for i in range(dry_run + num_batches):
            if i == dry_run:
                tic = time.time()
            mod.forward(batch, is_train=False)
            for output in mod.get_outputs():
                output.wait_to_read()

        avg_time = (time.time() - tic) / num_batches
        fps = bs / avg_time
        print("SSD-" + network + " with " + str(num_classes) + " classes and shape " + str(data_shape))
        print("batchsize=" + str(bs) + " " + str(1000*avg_time) + " ms")
        print("batchsize=" + str(bs) + " " + str(fps) + " imgs/s")
