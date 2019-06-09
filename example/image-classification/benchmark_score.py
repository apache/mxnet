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

"""
Benchmark the scoring performance on various CNNs
"""
from common import find_mxnet
from common.util import get_gpus
import mxnet as mx
import mxnet.gluon.model_zoo.vision as models
from importlib import import_module
import logging
import argparse
import time
import numpy as np
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='SymbolAPI-based CNN inference performance benchmark')
parser.add_argument('--network', type=str, default='all', 
                                 choices=['all', 'alexnet', 'vgg-16', 'resnetv1-50', 'resnet-50',
                                          'resnet-152', 'inception-bn', 'inception-v3', 
                                          'inception-v4', 'inception-resnet-v2', 'mobilenet',
                                          'densenet121', 'squeezenet1.1'])
parser.add_argument('--batch-size', type=int, default=0,
                     help='Batch size to use for benchmarking. Example: 32, 64, 128.'
                          'By default, runs benchmark for batch sizes - 1, 32, 64, 128, 256')

opt = parser.parse_args()

def get_symbol(network, batch_size, dtype):
    image_shape = (3,299,299) if network in ['inception-v3', 'inception-v4'] else (3,224,224)
    num_layers = 0
    if network == 'inception-resnet-v2':
        network = network
    elif 'resnet' in network:
        num_layers = int(network.split('-')[1])
        network = network.split('-')[0]
    if 'vgg' in network:
        num_layers = int(network.split('-')[1])
        network = 'vgg'
    if network in ['densenet121', 'squeezenet1.1']:
        sym = models.get_model(network)
        sym.hybridize()
        data = mx.sym.var('data')
        sym = sym(data)
        sym = mx.sym.SoftmaxOutput(sym, name='softmax')
    else:
        net = import_module('symbols.'+network)
        sym = net.get_symbol(num_classes=1000,
                             image_shape=','.join([str(i) for i in image_shape]),
                             num_layers=num_layers,
                             dtype=dtype)
    return (sym, [('data', (batch_size,)+image_shape)])

def score(network, dev, batch_size, num_batches, dtype):
    # get mod
    sym, data_shape = get_symbol(network, batch_size, dtype)
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)

if __name__ == '__main__':
    if opt.network == 'all':
        networks = ['alexnet', 'vgg-16', 'resnetv1-50', 'resnet-50',
                    'resnet-152', 'inception-bn', 'inception-v3', 
                    'inception-v4', 'inception-resnet-v2', 
                    'mobilenet', 'densenet121', 'squeezenet1.1']
        logging.info('It may take some time to run all models, '
                     'set --network to run a specific one')
    else:
        networks = [opt.network]
    devs = [mx.gpu(0)] if len(get_gpus()) > 0 else []
    # Enable USE_MKLDNN for better CPU performance
    devs.append(mx.cpu())

    if opt.batch_size == 0:
        batch_sizes = [1, 32, 64, 128, 256]
        logging.info('run batchsize [1, 32, 64, 128, 256] by default, '
                     'set --batch-size to run a specific one')
    else:
        batch_sizes = [opt.batch_size]

    for net in networks:
        logging.info('network: %s', net)
        if net in ['densenet121', 'squeezenet1.1']:
            logging.info('network: %s is converted from gluon modelzoo', net)
            logging.info('you can run benchmark/python/gluon/benchmark_gluon.py for more models')
        for d in devs:
            logging.info('device: %s', d)
            logged_fp16_warning = False
            for b in batch_sizes:
                for dtype in ['float32', 'float16']:
                    if d == mx.cpu() and dtype == 'float16':
                        #float16 is not supported on CPU
                        continue
                    elif net in ['inception-bn', 'alexnet'] and dtype == 'float16':
                        if not logged_fp16_warning:
                            logging.info('Model definition for {} does not support float16'.format(net))
                            logged_fp16_warning = True
                    else:
                        speed = score(network=net, dev=d, batch_size=b, num_batches=10, dtype=dtype)
                        logging.info('batch size %2d, dtype %s, images/sec: %f', b, dtype, speed)
