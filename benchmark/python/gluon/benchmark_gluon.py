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

import mxnet as mx
import mxnet.gluon.model_zoo.vision as models
import time
import logging
import argparse
import subprocess
import os
import errno

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Gluon modelzoo-based CNN performance benchmark')

parser.add_argument('--model', type=str, default='all',
                               choices=['all', 'alexnet', 'densenet121', 'densenet161',
                                        'densenet169', 'densenet201', 'inceptionv3', 'mobilenet0.25',
                                        'mobilenet0.5', 'mobilenet0.75', 'mobilenet1.0', 'mobilenetv2_0.25',
                                        'mobilenetv2_0.5', 'mobilenetv2_0.75', 'mobilenetv2_1.0', 'resnet101_v1',
                                        'resnet101_v2', 'resnet152_v1', 'resnet152_v2', 'resnet18_v1',
                                        'resnet18_v2', 'resnet34_v1', 'resnet34_v2', 'resnet50_v1',
                                        'resnet50_v2', 'squeezenet1.0', 'squeezenet1.1', 'vgg11',
                                        'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                                        'vgg19', 'vgg19_bn'])
parser.add_argument('--batch-size', type=int, default=0,
                     help='Batch size to use for benchmarking. Example: 32, 64, 128.'
                          'By default, runs benchmark for batch sizes - 1, 32, 64, 128, 256')
parser.add_argument('--num-batches', type=int, default=10)
parser.add_argument('--gpus', type=str, default='',
                    help='GPU IDs to use for this benchmark task. Example: --gpus=0,1,2,3 to use 4 GPUs.'
                         'By default, use CPU only.')
parser.add_argument('--type', type=str, default='inference', choices=['all', 'training', 'inference'])

opt = parser.parse_args()

num_batches = opt.num_batches
dry_run = 10  # use 10 iterations to warm up
batch_inf = [1, 32, 64, 128, 256]
batch_train = [1, 32, 64, 128, 256]
image_shapes = [(3, 224, 224), (3, 299, 299)]

def score(network, batch_size, ctx):
    assert (batch_size >= len(ctx)), "ERROR: batch size should not be smaller than num of GPUs."
    net = models.get_model(network)
    if 'inceptionv3' == network:
        data_shape = [('data', (batch_size,) + image_shapes[1])]
    else:
        data_shape = [('data', (batch_size,) + image_shapes[0])]

    data = mx.sym.var('data')
    out = net(data)
    softmax = mx.sym.SoftmaxOutput(out, name='softmax')
    mod = mx.mod.Module(softmax, context=ctx)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx[0]) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, [])
    for i in range(dry_run + num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
    fwd = time.time() - tic
    return fwd


def train(network, batch_size, ctx):
    assert (batch_size >= len(ctx)), "ERROR: batch size should not be smaller than num of GPUs."
    net = models.get_model(network)
    if 'inceptionv3' == network:
        data_shape = [('data', (batch_size,) + image_shapes[1])]
    else:
        data_shape = [('data', (batch_size,) + image_shapes[0])]

    data = mx.sym.var('data')
    out = net(data)
    softmax = mx.sym.SoftmaxOutput(out, name='softmax')
    mod = mx.mod.Module(softmax, context=ctx)
    mod.bind(for_training     = True,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    if len(ctx) > 1:
        mod.init_optimizer(kvstore='device', optimizer='sgd')
    else:
        mod.init_optimizer(kvstore='local', optimizer='sgd')
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx[0]) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, [])
    for i in range(dry_run + num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=True)
        for output in mod.get_outputs():
            output.wait_to_read()
        mod.backward()
        mod.update()
    bwd = time.time() - tic
    return bwd

if __name__ == '__main__':
    runtype = opt.type
    bs = opt.batch_size

    if opt.model == 'all':
        networks = ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
                    'inceptionv3', 'mobilenet0.25', 'mobilenet0.5', 'mobilenet0.75',
                    'mobilenet1.0', 'mobilenetv2_0.25', 'mobilenetv2_0.5', 'mobilenetv2_0.75',
                    'mobilenetv2_1.0', 'resnet101_v1', 'resnet101_v2', 'resnet152_v1', 'resnet152_v2',
                    'resnet18_v1', 'resnet18_v2', 'resnet34_v1', 'resnet34_v2', 'resnet50_v1',
                    'resnet50_v2', 'squeezenet1.0', 'squeezenet1.1', 'vgg11', 'vgg11_bn', 'vgg13',
                    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        logging.info('It may take some time to run all models, '
                     'set --network to run a specific one')
    else:
        networks = [opt.model]
    
    devs = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
    num_gpus = len(devs)

    for network in networks:
        logging.info('network: %s', network)
        logging.info('device: %s', devs)
        if runtype == 'inference' or runtype == 'all':
            if bs != 0:
                fwd_time = score(network, bs, devs)
                fps = (bs * num_batches)/fwd_time
                logging.info(network + ' inference perf for BS %d is %f img/s', bs, fps)
            else:
                logging.info('run batchsize [1, 2, 4, 8, 16, 32] by default, '
                             'set --batch-size to run a specific one')
                for batch_size in batch_inf:
                    fwd_time = score(network, batch_size, devs)
                    fps = (batch_size * num_batches) / fwd_time
                    logging.info(network + ' inference perf for BS %d is %f img/s', batch_size, fps)
        if runtype == 'training' or runtype == 'all':
            if bs != 0:
                bwd_time = train(network, bs, devs)
                fps = (bs * num_batches) / bwd_time
                logging.info(network + ' training perf for BS %d is %f img/s', bs, fps)
            else:
                logging.info('run batchsize [1, 2, 4, 8, 16, 32] by default, '
                             'set --batch-size to run a specific one')
                for batch_size in batch_train:
                    bwd_time = train(network, batch_size, devs)
                    fps = (batch_size * num_batches) / bwd_time
                    logging.info(network + ' training perf for BS %d is %f img/s', batch_size, fps)
