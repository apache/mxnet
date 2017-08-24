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

from __future__ import division

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet import autograd as ag

from data import *

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset to use. options are mnist, cifar10, and dummy.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate. default is 0.01.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123.')
parser.add_argument('--benchmark', action='store_true',
                    help='whether to run benchmark.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use_thumbnail', action='store_true',
                    help='use thumbnail or not in resnet. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--log-interval', type=int, default=50, help='Number of batches to wait before logging.')
opt = parser.parse_args()

print(opt)

mx.random.seed(opt.seed)

dataset_classes = {'mnist': 10, 'cifar10': 10, 'imagenet': 1000, 'dummy': 1000}

batch_size, dataset, classes = opt.batch_size, opt.dataset, dataset_classes[opt.dataset]

gpus = opt.gpus

if opt.benchmark:
    batch_size = 32
    dataset = 'dummy'
    classes = 1000

batch_size *= max(1, gpus)
context = [mx.gpu(i) for i in range(gpus)] if gpus > 0 else [mx.cpu()]

model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('resnet'):
    kwargs['thumbnail'] = opt.use_thumbnail
elif model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm

net = models.get_model(opt.model, **kwargs)

# get dataset iterators
if dataset == 'mnist':
    train_data, val_data = mnist_iterator(batch_size, (1, 32, 32))
elif dataset == 'cifar10':
    train_data, val_data = cifar10_iterator(batch_size, (3, 32, 32))
elif dataset == 'dummy':
    if model_name == 'inceptionv3':
        train_data, val_data = dummy_iterator(batch_size, (3, 299, 299))
    else:
        train_data, val_data = dummy_iterator(batch_size, (3, 224, 224))

def test(ctx):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': opt.lr, 'wd': opt.wd},
                            kvstore = opt.kvstore)
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if opt.log_interval and not (i+1)%opt.log_interval:
                name, acc = metric.get()
                logging.info('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f'%(
                               epoch, i, batch_size/(time.time()-btic), name, acc))
            btic = time.time()

        name, acc = metric.get()
        logging.info('[Epoch %d] training: %s=%f'%(epoch, name, acc))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        name, val_acc = test(ctx)
        logging.info('[Epoch %d] validation: %s=%f'%(epoch, name, val_acc))

    net.save_params('image-classifier-%s-%d.params'%(opt.model, epochs))

if __name__ == '__main__':
    if opt.mode == 'symbolic':
        data = mx.sym.var('data')
        out = net(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=[mx.gpu(i) for i in range(gpus)] if gpus > 0 else [mx.cpu()])
        mod.fit(train_data, num_epoch=opt.epochs, kvstore=opt.kvstore,
                batch_end_callback = mx.callback.Speedometer(batch_size, 1))
    else:
        if opt.mode == 'hybrid':
            net.hybridize()
        train(opt.epochs, context)
