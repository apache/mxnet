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
parser.add_argument('--train-data', type=str, default='',
                    help='training record file to use, required for imagenet.')
parser.add_argument('--val-data', type=str, default='',
                    help='validation record file to use, required for imagenet.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate. default is 0.01.')
parser.add_argument('-momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
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
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help='the ratio to reduce lr on each step')
parser.add_argument('--lr-step-epochs', type=str, default='30,60',
                   help='the epochs to reduce the lr, e.g. 30,60')
parser.add_argument('--num-examples', type=int, default=1281167,
                    help='the number of training examples')
parser.add_argument('--load-epoch', type=int,
                    help = 'load the model on an epoch using the model-load-prefix')
parser.add_argument('--model-prefix', type=str, default='model/',
                    help='model prefix')
parser.add_argument('--optimizer', type=str, default='sgd',
                   help='the optimizer type')
opt = parser.parse_args()

logging.info(opt)

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
elif dataset == 'imagenet':
    if model_name == 'inceptionv3':
        train_data, val_data = imagenet_iterator(opt.train_data, opt.val_data,
                                              batch_size, (3, 299, 299))
    else:
        train_data, val_data = imagenet_iterator(opt.train_data, opt.val_data,
                                                 batch_size, (3, 224, 224))
elif dataset == 'dummy':
    if model_name == 'inceptionv3':
        train_data, val_data = dummy_iterator(batch_size, (3, 299, 299))
    else:
        train_data, val_data = dummy_iterator(batch_size, (3, 224, 224))

kv = mx.kvstore.create(opt.kvstore)
def _get_lr_scheduler():
    if 'lr_factor' not in opt or opt.lr_factor >= 1:
        return (opt.lr, None)
    epoch_size = int(opt.num_examples / opt.batch_size / opt.gpus)
    if 'dist' in opt.kvstore:
        epoch_size /= kv.num_workers
    begin_epoch = opt.load_epoch if opt.load_epoch else 0
    step_epochs = [int(l) for l in opt.lr_step_epochs.split(',')]
    lr = opt.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= opt.lr_factor
    if lr != opt.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))
    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=opt.lr_factor))

def _load_model(rank=0):
    if 'load_epoch' not in opt or opt.load_epoch is None:
        return (None, None, None)
    assert opt.model_prefix is not None
    model_prefix = opt.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, opt.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, opt.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(rank=0):
    if opt.model_prefix is None:
        return None
    dst_dir = os.path.dirname(opt.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(opt.model_prefix if rank == 0 else "%s-%d" % (
        opt.model_prefix, rank))

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
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum},
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
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
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

        # load model
        if 'arg_params' in kwargs and 'aux_params' in kwargs:
            arg_params = kwargs['arg_params']
            aux_params = kwargs['aux_params']
        else:
            sym, arg_params, aux_params = _load_model(kv.rank)
            if sym is not None:
                assert sym.tojson() == softmax.tojson()

        # save model
        checkpoint = _save_model(kv.rank)

        # learning rate
        lr, lr_scheduler = _get_lr_scheduler()
        optimizer_params = {
            'learning_rate': lr,
            'wd': opt.wd,
            'lr_scheduler': lr_scheduler}
        # Add 'multi_precision' parameter only for SGD optimizer
        if opt.optimizer == 'sgd':
            optimizer_params['multi_precision'] = True

        # Only a limited number of optimizers have 'momentum' property
        has_momentum = {'sgd', 'dcasgd', 'nag'}
        if opt.optimizer in has_momentum:
            optimizer_params['momentum'] = opt.momentum
        mod.fit(train_data,
                begin_epoch=opt.load_epoch if opt.load_epoch else 0,
                eval_data = val_data,
                num_epoch=opt.epochs,
                kvstore=opt.kvstore,
                batch_end_callback =[mx.callback.Speedometer(batch_size, max(1, opt.log_interval))],
                epoch_end_callback = checkpoint,
                optimizer = 'sgd',
                arg_params = arg_params,
                aux_params = aux_params,
                optimizer_params = optimizer_params,
                initializer = mx.init.Xavier(magnitude=2))
        mod.save_params('image-classifier-%s-%d-final.params'%(opt.model, opt.epochs))
    else:
        if opt.mode == 'hybrid':
            net.hybridize()
        train(opt.epochs, context)
