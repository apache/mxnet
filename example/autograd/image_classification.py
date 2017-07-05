from __future__ import division

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import foo
from mxnet.foo import nn
from mxnet import autograd as ag
import vision_model as model

from data import *

# CLI
parser = argparse.ArgumentParser(description='Train a resnet model for image classification.')
parser.add_argument('--dataset', type=str, default='dummy', help='dataset to use. options are mnist, cifar10, and dummy.')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size per device (CPU/GPU).')
parser.add_argument('--gpus', type=int, default=0, help='number of gpus to use.')
parser.add_argument('--epochs', type=int, default=3, help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01, help='learning Rate. default is 0.01.')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123.')
parser.add_argument('--benchmark', action='store_true', default=True, help='whether to run benchmark.')
parser.add_argument('--symbolic', action='store_true', default=False, help='whether to train in symbolic way with module.')
parser.add_argument('--model', type=str, default='resnet50_v1', help='type of model to use. see foo.model for options.')
parser.add_argument('--use_thumbnail', action='store_true', default=False, help='use thumbnail or not in resnet. default is false.')
parser.add_argument('--batch_norm', action='store_true', default=False, help='enable batch normalization or not in vgg. default is false.')
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

model_name = opt.model

if model_name.startswith('resnet'):
    net = model.get_vision_model(opt.model, classes=classes, use_thumbnail=opt.use_thumbnail)
elif model_name.startswith('vgg'):
    net = model.get_vision_model(opt.model, classes=classes, batch_norm=opt.batch_norm)
else:
    net = model.get_vision_model(opt.model, classes=classes)

batch_size *= max(1, gpus)

# get dataset iterators
if dataset == 'mnist':
    train_data, val_data = mnist_iterator(batch_size, (1, 32, 32))
elif dataset == 'cifar10':
    train_data, val_data = cifar10_iterator(batch_size, (3, 32, 32))
elif dataset == 'dummy':
    train_data, val_data = dummy_iterator(batch_size, (3, 224, 224))

def test(ctx):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = foo.utils.load_data(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = foo.utils.load_data(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    logging.info('validation acc: %s=%f'%metric.get())


def train(epoch, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.all_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = foo.Trainer(net.all_params(), 'sgd', {'learning_rate': opt.lr})
    metric = mx.metric.Accuracy()

    for i in range(epoch):
        tic = time.time()
        train_data.reset()
        btic = time.time()
        for batch in train_data:
            data = foo.utils.load_data(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = foo.utils.load_data(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            losses = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    loss = foo.loss.softmax_cross_entropy_loss(z, y)
                    losses.append(loss)
                    outputs.append(z)
                for loss in losses:
                    loss.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            logging.info('speed: {} samples/s'.format(batch_size/(time.time()-btic)))
            btic = time.time()

        name, acc = metric.get()
        metric.reset()
        logging.info('training acc at epoch %d: %s=%f'%(i, name, acc))
        logging.info('time: %f'%(time.time()-tic))
        test(ctx)

    net.all_params().save('image-classifier-%s.params'%opt.model)

if __name__ == '__main__':
    if opt.symbolic:
        data = mx.sym.var('data')
        out = net(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=[mx.gpu(i) for i in range(gpus)] if gpus > 0 else [mx.cpu()])
        mod.fit(train_data, num_epoch=opt.epochs, batch_end_callback = mx.callback.Speedometer(batch_size, 1))
    else:
        net.hybridize()
        train(opt.epochs, [mx.gpu(i) for i in range(gpus)] if gpus > 0 else [mx.cpu()])
