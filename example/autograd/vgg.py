from __future__ import division

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import foo
from mxnet.foo import nn
from mxnet import autograd as ag

from data import *

# CLI
parser = argparse.ArgumentParser(description='Train a VGG model for image classification.')
parser.add_argument('--dataset', type=str, default='dummy', help='dataset to use. options are mnist, cifar10, and dummy.')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size per device (CPU/GPU).')
parser.add_argument('--batch_norm', action='store_true', default=False, help='whether to enable batch normalization. default false.')
parser.add_argument('--layers', type=int, default=13, help='layers of VGG net to use. options are 11, 13, 16, 19. default is 13.')
parser.add_argument('--gpus', type=int, default=0, help='number of gpus to use.')
parser.add_argument('--epochs', type=int, default=3, help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01, help='learning Rate. default is 0.01.')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123.')
parser.add_argument('--benchmark', action='store_true', default=True, help='whether to run benchmark.')
parser.add_argument('--symbolic', action='store_true', default=False, help='whether to train in symbolic way with module.')
opt = parser.parse_args()

print(opt)

class VGG(nn.HybridLayer):

    def __init__(self, layers, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.features = self._make_features(layers, batch_norm)
        self.classifier = nn.HSequential()
        self.classifier.add(nn.Dense(4096, activation='relu',
                                     kernel_initializer=mx.initializer.Normal(),
                                     bias_initializer='zeros'))
        self.classifier.add(nn.Dropout(rate=0.5))
        self.classifier.add(nn.Dense(4096, activation='relu',
                                     kernel_initializer=mx.initializer.Normal(),
                                     bias_initializer='zeros'))
        self.classifier.add(nn.Dropout(rate=0.5))
        self.classifier.add(nn.Dense(num_classes,
                                     kernel_initializer=mx.initializer.Normal(),
                                     bias_initializer='zeros'))

    def _make_features(self, layers, batch_norm):
        featurizer = nn.HSequential()
        filters = [64, 128, 256, 512, 512]
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         kernel_initializer=mx.initializer.Xavier(rnd_type='gaussian',
                                                                                  factor_type='out',
                                                                                  magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# construct net
vgg_spec = {
    11: [1, 1, 2, 2, 2],
    13: [2, 2, 2, 2, 2],
    16: [2, 2, 3, 3, 3],
    19: [2, 2, 4, 4, 4],
}

def get_vgg(num_layers, batch_norm, num_classes):
    layers = vgg_spec[num_layers]
    return VGG(layers, batch_norm, num_classes)

dataset_classes = {'mnist': 10, 'cifar10': 10, 'imagenet': 1000, 'dummy': 1000}

batch_size, dataset, classes = opt.batch_size, opt.dataset, dataset_classes[opt.dataset]

gpus, batch_norm = opt.gpus, opt.batch_norm

if opt.benchmark:
    batch_size = 32
    dataset = 'dummy'
    classes = 1000
    batch_norm = False

net = get_vgg(opt.layers, batch_norm, classes)

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
    trainer = foo.Trainer(net.all_params(), 'sgd', {'learning_rate': 0.1})
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

    net.all_params().save('mnist.params')

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
