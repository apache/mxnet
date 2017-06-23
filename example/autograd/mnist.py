# pylint: skip-file
from __future__ import print_function

from data import mnist_iterator
import mxnet as mx
from mxnet import foo
from mxnet.foo import nn
import numpy as np
import logging
from mxnet import autograd as ag
logging.basicConfig(level=logging.DEBUG)

# define network

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10))

# data

train_data, val_data = mnist_iterator(batch_size=100, input_shape = (784,))

# train

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
    print('validation acc: %s=%f'%metric.get())

def train(epoch, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.all_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = foo.Trainer(net.all_params(), 'sgd', {'learning_rate': 0.1})
    metric = mx.metric.Accuracy()

    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            data = foo.utils.load_data(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = foo.utils.load_data(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    loss = foo.loss.softmax_cross_entropy_loss(z, y)
                    ag.compute_gradient([loss])
                    outputs.append(z)
            metric.update(label, outputs)
            trainer.step(batch.data[0].shape[0])
        name, acc = metric.get()
        metric.reset()
        print('training acc at epoch %d: %s=%f'%(i, name, acc))
        test(ctx)

    net.all_params().save('mnist.params')


if __name__ == '__main__':
    train(10, [mx.cpu(0), mx.cpu(1)])
