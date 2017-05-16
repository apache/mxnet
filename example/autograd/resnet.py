from __future__ import division

import time
import mxnet as mx
from mxnet.contrib import nn
from mxnet.contrib import autograd as ag
from data import *

def conv3x3(filters, stride, in_filters):
    return nn.Conv2D(filters, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_filters=in_filters)

class BasicBlock(nn.Layer):
    def __init__(self, filters, stride, downsample=False, in_filters=0, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        with self.scope:
            self.bn1 = nn.BatchNorm(num_features=in_filters)
            self.conv1 = conv3x3(filters, stride, in_filters)
            self.bn2 = nn.BatchNorm(num_features=filters)
            self.conv2 = conv3x3(filters, 1, filters)
            if downsample:
                self.downsample = nn.Conv2D(filters, 1, stride, use_bias=False,
                                            in_filters=in_filters)
            else:
                self.downsample = None

    def generic_forward(self, domain, x):
        if not self.downsample:
            residual = x
        x = self.bn1(x)
        x = domain.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = domain.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class Bottleneck(nn.Layer):
    def __init__(self, filters, stride, downsample=False, in_filters=0, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        with self.scope:
            self.bn1 = nn.BatchNorm(num_features=in_filters)
            self.conv1 = conv3x3(filters//4, 1, in_filters)
            self.bn2 = nn.BatchNorm(num_features=filters//4)
            self.conv2 = conv3x3(filters//4, stride, filters//4)
            self.bn3 = nn.BatchNorm(num_features=filters//4)
            self.conv3 = conv3x3(filters, 1, filters//4)
            if downsample:
                self.downsample = nn.Conv2D(filters, 1, stride, use_bias=False,
                                            in_filters=in_filters)
            else:
                self.downsample = None

    def generic_forward(self, domain, x):
        if not self.downsample:
            residual = x
        x = self.bn1(x)
        x = domain.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = domain.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = domain.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual

class Resnet(nn.Layer):
    def __init__(self, block, classes, layers, filters, thumbnail=False, **kwargs):
        super(Resnet, self).__init__(**kwargs)
        with self.scope:
            assert len(layers) == len(filters) - 1
            self._thumbnail = thumbnail
            self.bn_data = nn.BatchNorm(num_features=3, scale=False, center=False)
            if thumbnail:
                self.conv0 = conv3x3(filters[0], 1, 3)
            else:
                self.conv0 = nn.Conv2D(filters[0], 7, 2, 3, use_bias=False,
                                       in_filters=3)
                self.bn0 = nn.BatchNorm(num_features=filters[0])
                self.pool0 = nn.MaxPool2D(3, 2, 1)

            self.body = nn.Sequential()
            in_filters = filters[0]
            for i in range(len(layers)):
                stride = 1 if i == 0 else 2
                self.body.add(self._make_layer(block, layers[i], filters[i+1],
                                               stride, in_filters=in_filters))
                in_filters = filters[i+1]

            self.bn1 = nn.BatchNorm(num_features=in_filters)
            self.pool1 = nn.GlobalAvgPool2D()
            self.dense1 = nn.Dense(classes, in_units=in_filters)

    def _make_layer(self, block, layers, filters, stride, in_filters=0):
        layer = nn.Sequential()
        layer.add(block(filters, stride, True, in_filters=in_filters))
        for i in range(layers-1):
            layer.add(block(filters, 1, False, in_filters=filters))
        return layer

    def generic_forward(self, domain, x):
        x = self.bn_data(x)
        x = self.conv0(x)
        if not self._thumbnail:
            x = self.bn0(x)
            x = domain.Activation(x, act_type='relu')
            x = self.pool0(x)

        x = self.body(x)

        x = self.bn1(x)
        x = domain.Activation(x, act_type='relu')
        x = self.pool1(x)
        x = x.reshape((0, -1))
        x = self.dense1(x)

        return x


def resnet18_cifar(classes):
    return Resnet(BasicBlock, classes, [2, 2, 2], [16, 16, 32, 64], True)

def resnet50_imagenet(classes):
    return Resnet(Bottleneck, classes, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], False)

net = resnet18_cifar(10)
batch_size = 32
train_data, val_data = cifar10_iterator(batch_size, (3, 32, 32))


def test(ctx):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = nn.utils.load_data(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = nn.utils.load_data(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    print 'validation acc: %s=%f'%metric.get()


def train(epoch, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.params.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    optim = nn.Optim(net.params, 'sgd', {'learning_rate': 0.1})
    metric = mx.metric.Accuracy()

    for i in range(epoch):
        tic = time.time()
        train_data.reset()
        for batch in train_data:
            data = nn.utils.load_data(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = nn.utils.load_data(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            with ag.train_section():
                for x, y in zip(data, label):
                    z = net(x)
                    loss = nn.loss.softmax_cross_entropy_loss(z, y)
                    ag.compute_gradient([loss])
                    outputs.append(z)
            metric.update(label, outputs)
            optim.step(batch.data[0].shape[0])
        name, acc = metric.get()
        metric.reset()
        print 'training acc at epoch %d: %s=%f'%(i, name, acc)
        print 'time: %f'%(time.time()-tic)
        test(ctx)

    net.params.save('mnist.params')

if __name__ == '__main__':
    train(200, [mx.gpu(0)])
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # data = mx.sym.var('data')
    # out = net(data)
    # softmax = mx.sym.SoftmaxOutput(out, name='softmax')
    # mod = mx.mod.Module(softmax, context=[mx.gpu(0)])
    # mod.fit(train_data, num_epoch=100, batch_end_callback = mx.callback.Speedometer(batch_size, 10))
