import mxnet as mx
from mxnet.contrib import nn
import numpy as np


def test_parameter():
    p = nn.Parameter('weight', shape=(10, 10))
    p.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assert len(p.list_data()) == 2
    assert len(p.list_grad()) == 2
    assert p.data(mx.cpu(1)).context == mx.cpu(1)
    assert p.data(mx.cpu(0)).shape == (10, 10)
    assert p.var().name == 'weight'


def test_paramdict():
    params = nn.ParameterDict('net_')
    params.get('weight', shape=(10, 10))
    assert list(params.keys()) == ['net_weight']
    params.initialize()
    params.save('test.params')
    params.load('test.params')

def test_basic():
    model = nn.Sequential()
    model.add(nn.Dense(128, activation='tanh', in_units=10))
    model.add(nn.Dropout(0.5))
    model.add(nn.Dense(64, activation='tanh', in_units=128))
    model.add(nn.Dense(32, in_units=64))
    model.add(nn.Activation('relu'))

    # symbol
    x = mx.sym.var('data')
    y = model(x)
    assert len(y.list_arguments()) == 7

    # ndarray
    model.params.initialize()
    x = model(mx.nd.zeros((32, 10)))
    assert x.shape == (32, 32)
    x.wait_to_read()


def check_layer_forward(layer, dshape):
    layer.params.initialize()
    with mx.contrib.autograd.train_section():
        out = layer(mx.nd.ones(shape=dshape))
    mx.contrib.autograd.compute_gradient([out])

def test_conv():
    layers1d = [
        nn.Conv1D(16, 3, in_filters=4),
        nn.Conv1D(16, 3, groups=2, in_filters=4),
        nn.Conv1D(16, 3, strides=3, groups=2, in_filters=4),
        ]
    for layer in layers1d:
        check_layer_forward(layer, (1, 4, 10))


    layers2d = [
        nn.Conv2D(16, (3, 4), in_filters=4),
        nn.Conv2D(16, (5, 4), in_filters=4),
        nn.Conv2D(16, (3, 4), groups=2, in_filters=4),
        nn.Conv2D(16, (3, 4), strides=4, in_filters=4),
        nn.Conv2D(16, (3, 4), dilation=4, in_filters=4),
        nn.Conv2D(16, (3, 4), padding=4, in_filters=4),
        ]
    for layer in layers2d:
        check_layer_forward(layer, (1, 4, 20, 20))


    layers3d = [
        nn.Conv3D(16, (1, 8, 4), in_filters=4),
        nn.Conv3D(16, (5, 4, 3), in_filters=4),
        nn.Conv3D(16, (3, 3, 3), groups=2, in_filters=4),
        nn.Conv3D(16, 4, strides=4, in_filters=4),
        nn.Conv3D(16, (3, 3, 3), padding=4, in_filters=4),
        ]
    for layer in layers3d:
        check_layer_forward(layer, (1, 4, 10, 10, 10))


    layer = nn.Conv2D(16, (3, 3), layout='NHWC', in_filters=4)
    # check_layer_forward(layer, (1, 10, 10, 4))

    layer = nn.Conv3D(16, (3, 3, 3), layout='NDHWC', in_filters=4)
    # check_layer_forward(layer, (1, 10, 10, 10, 4))


def test_pool():
    layers1d = [
        nn.MaxPool1D(),
        nn.MaxPool1D(3),
        nn.MaxPool1D(3, 2),
        nn.AvgPool1D(),
        nn.GlobalAvgPool1D(),
        ]
    for layer in layers1d:
        check_layer_forward(layer, (1, 2, 10))


    layers2d = [
        nn.MaxPool2D(),
        nn.MaxPool2D((3, 3)),
        nn.MaxPool2D(3, 2),
        nn.AvgPool2D(),
        nn.GlobalAvgPool2D(),
        ]
    for layer in layers2d:
        check_layer_forward(layer, (1, 2, 10, 10))

    layers3d = [
        nn.MaxPool3D(),
        nn.MaxPool3D((3, 3, 3)),
        nn.MaxPool3D(3, 2),
        nn.AvgPool3D(),
        nn.GlobalAvgPool3D(),
        ]
    for layer in layers3d:
        check_layer_forward(layer, (1, 2, 10, 10, 10))

def test_batchnorm():
    layer = nn.BatchNorm(num_features=10)
    check_layer_forward(layer, (2, 10, 10, 10))


def test_reshape():
    x = mx.nd.ones((2, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_filters=4)
    layer.params.initialize()
    with mx.contrib.autograd.train_section():
        x = layer(x)
        x = x.reshape((-1,))
        x = x + 10
    mx.contrib.autograd.compute_gradient([x])


def test_slice():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_filters=4)
    layer.params.initialize()
    with mx.contrib.autograd.train_section():
        x = layer(x)
        x = x[1:3]
        x = x + 10
    mx.contrib.autograd.compute_gradient([x])


def test_at():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_filters=4)
    layer.params.initialize()
    with mx.contrib.autograd.train_section():
        x = layer(x)
        x = x[1]
        x = x + 10
    mx.contrib.autograd.compute_gradient([x])


if __name__ == '__main__':
    import nose
    nose.runmodule()
