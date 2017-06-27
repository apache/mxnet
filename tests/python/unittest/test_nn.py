import mxnet as mx
from mxnet import foo
from mxnet.foo import nn
import numpy as np


def test_parameter():
    p = foo.Parameter('weight', shape=(10, 10))
    p.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    assert len(p.list_data()) == 2
    assert len(p.list_grad()) == 2
    assert p.data(mx.cpu(1)).context == mx.cpu(1)
    assert p.data(mx.cpu(0)).shape == (10, 10)
    assert p.var().name == 'weight'


def test_paramdict():
    params = foo.ParameterDict('net_')
    params.get('weight', shape=(10, 10))
    assert list(params.keys()) == ['net_weight']
    params.initialize(ctx=mx.cpu())
    params.save('test.params')
    params.load('test.params', mx.cpu())


def test_parameter_sharing():
    class Net(nn.Layer):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.dense0 = nn.Dense(5, in_units=5)
                self.dense1 = nn.Dense(5, in_units=5)

        def forward(self, x):
            return self.dense1(self.dense0(x))

    net1 = Net(prefix='net1_')
    net2 = Net(prefix='net2_', params=net1.all_params())
    net1.all_params().initialize()
    net2(mx.nd.zeros((3, 5)))


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
    model.all_params().initialize()
    x = model(mx.nd.zeros((32, 10)))
    assert x.shape == (32, 32)
    x.wait_to_read()


def check_layer_forward(layer, dshape):
    layer.all_params().initialize()
    with mx.autograd.record():
        out = layer(mx.nd.ones(shape=dshape))
    out.backward()

    layer.hybridize()

    with mx.autograd.record():
        out = layer(mx.nd.ones(shape=dshape))
    out.backward()

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


def test_deconv():
    # layers1d = [
    #     nn.Conv1DTranspose(16, 3, in_filters=4),
    #     nn.Conv1DTranspose(16, 3, groups=2, in_filters=4),
    #     nn.Conv1DTranspose(16, 3, strides=3, groups=2, in_filters=4),
    #     ]
    # for layer in layers1d:
    #     check_layer_forward(layer, (1, 4, 10))


    layers2d = [
        nn.Conv2DTranspose(16, (3, 4), in_filters=4),
        nn.Conv2DTranspose(16, (5, 4), in_filters=4),
        nn.Conv2DTranspose(16, (3, 4), groups=2, in_filters=4),
        nn.Conv2DTranspose(16, (3, 4), strides=4, in_filters=4),
        nn.Conv2DTranspose(16, (3, 4), dilation=4, in_filters=4),
        nn.Conv2DTranspose(16, (3, 4), padding=4, in_filters=4),
        nn.Conv2DTranspose(16, (3, 4), strides=4, output_padding=3, in_filters=4),
        ]
    for layer in layers2d:
        check_layer_forward(layer, (1, 4, 20, 20))


    # layers3d = [
    #     nn.Conv3DTranspose(16, (1, 8, 4), in_filters=4),
    #     nn.Conv3DTranspose(16, (5, 4, 3), in_filters=4),
    #     nn.Conv3DTranspose(16, (3, 3, 3), groups=2, in_filters=4),
    #     nn.Conv3DTranspose(16, 4, strides=4, in_filters=4),
    #     nn.Conv3DTranspose(16, (3, 3, 3), padding=4, in_filters=4),
    #     ]
    # for layer in layers3d:
    #     check_layer_forward(layer, (1, 4, 10, 10, 10))
    #
    #
    # layer = nn.Conv2DTranspose(16, (3, 3), layout='NHWC', in_filters=4)
    # # check_layer_forward(layer, (1, 10, 10, 4))
    #
    # layer = nn.Conv3DTranspose(16, (3, 3, 3), layout='NDHWC', in_filters=4)
    # # check_layer_forward(layer, (1, 10, 10, 10, 4))



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
    layer.all_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x.reshape((-1,))
        x = x + 10
    x.backward()


def test_slice():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_filters=4)
    layer.all_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x[1:3]
        x = x + 10
    x.backward()


def test_at():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2, in_filters=4)
    layer.all_params().initialize()
    with mx.autograd.record():
        x = layer(x)
        x = x[1]
        x = x + 10
    x.backward()


def test_defered_init():
    x = mx.nd.ones((5, 4, 10, 10))
    layer = nn.Conv2D(10, 2)
    layer.all_params().initialize()
    layer(x)


if __name__ == '__main__':
    import nose
    nose.runmodule()
