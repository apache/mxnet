import pytest
import mxnet as mx
import subprocess
from os.path import abspath

from mxnet import gluon
from mxnet.gluon import nn
from numpy.testing import assert_almost_equal


def dummy_data(ctx, batch_size=10):
    return [mx.nd.random.uniform(shape=shape, ctx=ctx) for shape in ([batch_size, 1, 28, 28], [batch_size])]


def binary_lenet(**qconv_kwargs):
    net = nn.HybridSequential(prefix="")
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=5))
        net.add(gluon.nn.Activation(activation='tanh'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))

        net.add(gluon.nn.QActivation())
        net.add(gluon.nn.QConv2D(**qconv_kwargs))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())

        net.add(gluon.nn.QActivation())
        net.add(gluon.nn.QDense(1000))
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(gluon.nn.Activation(activation='tanh'))

        net.add(gluon.nn.Dense(10))
    return net


@pytest.mark.parametrize("ctx", [mx.cpu()])
@pytest.mark.parametrize("qconv_kwargs", [{"channels": 64, "kernel_size": 5},
                                          {"channels": 64, "kernel_size": 3, "padding": 1}])
def test_save(ctx, qconv_kwargs):
    prefix, symbol_file, param_file = "binarized_", "test-symbol.json", "test-0000.params"
    orig_net = binary_lenet(**qconv_kwargs)
    orig_net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    # This dummy pass is needed to initialize binary layers correctly!
    data, _ = dummy_data(ctx)
    expected = orig_net(data)

    # Training loop would be here

    # Export as symbol, so it can be used with C API
    orig_net.hybridize()

    # This dummy pass is needed to make correct symbol export possible, but does not replace the first one
    data, label = dummy_data(ctx)
    _ = orig_net(data)
    orig_net.export("test", epoch=0)

    # Intermediate symbolic model, non-compressed
    net1 = mx.gluon.nn.SymbolBlock.imports(symbol_file, ['data'], param_file=param_file, ctx=ctx)
    out1 = net1(data)

    output = subprocess.check_output(["build/tools/binary_converter/model-converter", param_file])

    # Compressed symbolic model
    net2 = mx.gluon.nn.SymbolBlock.imports(prefix + symbol_file, ['data'], param_file=prefix + param_file, ctx=ctx)
    out2 = net2(data)

    assert_almost_equal(expected.asnumpy(), out1.asnumpy())
    assert_almost_equal(expected.asnumpy(), out2.asnumpy())
