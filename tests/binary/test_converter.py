import pytest
import mxnet as mx
import subprocess

from mxnet import gluon, autograd
from mxnet.gluon import nn
from numpy.testing import assert_almost_equal, assert_raises


def dummy_data(ctx, batch_size=10):
    return [mx.nd.random.uniform(shape=(batch_size, 1, 28, 28), ctx=ctx), mx.nd.array([1] + [0, ]*9)]


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


def export_symbol(orig_net, ctx):
    orig_net.hybridize()
    # This dummy pass is needed for internal mxnet purposes to make correct symbol export possible
    data, _ = dummy_data(ctx)
    _ = orig_net(data)
    orig_net.export("test", epoch=0)


@pytest.mark.parametrize("ctx", [mx.cpu()])
@pytest.mark.parametrize("conversion_should_fail,qconv_kwargs", [
    (False, {"channels": 64, "kernel_size": 5}),
    (False, {"channels": 64, "kernel_size": 3, "padding": 1}),
    (True, {"channels": 64, "kernel_size": 5, "no_offset": True}),
])
def test_save(ctx, qconv_kwargs, conversion_should_fail):
    prefix, symbol_file, param_file = "binarized_", "test-symbol.json", "test-0000.params"
    orig_net = binary_lenet(**qconv_kwargs)
    orig_net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    # This dummy pass is needed to initialize binary layers correctly!
    data, _ = dummy_data(ctx)
    _ = orig_net(data)

    # Simulate a few training iterations for BatchNorms to initialize
    kv = mx.kv.create()
    trainer = gluon.Trainer(orig_net.collect_params(), "adam",
                            {'learning_rate': 0.001, 'wd': 0.0, 'multi_precision': True}, kvstore=kv)
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(0, 100):
        with autograd.record():
            x, y = dummy_data(ctx)
            out = orig_net(x)
            L = loss(out, y)
            autograd.backward(L)
        trainer.step(10)

    # check that results are different for different input data
    result_1 = orig_net(dummy_data(ctx)[0])
    result_2 = orig_net(dummy_data(ctx)[0])
    assert_raises(AssertionError, assert_almost_equal, result_1.asnumpy(), result_2.asnumpy())

    # get expected test result
    test_data, _ = dummy_data(ctx)
    expected = orig_net(test_data)

    # Export the model as symbol, so it can be converted with model converter and used with C API
    export_symbol(orig_net, ctx)

    # Intermediate symbolic model, non-compressed
    net1 = mx.gluon.nn.SymbolBlock.imports(symbol_file, ['data'], param_file=param_file, ctx=ctx)
    out1 = net1(test_data)

    # Call model converter to compress symbolic model
    output = subprocess.check_output(["build/tools/binary_converter/model-converter", param_file])

    # Load compressed symbolic model
    net2 = mx.gluon.nn.SymbolBlock.imports(prefix + symbol_file, ['data'], param_file=prefix + param_file, ctx=ctx)
    out2 = net2(test_data)

    assert_almost_equal(expected.asnumpy(), out1.asnumpy())
    if conversion_should_fail:
        assert_raises(AssertionError, assert_almost_equal, expected.asnumpy(), out2.asnumpy())
    else:
        assert_almost_equal(expected.asnumpy(), out2.asnumpy())
