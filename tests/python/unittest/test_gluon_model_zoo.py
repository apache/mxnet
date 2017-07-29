import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.model_zoo.custom_layers import HybridConcurrent, Identity


def test_concurrent():
    model = HybridConcurrent(concat_dim=1)
    model.add(nn.Dense(128, activation='tanh', in_units=10))
    model.add(nn.Dense(64, activation='tanh', in_units=10))
    model.add(nn.Dense(32, in_units=10))

    # symbol
    x = mx.sym.var('data')
    y = model(x)
    assert len(y.list_arguments()) == 7

    # ndarray
    model.collect_params().initialize(mx.init.Xavier(magnitude=2.24))
    x = model(mx.nd.zeros((32, 10)))
    assert x.shape == (32, 224)
    x.wait_to_read()


def test_identity():
    model = Identity()
    x = mx.nd.random_uniform(shape=(128, 33, 64))
    mx.test_utils.assert_almost_equal(model(x).asnumpy(),
                                      x.asnumpy())


if __name__ == '__main__':
    import nose
    nose.runmodule()
