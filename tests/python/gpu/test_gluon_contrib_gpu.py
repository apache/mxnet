from __future__ import print_function
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon import contrib
from mxnet.gluon.contrib.nn import Deformable_Convolution


def test_deformable_convolution():
    net = nn.HybridSequential()
    net.add(
        Deformable_Convolution(10, kernel_size=(3, 3), strides=1, padding=0),
        Deformable_Convolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                               offset_use_bias=False, use_bias=False),
        Deformable_Convolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                               offset_use_bias=False),
        Deformable_Convolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                               use_bias=False),
        Deformable_Convolution(10, kernel_size=(3, 2), strides=1, padding=0, offset_use_bias=False, use_bias=False),
        Deformable_Convolution(10, kernel_size=(3, 2), strides=1, padding=0, offset_use_bias=False),
        Deformable_Convolution(10, kernel_size=(3, 2), strides=1, padding=0, use_bias=False),
    )

    try:
        ctx = mx.gpu()
        _ = mx.nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        print("deformable_convolution only supports GPU")
        return

    net.initialize(force_reinit=True, ctx=ctx)
    net.hybridize()

    x = mx.nd.random.uniform(shape=(8, 5, 30, 31), ctx=ctx)
    with mx.autograd.record():
        y = net(x)
        y.backward()


if __name__ == '__main__':
    import nose
    nose.runmodule()
