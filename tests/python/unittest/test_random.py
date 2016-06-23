import os
import mxnet as mx
import numpy as np

def same(a, b):
    return np.sum(a != b) == 0

def check_with_device(device):
    with mx.Context(device):
        a, b = -10, 10
        mu, sigma = 10, 2
        shape = (100, 100)
        mx.random.seed(128)
        ret1 = mx.random.normal(mu, sigma, shape)
        un1 = mx.random.uniform(a, b, shape)
        mx.random.seed(128)
        ret2 = mx.random.normal(mu, sigma, shape)
        un2 = mx.random.uniform(a, b, shape)
        assert same(ret1.asnumpy(), ret2.asnumpy())
        assert same(un1.asnumpy(), un2.asnumpy())
        assert abs(np.mean(ret1.asnumpy()) - mu) < 0.1
        assert abs(np.std(ret1.asnumpy()) - sigma) < 0.1
        assert abs(np.mean(un1.asnumpy()) - (a+b)/2) < 0.1


def check_symbolic_random(dev):
    a, b = -10, 10
    mu, sigma = 10, 2
    shape = (100, 100)
    X = mx.sym.Variable("X")
    Y = mx.sym.uniform(low=a, high=b, shape=shape) + X
    x = mx.nd.zeros(shape, ctx=dev)
    xgrad = mx.nd.zeros(shape, ctx=dev)
    yexec = Y.bind(dev, {'X' : x}, {'X': xgrad})
    mx.random.seed(128)
    yexec.forward()
    yexec.backward(yexec.outputs[0])
    un1 = (yexec.outputs[0] - x).copyto(dev)
    assert same(xgrad.asnumpy(), un1.asnumpy())
    mx.random.seed(128)
    yexec.forward()
    un2 = (yexec.outputs[0] - x).copyto(dev)
    assert same(un1.asnumpy(), un2.asnumpy())
    assert abs(np.mean(un1.asnumpy()) - (a+b)/2) < 0.1

    Y = mx.sym.normal(loc=mu, scale=sigma, shape=shape)
    yexec = Y.simple_bind(dev)
    mx.random.seed(128)
    yexec.forward()
    ret1 = yexec.outputs[0].copyto(dev)
    mx.random.seed(128)
    ret2 = mx.random.normal(mu, sigma, shape)
    assert same(ret1.asnumpy(), ret2.asnumpy())
    assert abs(np.mean(ret1.asnumpy()) - mu) < 0.1
    assert abs(np.std(ret1.asnumpy()) - sigma) < 0.1


def test_random():
    check_with_device(mx.cpu())
    check_symbolic_random(mx.cpu())


if __name__ == '__main__':
    test_random()
