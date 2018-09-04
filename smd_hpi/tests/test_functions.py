import mxnet as mx
import numpy as np
from mxnet import autograd
from mxnet.test_utils import assert_almost_equal


def test_det_sign():
    exp_y = np.array([1.0, 1.0, -1.0])
    exp_grad = np.array([1.0, 1.0, 1.0])

    x = mx.nd.array([0.0, 0.6, -0.3])
    x.attach_grad()
    with autograd.record():
        y = x.det_sign()
        assert_almost_equal(exp_y, y.asnumpy())
    y.backward()
    assert_almost_equal(exp_grad, x.grad.asnumpy())


def test_round_ste():
    npy = np.random.uniform(-10, 10, (2, 3, 4))

    exp_y = np.round(npy)
    exp_grad = np.ones_like(npy)

    x = mx.nd.array(npy)
    x.attach_grad()
    with autograd.record():
        y = x.round_ste()
        assert_almost_equal(exp_y, y.asnumpy())
    y.backward()
    assert_almost_equal(exp_grad, x.grad.asnumpy())
