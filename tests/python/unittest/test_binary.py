from functools import reduce
from operator import mul

import mxnet as mx
import numpy as np
from mxnet import autograd
from mxnet.test_utils import assert_almost_equal
from mxnet.gluon import nn
import pytest


def forward(x, *block_sequence):
    hidden = x
    with autograd.record():
        for block in block_sequence:
            hidden = block.forward(hidden)
        loss = 5 * hidden.mean()
    loss.backward()
    return x.grad.asnumpy(), hidden.asnumpy()


# input_shape,test_conv,layer,args,kwargs
TEST_PARAMS = [
    ((1, 2, 5, 5), True, nn.QConv2D, [16], {"kernel_size": 3, "strides": 1, "padding": 1, "in_channels": 0}),
    ((1, 2, 4, 4), True, nn.QConv2D, [16], {"kernel_size": 2, "strides": 2, "padding": 0, "in_channels": 0}),
    ((1, 2, 25), False, nn.QDense, [16], {})
]


@pytest.mark.parametrize("input_shape,test_conv,layer,args,kwargs", TEST_PARAMS)
def test_qconv_qdense(input_shape, test_conv, layer, args, kwargs):
    in_npy = np.sign(np.random.uniform(-1, 1, input_shape))
    in_npy[in_npy == 0] = 1
    in_data = mx.nd.array(in_npy)
    in_data.attach_grad()

    binary_layer = layer(*args, **kwargs)
    binary_layer.initialize(mx.init.Xavier(magnitude=2))
    with autograd.record():
        result1 = binary_layer.forward(in_data)
    result1.backward()
    gradients1 = in_data.grad

    with autograd.record():
        if test_conv:
            p = kwargs["padding"]
            padded_in = mx.ndarray.pad(in_data, mode="constant", pad_width=(0, 0, 0, 0, p, p, p, p), constant_value=-1)
        binary_weight = binary_layer.weight.data().det_sign()
        if test_conv:
            direct_result = mx.ndarray.Convolution(padded_in, binary_weight, **binary_layer._kwargs)
        else:
            direct_result = mx.ndarray.FullyConnected(in_data, binary_weight, None, no_bias=True, num_hidden=binary_layer._units)
        offset = reduce(mul, binary_weight.shape[1:], 1)
        result2 = (direct_result + offset) / 2
    result2.backward()
    gradients2 = in_data.grad

    assert_almost_equal(result1.asnumpy(), result2.asnumpy())
    assert_almost_equal(gradients1.asnumpy(), gradients2.asnumpy())


@pytest.mark.parametrize("input_shape", [(1, 2, 5, 5), (10, 1000)])
@pytest.mark.parametrize("threshold", [0.01, 0.1, 0.5, 1.0, 2.0])
def test_qconv_qdense(input_shape, threshold):
    in_npy = np.random.uniform(-2, 2, input_shape)
    in_data = mx.nd.array(in_npy)
    in_data.attach_grad()

    binary_layer = nn.QActivation(gradient_cancel_threshold=threshold)
    with autograd.record():
        result1 = binary_layer.forward(in_data)
    result1.backward()
    gradients1 = in_data.grad

    with autograd.record():
        cancelled = mx.ndarray.contrib.gradcancel(in_data, threshold=threshold)
        result2 = cancelled.det_sign()
    result2.backward()
    gradients2 = in_data.grad

    # check that correct functions are used
    assert_almost_equal(result1.asnumpy(), result2.asnumpy())
    assert_almost_equal(gradients1.asnumpy(), gradients2.asnumpy())

    # explicitly model cancelling
    grads_let_through = np.abs(np.sign(gradients2.asnumpy()))
    expected_let_through = np.zeros_like(in_npy)
    expected_let_through[np.abs(in_npy) <= threshold] = 1
    assert_almost_equal(grads_let_through, expected_let_through)

    # shape should be unchanged
    assert_almost_equal(result1.shape, in_data.shape)


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


@pytest.mark.parametrize("threshold", [0.01, 0.1, 0.5, 1.0, 2.0])
def test_grad_cancel(threshold):
    npy = np.random.uniform(-2, 2, (2, 3, 4))

    x = mx.nd.array(npy)
    x.attach_grad()
    with autograd.record():
        y = x ** 2
    y.backward()
    exp_grad = x.grad.asnumpy().copy()

    with autograd.record():
        cancelled = mx.nd.contrib.gradcancel(x, threshold=threshold)
        y = cancelled ** 2
    y.backward()
    exp_grad[np.abs(npy) > threshold] = 0
    np.testing.assert_almost_equal(exp_grad, x.grad.asnumpy())
