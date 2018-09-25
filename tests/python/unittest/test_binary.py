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


@pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("bits_a", [1])
@pytest.mark.parametrize("bits_w", [1])
@pytest.mark.parametrize("input_shape,test_conv,layer,args,kwargs", TEST_PARAMS)
def test_qactivation_qconvolution(threshold, bits_a, bits_w, input_shape, test_conv, layer, args, kwargs):
    in_data = mx.nd.array(np.random.uniform(-1, 1, input_shape))
    in_data.attach_grad()

    act1 = nn.QActivation(bits=bits_a, gradient_cancel_threshold=threshold, backward_only=False)
    layer1 = layer(*args, bits=bits_w, activation=0, **kwargs)
    layer1.initialize(mx.init.Xavier(magnitude=2))
    gradients1, result1 = forward(in_data, act1, layer1)

    act2 = nn.QActivation(bits=bits_a, gradient_cancel_threshold=threshold, backward_only=True)
    layer2 = layer(*args, bits=bits_w, activation=bits_a, params=layer1.params, **kwargs)
    gradients2, result2 = forward(in_data, act2, layer2)

    # test that defaults for backward_only and activation are sensible
    act3 = nn.QActivation(bits=bits_a, gradient_cancel_threshold=threshold)
    layer3 = layer(*args, bits=bits_w, params=layer1.params, **kwargs)
    gradients3, result3 = forward(in_data, act3, layer3)

    assert_almost_equal(result1, result2)
    assert_almost_equal(result1, result3)
    assert_almost_equal(gradients1, gradients2)
    assert_almost_equal(gradients1, gradients3)

    fractions = result1 - np.fix(result1)
    assert_almost_equal(fractions, np.zeros_like(result1))


@pytest.mark.parametrize("input_shape,test_conv,layer,args,kwargs", TEST_PARAMS)
def test_activation_in_qconv(input_shape, test_conv, layer, args, kwargs):
    in_data = mx.nd.array(np.random.uniform(-1, 1, input_shape))
    in_data.attach_grad()

    binary_layer = layer(*args, activation=1, **kwargs)
    binary_layer.initialize(mx.init.Xavier(magnitude=2))
    with autograd.record():
        result1 = binary_layer.forward(in_data)
    result1.backward()
    gradients1 = in_data.grad

    with autograd.record():
        sign_in = in_data.det_sign()
        if test_conv:
            p = kwargs["padding"]
            padded_in = mx.ndarray.pad(sign_in, mode="constant", pad_width=(0, 0, 0, 0, p, p, p, p), constant_value=-1)
        binary_weight = binary_layer.weight.data().det_sign()
        if test_conv:
            direct_result = mx.ndarray.Convolution(padded_in, binary_weight, **binary_layer._kwargs)
        else:
            direct_result = mx.ndarray.FullyConnected(sign_in, binary_weight, None, no_bias=True, num_hidden=binary_layer._units)
        offset = reduce(mul, binary_weight.shape[1:], 1)
        result2 = (direct_result + offset) / 2
    result2.backward()
    gradients2 = in_data.grad

    assert_almost_equal(result1.asnumpy(), result2.asnumpy())
    assert_almost_equal(gradients1.asnumpy(), gradients2.asnumpy())


@pytest.mark.parametrize("input_shape,test_conv,layer,args,kwargs", TEST_PARAMS)
def test_no_activation_in_qconv(input_shape, test_conv, layer, args, kwargs):
    in_data = mx.nd.array(np.random.uniform(-1, 1, input_shape))
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
