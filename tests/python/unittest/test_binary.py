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


@pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("bits_a", [1])
@pytest.mark.parametrize("bits_w", [1])
@pytest.mark.parametrize("input_shape,layer,args,kwargs", [
    ((1, 2, 5, 5), nn.QConv2D, [16], {"kernel_size": 3, "strides": 1, "padding": 1, "in_channels": 0}),
    ((1, 2, 25), nn.QDense, [16], {})
])
def test_qactivation_qconvolution(threshold, bits_a, bits_w, input_shape, layer, args, kwargs):
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
    assert_almost_equal(result2, result3)
    assert_almost_equal(gradients1, gradients2)
    assert_almost_equal(gradients2, gradients3)

    fractions = result1 - np.fix(result1)
    assert_almost_equal(fractions, np.zeros_like(result1))


def test_activation_in_qconv():
    in_data = mx.nd.array(np.random.uniform(-1, 1, (1, 2, 5, 5)))
    in_data.attach_grad()

    conv = nn.QConv2D(16, kernel_size=3, strides=1, padding=1, in_channels=0, activation=1)
    conv.initialize(mx.init.Xavier(magnitude=2))
    with autograd.record():
        result2 = conv.forward(in_data)
    result2.backward()
    gradients1 = in_data.grad

    with autograd.record():
        sign_in = in_data.det_sign()
        padded_in = mx.ndarray.pad(sign_in, mode="constant", pad_width=(0, 0, 0, 0, 1, 1, 1, 1), constant_value=-1)
        binary_weight = conv.weight.data().det_sign()
        direct_result = mx.ndarray.Convolution(padded_in, binary_weight, **conv._kwargs)
        offset = 2 * 3 * 3
        result1 = (direct_result + offset) / 2
    result1.backward()
    gradients2 = in_data.grad

    assert_almost_equal(result1.asnumpy(), result2.asnumpy())
    assert_almost_equal(gradients1.asnumpy(), gradients2.asnumpy())


def test_no_activation_in_qconv():
    in_data = mx.nd.array(np.random.uniform(-1, 1, (2, 3, 5, 5)))
    in_data.attach_grad()

    conv = nn.QConv2D(16, kernel_size=3, strides=1, padding=1, in_channels=0)
    conv.initialize(mx.init.Xavier(magnitude=2))
    with autograd.record():
        result2 = conv.forward(in_data)
    result2.backward()
    gradients1 = in_data.grad

    with autograd.record():
        padded_in = mx.ndarray.pad(in_data, mode="constant", pad_width=(0, 0, 0, 0, 1, 1, 1, 1), constant_value=-1)
        binary_weight = conv.weight.data().det_sign()
        direct_result = mx.ndarray.Convolution(padded_in, binary_weight, **conv._kwargs)
        offset = 3 * 3 * 3
        result1 = (direct_result + offset) / 2
    result1.backward()
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


def test_grad_cancel():
    npy = np.random.uniform(-0.5, 1, (2, 3, 4))

    x = mx.nd.array(npy)
    x.attach_grad()
    with autograd.record():
        y = x ** 2
    y.backward()
    exp_grad = x.grad.asnumpy()

    for threshold in [1.0, 0.5, 0.1]:
        with autograd.record():
            cancelled = mx.nd.contrib.gradcancel(x, threshold=threshold)
            y = cancelled ** 2
        y.backward()
        exp_grad[np.abs(exp_grad) > threshold] = 0
        np.testing.assert_almost_equal(exp_grad, x.grad.asnumpy())
