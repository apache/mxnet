from functools import reduce
from operator import mul

import mxnet as mx
import numpy as np
from mxnet import autograd
from mxnet.test_utils import assert_almost_equal, check_numeric_gradient, numeric_grad
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
    ((3, 50), False, nn.QDense, [16], {})
]


@pytest.mark.parametrize("input_shape,test_conv,layer,args,kwargs", TEST_PARAMS)
def test_binary_qconv_qdense(input_shape, test_conv, layer, args, kwargs):
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
        binary_weight = binary_layer.weight.data().det_sign()
        if test_conv:
            p = kwargs["padding"]
            padded_in = mx.ndarray.pad(in_data, mode="constant", pad_width=(0, 0, 0, 0, p, p, p, p), constant_value=-1)
            direct_result = mx.ndarray.Convolution(padded_in, binary_weight, **binary_layer._kwargs)
        else:
            direct_result = mx.ndarray.FullyConnected(in_data, binary_weight, None, no_bias=True, num_hidden=binary_layer._units)
        offset = reduce(mul, binary_weight.shape[1:], 1)
        result2 = (direct_result + offset) / 2
    result2.backward()
    gradients2 = in_data.grad

    assert_almost_equal(result1.asnumpy(), result2.asnumpy())
    assert_almost_equal(gradients1.asnumpy(), gradients2.asnumpy())

    fractions = result1.asnumpy() - np.fix(result1.asnumpy())
    assert_almost_equal(fractions, np.zeros_like(result1.asnumpy()))


@pytest.mark.parametrize("input_shape", [(1, 2, 5, 5), (10, 1000)])
@pytest.mark.parametrize("threshold", [0.01, 0.1, 0.5, 1.0, 2.0])
def test_qactivation(input_shape, threshold):
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


@pytest.mark.parametrize("input_shape,test_conv,layer,args,kwargs", TEST_PARAMS)
def test_qconv_binary_operation(input_shape, test_conv, layer, args, kwargs):
    in_npy = np.sign(np.random.uniform(-1, 1, input_shape))
    in_npy[in_npy == 0] = 1
    in_data = mx.nd.array(in_npy)

    binary_layer = layer(*args, **kwargs)
    binary_layer.initialize(mx.init.Xavier(magnitude=2))
    result1 = binary_layer.forward(in_data)

    if test_conv:
        p = kwargs["padding"]
        in_npy = mx.ndarray.pad(
            in_data, mode="constant", pad_width=(0, 0, 0, 0, p, p, p, p), constant_value=-1
        ).asnumpy()
    binary_weight_npy = ((binary_layer.weight.data().det_sign() + 1) / 2).asnumpy()
    in_data_converted_npy = (in_npy + 1) / 2
    if test_conv:
        kernel_size = kwargs["kernel_size"]
        stride = kwargs["strides"]
        shape = (
            in_npy.shape[0],
            binary_layer._channels,
            round((in_data_converted_npy.shape[2] - kernel_size + 1) / stride),
            round((in_data_converted_npy.shape[3] - kernel_size + 1) / stride)
        )
        assert_almost_equal(np.asarray(shape), np.asarray(result1.shape))
        result2 = np.zeros(shape)
        # naive convolution
        for sample_idx in range(0, shape[0]):
            for channel_idx in range(0, shape[1]):
                for out_idx1 in range(0, shape[2]):
                    for out_idx2 in range(0, shape[3]):
                        for in_channel_idx in range(0, in_data_converted_npy.shape[1]):
                            input_slice = in_data_converted_npy[
                                          sample_idx,
                                          in_channel_idx,
                                          out_idx1*stride:out_idx1*stride+kernel_size,
                                          out_idx2*stride:out_idx2*stride+kernel_size
                                          ]
                            weight_slice = binary_weight_npy[channel_idx, in_channel_idx]

                            xor = np.logical_xor(input_slice, weight_slice)
                            xnor = np.logical_not(xor)
                            popcount = np.sum(xnor)
                            result2[sample_idx, channel_idx, out_idx1, out_idx2] += popcount
    else:
        result2 = np.zeros((in_npy.shape[0], binary_layer._units))
        for sample_idx in range(0, in_npy.shape[0]):
            for output_idx in range(0, binary_layer._units):
                input_slice = in_data_converted_npy[sample_idx]
                weight_slice = binary_weight_npy[output_idx]

                xor = np.logical_xor(input_slice, weight_slice)
                xnor = np.logical_not(xor)
                popcount = np.sum(xnor)
                result2[sample_idx, output_idx] = popcount

    assert_almost_equal(result1.asnumpy(), result2)


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


def test_approx_sign():
    exp_y = np.array([1.0, 1.0, -1.0])
    exp_grad = np.array([2.0, 0.8, 1.4])

    x = mx.nd.array([0.0, 0.6, -0.3])
    x.attach_grad()
    with autograd.record():
        y = x.approx_sign()
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


"""
    Test DoReFa > 2 bit quantization
"""


@pytest.mark.parametrize("bits", [1])
@pytest.mark.parametrize("input_shape", [(1, 2, 4, 4)])
def test_qconvolution_scaling(input_shape, bits, channel=16, kernel=(3, 3)):
    d = np.random.uniform(-1, 1, input_shape)
    in_data = mx.nd.array(d)
    in_data.attach_grad()

    qconv_scaled = nn.QConv2D(channel, kernel, bits, use_bias=False, no_offset=True, apply_scaling=True)
    qconv_scaled.initialize(mx.init.Xavier(magnitude=2))
    qconv_std = nn.QConv2D(channel, kernel, bits, use_bias=False, no_offset=True, apply_scaling=False,
                           params=qconv_scaled.collect_params())
    conv = nn.Conv2D(channel, kernel, use_bias=False, params=qconv_scaled.collect_params())

    grad_scaled, result_scaled = forward(in_data, qconv_scaled)
    grad_std, result_std = forward(in_data, qconv_std)
    grad, result = forward(in_data, conv)

    def mse(a, b):
        return ((a - b)**2).mean()

    def sign_match(a, b):
        return np.mean(np.sign(a) * np.sign(b))

    assert mse(result, result_scaled) < mse(result, result_std)
    assert sign_match(result_std, result_scaled) > 0.95
    # assert sign_match(grad_std, grad_scaled) > 0.9


"""
    Test binary layer config
"""

@pytest.mark.parametrize("grad_cancel", [1.0, 0.2])
@pytest.mark.parametrize("bits,bits_a,method", [(1, 1, 'det_sign'), (2, 2, 'dorefa')])
def test_binary_layer_config_qact(grad_cancel, bits, bits_a, method, input_shape=(1, 2, 4, 4)):
    d = np.random.uniform(-1, 1, input_shape)
    in_data = mx.nd.array(d)
    in_data.attach_grad()

    qact = nn.QActivation(bits=bits_a, gradient_cancel_threshold=grad_cancel, method=method)
    with nn.set_binary_layer_config(grad_cancel=grad_cancel, bits=bits, bits_a=bits_a,
                                    method=method):
        qact_config = nn.QActivation()

    grad, y = forward(in_data, qact)
    grad_, y_ = forward(in_data, qact_config)

    np.testing.assert_almost_equal(y, y_)
    np.testing.assert_almost_equal(grad, grad_)
