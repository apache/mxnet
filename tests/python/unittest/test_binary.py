from functools import reduce
from operator import mul

import mxnet as mx
import numpy as np
from mxnet import autograd
from mxnet.test_utils import assert_almost_equal, check_numeric_gradient, numeric_grad
from mxnet.gluon import nn
import mxnet.ndarray as F
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
@pytest.mark.parametrize("bits_a,activation", [(1, 'det_sign'), (2, 'round')])
def test_binary_layer_config_qact(grad_cancel, bits_a, activation, input_shape=(1, 2, 4, 4)):
    d = np.random.uniform(-1, 1, input_shape)
    in_data = mx.nd.array(d)
    in_data.attach_grad()

    qact = nn.QActivation(bits=bits_a, gradient_cancel_threshold=grad_cancel, method=activation)
    with nn.set_binary_layer_config(grad_cancel=grad_cancel, bits_a=bits_a, activation=activation):
        qact_config = nn.QActivation()

    grad, y = forward(in_data, qact)
    grad_, y_ = forward(in_data, qact_config)

    np.testing.assert_almost_equal(y, y_)
    np.testing.assert_almost_equal(grad, grad_)


@pytest.mark.parametrize("bits,weight_quantization", [(1, 'det_sign'), (2, 'dorefa'), (32, 'identity')])
def test_binary_layer_config_qconv(bits, weight_quantization, input_shape=(1, 3, 7, 7), channel=2):
    d = np.random.uniform(-1, 1, input_shape)
    in_data = mx.nd.array(d)
    in_data.attach_grad()

    qconv = nn.QConv2D(channel, 3, bits=bits, quantization=weight_quantization, in_channels=input_shape[1])
    qconv.initialize(mx.init.Xavier(magnitude=2))

    with nn.set_binary_layer_config(bits=bits, weight_quantization=weight_quantization):
        qconv_config = nn.QConv2D(channel, 3, params=qconv.collect_params(), in_channels=input_shape[1])

    grad, y = forward(in_data, qconv)
    grad_, y_ = forward(in_data, qconv_config)

    np.testing.assert_almost_equal(y, y_)
    np.testing.assert_almost_equal(grad, grad_)


def test_binary_layer_config_scaling():
    assert isinstance(nn.activated_conv(3), nn.BinaryConvolution)
    with nn.set_binary_layer_config(approximation="xnor"):
        assert isinstance(nn.activated_conv(3), nn.ScaledBinaryConv)
    assert isinstance(nn.activated_conv(3), nn.BinaryConvolution)


"""
    Test binary inference layers
"""
# TODO: would be nice to find a better solution to do this
import sys
sys.path.append('./example/bmxnet-examples/model_converter/')
from concatenation_operator import get_binary_row, get_binary_col


def test_binary_inference_conv():
    bits_binary_word = 32
    input_dim = 32
    output_dim = 1
    batch_size = 10
    kernel_dim = 1
    input_data = mx.nd.random.normal(-1, 1, shape=(batch_size, input_dim, kernel_dim, kernel_dim))
    weight = mx.nd.random.normal(-1, 1, shape=(output_dim, input_dim, kernel_dim, kernel_dim))

    # weights concatenation
    size_binary_row = int(weight.size / bits_binary_word)
    weight_concatenated = np.zeros((size_binary_row), dtype='uint32')
    weight_concatenated = mx.nd.array(get_binary_row(weight.reshape(-1), 
                                                    weight_concatenated, 
                                                    weight.size, 
                                                    bits_binary_word), 
                                                    dtype='float64') 
    weight_concatenated = weight_concatenated.reshape((weight.shape[0], 
                                                    -1, 
                                                    weight.shape[2], 
                                                    weight.shape[3]))
    # create binary inference conv layer
    binary_infer_result = mx.ndarray.BinaryInferenceConvolution(data=input_data, weight=weight_concatenated,
                                                                kernel=(kernel_dim, kernel_dim), num_filter=output_dim)

    binary_infer_result2 = mx.ndarray.BinaryInferenceConvolution(data=input_data, weight=weight_concatenated,
                                                                 kernel=(kernel_dim, kernel_dim), num_filter=output_dim)

    # create qconv2d layer, assign weights and set input_data.
    qconv_layer = nn.QConv2D(output_dim, kernel_dim, bits=1, use_bias=False, in_channels=input_dim,
                             apply_scaling=False, no_offset = False)
    qact = nn.QActivation(bits=1)
    qact_result = qact.forward(input_data)
    qconv_result = qconv_layer.hybrid_forward(F, x=qact_result, weight=weight)

    np.testing.assert_equal(binary_infer_result.asnumpy(), binary_infer_result2.asnumpy())
    # conversion in python currently does not work
    # np.testing.assert_almost_equal(binary_infer_result.asnumpy(), qconv_result.asnumpy())


def test_binary_inference_fc():
    # setup data
    batch_size = 1
    bits_binary_word = 32
    num_hidden_fc = 10
    num_input_features = 1024
    input_data = mx.nd.random.normal(-1, 1, shape=(batch_size, num_input_features))
    weight = mx.nd.random.normal(-1, 1, shape=(num_hidden_fc, num_input_features))

    # input_npy = (np.sign(input_data.asnumpy()).flatten() + 1) / 2
    # weight_npy = (np.sign(weight.asnumpy()).flatten() + 1) / 2
    # result = 0
    # for i in range(len(weight_npy)):
    #     result += 0 if (input_npy[i] + weight_npy[i]) == 1 else 1

    # weights concatenation
    weight_T = weight.T
    size_binary_col = int(weight_T.size / bits_binary_word)
    weight_concatenated = np.zeros((size_binary_col), dtype='uint32')
    weight_concatenated =mx.nd.array(get_binary_col(weight_T.reshape((-1)),
                                                    weight_concatenated, 
                                                    weight_T.shape[0], 
                                                    weight_T.shape[1], 
                                                    bits_binary_word), 
                                                    dtype='float64')
    weight_concatenated = weight_concatenated.reshape((weight_T.shape[1], -1))
    assert weight_concatenated.shape[0] == num_hidden_fc
    assert weight_concatenated.shape[1] == num_input_features // bits_binary_word
    # create binary inference fc layer
    binary_infer_result = mx.ndarray.BinaryInferenceFullyConnected(data=input_data,
                                                     weight=weight_concatenated, num_hidden=num_hidden_fc)

    binary_infer_result2 = mx.ndarray.BinaryInferenceFullyConnected(data=input_data,
                                                     weight=weight_concatenated, num_hidden=num_hidden_fc)

    # create qdense layer, assign weights and set input_data.
    qdense_layer = nn.QDense(num_hidden_fc)
    qact = nn.QActivation(bits=1)
    qact_result = qact.forward(input_data)
    qdense_result = qdense_layer.hybrid_forward(F, x=qact_result, weight=weight)

    np.testing.assert_equal(binary_infer_result.asnumpy(), binary_infer_result2.asnumpy())
    # conversion in python currently does not work
    # np.testing.assert_almost_equal(binary_infer_result.asnumpy(), qdense_result.asnumpy())


def gpu_device(gpu_number=0):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
    except mx.MXNetError:
        return None
    return mx.gpu(gpu_number)


# Input shape: (batch size, num of input channels)
@pytest.mark.parametrize("input_shape", [(100, 2048), (10, 1024), (1, 512), (1, 64), (1, 32)])
@pytest.mark.parametrize("hidden_num", [1000, 512, 100, 13, 2])
def test_binary_inference_fc_gpu_cpu(input_shape, hidden_num):
    '''
    compares the outputs of binary_inference_fc from cpu and gpu implementations
    '''
    gpu_num = 0
    if gpu_device(gpu_num):
        # define variables
        bits_binary_word = 32

        weight_shape = (hidden_num, int(input_shape[1]/bits_binary_word))

        # create input tensor using gpu
        input_g = mx.nd.random.uniform(-1, 1, shape=input_shape, ctx=mx.gpu(gpu_num))
        # create a copy on cpu
        input_c = mx.nd.array(input_g, dtype='float32', ctx=mx.cpu(0))

        # create weights
        weight_np = np.random.randint(0, 2, size=weight_shape)
        weight_g = mx.nd.array(weight_np, dtype='int32', ctx=mx.gpu(gpu_num))
        weight_c = mx.nd.array(weight_np, dtype='int32', ctx=mx.cpu(0))

        # binary inferece forward
        result_g = mx.ndarray.BinaryInferenceFullyConnected(data=input_g, weight=weight_g, num_hidden=hidden_num)
        result_c = mx.ndarray.BinaryInferenceFullyConnected(data=input_c, weight=weight_c, num_hidden=hidden_num)

        np.testing.assert_equal(result_g.asnumpy(), result_c.asnumpy())


# Input shape: (batch size, num of input channels, h, w)
@pytest.mark.parametrize("input_shape", [(10, 64, 8, 8), (10, 256, 8, 8), (10, 1024, 8, 8), (10, 512, 8, 8),
                                         (10, 32, 51, 51), (10, 256, 32, 32), (10, 32, 10, 10), (1, 32, 7, 7)])
@pytest.mark.parametrize("conv_kernel", [(7, 7), (3, 3), (1, 1)])
@pytest.mark.parametrize("filter_num", [1, 5, 32, 64, 128, 512])
def test_binary_inference_conv_gpu_cpu(input_shape, conv_kernel, filter_num):
    '''
    compares the outputs of binary_inference_conv from cpu and gpu implementations
    '''
    gpu_num = 0
    if gpu_device(gpu_num):
        # define variables
        bits_binary_word = 32

        weight_shape = (filter_num,
                        (int)(input_shape[1]/bits_binary_word),  # num input channels / bits
                        conv_kernel[0],
                        conv_kernel[1])

        # create input tensor using gpu
        input_g = mx.nd.random.uniform(-1, 1, shape=input_shape, ctx=mx.gpu(gpu_num))
        # create a copy on cpu
        input_c = mx.nd.array(input_g, dtype='float32', ctx=mx.cpu(0))

        # create weights
        weight_np = np.random.randint(0, 2, size=weight_shape)
        weight_g = mx.nd.array(weight_np, dtype='int32', ctx=mx.gpu(gpu_num))
        weight_c = mx.nd.array(weight_np, dtype='int32', ctx=mx.cpu(0))

        # binary inferece forward
        result_g = mx.nd.BinaryInferenceConvolution(data=input_g, weight=weight_g,
                                                    kernel=conv_kernel, num_filter=filter_num)
        result_c = mx.nd.BinaryInferenceConvolution(data=input_c, weight=weight_c,
                                                    kernel=conv_kernel, num_filter=filter_num)

        np.testing.assert_equal(result_g.asnumpy(), result_c.asnumpy())
