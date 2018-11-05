from functools import reduce
from operator import mul

from .basic_layers import Dense, HybridBlock
from .conv_layers import _Conv
from ...base import numeric_types
from ...symbol import Symbol


def check_params(use_bias, activation):
    if use_bias:
        raise ValueError("Bias is not supported for a binary layer.")
    if activation is not None:
        raise ValueError("Activation '{}' is not supported for a binary layer.")


def _quantize_valid_params(bits, method):
    if bits == 32:
        return method in ['approx_sign', 'relu', 'clip', 'leakyclip']
    elif bits == 1:
        return method in ['det_sign', 'sign_approx_sign']
    else:
        return method in ['round', 'dorefa']


def _quantize_method_input_transform(F, method, x):
    if method == 'round':
        return F.clip(x, 0, 1)
    elif method == 'dorefa':
        return 0.5 * F.broadcast_div(F.tanh(x), F.max(F.abs(F.tanh(x)))) + 0.5
    return x


def _quantize_method(F, method, x, bits=None, leaky_slope=0.1):
    if method == 'clip':
        return F.clip(x, -1, 1)
    elif method == 'relu':
        return F.relu(x)
    elif method == 'leaky_clip':
        return F.where(x < -1, leaky_slope * x, F.where(x <= 1, x, leaky_slope * x))
    elif method == 'approx_sign':
        return F.where(x <= -1, -1 * F.ones_like(x),
                       F.where(x < 0, 2 * x + x ** 2,
                               F.where(x < 1, 2 * x - x ** 2,
                                       F.ones_like(x))))
    elif method == 'sign_approx_sign':
        return F.approx_sign(x)
    elif method == 'det_sign':
        return F.det_sign(x)
    elif method in ['round', 'dorefa']:
        vmax = 2 ** bits - 1
        return F.round_ste(x * vmax) / vmax
    return x


def _quantize_method_output_transform(F, method, x):
    if method == 'dorefa':
        return 2 * x - 1
    return x


def default_method(bits):
    if bits == 1:
        return 'det_sign'
    elif bits == 32:
        return 'relu'
    return 'dorefa'


def quantize(F, x, bits=1, method='det_sign'):
    """
    :param F: API (NDArray / Symbol)
    :param x: Data to be quantized (including grad_cancel)
    :param bits: bit resolution; 1 - binary, 2-31 quantized, 32 FP
    :param method: method of activation
           options: approx_sign, relu, clip, leaky_clip, det_sign, sign_approx_sign, round
    :return: activated (or quantized) data
    """
    assert _quantize_valid_params(bits, method)
    pre = _quantize_method_input_transform(F, method, x)
    out = _quantize_method(F, method, pre)
    return _quantize_method_output_transform(F, method, out)


class QActivation(HybridBlock):
    def __init__(self, *args, bits=1, gradient_cancel_threshold=1.0, method=None, **kwargs):
        super(QActivation, self).__init__(*args, **kwargs)
        self.bits = bits
        self.threshold = gradient_cancel_threshold
        self.method = method or default_method(self.bits)

    def hybrid_forward(self, F, x):
        x = F.contrib.gradcancel(x, threshold=self.threshold)
        x = quantize(F, x, self.bits, self.method)
        return x


class QDense(Dense):
    def __init__(self, *args, bits=1, activation=None, use_bias=False, **kwargs):
        check_params(use_bias, activation)
        super(QDense, self).__init__(*args, activation=None, use_bias=False, **kwargs)
        self._offset = 0
        self.bits = bits
        self.weight.wd_mult = 0.0

    def hybrid_forward(self, F, x, weight, bias=None):
        if not isinstance(weight, Symbol) and self._offset == 0:
            self._offset = reduce(mul, weight.shape[1:], 1)
        quantized_weight = quantize(F, weight, self.bits, default_method(self.bits))
        h = F.FullyConnected(x, quantized_weight, bias, no_bias=True,
                             num_hidden=self._units, flatten=self._flatten, name='fwd')
        return (h + self._offset) / 2


class _QConv(_Conv):
    def __init__(self, channels, kernel_size, bits, strides, padding, dilation, groups, layout, in_channels, activation,
                 use_bias, weight_initializer, bias_initializer, no_offset=False, apply_scaling=False, **kwargs):
        check_params(use_bias, activation)
        # set activation to None and padding to zero
        super(_QConv, self).__init__(
            channels, kernel_size, strides, 0, dilation, groups, layout,
            in_channels, None, use_bias, weight_initializer, bias_initializer, **kwargs)
        self._offset = 0
        self.no_offset = no_offset
        self.bits = bits
        if isinstance(padding, numeric_types):
            padding = (padding,) * len(kernel_size)
        self._pre_padding = padding
        self.weight.wd_mult = 0.0
        self.scaling = apply_scaling
        self._scaling_transpose = (1, 0, *range(2, len(kernel_size) + 2))

    def _alias(self):
        return 'qconv'

    def _apply_pre_padding(self, F, x):
        if sum(self._pre_padding) > 0:
            assert self._kwargs["layout"] == "NCHW", \
                "Padding with binary layers is currently only supported on NCHW layout."
            axis_padding = [0, 0, 0, 0]
            for pad_width in self._pre_padding:
                axis_padding.extend([pad_width, pad_width])
            x = F.pad(x, mode="constant", pad_width=axis_padding, constant_value=-1)
        return x

    def hybrid_forward(self, F, x, weight, bias=None):
        if not isinstance(weight, Symbol) and self._offset == 0:
            self._offset = reduce(mul, weight.shape[1:], 1)
        quantized_weight = quantize(F, weight, self.bits, default_method(self.bits))
        padded = self._apply_pre_padding(F, x)
        h = F.Convolution(padded, quantized_weight, name='fwd', **self._kwargs)
        if self.scaling:
            scale = weight.abs().mean(axis=0, exclude=True, keepdims=True).transpose(self._scaling_transpose)
            scale = F.stop_gradient(scale)
            h = F.broadcast_mul(h, scale)
        if self.bits == 1 and not self.no_offset and not self.scaling:
            h = (h + self._offset) / 2
        return h


class QConv1D(_QConv):
    def __init__(self, channels, kernel_size, bits=1, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCW', activation=None, use_bias=False,
                 weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)
        assert len(kernel_size) == 1, "kernel_size must be a number or a list of 1 ints"
        super(QConv1D, self).__init__(
            channels, kernel_size, bits, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


class QConv2D(_QConv):
    def __init__(self, channels, kernel_size, bits=1, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=False, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(QConv2D, self).__init__(
            channels, kernel_size, bits, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


class QConv3D(_QConv):
    def __init__(self, channels, kernel_size, bits=1, strides=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), groups=1, layout='NCDHW', activation=None,
                 use_bias=False, weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        super(QConv3D, self).__init__(
            channels, kernel_size, bits, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


class ScaledBinaryConv(HybridBlock):
    r"""ScaledBinaryConv implements scaled binarized 2D convolution,
        introduced by XNOR-Net Paper
    """

    def __init__(self, bits, bits_a, channels, kernel_size=3, stride=1, padding=0, in_channels=0, clip_threshold=1.0,
                 prefix=None, activation_method=None, **kwargs):
        super(ScaledBinaryConv, self).__init__(**kwargs)
        self.qact = QActivation(bits=bits_a, gradient_cancel_threshold=clip_threshold, method=activation_method)
        self.qconv = QConv2D(channels, bits=bits, kernel_size=kernel_size, strides=stride, padding=padding,
                             in_channels=in_channels, prefix=prefix, no_offset=True, apply_scaling=True)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels

    def hybrid_forward(self, F, x):
        y = self.qconv(self.qact(x))
        A = x.abs().mean(axis=1, keepdims=True)
        k = F.full((1, 1, self.kernel_size, self.kernel_size), 1 / self.kernel_size ** 2)
        K = F.Convolution(A, k, bias=None, name='scaling_conv', num_filter=1,
                          kernel=(self.kernel_size, self.kernel_size), no_bias=True, stride=(self.stride, self.stride),
                          pad=(self.padding, self.padding), layout='NCHW')
        K = F.stop_gradient(K)
        return F.broadcast_mul(K, y)
