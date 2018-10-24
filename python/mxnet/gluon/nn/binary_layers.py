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


def quantize(F, x, bits, use_dorefa_weight_activation=False):
    def quantize_k(x):
        vmax = 2 ** bits - 1
        return F.round_ste(x * vmax) / vmax

    if bits == 1:
        return F.det_sign(x)
    elif bits < 32:
        if use_dorefa_weight_activation:
            act_x = 0.5 * F.broadcast_div(F.tanh(x), F.max(F.abs(F.tanh(x)))) + 0.5
            return 2 * quantize_k(act_x) - 1
        else:
            return quantize_k(x.clip(0, 1))
    else:
        return x


class QActivation(HybridBlock):
    def __init__(self, *args, bits=1, gradient_cancel_threshold=1.0,
                 use_dorefa_weight_activation=False, **kwargs):
        super(QActivation, self).__init__(*args, **kwargs)
        self.bits = bits
        self.threshold = gradient_cancel_threshold
        self.use_dorefa = use_dorefa_weight_activation

    def hybrid_forward(self, F, x):
        x = F.contrib.gradcancel(x, threshold=self.threshold)
        x = quantize(F, x, self.bits, use_dorefa_weight_activation=self.use_dorefa)
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
        quantized_weight = quantize(F, weight, self.bits)
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
        quantized_weight = quantize(F, weight, self.bits, use_dorefa_weight_activation=True)
        padded = self._apply_pre_padding(F, x)
        h = F.Convolution(padded, quantized_weight, name='fwd', **self._kwargs)
        if self.scaling:
            scale = weight.abs().mean(axis=0, exclude=True, keepdims=True).transpose(self._scaling_transpose)
            scale = F.stop_gradient(scale)
            h = F.broadcast_mul(h, scale)
        if self.bits == 1 and not self.no_offset and not self.scaling:
            h = (h + self._offset) / 2
        if self.bits == 32:
            # non linearity for FP
            h = F.relu(h)
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
