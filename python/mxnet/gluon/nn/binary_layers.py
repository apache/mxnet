from .basic_layers import Dense, HybridBlock
from .conv_layers import _Conv
from ...base import numeric_types
from ...symbol import Symbol


def check_params(use_bias, activation):
    if use_bias:
        raise ValueError("Bias is not supported for a binary layer.")
    if isinstance(activation, int) and 0 <= activation <= 32:
        return
    raise ValueError("Activation '{}' is not supported for a binary layer. "
                     "Pass the number of bits (<=32) for activation instead (0 for no activation).".format(activation))


def quantize(F, x, bits):
    if bits == 1:
        return F.det_sign(x)
    elif bits < 32:
        raise NotImplementedError("Quantized not yet suported.")
    else:
        return x


class QActivation(HybridBlock):
    def __init__(self, *args, bits=1, backward_only=False, gradient_cancel_threshold=1.0, **kwargs):
        super(QActivation, self).__init__(*args, **kwargs)
        self.bits = bits
        self.do_forward = not backward_only
        self.threshold = gradient_cancel_threshold

    def hybrid_forward(self, F, x):
        x = F.contrib.gradcancel(x, threshold=self.threshold)
        if self.do_forward:
            x = quantize(F, x, self.bits)
        return x


class QDense(Dense):
    def __init__(self, *args, bits=1, activation=0, use_bias=False, **kwargs):
        check_params(use_bias, activation)
        super(QDense, self).__init__(*args, activation=None, use_bias=False, **kwargs)
        self._offset = 0
        self.bits = bits
        self.activation = activation

    def hybrid_forward(self, F, x, weight, bias=None):
        if not isinstance(weight, Symbol) and self._offset == 0:
            self._offset = 1
            for dim_size in weight.shape[1:]:
                self._offset *= dim_size
        if self.activation > 0:
            x = quantize(F, x, self.activation)
        quantized_weight = quantize(F, weight, self.bits)
        h = F.FullyConnected(x, quantized_weight, bias, no_bias=True,
                             num_hidden=self._units, flatten=self._flatten, name='fwd')
        return (h + self._offset) / 2


class _QConv(_Conv):
    def __init__(self, channels, kernel_size, bits, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs):
        # set activation to None and padding to zero
        super(_QConv, self).__init__(
            channels, kernel_size, strides, 0, dilation, groups, layout,
            in_channels, None, use_bias, weight_initializer, bias_initializer, **kwargs)
        check_params(use_bias, activation)
        self._offset = 0
        self.bits = bits
        self.activation = activation
        if isinstance(padding, numeric_types):
            padding = (padding,) * len(kernel_size)
        self._pre_padding = padding

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
            self._offset = 1
            for dim_size in weight.shape[1:]:
                self._offset *= dim_size
        if self.activation > 0:
            x = quantize(F, x, self.activation)
        quantized_weight = quantize(F, weight, self.bits)
        padded = self._apply_pre_padding(F, x)
        h = F.Convolution(padded, quantized_weight, name='fwd', **self._kwargs)
        return (h + self._offset) / 2


class QConv1D(_QConv):
    def __init__(self, channels, kernel_size, bits=1, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCW', activation=0, use_bias=False,
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
                 activation=0, use_bias=False, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(QConv2D, self).__init__(
            channels, kernel_size, bits, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


class QConv3D(_QConv):
    def __init__(self, channels, kernel_size, bits=1, strides=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), groups=1, layout='NCDHW', activation=0,
                 use_bias=False, weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        super(QConv3D, self).__init__(
            channels, kernel_size, bits, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)
