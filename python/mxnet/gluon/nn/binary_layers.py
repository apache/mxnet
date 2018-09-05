from .basic_layers import Dense, BatchNorm
from .conv_layers import _Conv
from ...base import numeric_types
from ...symbol import Symbol


def check_params(use_bias, activation):
    if use_bias:
        raise ValueError("Bias is not supported for a binary layer.")
    if activation is not None:
        raise ValueError("Activation is not supported for a binary layer.")


def binarize_inputs(F, x, batch_norm_layer):
    normalized = batch_norm_layer(x)
    grad_cancelling = F.contrib.gradcancel(normalized)
    return F.det_sign(grad_cancelling)


def binarize_weights(F, x):
    return F.det_sign(x)


class BDense(Dense):
    def __init__(self, *args, activation=None, use_bias=False, **kwargs):
        check_params(use_bias, activation)
        super(BDense, self).__init__(*args, activation=None, use_bias=False, **kwargs)
        self._pre_bn = BatchNorm()
        self._offset = 0

    def _alias(self):
        return 'bdense'

    def hybrid_forward(self, F, x, weight, bias=None):
        if not isinstance(weight, Symbol) and self._offset == 0:
            self._offset = 1
            for dim_size in weight.shape[1:]:
                self._offset *= dim_size
        binary_x = binarize_inputs(F, x, self._pre_bn)
        binary_weight = binarize_weights(F, weight)
        h = F.FullyConnected(binary_x, binary_weight, bias, no_bias=True,
                             num_hidden=self._units, flatten=self._flatten, name='fwd')
        return (h + self._offset) / 2


class _BConv(_Conv):
    def __init__(self, channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs):
        super(_BConv, self).__init__(
            channels, kernel_size, strides, 0, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)
        check_params(use_bias, activation)
        self._pre_bn = BatchNorm()
        self._offset = 0
        if isinstance(padding, numeric_types):
            padding = (padding,) * len(kernel_size)
        self._pre_padding = padding

    def _alias(self):
        return 'bconv'

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
        binary_x = binarize_inputs(F, self._apply_pre_padding(F, x), self._pre_bn)
        binary_weight = binarize_weights(F, weight)
        h = F.Convolution(binary_x, binary_weight, name='fwd', **self._kwargs)
        return (h + self._offset) / 2


class BConv1D(_BConv):
    def __init__(self, channels, kernel_size, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCW', activation=None, use_bias=False,
                 weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)
        assert len(kernel_size) == 1, "kernel_size must be a number or a list of 1 ints"
        super(BConv1D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


class BConv2D(_BConv):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=False, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(BConv2D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


class BConv3D(_BConv):
    def __init__(self, channels, kernel_size, strides=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), groups=1, layout='NCDHW', activation=None,
                 use_bias=False, weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        super(BConv3D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)
