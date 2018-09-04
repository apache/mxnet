from mxnet.gluon.nn import Dense, BatchNorm
from mxnet.gluon.nn.conv_layers import _Conv
from mxnet.base import numeric_types
from mxnet.symbol.symbol import Symbol


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

    def hybrid_forward(self, F, x, weight, bias=None):
        if not isinstance(weight, Symbol) and self._offset == 0:
            self._offset = 1
            for dim_size in weight.shape[1:]:
                self._offset *= dim_size
        h = F.FullyConnected(binarize_inputs(F, x, self._pre_bn), binarize_weights(F, weight), bias, no_bias=True,
                             num_hidden=self._units, flatten=self._flatten, name='fwd')
        return (h + self._offset) / 2


class _BConv(_Conv):
    def __init__(self, *args, **kwargs):
        super(_BConv, self).__init__(*args, **kwargs)
        self._pre_bn = BatchNorm()
        self._offset = 0

    def hybrid_forward(self, F, x, weight, bias=None):
        if not isinstance(weight, Symbol) and self._offset == 0:
            self._offset = 1
            for dim_size in weight.shape[1:]:
                self._offset *= dim_size
        h = F.Convolution(binarize_inputs(F, x, self._pre_bn), binarize_weights(F, weight), name='fwd', **self._kwargs)
        return (h + self._offset) / 2


class BConv1D(_BConv):
    def __init__(self, channels, kernel_size, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCW', activation=None, use_bias=False,
                 weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        check_params(use_bias, activation)
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
        check_params(use_bias, activation)
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
        check_params(use_bias, activation, in_channels)
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        super(BConv3D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)
