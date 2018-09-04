from mxnet.gluon.nn import Dense, BatchNorm
from mxnet.gluon.nn.conv_layers import _Conv
from mxnet.base import numeric_types


class BDense(Dense):
    def __init__(self, *args, activation=None, use_bias=False, **kwargs):
        if use_bias:
            raise RuntimeError("Bias is not supported for a binary Dense block.")
        if activation is not None:
            raise RuntimeError("Activation is not supported for a binary Dense block.")
        super(BDense, self).__init__(*args, activation=None, use_bias=False, **kwargs)
        self._pre_bn = BatchNorm()

    def hybrid_forward(self, F, x, weight, bias=None):
        normalized = self._pre_bn(x)
        binarized_input = F.det_sign(normalized)
        binarized_weight = F.det_sign(weight)
        h = F.FullyConnected(binarized_input, binarized_weight, bias, no_bias=True, num_hidden=self._units,
                               flatten=self._flatten, name='fwd')
        return (h + self._units) / 2


class _BConv(_Conv):
    def __init__(self, *args, **kwargs):
        super(_BConv, self).__init__(*args, **kwargs)
        self._pre_bn = BatchNorm()
        self._offset = 1
        for dim_size in self._kwargs["kernel"]:
            self._offset *= dim_size

    def hybrid_forward(self, F, x, weight, bias=None):
        normalized = self._pre_bn(x)
        binarized_input = F.det_sign(normalized)
        binarized_weight = F.det_sign(weight)
        h = F.Convolution(binarized_input, binarized_weight, name='fwd', **self._kwargs)
        return (h + self._offset) / 2


class BConv1D(_BConv):
    def __init__(self, channels, kernel_size, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCW', activation=None, use_bias=False,
                 weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        if use_bias:
            raise RuntimeError("Bias is not supported for a binary Dense block.")
        if activation is not None:
            raise RuntimeError("Activation is not supported for a binary Dense block.")
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
        if use_bias:
            raise RuntimeError("Bias is not supported for a binary Dense block.")
        if activation is not None:
            raise RuntimeError("Activation is not supported for a binary Dense block.")
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
        if use_bias:
            raise RuntimeError("Bias is not supported for a binary Dense block.")
        if activation is not None:
            raise RuntimeError("Activation is not supported for a binary Dense block.")
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        super(BConv3D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)
