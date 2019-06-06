from functools import reduce
from operator import mul

from .basic_layers import Dense, HybridBlock
from .conv_layers import _Conv
from ...base import numeric_types
from ...symbol import Symbol


class BinaryLayerConfig:
    def __init__(self, grad_cancel=1.0, bits=1, bits_a=1, activation='det_sign',
                 weight_quantization='det_sign', approximation=""):
        self.grad_cancel = grad_cancel
        self.bits = bits
        self.bits_a = bits_a
        self.activation = activation
        self.weight_quantization = weight_quantization
        assert approximation in ["", "binet", "xnor"]
        self.approximation = approximation

    def get_values(self):
        return dict(vars(self))

    def set_values(self, assure_consistency=True, **kwargs):
        for attr, value in kwargs.items():
            assert hasattr(self, attr), 'Attribute {} not in {}'.format(attr, self)
            value = value if value is not None else getattr(self, attr)
            setattr(self, attr, value)
        if assure_consistency:
            if not BinaryLayerConfig._valid_activation(self.bits_a, self.activation):
                self.activation = BinaryLayerConfig._default_activation(self.bits_a)
            if not BinaryLayerConfig._valid_weight_quantization(self.bits, self.weight_quantization):
                self.weight_quantization = BinaryLayerConfig._default_weight_quantization(self.bits)

    @staticmethod
    def _valid_activation(bits, activation):
        if bits == 32:
            return activation in ['identity', 'approx_sign', 'relu', 'clip', 'leaky_clip']
        elif bits == 1:
            return activation in ['det_sign', 'sign_approx_sign', 'round']
        return activation in ['round', 'dorefa']  # DoReFa paper only applies round activation

    @staticmethod
    def _default_activation(bits):
        if bits == 1:
            return 'det_sign'
        elif bits == 32:
            return 'relu'
        return 'round'

    @staticmethod
    def _valid_weight_quantization(bits, weight_quantization):
        if bits == 32:
            return weight_quantization in ['identity']
        elif bits == 1:
            return weight_quantization in ['det_sign', 'sign_approx_sign']
        return weight_quantization in ['dorefa']

    @staticmethod
    def _default_weight_quantization(bits):
        if bits == 1:
            return 'det_sign'
        elif bits == 32:
            return 'identity'
        return 'dorefa'

    def get_quantization_function(self, bits=None, method=None):
        """
        Returns quantization function
        Node:
            For specific use as activation or weight quantization, consider
            using `get_activation_function` or
            `get_weight_quantization_function` respectively
        :param bits: Bit resolution
        :param method: Quantzation method to be used
        :return: Function to be called in hybrid forward
        """
        def identity(F, x):
            return x

        def round(F, x):
            vmax = 2 ** bits - 1
            return F.round_ste(F.clip(x, 0, 1) * vmax) / vmax

        def dorefa(F, x):
            vmax = 2 ** bits - 1
            # tanh squashing
            x = 0.5 * F.broadcast_div(F.tanh(x), F.max(F.abs(F.tanh(x)))) + 0.5
            x = F.round_ste(x * vmax) / vmax
            return 2 * x - 1

        def clip(F, x):
            return F.clip(x, -1, 1)

        def relu(F, x):
            return F.relu(x)

        def leaky_clip(F, x, leaky_slope=0.1):
            return F.where(x < -1, leaky_slope * x, F.where(x <= 1, x, leaky_slope * x))

        def approx_sign(F, x):
            return F.where(x <= -1, -1 * F.ones_like(x),
                           F.where(x < 0, 2 * x + x ** 2,
                                   F.where(x < 1, 2 * x - x ** 2,
                                           F.ones_like(x))))

        def sign_approx_sign(F, x):
            return F.approx_sign(x)

        def det_sign(F, x):
            return F.det_sign(x)

        return locals()[method]
    
    def get_activation_function(self, bits=None, method=None):
        bits = bits or self.bits_a
        method = method or self.activation
        assert BinaryLayerConfig._valid_activation(bits, method), \
            'Combination of method {} using {} bit(s) for activation is not supported!'.format(method, bits)
        return self.get_quantization_function(bits=bits, method=method)

    def get_weight_quantization_function(self, bits=None, method=None):
        bits = bits or self.bits
        method = method or self.weight_quantization
        assert BinaryLayerConfig._valid_weight_quantization(bits, method), \
            'Combination of method {} using {} bit(s) for weight quantization is not supported!'.format(method, bits)
        return self.get_quantization_function(bits=bits, method=method)


binary_layer_config = BinaryLayerConfig()


class set_binary_layer_config():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.last_values = binary_layer_config.get_values()
        binary_layer_config.set_values(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        binary_layer_config.set_values(**self.last_values)


def check_params(use_bias, activation):
    if use_bias:
        raise ValueError("Bias is not supported for a binary layer.")
    if activation is not None:
        raise ValueError("Activation '{}' is not supported for a binary layer.")


class QActivation(HybridBlock):
    def __init__(self, *args, bits=None, gradient_cancel_threshold=None, method=None, **kwargs):
        super(QActivation, self).__init__(*args, **kwargs)
        self.quantize = binary_layer_config.get_activation_function(bits=bits, method=method)
        self.threshold = gradient_cancel_threshold or binary_layer_config.grad_cancel

    def hybrid_forward(self, F, x):
        x = F.contrib.gradcancel(x, threshold=self.threshold)
        x = self.quantize(F, x)
        return x


class QDense(Dense):
    def __init__(self, *args, bits=None, activation=None, use_bias=False, no_offset=False, **kwargs):
        check_params(use_bias, activation)
        super(QDense, self).__init__(*args, activation=None, use_bias=False, **kwargs)
        self._offset = 0
        self.quantize = binary_layer_config.get_weight_quantization_function(bits=bits)
        self.method = binary_layer_config.activation
        self.weight.wd_mult = 0.0
        self.no_offset = no_offset

    def hybrid_forward(self, F, x, weight, bias=None):
        if not isinstance(weight, Symbol) and self._offset == 0:
            self._offset = reduce(mul, weight.shape[1:], 1)
        quantized_weight = self.quantize(F, weight)
        h = F.FullyConnected(x, quantized_weight, bias, no_bias=True,
                             num_hidden=self._units, flatten=self._flatten, name='fwd')
        if self.no_offset:
            return h
        return (h + self._offset) / 2


class _QConv(_Conv):
    def __init__(self, channels, kernel_size, bits, strides, padding, dilation, groups, layout, in_channels, activation,
                 use_bias, weight_initializer, bias_initializer, quantization=None, no_offset=False,
                 apply_scaling=False, **kwargs):
        check_params(use_bias, activation)
        # set activation to None and padding to zero
        super(_QConv, self).__init__(
            channels, kernel_size, strides, 0, dilation, groups, layout,
            in_channels, None, use_bias, weight_initializer, bias_initializer, **kwargs)
        self._offset = 0
        self.no_offset = no_offset
        if isinstance(padding, numeric_types):
            padding = (padding,) * len(kernel_size)
        self._pre_padding = padding
        self.weight.wd_mult = 0.0
        self.scaling = apply_scaling
        self.stop_weight_scale_grad = binary_layer_config.approximation == "xnor"
        self._scaling_transpose = (1, 0, *range(2, len(kernel_size) + 2))
        self.bits = bits or binary_layer_config.bits
        self.quantize = binary_layer_config.get_weight_quantization_function(bits=self.bits, method=quantization)

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
        quantized_weight = self.quantize(F, weight)
        padded = self._apply_pre_padding(F, x)
        h = F.Convolution(padded, quantized_weight, name='fwd', **self._kwargs)
        if self.scaling:
            scale = weight.abs().mean(axis=0, exclude=True, keepdims=True).transpose(self._scaling_transpose)
            if self.stop_weight_scale_grad:
                scale = F.stop_gradient(scale)
            h = F.broadcast_mul(h, scale)
        if self.bits == 1 and not self.no_offset and not self.scaling:
            h = (h + self._offset) / 2
        return h


class QConv1D(_QConv):
    def __init__(self, channels, kernel_size, bits=None, strides=1, padding=0, dilation=1,
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
    def __init__(self, channels, kernel_size, bits=None, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=False, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, apply_scaling=False, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(QConv2D, self).__init__(
            channels, kernel_size, bits, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer,
            apply_scaling=apply_scaling, **kwargs)


class QConv3D(_QConv):
    def __init__(self, channels, kernel_size, bits=None, strides=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), groups=1, layout='NCDHW', activation=None,
                 use_bias=False, weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        super(QConv3D, self).__init__(
            channels, kernel_size, bits, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


"""
    Activated Convolution Blocks
"""


class BinaryConvolution(HybridBlock):
    r"""
        Typical binary (XNOR) convolution block with binarized activations and weights
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=0, in_channels=0, dilation=1, bits=None, bits_a=None,
                 clip_threshold=None, activation_method=None, prefix=None, **kwargs):
        super(BinaryConvolution, self).__init__(**kwargs)
        self.qact = QActivation(bits=bits_a, gradient_cancel_threshold=clip_threshold, method=activation_method)
        self.qconv = QConv2D(channels, bits=bits, kernel_size=kernel_size, strides=stride, padding=padding,
                             in_channels=in_channels, prefix=prefix, apply_scaling=False, dilation=dilation)

    def hybrid_forward(self, F, x):
        return self.qconv(self.qact(x))


class ScaledWeightsBinaryConv(HybridBlock):
    r"""
        This implements the magnitude aware gradients from Bi-Real Net paper. It is similiar to ScaledBinaryConv, 
    but it only scales the weights (not the feature maps) and the gradient of the scaling is not blocked.
    The later leads to the so-called "magnitude aware gradients" effect mentioned in the Bi-Real Net paper.
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=0, in_channels=0, dilation=1, bits=None, bits_a=None,
                 clip_threshold=None, activation_method=None, prefix=None, **kwargs):
        super(ScaledWeightsBinaryConv, self).__init__(**kwargs)
        self.qact = QActivation(bits=bits_a, gradient_cancel_threshold=clip_threshold, method=activation_method)
        self.qconv = QConv2D(channels, bits=bits, kernel_size=kernel_size, strides=stride, padding=padding,
                             in_channels=in_channels, prefix=prefix, apply_scaling=True, dilation=dilation)

    def hybrid_forward(self, F, x):
        return self.qconv(self.qact(x))


class ScaledBinaryConv(HybridBlock):
    r"""
        ScaledBinaryConv implements scaled binarized 2D convolution,
        introduced by XNOR-Net Paper
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=0, in_channels=0, dilation=1, bits=None, bits_a=None,
                 clip_threshold=None, activation_method=None, prefix=None, **kwargs):
        super(ScaledBinaryConv, self).__init__(**kwargs)
        self.qact = QActivation(bits=bits_a, gradient_cancel_threshold=clip_threshold, method=activation_method)
        self.qconv = QConv2D(channels, bits=bits, kernel_size=kernel_size, strides=stride, padding=padding,
                             in_channels=in_channels, prefix=prefix, no_offset=True, apply_scaling=True,
                             dilation=dilation)
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


class ActivatedConvolutionFactory:
    def __call__(self, *args, **kwargs):
        if binary_layer_config.approximation == "binet":
            return ScaledWeightsBinaryConv(*args, **kwargs)
        if binary_layer_config.approximation == "xnor":
            return ScaledBinaryConv(*args, **kwargs)
        return BinaryConvolution(*args, **kwargs)


activated_conv = ActivatedConvolutionFactory()
