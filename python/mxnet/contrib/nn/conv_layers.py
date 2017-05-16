# coding: utf-8
"""Convolutional neural network layers."""
from .layer import Layer
from ... import symbol
from ...base import numeric_types

def _infer_weight_shape(sym, data_shape, **kwargs):
    return sym(symbol.var('data', shape=data_shape), **kwargs).infer_shape_partial()[0]


class _Conv(Layer):
    """Abstract nD convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    Parameters
    ----------
    filters: Integer, the dimensionality of the output space
        (i.e. the number output of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
    strides: An integer or tuple/list of n integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: An integer or a tuple/list of n integers,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
    groups: int
        controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout: A string,
        Can be 'NCW', 'NWC', 'NCHW', 'NHWC', 'NCDHW', 'NDHWC', etc.
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively.
    in_filters: int, default 0
        The number of input channels to this layer. Only required when using
        NDArray API.
    activation: Activation function to use
        see mx.sym.Activation.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix
        see Initializer.
    bias_initializer: Initializer for the bias vector
        see Initializer.
    """
    def __init__(self, filters, kernel_size, strides, padding, dilation,
                 groups, layout, in_filters=0, activation=None, use_bias=True,
                 kernel_initializer=None, bias_initializer=None, **kwargs):
        super(_Conv, self).__init__(**kwargs)
        with self.scope:
            self._filters = filters
            self._in_filters = in_filters
            if isinstance(strides, numeric_types):
                strides = (strides,)*len(kernel_size)
            if isinstance(padding, numeric_types):
                padding = (padding,)*len(kernel_size)
            if isinstance(dilation, numeric_types):
                dilation = (dilation,)*len(kernel_size)
            self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': filters, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout}

            dshape = [0]*(len(kernel_size) + 2)
            dshape[layout.find('N')] = 1
            dshape[layout.find('C')] = in_filters
            wshapes = _infer_weight_shape(symbol.Convolution, dshape, **self._kwargs)
            self.weight = self.params.get('weight', shape=wshapes[1],
                                          init=kernel_initializer)
            if use_bias:
                self.bias = self.params.get('bias', shape=wshapes[2],
                                            init=bias_initializer)

            if activation is not None:
                self.act = Activation(activation, prefix=self.prefix+activation+'_',
                                      params=self.params.subdict(activation+'_'))
            else:
                self.act = None

    def generic_forward(self, F, x, **kwargs):
        self._kwargs.update(kwargs)
        act = F.Convolution(data=x, **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act


class Conv1D(_Conv):
    """1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer with NDArray API,
    provide an `in_filters` argument
    (integers, the number of input channels).


    Parameters
    ----------
    filters: Integer, the dimensionality of the output space
        (i.e. the number output of filters in the convolution).
    kernel_size: An integer or tuple/list of 1 integers, specifying the
        dimensions of the convolution window.
    strides: An integer or tuple/list of 1 integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: An integer or a tuple/list of 1 integers,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: An integer or tuple/list of 1 integers, specifying
        the dilation rate to use for dilated convolution.
    groups: int
        controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout: A string,
        Can be 'NCW', 'NWC', etc.
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively.
    in_filters: int, default 0
        The number of input channels to this layer. Only required when using
        NDArray API.
    activation: Activation function to use
        see mx.sym.Activation.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix
        see Initializer.
    bias_initializer: Initializer for the bias vector
        see Initializer.
    """
    def __init__(self, filters, kernel_size, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCW', in_filters=0, activation=None, use_bias=True,
                 kernel_initializer=None, bias_initializer=None, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)
        assert len(kernel_size) == 1, "kernel_size must be a number or a list of 1 ints"
        super(Conv1D, self).__init__(
            filters, kernel_size, strides, padding, dilation, groups, layout,
            in_filters, activation, use_bias, kernel_initializer, bias_initializer, **kwargs)


class Conv2D(_Conv):
    """2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer with NDArray API,
    provide an `in_filters` argument
    (integers, the number of input channels).


    Parameters
    ----------
    filters: Integer, the dimensionality of the output space
        (i.e. the number output of filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
        dimensions of the convolution window.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: An integer or a tuple/list of 2 integers,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
    groups: int
        controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout: A string,
        Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively.
    in_filters: int, default 0
        The number of input channels to this layer. Only required when using
        NDArray API.
    activation: Activation function to use
        see mx.sym.Activation.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix
        see Initializer.
    bias_initializer: Initializer for the bias vector
        see Initializer.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW', in_filters=0,
                 activation=None, use_bias=True,
                 kernel_initializer=None, bias_initializer=None, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(Conv2D, self).__init__(
            filters, kernel_size, strides, padding, dilation, groups, layout,
            in_filters, activation, use_bias, kernel_initializer, bias_initializer, **kwargs)


class Conv3D(_Conv):
    """3D convolution layer (e.g. spatial convolution over volumes).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer with NDArray API,
    provide an `in_filters` argument
    (integers, the number of input channels).


    Parameters
    ----------
    filters: Integer, the dimensionality of the output space
        (i.e. the number output of filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
        dimensions of the convolution window.
    strides: An integer or tuple/list of 3 integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: An integer or a tuple/list of 3 integers,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: An integer or tuple/list of 3 integers, specifying
        the dilation rate to use for dilated convolution.
    groups: int
        controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout: A string,
        Can be 'NCDHW', 'NDHWC', etc.
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively.
    in_filters: int, default 0
        The number of input channels to this layer. Only required when using
        NDArray API.
    activation: Activation function to use
        see mx.sym.Activation.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix
        see Initializer.
    bias_initializer: Initializer for the bias vector
        see Initializer.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), groups=1, layout='NCDHW', in_filters=0,
                 activation=None, use_bias=True,
                 kernel_initializer=None, bias_initializer=None, **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        super(Conv3D, self).__init__(
            filters, kernel_size, strides, padding, dilation, groups, layout,
            in_filters, activation, use_bias, kernel_initializer, bias_initializer, **kwargs)


class _Pooling(Layer):
    """Abstract class for different pooling layers.
    """
    def __init__(self, pool_size, strides, padding, global_pool, pool_type, **kwargs):
        super(_Pooling, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        if isinstance(strides, numeric_types):
            strides = (strides,)*len(pool_size)
        if isinstance(padding, numeric_types):
            padding = (padding,)*len(pool_size)
        self._kwargs = {
            'kernel': pool_size, 'stride': strides, 'pad': padding,
            'pooling_convention': 'full', 'global_pool': global_pool,
            'pool_type': pool_type}

    def generic_forward(self, F, x):
        return F.Pooling(x, **self._kwargs)


class MaxPool1D(_Pooling):
    """Max pooling operation for temporal data.

    Parameters
    ----------
    pool_size: Integer, size of the max pooling windows.
    strides: Integer, or None. Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
    padding: Integer,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points
    layout: A string,
        Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, and width (time) dimensions
        respectively. padding is applied on W dimension.
    """
    def __init__(self, pool_size=2, strides=None, padding=0, layout='NCW', **kwargs):
        assert layout == 'NCW', "Only supports NCW layout for now"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)
        assert len(pool_size) == 1, "pool_size must be a number or a list of 1 ints"
        super(MaxPool1D, self).__init__(
            pool_size, strides, padding, False, 'max', **kwargs)


class MaxPool2D(_Pooling):
    """Max pooling operation for spatial data.

    Parameters
    ----------
    pool_size: Integer or list/tuple of 2 Integers,
        size of the max pooling windows.
    strides: Integer, list/tuple of 2 Integers, or None.
        Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
    padding: Integer or list/tuple of 2 Integers,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points
    layout: A string,
        Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. padding is applied on 'H' and 'W' dimension.
    """
    def __init__(self, pool_size=(2, 2), strides=None, padding=0, layout='NCHW', **kwargs):
        assert layout == 'NCHW', "Only supports NCHW layout for now"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)*2
        assert len(pool_size) == 2, "pool_size must be a number or a list of 2 ints"
        super(MaxPool2D, self).__init__(
            pool_size, strides, padding, False, 'max', **kwargs)


class MaxPool3D(_Pooling):
    """Max pooling operation for 3D data (spatial or spatio-temporal).

    Parameters
    ----------
    pool_size: Integer or list/tuple of 3 Integers,
        size of the max pooling windows.
    strides: Integer, list/tuple of 3 Integers, or None.
        Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
    padding: Integer or list/tuple of 3 Integers,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points
    layout: A string,
        Can be 'NCDHW', 'NDHWC', etc.
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.
    """
    def __init__(self, pool_size=(2, 2, 2), strides=None, padding=0, layout='NCDHW', **kwargs):
        assert layout == 'NCDHW', "Only supports NCDHW layout for now"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)*3
        assert len(pool_size) == 3, "pool_size must be a number or a list of 3 ints"
        super(MaxPool3D, self).__init__(
            pool_size, strides, padding, False, 'max', **kwargs)


class AvgPool1D(_Pooling):
    """Average pooling operation for temporal data.

    Parameters
    ----------
    pool_size: Integer, size of the max pooling windows.
    strides: Integer, or None. Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
    padding: Integer,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points
    layout: A string,
        Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, and width (time) dimensions
        respectively. padding is applied on W dimension.
    """
    def __init__(self, pool_size=2, strides=None, padding=0, layout='NCW', **kwargs):
        assert layout == 'NCW', "Only supports NCW layout for now"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)
        assert len(pool_size) == 1, "pool_size must be a number or a list of 1 ints"
        super(AvgPool1D, self).__init__(
            pool_size, strides, padding, False, 'avg', **kwargs)


class AvgPool2D(_Pooling):
    """Average pooling operation for spatial data.

    Parameters
    ----------
    pool_size: Integer or list/tuple of 2 Integers,
        size of the max pooling windows.
    strides: Integer, list/tuple of 2 Integers, or None.
        Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
    padding: Integer or list/tuple of 2 Integers,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points
    layout: A string,
        Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. padding is applied on 'H' and 'W' dimension.
    """
    def __init__(self, pool_size=(2, 2), strides=None, padding=0, layout='NCHW', **kwargs):
        assert layout == 'NCHW', "Only supports NCHW layout for now"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)*2
        assert len(pool_size) == 2, "pool_size must be a number or a list of 2 ints"
        super(AvgPool2D, self).__init__(
            pool_size, strides, padding, False, 'avg', **kwargs)


class AvgPool3D(_Pooling):
    """Average pooling operation for 3D data (spatial or spatio-temporal).

    Parameters
    ----------
    pool_size: Integer or list/tuple of 3 Integers,
        size of the max pooling windows.
    strides: Integer, list/tuple of 3 Integers, or None.
        Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
    padding: Integer or list/tuple of 3 Integers,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points
    layout: A string,
        Can be 'NCDHW', 'NDHWC', etc.
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.
    """
    def __init__(self, pool_size=(2, 2, 2), strides=None, padding=0, layout='NCDHW', **kwargs):
        assert layout == 'NCDHW', "Only supports NCDHW layout for now"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)*3
        assert len(pool_size) == 3, "pool_size must be a number or a list of 3 ints"
        super(AvgPool3D, self).__init__(
            pool_size, strides, padding, False, 'avg', **kwargs)


class GlobalMaxPool1D(_Pooling):
    """Global max pooling operation for temporal data.
    """
    def __init__(self, layout='NCW', **kwargs):
        assert layout == 'NCW', "Only supports NCW layout for now"
        super(GlobalMaxPool1D, self).__init__(
            (1,), None, 0, True, 'max', **kwargs)


class GlobalMaxPool2D(_Pooling):
    """Global max pooling operation for spatial data.
    """
    def __init__(self, layout='NCHW', **kwargs):
        assert layout == 'NCHW', "Only supports NCW layout for now"
        super(GlobalMaxPool2D, self).__init__(
            (1, 1), None, 0, True, 'max', **kwargs)

class GlobalMaxPool3D(_Pooling):
    """Global max pooling operation for 3D data.
    """
    def __init__(self, layout='NCDHW', **kwargs):
        assert layout == 'NCDHW', "Only supports NCW layout for now"
        super(GlobalMaxPool3D, self).__init__(
            (1, 1, 1), None, 0, True, 'max', **kwargs)


class GlobalAvgPool1D(_Pooling):
    """Global average pooling operation for temporal data.
    """
    def __init__(self, layout='NCW', **kwargs):
        assert layout == 'NCW', "Only supports NCW layout for now"
        super(GlobalAvgPool1D, self).__init__(
            (1,), None, 0, True, 'avg', **kwargs)


class GlobalAvgPool2D(_Pooling):
    """Global average pooling operation for spatial data.
    """
    def __init__(self, layout='NCHW', **kwargs):
        assert layout == 'NCHW', "Only supports NCW layout for now"
        super(GlobalAvgPool2D, self).__init__(
            (1, 1), None, 0, True, 'avg', **kwargs)


class GlobalAvgPool3D(_Pooling):
    """Global max pooling operation for 3D data.
    """
    def __init__(self, layout='NCDHW', **kwargs):
        assert layout == 'NCDHW', "Only supports NCW layout for now"
        super(GlobalAvgPool3D, self).__init__(
            (1, 1, 1), None, 0, True, 'avg', **kwargs)
