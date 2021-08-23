# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ, too-many-lines
"""Convolutional neural network layers."""
__all__ = ['Conv1D', 'Conv2D', 'Conv3D',
           'Conv1DTranspose', 'Conv2DTranspose', 'Conv3DTranspose',
           'MaxPool1D', 'MaxPool2D', 'MaxPool3D',
           'AvgPool1D', 'AvgPool2D', 'AvgPool3D',
           'GlobalMaxPool1D', 'GlobalMaxPool2D', 'GlobalMaxPool3D',
           'GlobalAvgPool1D', 'GlobalAvgPool2D', 'GlobalAvgPool3D',
           'ReflectionPad2D', 'DeformableConvolution', 'ModulatedDeformableConvolution',
           'PixelShuffle1D', 'PixelShuffle2D', 'PixelShuffle3D']

from ..block import HybridBlock
from ..parameter import Parameter
from ... import np, npx
from ...base import numeric_types
from .activations import Activation
from ...util import use_np

@use_np
class _Conv(HybridBlock):
    """Abstract nD convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is `True`, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of n ints
        Specifies the dimensions of the convolution window.
    strides: int or tuple/list of n ints,
        Specifies the strides of the convolution.
    padding : int or tuple/list of n ints,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: int or tuple/list of n ints,
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str,
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.npx.activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    """
    def __init__(self, channels, kernel_size, strides, padding, dilation,
                 groups, layout, in_channels=0, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 op_name='convolution', adj=None):
        super(_Conv, self).__init__()
        self._channels = channels
        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._layout = layout
        self._groups = groups
        if isinstance(strides, numeric_types):
            strides = (strides,)*len(kernel_size)
        if isinstance(padding, numeric_types):
            padding = (padding,)*len(kernel_size)
        if isinstance(dilation, numeric_types):
            dilation = (dilation,)*len(kernel_size)
        self._op_name = op_name
        self._kwargs = {
            'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
            'pad': padding, 'num_filter': channels, 'num_group': groups,
            'no_bias': not use_bias, 'layout': layout}
        if adj is not None:
            self._kwargs['adj'] = adj

        self.weight = Parameter('weight', shape=self.pre_infer(),
                                init=weight_initializer,
                                allow_deferred_init=True)
        if use_bias:
            self.bias = Parameter('bias', shape=(channels,),
                                  init=bias_initializer,
                                  allow_deferred_init=True)
        else:
            self.bias = None

        if activation is not None:
            self.act = Activation(activation)
        else:
            self.act = None

    def forward(self, x):
        ctx = x.ctx
        if self.bias is None:
            act = getattr(npx, self._op_name)(x, self.weight.data(ctx), **self._kwargs)
        else:
            act = getattr(npx, self._op_name)(x, self.weight.data(ctx), self.bias.data(ctx),
                                              **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act

    def pre_infer(self):
        """
        Pre-infer the shape of weight parameter based on kernel size, group size and channels
        """
        wshape = [-1]*(len(self._kernel_size) + 2)
        if self._op_name == "convolution":
            if len(self._kernel_size) == 1:
                wshape[self._layout.find('N')] = self._channels // self._groups
                wshape[self._layout.find('W')] = self._kernel_size[0]
                wshape[0] *= self._groups
            elif len(self._kernel_size) == 2:
                wshape[self._layout.find('N')] = self._channels // self._groups
                wshape[self._layout.find('H')] = self._kernel_size[0]
                wshape[self._layout.find('W')] = self._kernel_size[1]
                wshape[0] *= self._groups
            else:
                assert len(self._kernel_size) == 3, "kernel_size must be 1, 2 or 3"
                wshape[self._layout.find('N')] = self._channels // self._groups
                wshape[self._layout.find('D')] = self._kernel_size[0]
                wshape[self._layout.find('H')] = self._kernel_size[1]
                wshape[self._layout.find('W')] = self._kernel_size[2]
                wshape[0] *= self._groups
        else:
            assert self._op_name == "deconvolution", \
                "Only support operator name with convolution and deconvolution"
            if len(self._kernel_size) == 1:
                wshape[self._layout.find('C')] = self._channels // self._groups
                wshape[self._layout.find('W')] = self._kernel_size[0]
            elif len(self._kernel_size) == 2:
                wshape[self._layout.find('C')] = self._channels // self._groups
                wshape[self._layout.find('H')] = self._kernel_size[0]
                wshape[self._layout.find('W')] = self._kernel_size[1]
            else:
                assert len(self._kernel_size) == 3, "kernel_size must be 1, 2 or 3"
                wshape[self._layout.find('C')] = self._channels // self._groups
                wshape[self._layout.find('D')] = self._kernel_size[0]
                wshape[self._layout.find('H')] = self._kernel_size[1]
                wshape[self._layout.find('W')] = self._kernel_size[2]
        return tuple(wshape)

    def infer_shape(self, x):
        dshape1 = x.shape[self._layout.find('C')]
        wshape = self.weight.shape
        if self._op_name == "convolution":
            wshape_list = list(wshape)
            wshape_list[self._layout.find('C')] = dshape1 // self._groups
        else:
            assert self._op_name == "deconvolution", \
                "Only support operator name with convolution and deconvolution"
            wshape_list = list(wshape)
            wshape_list[self._layout.find('N')] = dshape1
        self.weight.shape = tuple(wshape_list)

    def _alias(self):
        return 'conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.weight.shape
        if 'Transpose' in self.__class__.__name__:
            mapping = '{1} -> {0}'
        else:
            mapping = '{0} -> {1}'
        return s.format(name=self.__class__.__name__,
                        mapping=mapping.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)


class Conv1D(_Conv):
    r"""1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.


    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 1 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 1 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 1 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 1 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout: str, default 'NCW'
        Dimension ordering of data and weight. Only supports 'NCW' layout for now.
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. Convolution is applied on the 'W' dimension.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.npx.activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Inputs:
        - **data**: 3D input tensor with shape `(batch_size, in_channels, width)`
          when `layout` is `NCW`. For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 3D output tensor with shape `(batch_size, channels, out_width)`
          when `layout` is `NCW`. out_width is calculated as::

              out_width = floor((width+2*padding-dilation*(kernel_size-1)-1)/stride)+1
    """
    def __init__(self, channels, kernel_size, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCW', activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        assert layout == 'NCW', "Only supports 'NCW' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)
        assert len(kernel_size) == 1, "kernel_size must be a number or a list of 1 ints"
        op_name = 'convolution'
        super(Conv1D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer,
            op_name, **kwargs)


class Conv2D(_Conv):
    r"""2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 2 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 2 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 2 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 2 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCHW'
        Dimension ordering of data and weight. Only supports 'NCHW' and 'NHWC'
        layout for now. 'N', 'C', 'H', 'W' stands for batch, channel, height,
        and width dimensions respectively. Convolution is applied on the 'H' and
        'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.npx.activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
          out_height and out_width are calculated as::

              out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
              out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
    """
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        op_name = 'convolution'
        super(Conv2D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer,
            op_name, **kwargs)


class Conv3D(_Conv):
    """3D convolution layer (e.g. spatial convolution over volumes).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is `True`,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 3 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 3 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 3 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 3 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCDHW'
        Dimension ordering of data and weight. Only supports 'NCDHW' and 'NDHWC'
        layout for now. 'N', 'C', 'H', 'W', 'D' stands for batch, channel, height,
        width and depth dimensions respectively. Convolution is applied on the 'D',
        'H' and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.npx.activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, in_channels, depth, height, width)` when `layout` is `NCDHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, channels, out_depth, out_height, out_width)` when `layout` is `NCDHW`.
          out_depth, out_height and out_width are calculated as::

              out_depth = floor((depth+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
              out_height = floor((height+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
              out_width = floor((width+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2])+1
    """
    def __init__(self, channels, kernel_size, strides=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), groups=1, layout='NCDHW', activation=None,
                 use_bias=True, weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        assert layout in ('NCDHW', 'NDHWC'), "Only supports 'NCDHW' and 'NDHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        op_name = 'convolution'
        super(Conv3D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer,
            op_name, **kwargs)


class Conv1DTranspose(_Conv):
    """Transposed 1D convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 1 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 1 int
        Specify the strides of the convolution.
    padding : int or a tuple/list of 1 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    output_padding: int or a tuple/list of 1 int
        Controls the amount of implicit zero-paddings on both sides of the
        output for output_padding number of points for each dimension.
    dilation : int or tuple/list of 1 int
        Controls the spacing between the kernel points; also known as the
        a trous algorithm
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCW'
        Dimension ordering of data and weight. Only supports 'NCW' layout for now.
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. Convolution is applied on the 'W' dimension.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.npx.activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Inputs:
        - **data**: 3D input tensor with shape `(batch_size, in_channels, width)`
          when `layout` is `NCW`. For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 3D output tensor with shape `(batch_size, channels, out_width)`
          when `layout` is `NCW`. out_width is calculated as::

              out_width = (width-1)*strides-2*padding+kernel_size+output_padding
    """
    def __init__(self, channels, kernel_size, strides=1, padding=0, output_padding=0,
                 dilation=1, groups=1, layout='NCW', activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        assert layout == 'NCW', "Only supports 'NCW' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)
        if isinstance(output_padding, numeric_types):
            output_padding = (output_padding,)
        assert len(kernel_size) == 1, "kernel_size must be a number or a list of 1 ints"
        assert len(output_padding) == 1, "output_padding must be a number or a list of 1 ints"
        op_name = 'deconvolution'
        super(Conv1DTranspose, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer,
            bias_initializer, op_name=op_name, adj=output_padding, **kwargs)
        self.outpad = output_padding


class Conv2DTranspose(_Conv):
    """Transposed 2D convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.


    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 2 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 2 int
        Specify the strides of the convolution.
    padding : int or a tuple/list of 2 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    output_padding: int or a tuple/list of 2 int
        Controls the amount of implicit zero-paddings on both sides of the
        output for output_padding number of points for each dimension.
    dilation : int or tuple/list of 2 int
        Controls the spacing between the kernel points; also known as the
        a trous algorithm
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCHW'
        Dimension ordering of data and weight. Only supports 'NCHW' and 'NHWC'
        layout for now. 'N', 'C', 'H', 'W' stands for batch, channel, height,
        and width dimensions respectively. Convolution is applied on the 'H' and
        'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.npx.activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
          out_height and out_width are calculated as::

              out_height = (height-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
              out_width = (width-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]
    """
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 output_padding=(0, 0), dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        if isinstance(output_padding, numeric_types):
            output_padding = (output_padding,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        assert len(output_padding) == 2, "output_padding must be a number or a list of 2 ints"
        op_name = 'deconvolution'
        super(Conv2DTranspose, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer,
            bias_initializer, op_name=op_name, adj=output_padding, **kwargs)
        self.outpad = output_padding


class Conv3DTranspose(_Conv):
    """Transposed 3D convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.


    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 3 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 3 int
        Specify the strides of the convolution.
    padding : int or a tuple/list of 3 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    output_padding: int or a tuple/list of 3 int
        Controls the amount of implicit zero-paddings on both sides of the
        output for output_padding number of points for each dimension.
    dilation : int or tuple/list of 3 int
        Controls the spacing between the kernel points; also known as the
        a trous algorithm.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCDHW'
        Dimension ordering of data and weight. Only supports 'NCDHW' and 'NDHWC'
        layout for now. 'N', 'C', 'H', 'W', 'D' stands for batch, channel, height,
        width and depth dimensions respectively. Convolution is applied on the 'D',
        'H' and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.npx.activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, in_channels, depth, height, width)` when `layout` is `NCDHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, channels, out_depth, out_height, out_width)` when `layout` is `NCDHW`.
          out_depth, out_height and out_width are calculated as::

            out_depth = (depth-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
            out_height = (height-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]
            out_width = (width-1)*strides[2]-2*padding[2]+kernel_size[2]+output_padding[2]
    """
    def __init__(self, channels, kernel_size, strides=(1, 1, 1), padding=(0, 0, 0),
                 output_padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, layout='NCDHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCDHW', 'NDHWC'), "Only supports 'NCDHW' and 'NDHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        if isinstance(output_padding, numeric_types):
            output_padding = (output_padding,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        assert len(output_padding) == 3, "output_padding must be a number or a list of 3 ints"
        op_name = 'deconvolution'
        super(Conv3DTranspose, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer,
            op_name=op_name, adj=output_padding, **kwargs)
        self.outpad = output_padding


@use_np
class _Pooling(HybridBlock):
    """Abstract class for different pooling layers."""
    def __init__(self, pool_size, strides, padding, ceil_mode, global_pool,
                 pool_type, layout, count_include_pad=None, **kwargs):
        super(_Pooling, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        if isinstance(strides, numeric_types):
            strides = (strides,)*len(pool_size)
        if isinstance(padding, numeric_types):
            padding = (padding,)*len(pool_size)
        self._kwargs = {
            'kernel': pool_size, 'stride': strides, 'pad': padding,
            'global_pool': global_pool, 'pool_type': pool_type,
            'layout': layout,
            'pooling_convention': 'full' if ceil_mode else 'valid'}
        if count_include_pad is not None:
            self._kwargs['count_include_pad'] = count_include_pad

    def _alias(self):
        return 'pool'

    def forward(self, x):
        return npx.pooling(x, name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{name}(size={kernel}, stride={stride}, padding={pad}, ceil_mode={ceil_mode}'
        s += ', global_pool={global_pool}, pool_type={pool_type}, layout={layout})'
        return s.format(name=self.__class__.__name__,
                        ceil_mode=self._kwargs['pooling_convention'] == 'full',
                        **self._kwargs)


class MaxPool1D(_Pooling):
    """Max pooling operation for one dimensional data.


    Parameters
    ----------
    pool_size: int
        Size of the max pooling windows.
    strides: int, or None
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCW'
        Dimension ordering of data and out ('NCW' or 'NWC').
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. Pooling is applied on the W dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Inputs:
        - **data**: 3D input tensor with shape `(batch_size, in_channels, width)`
          when `layout` is `NCW`. For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 3D output tensor with shape `(batch_size, channels, out_width)`
          when `layout` is `NCW`. out_width is calculated as::

              out_width = floor((width+2*padding-pool_size)/strides)+1

          When `ceil_mode` is `True`, ceil will be used instead of floor in this
          equation.
    """
    def __init__(self, pool_size=2, strides=None, padding=0, layout='NCW',
                 ceil_mode=False, **kwargs):
        assert layout in ('NCW', 'NWC'),\
            "Only NCW and NWC layouts are valid for 1D Pooling"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)
        assert len(pool_size) == 1, "pool_size must be a number or a list of 1 ints"
        super(MaxPool1D, self).__init__(
            pool_size, strides, padding, ceil_mode, False, 'max', layout, **kwargs)


class MaxPool2D(_Pooling):
    """Max pooling operation for two dimensional (spatial) data.


    Parameters
    ----------
    pool_size: int or list/tuple of 2 ints,
        Size of the max pooling windows.
    strides: int, list/tuple of 2 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 2 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCHW'
        Dimension ordering of data and out ('NCHW' or 'NHWC').
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. padding is applied on 'H' and 'W' dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
          out_height and out_width are calculated as::

              out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
              out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1

          When `ceil_mode` is `True`, ceil will be used instead of floor in this
          equation.
    """
    def __init__(self, pool_size=(2, 2), strides=None, padding=0, layout='NCHW',
                 ceil_mode=False, **kwargs):
        assert layout in ('NCHW', 'NHWC'),\
            "Only NCHW and NHWC layouts are valid for 2D Pooling"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)*2
        assert len(pool_size) == 2, "pool_size must be a number or a list of 2 ints"
        super(MaxPool2D, self).__init__(
            pool_size, strides, padding, ceil_mode, False, 'max', layout, **kwargs)


class MaxPool3D(_Pooling):
    """Max pooling operation for 3D data (spatial or spatio-temporal).


    Parameters
    ----------
    pool_size: int or list/tuple of 3 ints,
        Size of the max pooling windows.
    strides: int, list/tuple of 3 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 3 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCDHW'
        Dimension ordering of data and out ('NCDHW' or 'NDHWC').
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, in_channels, depth, height, width)` when `layout` is `NCW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, channels, out_depth, out_height, out_width)` when `layout` is `NCDHW`.
          out_depth, out_height and out_width are calculated as::

              out_depth = floor((depth+2*padding[0]-pool_size[0])/strides[0])+1
              out_height = floor((height+2*padding[1]-pool_size[1])/strides[1])+1
              out_width = floor((width+2*padding[2]-pool_size[2])/strides[2])+1

          When `ceil_mode` is `True`, ceil will be used instead of floor in this
          equation.
    """
    def __init__(self, pool_size=(2, 2, 2), strides=None, padding=0,
                 ceil_mode=False, layout='NCDHW', **kwargs):
        assert layout in ('NCDHW', 'NDHWC'),\
            "Only NCDHW and NDHWC layouts are valid for 3D Pooling"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)*3
        assert len(pool_size) == 3, "pool_size must be a number or a list of 3 ints"
        super(MaxPool3D, self).__init__(
            pool_size, strides, padding, ceil_mode, False, 'max', layout, **kwargs)


class AvgPool1D(_Pooling):
    """Average pooling operation for temporal data.

    Parameters
    ----------
    pool_size: int
        Size of the average pooling windows.
    strides: int, or None
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCW'
        Dimension ordering of data and out ('NCW' or 'NWC').
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. padding is applied on 'W' dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.
    count_include_pad : bool, default True
        When 'False', will exclude padding elements when computing the average value.


    Inputs:
        - **data**: 3D input tensor with shape `(batch_size, in_channels, width)`
          when `layout` is `NCW`. For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 3D output tensor with shape `(batch_size, channels, out_width)`
          when `layout` is `NCW`. out_width is calculated as::

              out_width = floor((width+2*padding-pool_size)/strides)+1

          When `ceil_mode` is `True`, ceil will be used instead of floor in this
          equation.
    """
    def __init__(self, pool_size=2, strides=None, padding=0, layout='NCW',
                 ceil_mode=False, count_include_pad=True, **kwargs):
        assert layout in ('NCW', 'NWC'),\
            "Only NCW and NWC layouts are valid for 1D Pooling"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)
        assert len(pool_size) == 1, "pool_size must be a number or a list of 1 ints"
        super(AvgPool1D, self).__init__(
            pool_size, strides, padding, ceil_mode, False, 'avg', layout, count_include_pad,
            **kwargs)


class AvgPool2D(_Pooling):
    """Average pooling operation for spatial data.

    Parameters
    ----------
    pool_size: int or list/tuple of 2 ints,
        Size of the average pooling windows.
    strides: int, list/tuple of 2 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 2 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCHW'
        Dimension ordering of data and out ('NCHW' or 'NHWC').
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. padding is applied on 'H' and 'W' dimension.
    ceil_mode : bool, default False
        When True, will use ceil instead of floor to compute the output shape.
    count_include_pad : bool, default True
        When 'False', will exclude padding elements when computing the average value.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
          out_height and out_width are calculated as::

              out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
              out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1

          When `ceil_mode` is `True`, ceil will be used instead of floor in this
          equation.
    """
    def __init__(self, pool_size=(2, 2), strides=None, padding=0,
                 ceil_mode=False, layout='NCHW', count_include_pad=True, **kwargs):
        assert layout in ('NCHW', 'NHWC'),\
            "Only NCHW and NHWC layouts are valid for 2D Pooling"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)*2
        assert len(pool_size) == 2, "pool_size must be a number or a list of 2 ints"
        super(AvgPool2D, self).__init__(
            pool_size, strides, padding, ceil_mode, False, 'avg', layout, count_include_pad,
            **kwargs)


class AvgPool3D(_Pooling):
    """Average pooling operation for 3D data (spatial or spatio-temporal).

    Parameters
    ----------
    pool_size: int or list/tuple of 3 ints,
        Size of the average pooling windows.
    strides: int, list/tuple of 3 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 3 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCDHW'
        Dimension ordering of data and out ('NCDHW' or 'NDHWC').
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.
    ceil_mode : bool, default False
        When True, will use ceil instead of floor to compute the output shape.
    count_include_pad : bool, default True
        When 'False', will exclude padding elements when computing the average value.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, in_channels, depth, height, width)` when `layout` is `NCDHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, channels, out_depth, out_height, out_width)` when `layout` is `NCDHW`.
          out_depth, out_height and out_width are calculated as::

              out_depth = floor((depth+2*padding[0]-pool_size[0])/strides[0])+1
              out_height = floor((height+2*padding[1]-pool_size[1])/strides[1])+1
              out_width = floor((width+2*padding[2]-pool_size[2])/strides[2])+1

          When `ceil_mode` is `True,` ceil will be used instead of floor in this
          equation.
    """
    def __init__(self, pool_size=(2, 2, 2), strides=None, padding=0,
                 ceil_mode=False, layout='NCDHW', count_include_pad=True, **kwargs):
        assert layout in ('NCDHW', 'NDHWC'),\
            "Only NCDHW and NDHWC layouts are valid for 3D Pooling"
        if isinstance(pool_size, numeric_types):
            pool_size = (pool_size,)*3
        assert len(pool_size) == 3, "pool_size must be a number or a list of 3 ints"
        super(AvgPool3D, self).__init__(
            pool_size, strides, padding, ceil_mode, False, 'avg', layout, count_include_pad,
            **kwargs)


class GlobalMaxPool1D(_Pooling):
    """Gloabl max pooling operation for one dimensional (temporal) data.


    Parameters
    ----------
    layout : str, default 'NCW'
        Dimension ordering of data and out ('NCW' or 'NWC').
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. Pooling is applied on the W dimension.


    Inputs:
        - **data**: 3D input tensor with shape `(batch_size, in_channels, width)`
          when `layout` is `NCW`. For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 3D output tensor with shape `(batch_size, channels, 1)`
          when `layout` is `NCW`.
    """
    def __init__(self, layout='NCW', **kwargs):
        assert layout in ('NCW', 'NWC'),\
            "Only NCW and NWC layouts are valid for 1D Pooling"
        super(GlobalMaxPool1D, self).__init__(
            (1,), None, 0, True, True, 'max', layout, **kwargs)


class GlobalMaxPool2D(_Pooling):
    """Global max pooling operation for two dimensional (spatial) data.


    Parameters
    ----------
    layout : str, default 'NCHW'
        Dimension ordering of data and out ('NCHW' or 'NHWC').
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. padding is applied on 'H' and 'W' dimension.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, 1, 1)` when `layout` is `NCHW`.
    """
    def __init__(self, layout='NCHW', **kwargs):
        assert layout in ('NCHW', 'NHWC'),\
            "Only NCHW and NHWC layouts are valid for 2D Pooling"
        super(GlobalMaxPool2D, self).__init__(
            (1, 1), None, 0, True, True, 'max', layout, **kwargs)


class GlobalMaxPool3D(_Pooling):
    """Global max pooling operation for 3D data (spatial or spatio-temporal).


    Parameters
    ----------
    layout : str, default 'NCDHW'
        Dimension ordering of data and out ('NCDHW' or 'NDHWC').
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, in_channels, depth, height, width)` when `layout` is `NCW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, channels, 1, 1, 1)` when `layout` is `NCDHW`.
    """
    def __init__(self, layout='NCDHW', **kwargs):
        assert layout in ('NCDHW', 'NDHWC'),\
            "Only NCDHW and NDHWC layouts are valid for 3D Pooling"
        super(GlobalMaxPool3D, self).__init__(
            (1, 1, 1), None, 0, True, True, 'max', layout, **kwargs)


class GlobalAvgPool1D(_Pooling):
    """Global average pooling operation for temporal data.

    Parameters
    ----------
    layout : str, default 'NCW'
        Dimension ordering of data and out ('NCW' or 'NWC').
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. padding is applied on 'W' dimension.


    Inputs:
        - **data**: 3D input tensor with shape `(batch_size, in_channels, width)`
          when `layout` is `NCW`. For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 3D output tensor with shape `(batch_size, channels, 1)`.
    """
    def __init__(self, layout='NCW', **kwargs):
        assert layout in ('NCW', 'NWC'),\
            "Only NCW and NWC layouts are valid for 1D Pooling"
        super(GlobalAvgPool1D, self).__init__(
            (1,), None, 0, True, True, 'avg', layout, **kwargs)


class GlobalAvgPool2D(_Pooling):
    """Global average pooling operation for spatial data.

    Parameters
    ----------
    layout : str, default 'NCHW'
        Dimension ordering of data and out ('NCHW' or 'NHWC').
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, 1, 1)` when `layout` is `NCHW`.
    """
    def __init__(self, layout='NCHW', **kwargs):
        assert layout in ('NCHW', 'NHWC'),\
            "Only NCHW and NHWC layouts are valid for 2D Pooling"
        super(GlobalAvgPool2D, self).__init__(
            (1, 1), None, 0, True, True, 'avg', layout, **kwargs)


class GlobalAvgPool3D(_Pooling):
    """Global average pooling operation for 3D data (spatial or spatio-temporal).

    Parameters
    ----------
    layout : str, default 'NCDHW'
        Dimension ordering of data and out ('NCDHW' or 'NDHWC').
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, in_channels, depth, height, width)` when `layout` is `NCDHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, channels, 1, 1, 1)` when `layout` is `NCDHW`.
    """
    def __init__(self, layout='NCDHW', **kwargs):
        assert layout in ('NCDHW', 'NDHWC'),\
            "Only NCDHW and NDHWC layouts are valid for 3D Pooling"
        super(GlobalAvgPool3D, self).__init__(
            (1, 1, 1), None, 0, True, True, 'avg', layout, **kwargs)


@use_np
class ReflectionPad2D(HybridBlock):
    r"""Pads the input tensor using the reflection of the input boundary.

    Parameters
    ----------
    padding: int
        An integer padding size


    Inputs:
        - **data**: input tensor with the shape :math:`(N, C, H_{in}, W_{in})`.

    Outputs:
        - **out**: output tensor with the shape :math:`(N, C, H_{out}, W_{out})`, where

          .. math::

            H_{out} = H_{in} + 2 \cdot padding

            W_{out} = W_{in} + 2 \cdot padding


    Examples
    --------
    >>> m = nn.ReflectionPad2D(3)
    >>> input = mx.np.random.normal(size=(16, 3, 224, 224))
    >>> output = m(input)
    """
    def __init__(self, padding=0, **kwargs):
        super(ReflectionPad2D, self).__init__(**kwargs)
        if isinstance(padding, numeric_types):
            padding = (0, 0, 0, 0, padding, padding, padding, padding)
        assert(len(padding) == 8)
        self._padding = padding

    def forward(self, x):
        """
        Use pad operator in numpy extension module,
        which has backward support for reflect mode
        """
        return npx.pad(x, mode='reflect', pad_width=self._padding)


@use_np
class DeformableConvolution(HybridBlock):
    """2-D Deformable Convolution v_1 (Dai, 2017).
    Normal Convolution uses sampling points in a regular grid, while the sampling
    points of Deformablem Convolution can be offset. The offset is learned with a
    separate convolution layer during the training. Both the convolution layer for
    generating the output features and the offsets are included in this gluon layer.

    Parameters
    ----------
    channels : int,
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the strides of the convolution.
    padding : int or tuple/list of 2 ints, (Default value = (0,0))
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.
    dilation : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the dilation rate to use for dilated convolution.
    groups : int, (Default value = 1)
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    num_deformable_group : int, (Default value = 1)
        Number of deformable group partitions.
    layout : str, (Default value = NCHW)
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    use_bias : bool, (Default value = True)
        Whether the layer for generating the output features uses a bias vector.
    in_channels : int, (Default value = 0)
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and input channels will be inferred from the shape of input data.
    activation : str, (Default value = None)
        Activation function to use. See :func:`~mxnet.npx.activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    weight_initializer : str or `Initializer`, (Default value = None)
        Initializer for the `weight` weights matrix for the convolution layer
        for generating the output features.
    bias_initializer : str or `Initializer`, (Default value = zeros)
        Initializer for the bias vector for the convolution layer
        for generating the output features.
    offset_weight_initializer : str or `Initializer`, (Default value = zeros)
        Initializer for the `weight` weights matrix for the convolution layer
        for generating the offset.
    offset_bias_initializer : str or `Initializer`, (Default value = zeros),
        Initializer for the bias vector for the convolution layer
        for generating the offset.
    offset_use_bias: bool, (Default value = True)
        Whether the layer for generating the offset uses a bias vector.

    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
          out_height and out_width are calculated as::

              out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
              out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
    """

    def __init__(self, channels, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,
                 num_deformable_group=1, layout='NCHW', use_bias=True, in_channels=0, activation=None,
                 weight_initializer=None, bias_initializer='zeros',
                 offset_weight_initializer='zeros', offset_bias_initializer='zeros', offset_use_bias=True,
                 op_name='DeformableConvolution', adj=None):
        super(DeformableConvolution, self).__init__()
        self._channels = channels
        self._in_channels = in_channels

        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,) * 2
        if isinstance(strides, numeric_types):
            strides = (strides,) * len(kernel_size)
        if isinstance(padding, numeric_types):
            padding = (padding,) * len(kernel_size)
        if isinstance(dilation, numeric_types):
            dilation = (dilation,) * len(kernel_size)
        self._op_name = op_name
        self._kernel_size = kernel_size
        self._layout = layout
        self._groups = groups

        offset_channels = 2 * kernel_size[0] * kernel_size[1] * num_deformable_group
        self._offset_channels = offset_channels
        self._kwargs_offset = {
            'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
            'pad': padding, 'num_filter': offset_channels, 'num_group': groups,
            'no_bias': not offset_use_bias, 'layout': layout}

        self._kwargs_deformable_conv = {
            'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
            'pad': padding, 'num_filter': channels, 'num_group': groups,
            'num_deformable_group': num_deformable_group,
            'no_bias': not use_bias, 'layout': layout}

        if adj:
            self._kwargs_offset['adj'] = adj
            self._kwargs_deformable_conv['adj'] = adj

        self.offset_weight = Parameter('offset_weight', shape=self.pre_infer_offset_weight(),
                                       init=offset_weight_initializer,
                                       allow_deferred_init=True)

        if offset_use_bias:
            self.offset_bias = Parameter('offset_bias', shape=(offset_channels,),
                                         init=offset_bias_initializer,
                                         allow_deferred_init=True)
        else:
            self.offset_bias = None

        self.deformable_conv_weight = Parameter('deformable_conv_weight',
                                                shape=self.pre_infer_weight(),
                                                init=weight_initializer,
                                                allow_deferred_init=True)

        if use_bias:
            self.deformable_conv_bias = Parameter('deformable_conv_bias', shape=(channels,),
                                                  init=bias_initializer,
                                                  allow_deferred_init=True)
        else:
            self.deformable_conv_bias = None

        if activation:
            self.act = Activation(activation)
        else:
            self.act = None

    def forward(self, x):
        ctx = x.ctx
        if self.offset_bias is None:
            offset = npx.convolution(x, self.offset_weight.data(ctx), cudnn_off=True, **self._kwargs_offset)
        else:
            offset = npx.convolution(x, self.offset_weight.data(ctx), self.offset_bias.data(ctx),
                                     cudnn_off=True, **self._kwargs_offset)

        if self.deformable_conv_bias is None:
            act = npx.deformable_convolution(data=x, offset=offset,
                                             weight=self.deformable_conv_weight.data(ctx),
                                             name='fwd', **self._kwargs_deformable_conv)
        else:
            act = npx.deformable_convolution(data=x, offset=offset,
                                             weight=self.deformable_conv_weight.data(ctx),
                                             bias=self.deformable_conv_bias.data(ctx), name='fwd',
                                             **self._kwargs_deformable_conv)

        if self.act:
            act = self.act(act)
        return act


    def pre_infer_offset_weight(self):
        """
        Pre-infer the shape of offsite weight parameter based on kernel size,
        group size and offset channels
        """
        wshape = [-1]*(len(self._kernel_size) + 2)
        wshape[self._layout.find('N')] = self._offset_channels // self._groups
        wshape[self._layout.find('H')] = self._kernel_size[0]
        wshape[self._layout.find('W')] = self._kernel_size[1]
        wshape[0] *= self._groups
        return tuple(wshape)

    def pre_infer_weight(self):
        """
        Pre-infer the shape of weight parameter based on kernel size, group size and channels
        """
        wshape = [-1]*(len(self._kernel_size) + 2)
        wshape[self._layout.find('N')] = self._channels // self._groups
        wshape[self._layout.find('H')] = self._kernel_size[0]
        wshape[self._layout.find('W')] = self._kernel_size[1]
        wshape[0] *= self._groups
        return tuple(wshape)

    def infer_shape(self, x):
        dshape1 = x.shape[self._layout.find('C')]
        wshape = self.deformable_conv_weight.shape
        wshape_offset = self.offset_weight.shape
        wshape_list = list(wshape)
        wshape_offset_list = list(wshape_offset)
        wshape_list[self._layout.find('C')] = dshape1 // self._groups
        wshape_offset_list[self._layout.find('C')] = dshape1 // self._groups
        self.deformable_conv_weight.shape = tuple(wshape_list)
        self.offset_weight.shape = tuple(wshape_offset_list)

    def _alias(self):
        return 'deformable_conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs_deformable_conv['kernel'])
        if self._kwargs_deformable_conv['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs_deformable_conv['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs_deformable_conv['num_group'] != 1:
            s += ', groups={num_group}'
        if self.deformable_conv_bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.deformable_conv_weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs_deformable_conv)


@use_np
class ModulatedDeformableConvolution(HybridBlock):
    """2-D Deformable Convolution v2 (Dai, 2018).

    The modulated deformable convolution operation is described in https://arxiv.org/abs/1811.11168

    Parameters
    ----------
    channels : int,
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the strides of the convolution.
    padding : int or tuple/list of 2 ints, (Default value = (0,0))
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.
    dilation : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the dilation rate to use for dilated convolution.
    groups : int, (Default value = 1)
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    num_deformable_group : int, (Default value = 1)
        Number of deformable group partitions.
    layout : str, (Default value = NCHW)
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    use_bias : bool, (Default value = True)
        Whether the layer for generating the output features uses a bias vector.
    in_channels : int, (Default value = 0)
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and input channels will be inferred from the shape of input data.
    activation : str, (Default value = None)
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    weight_initializer : str or `Initializer`, (Default value = None)
        Initializer for the `weight` weights matrix for the convolution layer
        for generating the output features.
    bias_initializer : str or `Initializer`, (Default value = zeros)
        Initializer for the bias vector for the convolution layer
        for generating the output features.
    offset_weight_initializer : str or `Initializer`, (Default value = zeros)
        Initializer for the `weight` weights matrix for the convolution layer
        for generating the offset.
    offset_bias_initializer : str or `Initializer`, (Default value = zeros),
        Initializer for the bias vector for the convolution layer
        for generating the offset.
    offset_use_bias: bool, (Default value = True)
        Whether the layer for generating the offset uses a bias vector.

    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
          out_height and out_width are calculated as::

              out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
              out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
    """

    def __init__(self, channels, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,
                 num_deformable_group=1, layout='NCHW', use_bias=True, in_channels=0, activation=None,
                 weight_initializer=None, bias_initializer='zeros',
                 offset_weight_initializer='zeros', offset_bias_initializer='zeros', offset_use_bias=True,
                 op_name='ModulatedDeformableConvolution', adj=None):
        super(ModulatedDeformableConvolution, self).__init__()
        self._channels = channels
        self._in_channels = in_channels

        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,) * 2
        if isinstance(strides, numeric_types):
            strides = (strides,) * len(kernel_size)
        if isinstance(padding, numeric_types):
            padding = (padding,) * len(kernel_size)
        if isinstance(dilation, numeric_types):
            dilation = (dilation,) * len(kernel_size)
        self._op_name = op_name

        offset_channels = num_deformable_group * 3 * kernel_size[0] * kernel_size[1]
        self.offset_split_index = num_deformable_group * 2 * kernel_size[0] * kernel_size[1]
        self._layout = layout
        self._groups = groups
        self._offset_channels = offset_channels
        self._kernel_size = kernel_size
        self._kwargs_offset = {
            'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
            'pad': padding, 'num_filter': offset_channels, 'num_group': groups,
            'no_bias': not offset_use_bias, 'layout': layout}

        self._kwargs_deformable_conv = {
            'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
            'pad': padding, 'num_filter': channels, 'num_group': groups,
            'num_deformable_group': num_deformable_group,
            'no_bias': not use_bias, 'layout': layout}

        if adj:
            self._kwargs_offset['adj'] = adj
            self._kwargs_deformable_conv['adj'] = adj

        self.deformable_conv_weight = Parameter('deformable_conv_weight',
                                                shape=self.pre_infer_weight(),
                                                init=weight_initializer,
                                                allow_deferred_init=True)

        if use_bias:
            self.deformable_conv_bias = Parameter('deformable_conv_bias', shape=(channels,),
                                                  init=bias_initializer,
                                                  allow_deferred_init=True)
        else:
            self.deformable_conv_bias = None

        self.offset_weight = Parameter('offset_weight', shape=self.pre_infer_offset_weight(),
                                       init=offset_weight_initializer,
                                       allow_deferred_init=True)

        if offset_use_bias:
            self.offset_bias = Parameter('offset_bias', shape=(offset_channels,),
                                         init=offset_bias_initializer,
                                         allow_deferred_init=True)
        else:
            self.offset_bias = None

        if activation:
            self.act = Activation(activation)
        else:
            self.act = None

    def forward(self, x):
        ctx = x.ctx
        if self.offset_bias is None:
            offset = npx.convolution(x, self.offset_weight.data(ctx),
                                     cudnn_off=True, **self._kwargs_offset)
        else:
            offset = npx.convolution(x, self.offset_weight.data(ctx),
                                     self.offset_bias.data(ctx), cudnn_off=True, **self._kwargs_offset)

        offset_t = npx.slice_axis(offset, axis=1, begin=0, end=self.offset_split_index)
        mask = npx.slice_axis(offset, axis=1, begin=self.offset_split_index, end=None)
        mask = npx.sigmoid(mask) * 2

        if self.deformable_conv_bias is None:
            act = npx.modulated_deformable_convolution(data=x, offset=offset_t, mask=mask,
                                                       weight=self.deformable_conv_weight.data(ctx),
                                                       name='fwd', **self._kwargs_deformable_conv)
        else:
            act = npx.modulated_deformable_convolution(data=x, offset=offset_t, mask=mask,
                                                       weight=self.deformable_conv_weight.data(ctx),
                                                       bias=self.deformable_conv_bias.data(ctx), name='fwd',
                                                       **self._kwargs_deformable_conv)

        if self.act:
            act = self.act(act)
        return act

    def pre_infer_offset_weight(self):
        """
        Pre-infer the shape of offsite weight parameter based on kernel size,
        group size and offset channels
        """
        wshape = [-1]*(len(self._kernel_size) + 2)
        wshape[self._layout.find('N')] = self._offset_channels // self._groups
        wshape[self._layout.find('H')] = self._kernel_size[0]
        wshape[self._layout.find('W')] = self._kernel_size[1]
        wshape[0] *= self._groups
        return tuple(wshape)

    def pre_infer_weight(self):
        """
        Pre-infer the shape of weight parameter based on kernel size, group size and channels
        """
        wshape = [-1]*(len(self._kernel_size) + 2)
        wshape[self._layout.find('N')] = self._channels // self._groups
        wshape[self._layout.find('H')] = self._kernel_size[0]
        wshape[self._layout.find('W')] = self._kernel_size[1]
        wshape[0] *= self._groups
        return tuple(wshape)

    def infer_shape(self, x):
        dshape1 = x.shape[self._layout.find('C')]
        wshape = self.deformable_conv_weight.shape
        wshape_offset = self.offset_weight.shape
        wshape_list = list(wshape)
        wshape_offset_list = list(wshape_offset)
        wshape_list[self._layout.find('C')] = dshape1 // self._groups
        wshape_offset_list[self._layout.find('C')] = dshape1 // self._groups
        self.deformable_conv_weight.shape = tuple(wshape_list)
        self.offset_weight.shape = tuple(wshape_offset_list)

    def _alias(self):
        return 'modulated_deformable_conv'


@use_np
class PixelShuffle1D(HybridBlock):

    r"""Pixel-shuffle layer for upsampling in 1 dimension.

    Pixel-shuffling is the operation of taking groups of values along
    the *channel* dimension and regrouping them into blocks of pixels
    along the ``W`` dimension, thereby effectively multiplying that dimension
    by a constant factor in size.

    For example, a feature map of shape :math:`(fC, W)` is reshaped
    into :math:`(C, fW)` by forming little value groups of size :math:`f`
    and arranging them in a grid of size :math:`W`.

    Parameters
    ----------
    factor : int or 1-tuple of int
        Upsampling factor, applied to the ``W`` dimension.

    Inputs:
        - **data**: Tensor of shape ``(N, f*C, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, W*f)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle1D(2)
    >>> x = mx.np.zeros((1, 8, 3))
    >>> pxshuf(x).shape
    (1, 4, 6)
    """

    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self._factor = int(factor)

    def forward(self, x):
        """Perform pixel-shuffling on the input."""
        f = self._factor                                             # (N, C*f, W)
        x = npx.reshape(x, (-2, -6, -1, f, -2))  # (N, C, f, W)
        x = np.transpose(x, (0, 1, 3, 2))     # (N, C, W, f)
        x = npx.reshape(x, (-2, -2, -5))         # (N, C, W*f)
        return x

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._factor)


@use_np
class PixelShuffle2D(HybridBlock):

    r"""Pixel-shuffle layer for upsampling in 2 dimensions.

    Pixel-shuffling is the operation of taking groups of values along
    the *channel* dimension and regrouping them into blocks of pixels
    along the ``H`` and ``W`` dimensions, thereby effectively multiplying
    those dimensions by a constant factor in size.

    For example, a feature map of shape :math:`(f^2 C, H, W)` is reshaped
    into :math:`(C, fH, fW)` by forming little :math:`f \times f` blocks
    of pixels and arranging them in an :math:`H \times W` grid.

    Pixel-shuffling together with regular convolution is an alternative,
    learnable way of upsampling an image by arbitrary factors. It is reported
    to help overcome checkerboard artifacts that are common in upsampling with
    transposed convolutions (also called deconvolutions). See the paper
    `Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_
    for further details.

    Parameters
    ----------
    factor : int or 2-tuple of int
        Upsampling factors, applied to the ``H`` and ``W`` dimensions,
        in that order.

    Inputs:
        - **data**: Tensor of shape ``(N, f1*f2*C, H, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, H*f1, W*f2)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle2D((2, 3))
    >>> x = mx.np.zeros((1, 12, 3, 5))
    >>> pxshuf(x).shape
    (1, 2, 6, 15)
    """

    def __init__(self, factor):
        super(PixelShuffle2D, self).__init__()
        try:
            self._factors = (int(factor),) * 2
        except TypeError:
            self._factors = tuple(int(fac) for fac in factor)
            assert len(self._factors) == 2, "wrong length {}".format(len(self._factors))

    def forward(self, x):
        """Perform pixel-shuffling on the input."""
        f1, f2 = self._factors
                                                      # (N, f1*f2*C, H, W)
        x = npx.reshape(x, (-2, -6, -1, f1 * f2, -2, -2))  # (N, C, f1*f2, H, W)
        x = npx.reshape(x, (-2, -2, -6, f1, f2, -2, -2))    # (N, C, f1, f2, H, W)
        x = np.transpose(x, (0, 1, 4, 2, 5, 3))        # (N, C, H, f1, W, f2)
        x = npx.reshape(x, (-2, -2, -5, -5))              # (N, C, H*f1, W*f2)
        return x

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._factors)


@use_np
class PixelShuffle3D(HybridBlock):

    r"""Pixel-shuffle layer for upsampling in 3 dimensions.

    Pixel-shuffling (or voxel-shuffling in 3D) is the operation of taking
    groups of values along the *channel* dimension and regrouping them into
    blocks of voxels along the ``D``, ``H`` and ``W`` dimensions, thereby
    effectively multiplying those dimensions by a constant factor in size.

    For example, a feature map of shape :math:`(f^3 C, D, H, W)` is reshaped
    into :math:`(C, fD, fH, fW)` by forming little :math:`f \times f \times f`
    blocks of voxels and arranging them in a :math:`D \times H \times W` grid.

    Pixel-shuffling together with regular convolution is an alternative,
    learnable way of upsampling an image by arbitrary factors. It is reported
    to help overcome checkerboard artifacts that are common in upsampling with
    transposed convolutions (also called deconvolutions). See the paper
    `Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_
    for further details.

    Parameters
    ----------
    factor : int or 3-tuple of int
        Upsampling factors, applied to the ``D``, ``H`` and ``W``
        dimensions, in that order.

    Inputs:
        - **data**: Tensor of shape ``(N, f1*f2*f3*C, D, H, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, D*f1, H*f2, W*f3)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle3D((2, 3, 4))
    >>> x = mx.np.zeros((1, 48, 3, 5, 7))
    >>> pxshuf(x).shape
    (1, 2, 6, 15, 28)
    """

    def __init__(self, factor):
        super(PixelShuffle3D, self).__init__()
        try:
            self._factors = (int(factor),) * 3
        except TypeError:
            self._factors = tuple(int(fac) for fac in factor)
            assert len(self._factors) == 3, "wrong length {}".format(len(self._factors))

    def forward(self, x):
        """Perform pixel-shuffling on the input."""
        # `transpose` doesn't support 8D, need other implementation
        f1, f2, f3 = self._factors
                                                              # (N, C*f1*f2*f3, D, H, W)
        x = npx.reshape(x, (-2, -6, -1, f1 * f2 * f3, -2, -2, -2))  # (N, C, f1*f2*f3, D, H, W)
        x = np.swapaxes(x, 2, 3)                               # (N, C, D, f1*f2*f3, H, W)
        x = npx.reshape(x, (-2, -2, -2, -6, f1, f2*f3, -2, -2))      # (N, C, D, f1, f2*f3, H, W)
        x = npx.reshape(x, (-2, -2, -5, -2, -2, -2))                 # (N, C, D*f1, f2*f3, H, W)
        x = np.swapaxes(x, 3, 4)                               # (N, C, D*f1, H, f2*f3, W)
        x = npx.reshape(x, (-2, -2, -2, -2, -6, f2, f3, -2))         # (N, C, D*f1, H, f2, f3, W)
        x = npx.reshape(x, (-2, -2, -2, -5, -2, -2))                 # (N, C, D*f1, H*f2, f3, W)
        x = np.swapaxes(x, 4, 5)                               # (N, C, D*f1, H*f2, W, f3)
        x = npx.reshape(x, (-2, -2, -2, -2, -5))                    # (N, C, D*f1, H*f2, W*f3)
        return x

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._factors)
