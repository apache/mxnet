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

# pylint: disable=arguments-differ, too-many-lines
# coding: utf-8
"""Definition of various recurrent neural network cells."""
__all__ = ['Conv1DRNNCell', 'Conv2DRNNCell', 'Conv3DRNNCell',
           'Conv1DLSTMCell', 'Conv2DLSTMCell', 'Conv3DLSTMCell',
           'Conv1DGRUCell', 'Conv2DGRUCell', 'Conv3DGRUCell']


from math import floor

from ...base import numeric_types
from .rnn_cell import HybridRecurrentCell
from ..parameter import Parameter
from ... import np, npx
from ...util import use_np


def _get_conv_out_size(dimensions, kernels, paddings, dilations):
    return tuple(int(floor(x+2*p-d*(k-1)-1)+1) if x else 0 for x, k, p, d in
                 zip(dimensions, kernels, paddings, dilations))


@use_np
class _BaseConvRNNCell(HybridRecurrentCell):
    """Abstract base class for convolutional RNNs"""
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad, i2h_dilate, h2h_dilate,
                 i2h_weight_initializer, h2h_weight_initializer,
                 i2h_bias_initializer, h2h_bias_initializer,
                 dims,
                 conv_layout, activation):
        super(_BaseConvRNNCell, self).__init__()

        self._hidden_channels = hidden_channels
        self._input_shape = input_shape
        self._conv_layout = conv_layout
        self._activation = activation

        # Convolution setting
        assert all(isinstance(spec, int) or len(spec) == dims
                   for spec in [i2h_kernel, i2h_pad, i2h_dilate,
                                h2h_kernel, h2h_dilate]), \
               "For {dims}D convolution, the convolution settings can only be either int " \
               "or list/tuple of length {dims}".format(dims=dims)

        self._i2h_kernel = (i2h_kernel,) * dims if isinstance(i2h_kernel, numeric_types) \
                           else i2h_kernel
        self._stride = (1,) * dims
        self._i2h_pad = (i2h_pad,) * dims if isinstance(i2h_pad, numeric_types) \
                        else i2h_pad
        self._i2h_dilate = (i2h_dilate,) * dims if isinstance(i2h_dilate, numeric_types) \
                           else i2h_dilate
        self._h2h_kernel = (h2h_kernel,) * dims if isinstance(h2h_kernel, numeric_types) \
                           else h2h_kernel
        assert all(k % 2 == 1 for k in self._h2h_kernel), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_dilate = (h2h_dilate,) * dims if isinstance(h2h_dilate, numeric_types) \
                           else h2h_dilate

        self._channel_axis, \
        self._in_channels, \
        i2h_param_shape, \
        h2h_param_shape, \
        self._h2h_pad, \
        self._state_shape = self._decide_shapes()

        self.i2h_weight = Parameter('i2h_weight', shape=i2h_param_shape,
                                    init=i2h_weight_initializer,
                                    allow_deferred_init=True)
        self.h2h_weight = Parameter('h2h_weight', shape=h2h_param_shape,
                                    init=h2h_weight_initializer,
                                    allow_deferred_init=True)
        self.i2h_bias = Parameter('i2h_bias', shape=(hidden_channels*self._num_gates,),
                                  init=i2h_bias_initializer,
                                  allow_deferred_init=True)
        self.h2h_bias = Parameter('h2h_bias', shape=(hidden_channels*self._num_gates,),
                                  init=h2h_bias_initializer,
                                  allow_deferred_init=True)

    def _decide_shapes(self):
        channel_axis = self._conv_layout.find('C')
        input_shape = self._input_shape
        in_channels = input_shape[channel_axis - 1]
        hidden_channels = self._hidden_channels
        if channel_axis == 1:
            dimensions = input_shape[1:]
        else:
            dimensions = input_shape[:-1]

        total_out = hidden_channels * self._num_gates

        i2h_param_shape = (total_out,)
        h2h_param_shape = (total_out,)
        state_shape = (hidden_channels,)
        conv_out_size = _get_conv_out_size(dimensions,
                                           self._i2h_kernel,
                                           self._i2h_pad,
                                           self._i2h_dilate)
        h2h_pad = tuple(d*(k-1)//2 for d, k in zip(self._h2h_dilate, self._h2h_kernel))
        if channel_axis == 1:
            i2h_param_shape += (in_channels,) + self._i2h_kernel
            h2h_param_shape += (hidden_channels,) + self._h2h_kernel
            state_shape += conv_out_size
        else:
            i2h_param_shape += self._i2h_kernel + (in_channels,)
            h2h_param_shape += self._h2h_kernel + (hidden_channels,)
            state_shape = conv_out_size + state_shape

        return channel_axis, in_channels, i2h_param_shape, \
               h2h_param_shape, h2h_pad, state_shape

    def __repr__(self):
        s = '{name}({mapping}'
        if hasattr(self, '_activation'):
            s += ', {_activation}'
        s += ', {_conv_layout}'
        s += ')'
        attrs = self.__dict__
        shape = self.i2h_weight.shape
        in_channels = shape[1 if self._channel_axis == 1 else -1]
        mapping = ('{0} -> {1}'.format(in_channels if in_channels else None, shape[0]))
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **attrs)

    @property
    def _num_gates(self):
        return len(self._gate_names)

    def _conv_forward(self, inputs, states):
        ctx = inputs.ctx
        i2h = npx.convolution(data=inputs,
                              num_filter=self._hidden_channels*self._num_gates,
                              kernel=self._i2h_kernel,
                              stride=self._stride,
                              pad=self._i2h_pad,
                              dilate=self._i2h_dilate,
                              weight=self.i2h_weight.data(ctx),
                              bias=self.i2h_bias.data(ctx),
                              layout=self._conv_layout)
        h2h = npx.convolution(data=states[0].as_in_ctx(ctx),
                              num_filter=self._hidden_channels*self._num_gates,
                              kernel=self._h2h_kernel,
                              dilate=self._h2h_dilate,
                              pad=self._h2h_pad,
                              stride=self._stride,
                              weight=self.h2h_weight.data(ctx),
                              bias=self.h2h_bias.data(ctx),
                              layout=self._conv_layout)
        return i2h, h2h

    def state_info(self, batch_size=0):
        raise NotImplementedError("_BaseConvRNNCell is abstract class for convolutional RNN")

    def forward(self, inputs, states):
        raise NotImplementedError("_BaseConvRNNCell is abstract class for convolutional RNN")

    # pylint: disable=unused-argument
    def infer_shape(self, i, x, is_bidirect):
        channel_axis = self._conv_layout.find('C')
        shape_c = x.shape[-len(self._i2h_kernel)-1:][channel_axis-1]
        wshape = self.i2h_weight.shape
        wshape_list = list(wshape)
        wshape_list[self._conv_layout.find('C')] = shape_c
        self.i2h_weight.shape = tuple(wshape_list)


@use_np
class _ConvRNNCell(_BaseConvRNNCell):
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel, i2h_pad, i2h_dilate, h2h_dilate,
                 i2h_weight_initializer, h2h_weight_initializer,
                 i2h_bias_initializer, h2h_bias_initializer,
                 dims, conv_layout, activation):
        super(_ConvRNNCell, self).__init__(input_shape=input_shape,
                                           hidden_channels=hidden_channels,
                                           activation=activation,
                                           i2h_kernel=i2h_kernel,
                                           i2h_pad=i2h_pad, i2h_dilate=i2h_dilate,
                                           h2h_kernel=h2h_kernel, h2h_dilate=h2h_dilate,
                                           i2h_weight_initializer=i2h_weight_initializer,
                                           h2h_weight_initializer=h2h_weight_initializer,
                                           i2h_bias_initializer=i2h_bias_initializer,
                                           h2h_bias_initializer=h2h_bias_initializer,
                                           dims=dims,
                                           conv_layout=conv_layout)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size,)+self._state_shape, '__layout__': self._conv_layout}]

    def _alias(self):
        return 'conv_rnn'

    @property
    def _gate_names(self):
        return ('',)

    def forward(self, inputs, states):
        i2h, h2h = self._conv_forward(inputs, states)
        output = self._get_activation(i2h + h2h, self._activation)
        return output, [output]


class Conv1DRNNCell(_ConvRNNCell):
    r"""1D Convolutional RNN cell.

    .. math::

        h_t = tanh(W_i \ast x_t + R_i \ast h_{t-1} + b_i)

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCW' the shape should be (C, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0,)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1,)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1,)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCW' and 'NWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0,), i2h_dilate=(1,), h2h_dilate=(1,),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCW', activation='tanh'):
        super(Conv1DRNNCell, self).__init__(input_shape=input_shape,
                                            hidden_channels=hidden_channels,
                                            i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                            i2h_pad=i2h_pad,
                                            i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                            i2h_weight_initializer=i2h_weight_initializer,
                                            h2h_weight_initializer=h2h_weight_initializer,
                                            i2h_bias_initializer=i2h_bias_initializer,
                                            h2h_bias_initializer=h2h_bias_initializer,
                                            dims=1,
                                            conv_layout=conv_layout,
                                            activation=activation)


class Conv2DRNNCell(_ConvRNNCell):
    r"""2D Convolutional RNN cell.

    .. math::

        h_t = tanh(W_i \ast x_t + R_i \ast h_{t-1} + b_i)

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCHW' the shape should be (C, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCHW' and 'NHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0, 0), i2h_dilate=(1, 1), h2h_dilate=(1, 1),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCHW', activation='tanh'):
        super(Conv2DRNNCell, self).__init__(input_shape=input_shape,
                                            hidden_channels=hidden_channels,
                                            i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                            i2h_pad=i2h_pad,
                                            i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                            i2h_weight_initializer=i2h_weight_initializer,
                                            h2h_weight_initializer=h2h_weight_initializer,
                                            i2h_bias_initializer=i2h_bias_initializer,
                                            h2h_bias_initializer=h2h_bias_initializer,
                                            dims=2,
                                            conv_layout=conv_layout,
                                            activation=activation)


class Conv3DRNNCell(_ConvRNNCell):
    r"""3D Convolutional RNN cells

    .. math::

        h_t = tanh(W_i \ast x_t + R_i \ast h_{t-1} + b_i)

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCDHW' the shape should be (C, D, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCDHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCDHW' and 'NDHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0, 0, 0),
                 i2h_dilate=(1, 1, 1), h2h_dilate=(1, 1, 1),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCDHW', activation='tanh'):
        super(Conv3DRNNCell, self).__init__(input_shape=input_shape,
                                            hidden_channels=hidden_channels,
                                            i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                            i2h_pad=i2h_pad,
                                            i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                            i2h_weight_initializer=i2h_weight_initializer,
                                            h2h_weight_initializer=h2h_weight_initializer,
                                            i2h_bias_initializer=i2h_bias_initializer,
                                            h2h_bias_initializer=h2h_bias_initializer,
                                            dims=3,
                                            conv_layout=conv_layout,
                                            activation=activation)


@use_np
class _ConvLSTMCell(_BaseConvRNNCell):
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad, i2h_dilate, h2h_dilate,
                 i2h_weight_initializer, h2h_weight_initializer,
                 i2h_bias_initializer, h2h_bias_initializer,
                 dims, conv_layout, activation):
        super(_ConvLSTMCell, self).__init__(input_shape=input_shape,
                                            hidden_channels=hidden_channels,
                                            i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                            i2h_pad=i2h_pad,
                                            i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                            i2h_weight_initializer=i2h_weight_initializer,
                                            h2h_weight_initializer=h2h_weight_initializer,
                                            i2h_bias_initializer=i2h_bias_initializer,
                                            h2h_bias_initializer=h2h_bias_initializer,
                                            dims=dims,
                                            conv_layout=conv_layout,
                                            activation=activation)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size,)+self._state_shape, '__layout__': self._conv_layout},
                {'shape': (batch_size,)+self._state_shape, '__layout__': self._conv_layout}]

    def _alias(self):
        return 'conv_lstm'

    @property
    def _gate_names(self):
        return ['_i', '_f', '_c', '_o']

    def forward(self, inputs, states):
        i2h, h2h = self._conv_forward(inputs, states)
        gates = i2h + h2h
        slice_gates = npx.slice_channel(gates, num_outputs=4, axis=self._channel_axis)
        in_gate = npx.activation(slice_gates[0], act_type="sigmoid")
        forget_gate = npx.activation(slice_gates[1], act_type="sigmoid")
        in_transform = self._get_activation(slice_gates[2], self._activation)
        out_gate = npx.activation(slice_gates[3], act_type="sigmoid")
        next_c = forget_gate * states[1].as_in_ctx(inputs.ctx) + in_gate * in_transform
        next_h = np.multiply(out_gate, self._get_activation(next_c, self._activation))

        return next_h, [next_h, next_c]


class Conv1DLSTMCell(_ConvLSTMCell):
    r"""1D Convolutional LSTM network cell.

    `"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    <https://arxiv.org/abs/1506.04214>`_ paper. Xingjian et al. NIPS2015

    .. math::
        \begin{array}{ll}
        i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} + b_i) \\
        f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
        o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
        c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\
        c_t = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\
        h_t = o_t \circ tanh(c_t) \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCW' the shape should be (C, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0,)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1,)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1,)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCW' and 'NWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in c^\prime_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0,),
                 i2h_dilate=(1,), h2h_dilate=(1,),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCW', activation='tanh'):
        super(Conv1DLSTMCell, self).__init__(input_shape=input_shape,
                                             hidden_channels=hidden_channels,
                                             i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                             i2h_pad=i2h_pad,
                                             i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                             i2h_weight_initializer=i2h_weight_initializer,
                                             h2h_weight_initializer=h2h_weight_initializer,
                                             i2h_bias_initializer=i2h_bias_initializer,
                                             h2h_bias_initializer=h2h_bias_initializer,
                                             dims=1,
                                             conv_layout=conv_layout,
                                             activation=activation)


class Conv2DLSTMCell(_ConvLSTMCell):
    r"""2D Convolutional LSTM network cell.

    `"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    <https://arxiv.org/abs/1506.04214>`_ paper. Xingjian et al. NIPS2015

    .. math::
        \begin{array}{ll}
        i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} + b_i) \\
        f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
        o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
        c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\
        c_t = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\
        h_t = o_t \circ tanh(c_t) \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCHW' the shape should be (C, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCHW' and 'NHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in c^\prime_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0, 0),
                 i2h_dilate=(1, 1), h2h_dilate=(1, 1),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCHW', activation='tanh'):
        super(Conv2DLSTMCell, self).__init__(input_shape=input_shape,
                                             hidden_channels=hidden_channels,
                                             i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                             i2h_pad=i2h_pad,
                                             i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                             i2h_weight_initializer=i2h_weight_initializer,
                                             h2h_weight_initializer=h2h_weight_initializer,
                                             i2h_bias_initializer=i2h_bias_initializer,
                                             h2h_bias_initializer=h2h_bias_initializer,
                                             dims=2,
                                             conv_layout=conv_layout,
                                             activation=activation)


class Conv3DLSTMCell(_ConvLSTMCell):
    r"""3D Convolutional LSTM network cell.

    `"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    <https://arxiv.org/abs/1506.04214>`_ paper. Xingjian et al. NIPS2015

    .. math::
        \begin{array}{ll}
        i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} + b_i) \\
        f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
        o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
        c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\
        c_t = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\
        h_t = o_t \circ tanh(c_t) \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCDHW' the shape should be (C, D, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCDHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCDHW' and 'NDHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in c^\prime_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0, 0, 0),
                 i2h_dilate=(1, 1, 1), h2h_dilate=(1, 1, 1),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCDHW', activation='tanh'):
        super(Conv3DLSTMCell, self).__init__(input_shape=input_shape,
                                             hidden_channels=hidden_channels,
                                             i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                             i2h_pad=i2h_pad,
                                             i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                             i2h_weight_initializer=i2h_weight_initializer,
                                             h2h_weight_initializer=h2h_weight_initializer,
                                             i2h_bias_initializer=i2h_bias_initializer,
                                             h2h_bias_initializer=h2h_bias_initializer,
                                             dims=3,
                                             conv_layout=conv_layout,
                                             activation=activation)


@use_np
class _ConvGRUCell(_BaseConvRNNCell):
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel, i2h_pad, i2h_dilate, h2h_dilate,
                 i2h_weight_initializer, h2h_weight_initializer,
                 i2h_bias_initializer, h2h_bias_initializer,
                 dims, conv_layout, activation):
        super(_ConvGRUCell, self).__init__(input_shape=input_shape,
                                           hidden_channels=hidden_channels,
                                           i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                           i2h_pad=i2h_pad,
                                           i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                           i2h_weight_initializer=i2h_weight_initializer,
                                           h2h_weight_initializer=h2h_weight_initializer,
                                           i2h_bias_initializer=i2h_bias_initializer,
                                           h2h_bias_initializer=h2h_bias_initializer,
                                           dims=dims,
                                           conv_layout=conv_layout,
                                           activation=activation)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size,)+self._state_shape, '__layout__': self._conv_layout}]

    def _alias(self):
        return 'conv_gru'

    @property
    def _gate_names(self):
        return ['_r', '_z', '_o']

    def forward(self, inputs, states):
        i2h, h2h = self._conv_forward(inputs, states)

        i2h_r, i2h_z, i2h = npx.slice_channel(i2h, num_outputs=3,
                                              axis=self._channel_axis)
        h2h_r, h2h_z, h2h = npx.slice_channel(h2h, num_outputs=3,
                                              axis=self._channel_axis)

        reset_gate = npx.activation(i2h_r + h2h_r, act_type="sigmoid")
        update_gate = npx.activation(i2h_z + h2h_z, act_type="sigmoid")

        next_h_tmp = self._get_activation(i2h + reset_gate * h2h, self._activation)

        next_h = (1. - update_gate) * next_h_tmp + update_gate * \
            states[0].as_in_ctx(inputs.ctx)

        return next_h, [next_h]


class Conv1DGRUCell(_ConvGRUCell):
    r"""1D Convolutional Gated Rectified Unit (GRU) network cell.

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_r \ast x_t + R_r \ast h_{t-1} + b_r) \\
        z_t = \sigma(W_z \ast x_t + R_z \ast h_{t-1} + b_z) \\
        n_t = tanh(W_i \ast x_t + b_i + r_t \circ (R_n \ast h_{t-1} + b_n)) \\
        h^\prime_t = (1 - z_t) \circ n_t + z_t \circ h \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCW' the shape should be (C, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0,)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1,)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1,)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCW' and 'NWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in n_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0,),
                 i2h_dilate=(1,), h2h_dilate=(1,),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCW', activation='tanh'):
        super(Conv1DGRUCell, self).__init__(input_shape=input_shape,
                                            hidden_channels=hidden_channels,
                                            i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                            i2h_pad=i2h_pad,
                                            i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                            i2h_weight_initializer=i2h_weight_initializer,
                                            h2h_weight_initializer=h2h_weight_initializer,
                                            i2h_bias_initializer=i2h_bias_initializer,
                                            h2h_bias_initializer=h2h_bias_initializer,
                                            dims=1,
                                            conv_layout=conv_layout,
                                            activation=activation)


class Conv2DGRUCell(_ConvGRUCell):
    r"""2D Convolutional Gated Rectified Unit (GRU) network cell.

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_r \ast x_t + R_r \ast h_{t-1} + b_r) \\
        z_t = \sigma(W_z \ast x_t + R_z \ast h_{t-1} + b_z) \\
        n_t = tanh(W_i \ast x_t + b_i + r_t \circ (R_n \ast h_{t-1} + b_n)) \\
        h^\prime_t = (1 - z_t) \circ n_t + z_t \circ h \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCHW' the shape should be (C, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCHW' and 'NHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in n_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0, 0),
                 i2h_dilate=(1, 1), h2h_dilate=(1, 1),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCHW', activation='tanh'):
        super(Conv2DGRUCell, self).__init__(input_shape=input_shape,
                                            hidden_channels=hidden_channels,
                                            i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                            i2h_pad=i2h_pad,
                                            i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                            i2h_weight_initializer=i2h_weight_initializer,
                                            h2h_weight_initializer=h2h_weight_initializer,
                                            i2h_bias_initializer=i2h_bias_initializer,
                                            h2h_bias_initializer=h2h_bias_initializer,
                                            dims=2,
                                            conv_layout=conv_layout,
                                            activation=activation)


class Conv3DGRUCell(_ConvGRUCell):
    r"""3D Convolutional Gated Rectified Unit (GRU) network cell.

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_r \ast x_t + R_r \ast h_{t-1} + b_r) \\
        z_t = \sigma(W_z \ast x_t + R_z \ast h_{t-1} + b_z) \\
        n_t = tanh(W_i \ast x_t + b_i + r_t \circ (R_n \ast h_{t-1} + b_n)) \\
        h^\prime_t = (1 - z_t) \circ n_t + z_t \circ h \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCDHW' the shape should be (C, D, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCDHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCDHW' and 'NDHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in n_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0, 0, 0),
                 i2h_dilate=(1, 1, 1), h2h_dilate=(1, 1, 1),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCDHW', activation='tanh'):
        super(Conv3DGRUCell, self).__init__(input_shape=input_shape,
                                            hidden_channels=hidden_channels,
                                            i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                            i2h_pad=i2h_pad,
                                            i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                            i2h_weight_initializer=i2h_weight_initializer,
                                            h2h_weight_initializer=h2h_weight_initializer,
                                            i2h_bias_initializer=i2h_bias_initializer,
                                            h2h_bias_initializer=h2h_bias_initializer,
                                            dims=3,
                                            conv_layout=conv_layout,
                                            activation=activation)
