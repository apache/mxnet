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
# pylint: disable=no-member, invalid-name, protected-access, no-self-use
# pylint: disable=too-many-branches, too-many-arguments, no-self-use
# pylint: disable=too-many-lines, arguments-differ
"""Definition of various recurrent neural network layers."""

__all__ = ['RNN', 'LSTM', 'GRU']

from ... import np, npx, initializer, cpu
from .. import HybridBlock, tensor_types
from ..parameter import Parameter
from ...util import use_np


@use_np
class _RNNLayer(HybridBlock):
    """Implementation of recurrent layers."""
    def __init__(self, hidden_size, num_layers, layout,
                 dropout, bidirectional, input_size,
                 i2h_weight_initializer, h2h_weight_initializer,
                 i2h_bias_initializer, h2h_bias_initializer,
                 mode, projection_size, h2r_weight_initializer,
                 lstm_state_clip_min, lstm_state_clip_max, lstm_state_clip_nan,
                 dtype, use_sequence_length=False, **kwargs):
        super(_RNNLayer, self).__init__(**kwargs)
        assert layout in ('TNC', 'NTC'), \
            f"Invalid layout {layout}; must be one of ['TNC' or 'NTC']"
        self._hidden_size = hidden_size
        self._projection_size = projection_size if projection_size else None
        self._num_layers = num_layers
        self._mode = mode
        self._layout = layout
        self._dropout = dropout
        self._dir = 2 if bidirectional else 1
        self._input_size = input_size
        self._lstm_state_clip_min = lstm_state_clip_min
        self._lstm_state_clip_max = lstm_state_clip_max
        self._lstm_state_clip_nan = lstm_state_clip_nan
        self._dtype = dtype
        self._use_sequence_length = use_sequence_length
        self.skip_states = None

        self._gates = {'rnn_relu': 1, 'rnn_tanh': 1, 'lstm': 4, 'gru': 3}[mode]

        param_initializer = initializer.RNNFused(
            mode, num_layers, hidden_size,
            bidirectional, projection_size,
            i2h_weight_initializer=i2h_weight_initializer,
            h2h_weight_initializer=h2h_weight_initializer,
            i2h_bias_initializer=i2h_bias_initializer,
            h2h_bias_initializer=h2h_bias_initializer,
            h2r_weight_initializer=h2r_weight_initializer)

        self.rnn_param = Parameter('rnn_param', shape=(-1,), init=param_initializer,
                                   allow_deferred_init=True, dtype=dtype)

    def __repr__(self):
        s = '{name}({mapping}, {_layout}'
        if self._num_layers != 1:
            s += ', num_layers={_num_layers}'
        if self._dropout != 0:
            s += ', dropout={_dropout}'
        if self._dir == 2:
            s += ', bidirectional'
        s += ')'
        mapping = '{0} -> {1}'.format(self._input_size if self._input_size else None, self._hidden_size)
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

    def state_info(self, batch_size=0):
        raise NotImplementedError

    def cast(self, dtype):
        super(_RNNLayer, self).cast(dtype)
        self._dtype = dtype

    def begin_state(self, batch_size=0, func=np.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        batch_size: int
            Only required for `NDArray` API. Size of the batch ('N' in layout).
            Dimension of the input.
        func : callable, default `ndarray.zeros`
            Function for creating initial state.

            For Symbol API, func can be `symbol.zeros`, `symbol.uniform`,
            `symbol.var` etc. Use `symbol.var` if you want to directly
            feed input as states.

            For NDArray API, func can be `ndarray.zeros`, `ndarray.ones`, etc.

        **kwargs :
            Additional keyword arguments passed to func. For example
            `mean`, `std`, `dtype`, etc.

        Returns
        -------
        states : nested list of Symbol
            Starting states for the first RNN step.
        """
        states = []
        for info in self.state_info(batch_size):
            if info is not None:
                info.update(kwargs)
            else:
                info = kwargs
            state = func(shape=info.pop("shape", ()),
                         device=info.pop("device", cpu()),
                         dtype=info.pop("dtype", "float32"))
            states.append(state)
        return states

    def __call__(self, inputs, states=None, sequence_length=None, **kwargs):
        self.skip_states = states is None
        if states is None:
            batch_size = inputs.shape[self._layout.find('N')]
            states = self.begin_state(batch_size, device=inputs.device, dtype=inputs.dtype)
        if isinstance(states, tensor_types):
            states = [states]

        if self._use_sequence_length:
            return super(_RNNLayer, self).__call__(inputs, states, sequence_length, **kwargs)
        else:
            return super(_RNNLayer, self).__call__(inputs, states, **kwargs)

    def forward(self, inputs, states, sequence_length=None):
        batch_size = inputs.shape[self._layout.find('N')]

        for state, info in zip(states, self.state_info(batch_size)):
            if state.shape != info['shape']:
                raise ValueError(
                    f"Invalid recurrent state shape. Expecting {str(info['shape'])}, got {str(state.shape)}.")
        out = self._forward_kernel(inputs, states, sequence_length)

        # out is (output, state)
        return out[0] if self.skip_states else out

    def infer_shape(self, inputs, *args):
        assert inputs.ndim == 3, \
            "Input data should be rank-3 tensor of dim [sequence length, batch size, input size]"
        self._input_size = inputs.shape[2]
        ng, ni, nh = self._gates, inputs.shape[2], self._hidden_size

        size = nh * self._dir * ng
        size1 = (ni + nh + 2) * size  # first layer size
        size2 = (nh * self._dir + nh + 2) * size  # second layer size
        if self._projection_size:
            size1 = (ni + self._projection_size + 2) * size  # first layer size
            size2 = (self._projection_size * self._dir + \
                self._projection_size + 2) * size  # second layer size
        param_size = size1 + (self._num_layers - 1) * size2
        if self._projection_size:
            param_size += self._projection_size * nh * self._num_layers * self._dir
        self.rnn_param.shape = (param_size, )

    def _forward_kernel(self, inputs, states, sequence_length):
        """ forward using CUDNN or CPU kenrel"""
        device = inputs.device
        if self._layout == 'NTC':
            inputs = np.swapaxes(inputs, 0, 1)

        if self._use_sequence_length:
            rnn_args = states + [sequence_length]
        else:
            rnn_args = states

        rnn_args_device = []
        for args in rnn_args:
            new_args = args.to_device(device)
            rnn_args_device.append(new_args)

        rnn = npx.rnn(inputs, self.rnn_param.data(device), *rnn_args_device,
                      use_sequence_length=self._use_sequence_length,
                      state_size=self._hidden_size, projection_size=self._projection_size,
                      num_layers=self._num_layers, bidirectional=self._dir == 2,
                      p=self._dropout, state_outputs=True, mode=self._mode,
                      lstm_state_clip_min=self._lstm_state_clip_min,
                      lstm_state_clip_max=self._lstm_state_clip_max,
                      lstm_state_clip_nan=self._lstm_state_clip_nan)

        if self._mode == 'lstm':
            outputs, states = rnn[0], [rnn[1], rnn[2]]
        else:
            outputs, states = rnn[0], [rnn[1]]

        if self._layout == 'NTC':
            outputs = np.swapaxes(outputs, 0, 1)

        return outputs, states


class RNN(_RNNLayer):
    r"""Applies a multi-layer Elman RNN with `tanh` or `ReLU` non-linearity to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is the output
    of the previous layer at time `t` or :math:`input_t` for the first layer.
    If nonlinearity='relu', then `ReLU` is used instead of `tanh`.

    Parameters
    ----------
    hidden_size: int
        The number of features in the hidden state h.
    num_layers: int, default 1
        Number of recurrent layers.
    activation: {'relu' or 'tanh'}, default 'relu'
        The activation function to use.
    layout : str, default 'TNC'
        The format of input and output tensors. T, N and C stand for
        sequence length, batch size, and feature dimensions respectively.
    dropout: float, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    bidirectional: bool, default False
        If `True`, becomes a bidirectional RNN.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.
    dtype : str, default 'float32'
        Type to initialize the parameters and default states to


    Inputs:
        - **data**: input tensor with shape `(sequence_length, batch_size, input_size)`
          when `layout` is "TNC". For other layouts, dimensions are permuted accordingly
          using transpose() operator which adds performance overhead. Consider creating
          batches in TNC layout during data batching step.

        - **states**: initial recurrent state tensor with shape
          `(num_layers, batch_size, num_hidden)`. If `bidirectional` is True,
          shape will instead be `(2*num_layers, batch_size, num_hidden)`. If
          `states` is None, zeros will be used as default begin states.

    Outputs:
        - **out**: output tensor with shape `(sequence_length, batch_size, num_hidden)`
          when `layout` is "TNC". If `bidirectional` is True, output shape will instead
          be `(sequence_length, batch_size, 2*num_hidden)`
        - **out_states**: output recurrent state tensor with the same shape as `states`.
          If `states` is None `out_states` will not be returned.


    Examples
    --------
    >>> layer = mx.gluon.rnn.RNN(100, 3)
    >>> layer.initialize()
    >>> input = mx.np.random.uniform(size=(5, 3, 10))
    >>> # by default zeros are used as begin state
    >>> output = layer(input)
    >>> # manually specify begin state.
    >>> h0 = mx.np.random.uniform(size=(3, 3, 100))
    >>> output, hn = layer(input, h0)
    """
    def __init__(self, hidden_size, num_layers=1, activation='relu',
                 layout='TNC', dropout=0, bidirectional=False,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0, dtype='float32', **kwargs):
        super(RNN, self).__init__(hidden_size, num_layers, layout,
                                  dropout, bidirectional, input_size,
                                  i2h_weight_initializer, h2h_weight_initializer,
                                  i2h_bias_initializer, h2h_bias_initializer,
                                  'rnn_'+activation, None, None, None, None, False,
                                  dtype, **kwargs)

    def state_info(self, batch_size=0):
        return [{'shape': (self._num_layers * self._dir, batch_size, self._hidden_size),
                 '__layout__': 'LNC', 'dtype': self._dtype}]


class LSTM(_RNNLayer):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
        i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        f_t = sigmoid(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
        o_t = sigmoid(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
        c_t = f_t * c_{(t-1)} + i_t * g_t \\
        h_t = o_t * \tanh(c_t)
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the
    cell state at time `t`, :math:`x_t` is the hidden state of the previous
    layer at time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell, and
    out gates, respectively.

    Parameters
    ----------
    hidden_size: int
        The number of features in the hidden state h.
    num_layers: int, default 1
        Number of recurrent layers.
    layout : str, default 'TNC'
        The format of input and output tensors. T, N and C stand for
        sequence length, batch size, and feature dimensions respectively.
    dropout: float, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    bidirectional: bool, default False
        If `True`, becomes a bidirectional RNN.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'lstmbias'
        Initializer for the bias vector. By default, bias for the forget
        gate is initialized to 1 while all other biases are initialized
        to zero.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    projection_size: int, default None
        The number of features after projection.
    h2r_weight_initializer : str or Initializer, default None
        Initializer for the projected recurrent weights matrix, used for the linear
        transformation of the recurrent state to the projected space.
    state_clip_min : float or None, default None
        Minimum clip value of LSTM states. This option must be used together with
        state_clip_max. If None, clipping is not applied.
    state_clip_max : float or None, default None
        Maximum clip value of LSTM states. This option must be used together with
        state_clip_min. If None, clipping is not applied.
    state_clip_nan : boolean, default False
        Whether to stop NaN from propagating in state by clipping it to min/max.
        If the clipping range is not specified, this option is ignored.
    dtype : str, default 'float32'
        Type to initialize the parameters and default states to
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.


    Inputs:
        - **data**: input tensor with shape `(sequence_length, batch_size, input_size)`
          when `layout` is "TNC". For other layouts, dimensions are permuted accordingly
          using transpose() operator which adds performance overhead. Consider creating
          batches in TNC layout during data batching step.
        - **states**: a list of two initial recurrent state tensors. Each has shape
          `(num_layers, batch_size, num_hidden)`. If `bidirectional` is True,
          shape will instead be `(2*num_layers, batch_size, num_hidden)`. If
          `states` is None, zeros will be used as default begin states.

    Outputs:
        - **out**: output tensor with shape `(sequence_length, batch_size, num_hidden)`
          when `layout` is "TNC". If `bidirectional` is True, output shape will instead
          be `(sequence_length, batch_size, 2*num_hidden)`
        - **out_states**: a list of two output recurrent state tensors with the same
          shape as in `states`. If `states` is None `out_states` will not be returned.


    Examples
    --------
    >>> layer = mx.gluon.rnn.LSTM(100, 3)
    >>> layer.initialize()
    >>> input = mx.np.random.uniform(size=(5, 3, 10))
    >>> # by default zeros are used as begin state
    >>> output = layer(input)
    >>> # manually specify begin state.
    >>> h0 = mx.np.random.uniform(size=(3, 3, 100))
    >>> c0 = mx.np.random.uniform(size=(3, 3, 100))
    >>> output, hn = layer(input, [h0, c0])
    """
    def __init__(self, hidden_size, num_layers=1, layout='TNC',
                 dropout=0, bidirectional=False, input_size=0,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 projection_size=None, h2r_weight_initializer=None,
                 state_clip_min=None, state_clip_max=None, state_clip_nan=False,
                 dtype='float32', **kwargs):
        super(LSTM, self).__init__(hidden_size, num_layers, layout,
                                   dropout, bidirectional, input_size,
                                   i2h_weight_initializer, h2h_weight_initializer,
                                   i2h_bias_initializer, h2h_bias_initializer,
                                   'lstm', projection_size, h2r_weight_initializer,
                                   state_clip_min, state_clip_max, state_clip_nan,
                                   dtype, **kwargs)

    def state_info(self, batch_size=0):
        if self._projection_size is None:
            return [{'shape': (self._num_layers * self._dir, batch_size, self._hidden_size),
                     '__layout__': 'LNC', 'dtype': self._dtype},
                    {'shape': (self._num_layers * self._dir, batch_size, self._hidden_size),
                     '__layout__': 'LNC', 'dtype': self._dtype}]
        else:
            return [{'shape': (self._num_layers * self._dir, batch_size, self._projection_size),
                     '__layout__': 'LNC', 'dtype': self._dtype},
                    {'shape': (self._num_layers * self._dir, batch_size, self._hidden_size),
                     '__layout__': 'LNC', 'dtype': self._dtype}]


class GRU(_RNNLayer):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
    Note: this is an implementation of the cuDNN version of GRUs
    (slight modification compared to Cho et al. 2014; the reset gate :math:`r_t`
    is applied after matrix multiplication).

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
        r_t = sigmoid(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
        i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)} + b_{hn})) \\
        h_t = (1 - i_t) * n_t + i_t * h_{(t-1)} \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first layer,
    and :math:`r_t`, :math:`i_t`, :math:`n_t` are the reset, input, and new gates, respectively.

    Parameters
    ----------
    hidden_size: int
        The number of features in the hidden state h
    num_layers: int, default 1
        Number of recurrent layers.
    layout : str, default 'TNC'
        The format of input and output tensors. T, N and C stand for
        sequence length, batch size, and feature dimensions respectively.
    dropout: float, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer
    bidirectional: bool, default False
        If True, becomes a bidirectional RNN.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    dtype : str, default 'float32'
        Type to initialize the parameters and default states to
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.


    Inputs:
        - **data**: input tensor with shape `(sequence_length, batch_size, input_size)`
          when `layout` is "TNC". For other layouts, dimensions are permuted accordingly
          using transpose() operator which adds performance overhead. Consider creating
          batches in TNC layout during data batching step.
        - **states**: initial recurrent state tensor with shape
          `(num_layers, batch_size, num_hidden)`. If `bidirectional` is True,
          shape will instead be `(2*num_layers, batch_size, num_hidden)`. If
          `states` is None, zeros will be used as default begin states.

    Outputs:
        - **out**: output tensor with shape `(sequence_length, batch_size, num_hidden)`
          when `layout` is "TNC". If `bidirectional` is True, output shape will instead
          be `(sequence_length, batch_size, 2*num_hidden)`
        - **out_states**: output recurrent state tensor with the same shape as `states`.
          If `states` is None `out_states` will not be returned.


    Examples
    --------
    >>> layer = mx.gluon.rnn.GRU(100, 3)
    >>> layer.initialize()
    >>> input = mx.np.random.uniform(size=(5, 3, 10))
    >>> # by default zeros are used as begin state
    >>> output = layer(input)
    >>> # manually specify begin state.
    >>> h0 = mx.np.random.uniform(size=(3, 3, 100))
    >>> output, hn = layer(input, h0)
    """
    def __init__(self, hidden_size, num_layers=1, layout='TNC',
                 dropout=0, bidirectional=False, input_size=0,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 dtype='float32', **kwargs):
        super(GRU, self).__init__(hidden_size, num_layers, layout,
                                  dropout, bidirectional, input_size,
                                  i2h_weight_initializer, h2h_weight_initializer,
                                  i2h_bias_initializer, h2h_bias_initializer,
                                  'gru', None, None, None, None, False,
                                  dtype, **kwargs)

    def state_info(self, batch_size=0):
        return [{'shape': (self._num_layers * self._dir, batch_size, self._hidden_size),
                 '__layout__': 'LNC', 'dtype': self._dtype}]
