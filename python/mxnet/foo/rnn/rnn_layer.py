# coding: utf-8
# pylint: disable=no-member, invalid-name, protected-access, no-self-use
# pylint: disable=too-many-branches, too-many-arguments, no-self-use
# pylint: disable=too-many-lines, arguments-differ
"""Definition of various recurrent neural network layers."""
from __future__ import print_function

import warnings

from ... import symbol, init, ndarray
from ...base import string_types, numeric_types
from ..nn import Layer, HybridLayer
from .. import tensor_types
from . import rnn_cell


class _RNNLayer(Layer):
    def __init__(self, hidden_size, num_layers, layout, dropout,
                 bidirectional, input_size, mode, **kwargs):
        super(_RNNLayer, self).__init__(**kwargs)
        assert layout == 'TNC' or layout == 'NTC', \
            "Invalid layout %s; must be one of ['TNC' or 'NTC']"%layout
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._mode = mode
        self._layout = layout
        self._dropout = dropout
        self._bi = bidirectional
        self._input_size = input_size

        self._gates = {'rnn_relu': 1, 'rnn_tanh': 1, 'lstm': 4, 'gru': 3}[mode]

        self.i2h_weight = []
        self.h2h_weight = []
        self.i2h_bias = []
        self.h2h_bias = []

        ng, ni, nh = self._gates, input_size, hidden_size
        for i in range(num_layers):
            self.i2h_weight.append(self.params.get('l%d_i2h_weight'%i, shape=(ng*nh, ni)))
            self.h2h_weight.append(self.params.get('l%d_h2h_weight'%i, shape=(ng*nh, nh)))
            self.i2h_bias.append(self.params.get('l%d_i2h_bias'%i, shape=(ng*nh,)))
            self.h2h_bias.append(self.params.get('l%d_h2h_bias'%i, shape=(ng*nh,)))
            ni = nh

        self._unfused = self._unfuse()
        #self._unfused.hybridize()

    def _unfuse(self):
        """Unfuse the fused RNN in to a stack of rnn cells."""
        get_cell = {'rnn_relu': lambda **kwargs: rnn_cell.RNNCell(self._hidden_size,
                                                                  activation='relu',
                                                                  **kwargs),
                    'rnn_tanh': lambda **kwargs: rnn_cell.RNNCell(self._hidden_size,
                                                                  activation='tanh',
                                                                  **kwargs),
                    'lstm': lambda **kwargs: rnn_cell.LSTMCell(self._hidden_size,
                                                               **kwargs),
                    'gru': lambda **kwargs: rnn_cell.GRUCell(self._hidden_size,
                                                             **kwargs)}[self._mode]

        stack = rnn_cell.SequentialRNNCell(prefix=self.prefix, params=self.params)
        with stack.name_scope():
            ni = self._input_size
            for i in range(self._num_layers):
                if self._bi:
                    stack.add(rnn_cell.BidirectionalCell(
                        get_cell(prefix='l%d_'%i, input_size=ni),
                        get_cell(prefix='r%d_'%i, input_size=ni)))
                else:
                    stack.add(get_cell(prefix='l%d_'%i, input_size=ni))

                if self._dropout > 0 and i != self._num_layers - 1:
                    stack.add(DropoutCell(self._dropout))

                ni = self._hidden_size

        return stack

    def forward(self, inputs, states):
        if self._input_size == 0:
            self.i2h_weight[0].shape = (self._gates*self._hidden_size, inputs.shape[2])
            self.i2h_weight[0]._finish_deferred_init()
        if inputs.context.device_type == 'gpu':
            return self._forward_gpu(inputs, states)
        return self._forward_cpu(inputs, states)

    def _forward_cpu(self, inputs, states):
        ns = len(states)
        axis = self._layout.find('T')
        no = (self._bi + 1)*self._num_layers
        states = sum(zip(*((j for j in i) for i in states)), ())
        outputs, states = self._unfused.unroll(
            inputs.shape[axis], inputs, states,
            layout=self._layout, merge_outputs=True)
        new_states = []
        for i in range(ns):
            state = ndarray.concat(*(i.reshape((1,)+i.shape) for i in states[i::ns]), dim=0)
            new_states.append(state)

        return outputs, new_states

    def _forward_gpu(self, inputs, states):
        if self._layout == 'NTC':
            inputs = ndarray.swapaxes(inputs, dim1=0, dim2=1)
        ctx = inputs.context
        params = sum(zip(self.i2h_weight, self.h2h_weight), ())
        params += sum(zip(self.i2h_bias, self.h2h_bias), ())
        params = (i.data(ctx).reshape((-1,)) for i in params)
        params = ndarray.concat(*params, dim=0)

        rnn = ndarray.RNN(inputs, params, *states, state_size=self._hidden_size,
                          num_layers=self._num_layers, bidirectional=self._bi,
                          p=self._dropout, state_outputs=True, mode=self._mode)

        if self._mode == 'lstm':
            outputs, states = rnn[0], [rnn[1], rnn[2]]
        else:
            outputs, states = rnn[0], [rnn[1]]

        if self._layout == 'NTC':
            outputs = ndarray.swapaxes(outputs, dim1=0, dim2=1)

        return outputs, states


class RNN(_RNNLayer):
    def __init__(self, hidden_size, activation='relu', num_layers=1,
                 layout='TNC', dropout=0, bidirectional=False,
                 input_size=0, **kwargs):
        super(LSTM, self).__init__(hidden_size, num_layers, layout, dropout,
                                   bidirectional, input_size, 'rnn_'+activation,
                                   **kwargs)


class LSTM(_RNNLayer):
    def __init__(self, hidden_size, num_layers=1, layout='TNC', dropout=0,
                 bidirectional=False, input_size=0, **kwargs):
        super(LSTM, self).__init__(hidden_size, num_layers, layout, dropout,
                                   bidirectional, input_size, 'lstm', **kwargs)


class GRU(_RNNLayer):
    def __init__(self, hidden_size, num_layers=1, layout='TNC', dropout=0,
                 bidirectional=False, input_size=0, **kwargs):
        super(LSTM, self).__init__(hidden_size, num_layers, layout, dropout,
                                   bidirectional, input_size, 'gru', **kwargs)
