# coding: utf-8
"""Definition of various recurrent neural network cells."""

import copy

from .. import symbol
from .. import ndarray
from ..base import numeric_types, string_types

class RNNParams(object):
    def __init__(self):
        self._params = {}

    def get(self, name, **kwargs):
        if name not in self._params:
            self._params[name] = symbol.Variable(name, **kwargs)
        return self._params[name]


class BaseRNNCell(object):
    """Abstract base class for RNN cells"""
    def __call__(self, inputs, states, params, prefix=''):
        """construct symbol"""
        raise NotImplementedError()

    @property
    def state_shape(self):
        """shape(s) of states"""
        raise NotImplementedError()

    @property
    def output_shape(self):
        """shape(s) of output"""
        raise NotImplementedError()

    def begin_state(self, prefix='', init_sym=symbol.zeros, **kwargs):
        """initial state"""
        state_shape = self.state_shape
        def recursive(shape, c):
            if shape is None:
                c[0] += 1
                return init_sym(name='%sbegin_state_%d'%(prefix, c[0]), **kwargs)
            elif isinstance(shape, tuple):
                assert len(shape) == 0 or isinstance(shape[0], numeric_types)
                c[0] += 1
                return init_sym(name='%sbegin_state_%d'%(prefix, c[0]), shape=shape, **kwargs)
            else:
                assert isinstance(shape, list)
                return [recursive(i, c) for i in shape]

        return recursive(state_shape, [-1])

    def _get_activation(self, x, activation, **kwargs):
        if isinstance(activation, string_types):
            return symbol.Activation(x, act_type=activation, **kwargs)
        else:
            return activation(x, **kwargs)

class RNNCell(BaseRNNCell):
    """Simple recurrent neural network cell"""
    def __init__(self, num_hidden, activation='tanh'):
        self._num_hidden = num_hidden
        self._activation = activation
        self._counter = 0

    @property
    def state_shape(self):
        return (0, self._num_hidden)

    @property
    def output_shape(self):
        return (0, self._num_hidden)

    def __call__(self, inputs, states, params, prefix=''):
        W = params.get('%si2h_weight'%prefix)
        B = params.get('%si2h_bias'%prefix)
        U = params.get('%sh2h_weight'%prefix)
        name = '%st%d_'%(prefix, self._counter)
        i2h = symbol.FullyConnected(data=inputs, weight=W, bias=B,
                                    num_hidden=self._num_hidden,
                                    name='%si2h'%name)
        h2h = symbol.FullyConnected(data=states, weight=U, no_bias=True,
                                    num_hidden=self._num_hidden,
                                    name='%sh2h'%name)
        output = self._get_activation(i2h + h2h, self._activation,
                                      name='%sout'%name)

        self._counter += 1
        return output, output

class LSTMCell(BaseRNNCell):
    """LSTM cell"""
    def __init__(self, num_hidden):
        self._num_hidden = num_hidden
        self._counter = 0

    @property
    def state_shape(self):
        return [(0, self._num_hidden), (0, self._num_hidden)]

    @property
    def output_shape(self):
        return (0, self._num_hidden)

    def __call__(self, inputs, states, params, prefix=''):
        iW = params.get('%si2h_weight'%prefix)
        iB = params.get('%si2h_bias'%prefix)
        hW = params.get('%sh2h_weight'%prefix)
        name = '%st%d_'%(prefix, self._counter)
        i2h = symbol.FullyConnected(data=inputs, weight=iW, bias=iB,
                                    num_hidden=self._num_hidden*4,
                                    name='%si2h'%name)
        h2h = symbol.FullyConnected(data=states[0], weight=hW, no_bias=True,
                                    num_hidden=self._num_hidden*4,
                                    name='%sh2h'%name)
        gates = i2h + h2h
        slice_gates = symbol.SliceChannel(gates, num_outputs=4,
                                          name="%sslice"%name)
        in_gate = symbol.Activation(slice_gates[0], act_type="sigmoid",
                                    name='%si'%name)
        in_transform = symbol.Activation(slice_gates[1], act_type="tanh",
                                         name='%sc'%name)
        forget_gate = symbol.Activation(slice_gates[2], act_type="sigmoid",
                                        name='%sf'%name)
        out_gate = symbol.Activation(slice_gates[3], act_type="sigmoid",
                                     name='%so'%name)
        next_c = symbol._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                        name='%sstate'%name)
        next_h = symbol._internal._mul(out_gate, symbol.Activation(next_c, act_type="tanh"),
                                        name='%sout'%name)

        self._counter += 1
        return next_h, [next_h, next_c]


class StackedRNNCell(BaseRNNCell):
    """Stacked multple rnn cels"""
    def __init__(self, cells):
        self._cells = [copy.copy(c) for c in cells]
        self._counter = 0

    @property
    def state_shape(self):
        return [c.state_shape for c in self._cells]

    @property
    def output_shape(self):
        return self._cells[-1].output_shape

    def begin_state(self, prefix='', **kwargs):
        return [c.begin_state(prefix='%sstack%d_'%(prefix, i)) for i, c in enumerate(self._cells)]


    def __call__(self, inputs, states, params, prefix=''):
        next_states = []
        for i, (cell, state) in enumerate(zip(self._cells, states)):
            inputs, state = cell(inputs, state, params, prefix='%sstack%d_'%(prefix, i))
            next_states.append(state)
        return inputs, next_states













