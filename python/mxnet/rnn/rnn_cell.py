# coding: utf-8
"""Definition of various recurrent neural network cells."""
from __future__ import print_function
import copy

from .. import symbol
from .. import ndarray
from ..base import numeric_types, string_types

class RNNParams(object):
    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def get(self, name, **kwargs):
        name = self._prefix + name
        if name not in self._params:
            self._params[name] = symbol.Variable(name, **kwargs)
        return self._params[name]


class BaseRNNCell(object):
    """Abstract base class for RNN cells"""
    def __init__(self, prefix='', params=None):
        if params is None:
            params = RNNParams(prefix)
            self._own_params = True
        else:
            self._own_params = False
        self._prefix = prefix
        self._params = params
        self._init_counter = -1
        self._counter = -1

    def __call__(self, inputs, states):
        """construct symbol"""
        raise NotImplementedError()

    @property
    def params(self):
        """Parameters of this cell"""
        self._own_params = False
        return self._params

    @property
    def state_shape(self):
        """shape(s) of states"""
        raise NotImplementedError()

    @property
    def output_shape(self):
        """shape(s) of output"""
        raise NotImplementedError()

    def begin_state(self, init_sym=symbol.zeros, **kwargs):
        """initial state"""
        state_shape = self.state_shape
        def recursive(shape):
            if isinstance(shape, tuple):
                assert len(shape) == 0 or isinstance(shape[0], numeric_types)
                self._init_counter += 1
                return init_sym(name='%sinit_%d'%(self._prefix, self._init_counter),
                                shape=shape, **kwargs)
            else:
                assert isinstance(shape, list)
                return [recursive(i) for i in shape]

        return recursive(state_shape)

    def _get_activation(self, x, activation, **kwargs):
        if isinstance(activation, string_types):
            return symbol.Activation(x, act_type=activation, **kwargs)
        else:
            return activation(x, **kwargs)

class RNNCell(BaseRNNCell):
    """Simple recurrent neural network cell"""
    def __init__(self, num_hidden, activation='tanh', prefix='rnn_', params=None):
        super(RNNCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._activation = activation
        self._iW = self.params.get('i2h_weight')
        self._iB = self.params.get('i2h_bias')
        self._hW = self.params.get('h2h_weight')
        self._hB = self.params.get('h2h_bias')

    @property
    def state_shape(self):
        return (0, self._num_hidden)

    @property
    def output_shape(self):
        return (0, self._num_hidden)

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_'%(self._prefix, self._counter)
        i2h = symbol.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden,
                                    name='%si2h'%name)
        h2h = symbol.FullyConnected(data=states, weight=self._hW, bias=self._hB,
                                    num_hidden=self._num_hidden,
                                    name='%sh2h'%name)
        output = self._get_activation(i2h + h2h, self._activation,
                                      name='%sout'%name)

        return output, output

class LSTMCell(BaseRNNCell):
    """LSTM cell"""
    def __init__(self, num_hidden, prefix='lstm_', params=None):
        super(LSTMCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._iW = self.params.get('i2h_weight')
        self._iB = self.params.get('i2h_bias')
        self._hW = self.params.get('h2h_weight')
        self._hB = self.params.get('h2h_bias')

    @property
    def state_shape(self):
        return [(0, self._num_hidden), (0, self._num_hidden)]

    @property
    def output_shape(self):
        return (0, self._num_hidden)

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_'%(self._prefix, self._counter)
        i2h = symbol.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden*4,
                                    name='%si2h'%name)
        h2h = symbol.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
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

        return next_h, [next_h, next_c]


class SequentialRNNCell(BaseRNNCell):
    """Stacked multple rnn cels"""
    def __init__(self, params=None):
        super(SequentialRNNCell, self).__init__(prefix='', params=params)
        self._override_cell_params = params is not None
        self._cells = []

    def add(self, cell):
        self._cells.append(cell)
        if self._override_cell_params:
            assert cell._own_params, \
                "Either specify params for SequentialRNNCell " \
                "or child cells, not both."
            cell.params._params.update(self.params._params)
        self.params._params.update(cell.params._params)

    @property
    def state_shape(self):
        return [c.state_shape for c in self._cells]

    @property
    def output_shape(self):
        return self._cells[-1].output_shape

    def begin_state(self, **kwargs):
        return [c.begin_state(**kwargs) for c in self._cells]

    def __call__(self, inputs, states):
        self._counter += 1
        next_states = []
        for cell, state in zip(self._cells, states):
            inputs, state = cell(inputs, state)
            next_states.append(state)
        return inputs, next_states













