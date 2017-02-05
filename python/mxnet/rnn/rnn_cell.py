# coding: utf-8
# pylint: disable=no-member, invalid-name, protected-access
"""Definition of various recurrent neural network cells."""
from __future__ import print_function

from .. import symbol
from ..base import numeric_types, string_types

class RNNParams(object):
    """Container for holding variables.
    Used by RNN cells for parameter sharing between cells.

    Parameters
    ----------
    prefix : str
        All variables' name created by this container will
        be prepended with prefix
    """
    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def get(self, name, **kwargs):
        """Get a variable with name or create a new one if missing.

        Parameters
        ----------
        name : str
            name of the variable
        **kwargs :
            more arguments that's passed to symbol.Variable
        """
        name = self._prefix + name
        if name not in self._params:
            self._params[name] = symbol.Variable(name, **kwargs)
        return self._params[name]


class BaseRNNCell(object):
    """Abstract base class for RNN cells

    Parameters
    ----------
    prefix : str
        prefix for name of layers
        (and name of weight if params is None)
    params : RNNParams or None
        container for weight sharing between cells.
        created if None.
    """
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
        self._modified = False

    def __call__(self, inputs, states):
        """Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        """
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
        """Initial state for this cell.

        Parameters
        ----------
        init_sym : Symbol, default symbol.zeros
            Symbol for generating initial state. Can be zeros,
            ones, uniform, normal, etc.
        **kwargs :
            more keyword arguments passed to init_sym. For example
            mean, std, dtype, etc.

        Returns
        -------
        states : nested list of Symbol
            starting states for first RNN step
        """
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        state_shape = self.state_shape
        def recursive(shape):
            """Recursively construct input states"""
            if isinstance(shape, tuple):
                assert len(shape) == 0 or isinstance(shape[0], numeric_types)
                self._init_counter += 1
                return init_sym(name='%sinit_%d'%(self._prefix, self._init_counter),
                                shape=shape, **kwargs)
            else:
                assert isinstance(shape, list)
                return [recursive(i) for i in shape]

        return recursive(state_shape)

    #pylint: disable=no-self-use
    def _get_activation(self, inputs, activation, **kwargs):
        """Get activation function. Convert if is string"""
        if isinstance(activation, string_types):
            return symbol.Activation(inputs, act_type=activation, **kwargs)
        else:
            return activation(inputs, **kwargs)


class RNNCell(BaseRNNCell):
    """Simple recurrent neural network cell

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    activation : str or Symbol, default 'tanh'
        type of activation function
    prefix : str, default 'rnn_'
        prefix for name of layers
        (and name of weight if params is None)
    params : RNNParams or None
        container for weight sharing between cells.
        created if None.
    """
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
        """shape(s) of states"""
        return (0, self._num_hidden)

    @property
    def output_shape(self):
        """shape(s) of output"""
        return (0, self._num_hidden)

    def __call__(self, inputs, states):
        """Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        """
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
    """Long-Short Term Memory (LSTM) network cell.

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    prefix : str, default 'rnn_'
        prefix for name of layers
        (and name of weight if params is None)
    params : RNNParams or None
        container for weight sharing between cells.
        created if None.
    """
    def __init__(self, num_hidden, prefix='lstm_', params=None):
        super(LSTMCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._iW = self.params.get('i2h_weight')
        self._iB = self.params.get('i2h_bias')
        self._hW = self.params.get('h2h_weight')
        self._hB = self.params.get('h2h_bias')

    @property
    def state_shape(self):
        """shape(s) of states"""
        return [(0, self._num_hidden), (0, self._num_hidden)]

    @property
    def output_shape(self):
        """shape(s) of output"""
        return (0, self._num_hidden)

    def __call__(self, inputs, states):
        """Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        """
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
    """Sequantially stacking multiple RNN cells

    Parameters
    ----------
    params : RNNParams or None
        container for weight sharing between cells.
        created if None.
    """
    def __init__(self, params=None):
        super(SequentialRNNCell, self).__init__(prefix='', params=params)
        self._override_cell_params = params is not None
        self._cells = []

    def add(self, cell):
        """Append a cell into the stack.

        Parameters
        ----------
        cell : rnn cell
        """
        self._cells.append(cell)
        if self._override_cell_params:
            assert cell._own_params, \
                "Either specify params for SequentialRNNCell " \
                "or child cells, not both."
            cell.params._params.update(self.params._params)
        self.params._params.update(cell.params._params)

    @property
    def state_shape(self):
        """shape(s) of states"""
        return [c.state_shape for c in self._cells]

    @property
    def output_shape(self):
        """shape(s) of output"""
        return self._cells[-1].output_shape

    def begin_state(self, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        init_sym : Symbol, default symbol.zeros
            Symbol for generating initial state. Can be zeros,
            ones, uniform, normal, etc.
        **kwargs :
            more keyword arguments passed to init_sym. For example
            mean, std, dtype, etc.

        Returns
        -------
        states : nested list of Symbol
            starting states for first RNN step
        """
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        return [c.begin_state(**kwargs) for c in self._cells]

    def __call__(self, inputs, states):
        """Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        """
        self._counter += 1
        next_states = []
        for cell, state in zip(self._cells, states):
            inputs, state = cell(inputs, state)
            next_states.append(state)
        return inputs, next_states

class ModifierCell(BaseRNNCell):
    """Base class for modifier cells. A modifier
    cell takes a base cell, apply modifications
    on it (e.g. Dropout), and returns a new cell.

    After applying modifiers the base cell should
    no longer be called directly. The modifer cell
    should be used instead.
    """
    def __init__(self, base_cell):
        super(ModifierCell, self).__init__()
        base_cell._modified = True
        self.base_cell = base_cell

    @property
    def params(self):
        """Parameters of this cell"""
        self._own_params = False
        return self.base_cell.params

    @property
    def state_shape(self):
        """shape(s) of states"""
        return self.base_cell.state_shape

    @property
    def output_shape(self):
        """shape(s) of output"""
        return self.base_cell.output_shape

    def begin_state(self, init_sym=symbol.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        init_sym : Symbol, default symbol.zeros
            Symbol for generating initial state. Can be zeros,
            ones, uniform, normal, etc.
        **kwargs :
            more keyword arguments passed to init_sym. For example
            mean, std, dtype, etc.

        Returns
        -------
        states : nested list of Symbol
            starting states for first RNN step
        """
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        self.base_cell._modified = False
        begin = self.base_cell.begin_state(init_sym, **kwargs)
        self.base_cell._modified = True
        return begin

    def __call__(self, inputs, states):
        """Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        """
        raise NotImplementedError


class DropoutCell(ModifierCell):
    """Apply dropout on base cell"""
    def __init__(self, base_cell, dropout_outputs=0., dropout_states=0.):
        super(DropoutCell, self).__init__(base_cell)
        self.dropout_outputs = dropout_outputs
        self.dropout_states = dropout_states

    def __call__(self, inputs, states):
        """Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        """
        output, states = self.base_cell(inputs, states)
        if self.dropout_outputs > 0:
            output = symbol.Dropout(data=output, p=self.dropout_outputs)
        if self.dropout_states > 0:
            states = symbol.Dropout(data=states, p=self.dropout_states)
        return output, states


class ZoneoutCell(ModifierCell):
    """Apply Zoneout on base cell"""
    def __init__(self, base_cell, zoneout_outputs=0., zoneout_states=0.):
        super(ZoneoutCell, self).__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self.prev_output = None

    def __call__(self, inputs, states):
        """Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        """
        raise NotImplementedError



