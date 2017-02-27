# coding: utf-8
# pylint: disable=no-member, invalid-name, protected-access, no-self-use
# pylint: disable=too-many-branches, too-many-arguments, no-self-use
"""Definition of various recurrent neural network cells."""
from __future__ import print_function

import warnings

from .. import symbol, init, ndarray
from ..base import string_types

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
        self._modified = False

        self.reset()

    def reset(self):
        """Reset before re-using the cell for another graph"""
        self._init_counter = -1
        self._counter = -1

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

    def begin_state(self, func=symbol.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        func : callable, default symbol.zeros
            Function for creating initial state. Can be symbol.zeros,
            symbol.uniform, symbol.Variable etc.
            Use symbol.Variable if you want to directly
            feed input as states.
        **kwargs :
            more keyword arguments passed to func. For example
            mean, std, dtype, etc.

        Returns
        -------
        states : nested list of Symbol
            starting states for first RNN step
        """
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        states = []
        for shape in self.state_shape:
            self._init_counter += 1
            if shape is None:
                state = func(name='%sbegin_state_%d'%(self._prefix, self._init_counter),
                             **kwargs)
            else:
                state = func(name='%sbegin_state_%d'%(self._prefix, self._init_counter),
                             shape=shape, **kwargs)
            states.append(state)
        return states

    def unpack_weights(self, args):
        """Unpack fused weight matrices into separate
        weight matrices

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing packed weights.
            usually from Module.get_output()

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell unpacked.
        """
        #pylint: disable=R0201
        return args.copy()

    def pack_weights(self, args):
        """Pack separate weight matrices into fused
        weight.

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing unpacked weights.

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell packed.
        """
        #pylint: disable=R0201
        return args.copy()

    def unroll(self, length, inputs=None, begin_state=None,
               input_prefix='', layout='NTC', merge_outputs=False):
        """Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            number of steps to unroll
        inputs : Symbol, list of Symbol, or None
            if inputs is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If inputs is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).

            If inputs is None, Placeholder variables are
            automatically created.
        begin_state : nested list of Symbol
            input states. Created by begin_state()
            or output state of another cell. Created
            from begin_state() if None.
        input_prefix : str
            prefix for automatically created input
            placehodlers.
        layout : str
            layout of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool
            if False, return outputs as a list of Symbols.
            If True, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

        Returns
        -------
        outputs : list of Symbol
            output symbols.
        states : Symbol or nested list of Symbol
            has the same structure as begin_state()
        """
        self.reset()

        axis = layout.find('T')
        if inputs is None:
            inputs = [symbol.Variable('%st%d_data'%(input_prefix, i))
                      for i in range(length)]
        elif isinstance(inputs, symbol.Symbol):
            assert len(inputs.list_outputs()) == 1, \
                "unroll doesn't allow grouped symbol as input. Please " \
                "convert to list first or let unroll handle slicing"
            inputs = symbol.SliceChannel(inputs, axis=axis, num_outputs=length,
                                         squeeze_axis=1)
        else:
            assert len(inputs) == length
        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        outputs = []
        for i in range(length):
            output, states = self(inputs[i], states)
            outputs.append(output)

        if merge_outputs:
            outputs = [symbol.expand_dims(i, axis=axis) for i in outputs]
            outputs = symbol.Concat(*outputs, dim=axis)
        return outputs, states

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
        return [(0, self._num_hidden)]

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
        h2h = symbol.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
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

    def unpack_weights(self, args):
        """Unpack fused weight matrices into separate
        weight matrices

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing packed weights.
            usually from Module.get_output()

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell unpacked.
        """
        args = args.copy()
        outs = ['_i', '_f', '_c', '_o']
        h = self._num_hidden
        for i in ['i2h', 'h2h']:
            weight = args.pop('%s%s_weight'%(self._prefix, i))
            bias = args.pop('%s%s_bias'%(self._prefix, i))
            for j, name in enumerate(outs):
                wname = '%s%s%s_weight'%(self._prefix, i, name)
                args[wname] = weight[j*h:(j+1)*h].copy()
                bname = '%s%s%s_bias'%(self._prefix, i, name)
                args[bname] = bias[j*h:(j+1)*h].copy()
        return args

    def pack_weights(self, args):
        """Pack separate weight matrices into fused
        weight.

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing unpacked weights.

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell packed.
        """
        args = args.copy()
        outs = ['_i', '_f', '_c', '_o']
        for i in ['i2h', 'h2h']:
            weight = []
            bias = []
            for name in outs:
                wname = '%s%s%s_weight'%(self._prefix, i, name)
                weight.append(args.pop(wname))
                bname = '%s%s%s_bias'%(self._prefix, i, name)
                bias.append(args.pop(bname))
            args['%s%s_weight'%(self._prefix, i)] = ndarray.concatenate(weight)
            args['%s%s_bias'%(self._prefix, i)] = ndarray.concatenate(bias)
        return args

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
        forget_gate = symbol.Activation(slice_gates[1], act_type="sigmoid",
                                        name='%sf'%name)
        in_transform = symbol.Activation(slice_gates[2], act_type="tanh",
                                         name='%sc'%name)
        out_gate = symbol.Activation(slice_gates[3], act_type="sigmoid",
                                     name='%so'%name)
        next_c = symbol._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                        name='%sstate'%name)
        next_h = symbol._internal._mul(out_gate, symbol.Activation(next_c, act_type="tanh"),
                                       name='%sout'%name)

        return next_h, [next_h, next_c]


class FusedRNNCell(BaseRNNCell):
    """Fusing RNN layers across time step into one kernel.
    Improves speed but is less flexible. Currently only
    supported if using cuDNN on GPU.

    Parameters
    ----------
    """
    def __init__(self, num_hidden, num_layers=1, mode='lstm', bidirectional=False,
                 dropout=0., get_next_state=False, initializer=None,
                 prefix=None, params=None):
        if prefix is None:
            prefix = '%s_'%mode
        super(FusedRNNCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._mode = mode
        self._bidirectional = bidirectional
        self._dropout = dropout
        self._get_next_state = get_next_state
        if initializer is None:
            initializer = init.Xavier(factor_type='in', magnitude=2.34)
        if not isinstance(initializer, init.FusedRNN):
            initializer = init.FusedRNN(initializer, num_hidden, num_layers,
                                        mode, bidirectional)
        self._parameter = self.params.get('parameters', init=initializer)

        self._directions = self._bidirectional + 1
        self._weight_names = {'rnn_relu': [''],
                              'rnn_tanh': [''],
                              'lstm': ['_i', '_f', '_c', '_o'],
                              'gru': ['_r', '_z', '_o']}[self._mode]
        self._num_weights = len(self._weight_names)

    @property
    def state_shape(self):
        """shape(s) of states"""
        b = self._bidirectional + 1
        n = (self._mode == 'lstm') + 1
        return [(b*self._num_layers, 0, self._num_hidden)]*n

    def _slice_weights(self, arr, li, lh):
        """slice fused rnn weights"""
        args = {}
        b = self._directions
        m = self._num_weights
        c = self._weight_names
        d = ['l', 'r']

        p = 0
        for i in range(self._num_layers):
            for j in range(b):
                for k in range(m):
                    name = '%s%s%d_i2h%s_weight'%(self._prefix, d[j], i, c[k])
                    if i > 0:
                        size = b*lh*lh
                        args[name] = arr[p:p+size].reshape((lh, b*lh))
                    else:
                        size = li*lh
                        args[name] = arr[p:p+size].reshape((lh, li))
                    p += size
                for k in range(m):
                    name = '%s%s%d_h2h%s_weight'%(self._prefix, d[j], i, c[k])
                    size = lh**2
                    args[name] = arr[p:p+size].reshape((lh, lh))
                    p += size

        for i in range(self._num_layers):
            for j in range(b):
                for k in range(m):
                    name = '%s%s%d_i2h%s_bias'%(self._prefix, d[j], i, c[k])
                    args[name] = arr[p:p+lh]
                    p += lh
                for k in range(m):
                    name = '%s%s%d_h2h%s_bias'%(self._prefix, d[j], i, c[k])
                    args[name] = arr[p:p+lh]
                    p += lh

        assert p == arr.size, "Invalid parameters size for FusedRNNCell"
        return args

    def unpack_weights(self, args):
        """Unpack fused weight matrices into separate
        weight matrices

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing packed weights.
            usually from Module.get_output()

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell unpacked.
        """
        args = args.copy()
        arr = args.pop(self._parameter.name)
        b = self._directions
        m = self._num_weights
        h = self._num_hidden
        num_input = arr.size//b//h//m - (self._num_layers - 1)*(h+b*h+2) - h - 2

        nargs = self._slice_weights(arr, num_input, self._num_hidden)
        args.update({name: nd.copy() for name, nd in nargs.items()})
        return args

    def pack_weights(self, args):
        """Pack separate weight matrices into fused
        weight.

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing unpacked weights.

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell packed.
        """
        args = args.copy()
        b = self._bidirectional + 1
        m = self._num_weights
        c = self._weight_names
        h = self._num_hidden
        w0 = args['%sl0_i2h%s_weight'%(self._prefix, c[0])]
        num_input = w0.shape[1]
        total = (num_input+h+2)*h*m*b + (self._num_layers-1)*m*h*(h+b*h+2)*b

        arr = ndarray.zeros((total,), ctx=w0.context, dtype=w0.dtype)
        for name, nd in self._slice_weights(arr, num_input, h).items():
            nd[:] = args.pop(name)
        args[self._parameter.name] = arr
        return args

    def __call__(self, inputs, states):
        raise NotImplementedError("FusedRNNCell cannot be stepped. Please use unroll")

    def unroll(self, length, inputs=None, begin_state=None,
               input_prefix='', layout='NTC', merge_outputs=False):
        """Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            number of steps to unroll
        inputs : Symbol, list of Symbol, or None
            if inputs is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.
            using 'TNC' is more efficient for FusedRNNCell.

            If inputs is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...). using single symbol is
            more efficient for FusedRNNCell.

            If inputs is None, a single placeholder variable is
            automatically created.
        begin_state : nested list of Symbol
            input states. Created by begin_state()
            or output state of another cell. Created
            from begin_state() if None.
        input_prefix : str
            prefix for automatically created input
            placehodlers.
        layout : str
            layout of input/output symbol.

        Returns
        -------
        outputs : list of Symbol
            output symbols.
        states : Symbol or nested list of Symbol
            has the same structure as begin_state()
        """
        self.reset()

        axis = layout.find('T')
        if inputs is None:
            inputs = symbol.Variable('%sdata'%input_prefix)
        if isinstance(inputs, symbol.Symbol):
            assert len(inputs.list_outputs()) == 1, \
                "unroll doesn't allow grouped symbol as input. Please " \
                "convert to list first or let unroll handle slicing"
            if axis == 1:
                warnings.warn("NTC layout detected. Consider using "
                              "TNC for FusedRNNCell for faster speed")
                inputs = symbol.SwapAxis(inputs, dim1=0, dim2=1)
            else:
                assert axis == 0, "Unsupported layout %s"%layout
        else:
            assert len(inputs) == length
            inputs = [symbol.expand_dims(i, axis=1) for i in inputs]
            inputs = symbol.Concat(*inputs, dim=1)
        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        if self._mode == 'lstm':
            states = {'state': states[0], 'state_cell': states[1]}
        else:
            states = {'state': states}

        rnn = symbol.RNN(data=inputs, parameters=self._parameter,
                         state_size=self._num_hidden, num_layers=self._num_layers,
                         bidirectional=self._bidirectional, p=self._dropout,
                         state_outputs=self._get_next_state,
                         mode=self._mode, name=self._prefix+'rnn',
                         **states)

        if not self._get_next_state:
            outputs, states = rnn, []
        elif self._mode == 'lstm':
            outputs, states = rnn[0], [rnn[1], rnn[2]]
        else:
            outputs, states = rnn[0], rnn[1]

        if not merge_outputs:
            warnings.warn("Call FusedRNNCell.unroll with merge_outputs=True "
                          "for faster speed")
            outputs = list(symbol.SliceChannel(outputs, axis=axis, num_outputs=length,
                                               squeeze_axis=1))
        elif axis == 1:
            outputs = symbol.SwapAxis(outputs, dim1=0, dim2=1)

        return outputs, states


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
        return sum([c.state_shape for c in self._cells], [])

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
        return sum([c.begin_state(**kwargs) for c in self._cells], [])

    def unpack_weights(self, args):
        for cell in self._cells:
            args = cell.unpack_weights(args)
        return args

    def pack_weights(self, args):
        for cell in self._cells:
            args = cell.pack_weights(args)
        return args

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
        p = 0
        for cell in self._cells:
            n = len(cell.state_shape)
            state = states[p:p+n]
            p += n
            inputs, state = cell(inputs, state)
            next_states.append(state)
        return inputs, sum(next_states, [])

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

    def unpack_weights(self, args):
        """Unpack fused weight matrices into separate
        weight matrices

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing packed weights.
            usually from Module.get_output()

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell unpacked.
        """
        return self.base_cell.unpack_weights(args)

    def pack_weights(self, args):
        """Pack separate weight matrices into fused
        weight.

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing unpacked weights.

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell packed.
        """
        return self.base_cell.pack_weights(args)

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

