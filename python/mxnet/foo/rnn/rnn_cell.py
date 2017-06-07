# coding: utf-8
# pylint: disable=no-member, invalid-name, protected-access, no-self-use
# pylint: disable=too-many-branches, too-many-arguments, no-self-use
# pylint: disable=too-many-lines, arguments-differ
"""Definition of various recurrent neural network cells."""
from __future__ import print_function

import warnings

from ... import symbol, init, ndarray
from ...base import string_types, numeric_types
from ..nn import Layer
from .. import tensor_types


def _cells_state_shape(cells):
    return sum([c.state_shape for c in cells], [])

def _cells_state_info(cells, batch_size):
    return sum([c.state_info(batch_size) for c in cells], [])

def _cells_begin_state(cells, **kwargs):
    return sum([c.begin_state(**kwargs) for c in cells], [])

def _cells_unpack_weights(cells, args):
    for cell in cells:
        args = cell.unpack_weights(args)
    return args

def _cells_pack_weights(cells, args):
    for cell in cells:
        args = cell.pack_weights(args)
    return args

def _get_begin_state(cell, F, begin_state, inputs, batch_size):
    if begin_state is None:
        if F is ndarray:
            ctx = inputs.context if isinstance(inputs, tensor_types) else inputs[0].context
            with ctx:
                begin_state = cell.begin_state(func=F.zeros, batch_size=batch_size)
        else:
            begin_state = cell.begin_state(func=F.zeros, batch_size=batch_size)
    return begin_state

def _format_sequence(length, inputs, layout, merge, in_layout=None):
    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    batch_axis = layout.find('N')
    batch_size = 0
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, symbol.Symbol):
        F = symbol
        if merge is False:
            assert len(inputs.list_outputs()) == 1, \
                "unroll doesn't allow grouped symbol as input. Please convert " \
                "to list with list(inputs) first or let unroll handle splitting."
            inputs = list(symbol.split(inputs, axis=in_axis, num_outputs=length,
                                       squeeze_axis=1))
    elif isinstance(inputs, ndarray.NDArray):
        F = ndarray
        batch_size = inputs.shape[batch_axis]
        if merge is False:
            assert length is None or length == inputs.shape[in_axis]
            inputs = ndarray.split(inputs, axis=in_axis, num_outputs=inputs.shape[in_axis],
                                   squeeze_axis=1)
    else:
        assert length is None or len(inputs) == length
        if isinstance(inputs[0], symbol.Symbol):
            F = symbol
        else:
            F = ndarray
            batch_size = inputs[0].shape[batch_axis]
        if merge is True:
            inputs = [F.expand_dims(i, axis=axis) for i in inputs]
            inputs = F.concat(*inputs, dim=axis)
            in_axis = axis

    if isinstance(inputs, tensor_types) and axis != in_axis:
        inputs = F.swapaxes(inputs, dim1=axis, dim2=in_axis)

    return inputs, axis, F, batch_size


class RecurrentCell(Layer):
    """Abstract base class for RNN cells

    Parameters
    ----------
    prefix : str, optional
        Prefix for names of layers
        (this prefix is also used for names of weights if `params` is None
        i.e. if `params` are being created and not reused)
    params : Parameter or None, optional
        Container for weight sharing between cells.
        A new Parameter container is created if `params` is None.
    """
    def __init__(self, prefix=None, params=None):
        super(RecurrentCell, self).__init__(prefix=prefix, params=params)
        self._modified = False
        self.reset()

    def reset(self):
        """Reset before re-using the cell for another graph."""
        self._init_counter = -1
        self._counter = -1

    def state_info(self, batch_size=0):
        """shape and layout information of states"""
        raise NotImplementedError()

    @property
    def state_shape(self):
        """shape(s) of states"""
        return [ele['shape'] for ele in self.state_info()]

    @property
    def _gate_names(self):
        """name(s) of gates"""
        return ()

    @property
    def _curr_prefix(self):
        return '%st%d_'%(self.prefix, self._counter)

    def begin_state(self, func=symbol.zeros, batch_size=0, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        func : callable, default symbol.zeros
            Function for creating initial state.

            For Symbol API, func can be symbol.zeros, symbol.uniform,
            symbol.var etc. Use symbol.var if you want to directly
            feed input as states.

            For NDArray API, func can be ndarray.zeros, ndarray.ones, etc.
        batch_size: int, default 0
            Only required for NDArray API. Size of the batch ('N' in layout)
            dimension of input.

        **kwargs :
            additional keyword arguments passed to func. For example
            mean, std, dtype, etc.

        Returns
        -------
        states : nested list of Symbol
            Starting states for the first RNN step.
        """
        assert not self._modified, \
            "After applying modifier cells (e.g. ZoneoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        states = []
        for info in self.state_info(batch_size):
            self._init_counter += 1
            if info is not None:
                info.update(kwargs)
            else:
                info = kwargs
            state = func(name='%sbegin_state_%d'%(self._prefix, self._init_counter),
                         **info)
            states.append(state)
        return states

    def unpack_weights(self, args):
        """Unpack fused weight matrices into separate
        weight matrices.

        For example, say you use a module object `mod` to run a network that has an lstm cell.
        In `mod.get_params()[0]`, the lstm parameters are all represented as a single big vector.
        `cell.unpack_weights(mod.get_params()[0])` will unpack this vector into a dictionary of
        more readable lstm parameters - c, f, i, o gates for i2h (input to hidden) and
        h2h (hidden to hidden) weights.

        Parameters
        ----------
        args : dict of str -> NDArray
            Dictionary containing packed weights.
            usually from `Module.get_params()[0]`.

        Returns
        -------
        args : dict of str -> NDArray
            Dictionary with unpacked weights associated with
            this cell.

        See Also
        --------
        pack_weights: Performs the reverse operation of this function.
        """
        args = args.copy()
        if not self._gate_names:
            return args
        h = self._num_hidden
        for group_name in ['i2h', 'h2h']:
            weight = args.pop('%s%s_weight'%(self._prefix, group_name))
            bias = args.pop('%s%s_bias' % (self._prefix, group_name))
            for j, gate in enumerate(self._gate_names):
                wname = '%s%s%s_weight' % (self._prefix, group_name, gate)
                args[wname] = weight[j*h:(j+1)*h].copy()
                bname = '%s%s%s_bias' % (self._prefix, group_name, gate)
                args[bname] = bias[j*h:(j+1)*h].copy()
        return args

    def pack_weights(self, args):
        """Pack separate weight matrices into a single packed
        weight.

        Parameters
        ----------
        args : dict of str -> NDArray
            Dictionary containing unpacked weights.

        Returns
        -------
        args : dict of str -> NDArray
            Dictionary with packed weights associated with
            this cell.
        """
        args = args.copy()
        if not self._gate_names:
            return args
        for group_name in ['i2h', 'h2h']:
            weight = []
            bias = []
            for gate in self._gate_names:
                wname = '%s%s%s_weight'%(self._prefix, group_name, gate)
                weight.append(args.pop(wname))
                bname = '%s%s%s_bias'%(self._prefix, group_name, gate)
                bias.append(args.pop(bname))
            args['%s%s_weight'%(self._prefix, group_name)] = ndarray.concatenate(weight)
            args['%s%s_bias'%(self._prefix, group_name)] = ndarray.concatenate(bias)
        return args

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        """Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            number of steps to unroll
        inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol, optional
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if None.
        layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool, optional
            If False, return outputs as a list of Symbols.
            If True, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.
            If None, output whatever is faster

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of begin_state().
        """
        self.reset()

        inputs, _, F, batch_size = _format_sequence(length, inputs, layout, False)
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)

        states = begin_state
        outputs = []
        for i in range(length):
            output, states = self(inputs[i], states)
            outputs.append(output)

        outputs, _, _, _ = _format_sequence(length, outputs, layout, merge_outputs)

        return outputs, states

    #pylint: disable=no-self-use
    def _get_activation(self, F, inputs, activation, **kwargs):
        """Get activation function. Convert if is string"""
        if isinstance(activation, string_types):
            return F.Activation(inputs, act_type=activation, **kwargs)
        else:
            return activation(inputs, **kwargs)

    def forward(self, inputs, states):
        """Unroll the recurrent cell for one time step.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch_size * num_units
        states : list of sym.Variable
            RNN state from previous step or the output of begin_state().

        Returns
        -------
        output : Symbol
            Symbol corresponding to the output from the RNN when unrolling
            for a single time step.
        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of begin_state().
            This can be used as input state to the next time step
            of this RNN.

        See Also
        --------
        begin_state: This function can provide the states for the first time step.
        unroll: This function unrolls an RNN for a given number of (>=1) time steps.
        """
        # pylint: disable= arguments-differ
        self._counter += 1
        return super(RecurrentCell, self).forward(inputs, states)



class RNNCell(RecurrentCell):
    """Simple recurrent neural network cell.

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    activation : str or Symbol, default 'tanh'
        type of activation function
    prefix : str, default 'rnn_'
        prefix for name of layers
        (and name of weight if params is None)
    params : Parameter or None
        container for weight sharing between cells.
        created if None.
    """
    def __init__(self, num_hidden, activation='tanh', num_input=0,
                 prefix=None, params=None):
        super(RNNCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._activation = activation
        self._num_input = num_input
        self.i2h_weight = self.params.get('i2h_weight', shape=(num_hidden, num_input))
        self.i2h_bias = self.params.get('i2h_bias', shape=(num_hidden,))
        self.h2h_weight = self.params.get('h2h_weight', shape=(num_hidden, num_hidden))
        self.h2h_bias = self.params.get('h2h_bias', shape=(num_hidden,))

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._num_hidden), '__layout__': 'NC'}]

    @property
    def _gate_names(self):
        return ('',)

    def _alias(self):
        return 'rnn'

    def generic_forward(self, F, inputs, states, i2h_weight, i2h_bias,
                        h2h_weight, h2h_bias):
        name = self._curr_prefix
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._num_hidden,
                               name='%si2h'%name)
        h2h = F.FullyConnected(data=states[0], weight=h2h_weight, bias=h2h_bias,
                               num_hidden=self._num_hidden,
                               name='%sh2h'%name)
        output = self._get_activation(F, i2h + h2h, self._activation,
                                      name='%sout'%name)

        return output, [output]


class LSTMCell(RecurrentCell):
    """Long-Short Term Memory (LSTM) network cell.

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    prefix : str, default 'lstm_'
        prefix for name of layers
        (and name of weight if params is None)
    params : Parameter or None
        container for weight sharing between cells.
        created if None.
    forget_bias : bias added to forget gate, default 1.0.
        Jozefowicz et al. 2015 recommends setting this to 1.0
    """
    def __init__(self, num_hidden, forget_bias=1.0, num_input=0,
                 prefix=None, params=None):
        super(LSTMCell, self).__init__(prefix=prefix, params=params)

        self._num_hidden = num_hidden
        self._num_input = num_input
        self.i2h_weight = self.params.get('i2h_weight', shape=(4*num_hidden, num_input))
        self.h2h_weight = self.params.get('h2h_weight', shape=(4*num_hidden, num_hidden))
        # we add the forget_bias to i2h_bias, this adds the bias to the forget gate activation
        self.i2h_bias = self.params.get('i2h_bias', shape=(4*num_hidden,),
                                        init=init.LSTMBias(forget_bias=forget_bias))
        self.h2h_bias = self.params.get('h2h_bias', shape=(4*num_hidden,))

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._num_hidden), '__layout__': 'NC'},
                {'shape': (batch_size, self._num_hidden), '__layout__': 'NC'}]

    @property
    def _gate_names(self):
        return ['_i', '_f', '_c', '_o']

    def _alias(self):
        return 'lstm'

    def generic_forward(self, F, inputs, states, i2h_weight, i2h_bias,
                        h2h_weight, h2h_bias):
        name = self._curr_prefix
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._num_hidden*4,
                               name='%si2h'%name)
        h2h = F.FullyConnected(data=states[0], weight=h2h_weight, bias=h2h_bias,
                               num_hidden=self._num_hidden*4,
                               name='%sh2h'%name)
        gates = i2h + h2h
        slice_gates = F.SliceChannel(gates, num_outputs=4,
                                     name="%sslice"%name)
        in_gate = F.Activation(slice_gates[0], act_type="sigmoid",
                               name='%si'%name)
        forget_gate = F.Activation(slice_gates[1], act_type="sigmoid",
                                   name='%sf'%name)
        in_transform = F.Activation(slice_gates[2], act_type="tanh",
                                    name='%sc'%name)
        out_gate = F.Activation(slice_gates[3], act_type="sigmoid",
                                name='%so'%name)
        next_c = F._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                   name='%sstate'%name)
        next_h = F._internal._mul(out_gate, F.Activation(next_c, act_type="tanh"),
                                  name='%sout'%name)

        return next_h, [next_h, next_c]


class GRUCell(RecurrentCell):
    """Gated Rectified Unit (GRU) network cell.
    Note: this is an implementation of the cuDNN version of GRUs
    (slight modification compared to Cho et al. 2014).

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    prefix : str, default 'gru_'
        prefix for name of layers
        (and name of weight if params is None)
    params : Parameter or None
        container for weight sharing between cells.
        created if None.
    """
    def __init__(self, num_hidden, num_input=0, prefix=None, params=None):
        super(GRUCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self.i2h_weight = self.params.get('i2h_weight', shape=(3*num_hidden, num_input))
        self.h2h_weight = self.params.get('h2h_weight', shape=(3*num_hidden, num_hidden))
        self.i2h_bias = self.params.get('i2h_bias', shape=(3*num_hidden))
        self.h2h_bias = self.params.get('h2h_bias', shape=(3*num_hidden))

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._num_hidden), '__layout__': 'NC'}]

    @property
    def _gate_names(self):
        return ['_r', '_z', '_o']

    def _alias(self):
        return 'gru'

    def generic_forward(self, F, inputs, states, i2h_weight, i2h_bias,
                        h2h_weight, h2h_bias):
        # pylint: disable=too-many-locals
        name = self._curr_prefix
        prev_state_h = states[0]
        i2h = F.FullyConnected(data=inputs,
                               weight=i2h_weight,
                               bias=i2h_bias,
                               num_hidden=self._num_hidden * 3,
                               name="%si2h" % name)
        h2h = F.FullyConnected(data=prev_state_h,
                               weight=h2h_weight,
                               bias=h2h_bias,
                               num_hidden=self._num_hidden * 3,
                               name="%sh2h" % name)

        i2h_r, i2h_z, i2h = F.SliceChannel(i2h, num_outputs=3, name="%si2h_slice" % name)
        h2h_r, h2h_z, h2h = F.SliceChannel(h2h, num_outputs=3, name="%sh2h_slice" % name)

        reset_gate = F.Activation(i2h_r + h2h_r, act_type="sigmoid",
                                  name="%sr_act" % name)
        update_gate = F.Activation(i2h_z + h2h_z, act_type="sigmoid",
                                   name="%sz_act" % name)

        next_h_tmp = F.Activation(i2h + reset_gate * h2h, act_type="tanh",
                                  name="%sh_act" % name)

        next_h = F._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                   name='%sout' % name)

        return next_h, [next_h]


class FusedRNNCell(RecurrentCell):
    """Fusing RNN layers across time step into one kernel.
    Improves speed but is less flexible. Currently only
    supported if using cuDNN on GPU.

    Parameters
    ----------
    """
    def __init__(self, num_hidden, num_layers=1, mode='lstm', bidirectional=False,
                 dropout=0., get_next_state=False, forget_bias=1.0, num_input=0,
                 prefix=None, params=None):
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._mode = mode
        self._bidirectional = bidirectional
        self._dropout = dropout
        self._get_next_state = get_next_state
        self._directions = ['l', 'r'] if bidirectional else ['l']
        super(FusedRNNCell, self).__init__(prefix=prefix, params=params)

        initializer = init.FusedRNN(None, num_hidden, num_layers, mode,
                                    bidirectional, forget_bias)
        self.parameters = self.params.get('parameters', init=initializer,
                                          shape=(self._num_input_to_size(num_input),))

    def state_info(self, batch_size=0):
        b = self._bidirectional + 1
        n = (self._mode == 'lstm') + 1
        return [{'shape': (b*self._num_layers, batch_size, self._num_hidden),
                 '__layout__': 'LNC'} for _ in range(n)]

    @property
    def _gate_names(self):
        return {'rnn_relu': [''],
                'rnn_tanh': [''],
                'lstm': ['_i', '_f', '_c', '_o'],
                'gru': ['_r', '_z', '_o']}[self._mode]

    @property
    def _num_gates(self):
        return len(self._gate_names)

    def _alias(self):
        return self._mode

    def _size_to_num_input(self, size):
        b = len(self._directions)
        m = self._num_gates
        h = self._num_hidden
        return size//b//h//m - (self._num_layers - 1)*(h+b*h+2) - h - 2

    def _num_input_to_size(self, num_input):
        if num_input == 0:
            return 0
        b = self._bidirectional + 1
        m = self._num_gates
        h = self._num_hidden
        return (num_input+h+2)*h*m*b + (self._num_layers-1)*m*h*(h+b*h+2)*b

    def _slice_weights(self, arr, li, lh):
        """slice fused rnn weights"""
        args = {}
        gate_names = self._gate_names
        directions = self._directions

        b = len(directions)
        p = 0
        for layer in range(self._num_layers):
            for direction in directions:
                for gate in gate_names:
                    name = '%s%s%d_i2h%s_weight'%(self._prefix, direction, layer, gate)
                    if layer > 0:
                        size = b*lh*lh
                        args[name] = arr[p:p+size].reshape((lh, b*lh))
                    else:
                        size = li*lh
                        args[name] = arr[p:p+size].reshape((lh, li))
                    p += size
                for gate in gate_names:
                    name = '%s%s%d_h2h%s_weight'%(self._prefix, direction, layer, gate)
                    size = lh**2
                    args[name] = arr[p:p+size].reshape((lh, lh))
                    p += size

        for layer in range(self._num_layers):
            for direction in directions:
                for gate in gate_names:
                    name = '%s%s%d_i2h%s_bias'%(self._prefix, direction, layer, gate)
                    args[name] = arr[p:p+lh]
                    p += lh
                for gate in gate_names:
                    name = '%s%s%d_h2h%s_bias'%(self._prefix, direction, layer, gate)
                    args[name] = arr[p:p+lh]
                    p += lh

        assert p == arr.size, "Invalid parameters size for FusedRNNCell"
        return args

    def unpack_weights(self, args):
        args = args.copy()
        arr = args.pop(self.parameters.name)
        num_input = self._size_to_num_input(arr.size)
        nargs = self._slice_weights(arr, num_input, self._num_hidden)
        args.update({name: nd.copy() for name, nd in nargs.items()})
        return args

    def pack_weights(self, args):
        args = args.copy()
        w0 = args['%sl0_i2h%s_weight'%(self._prefix, self._gate_names[0])]
        num_input = w0.shape[1]
        total = self._num_input_to_size(num_input)

        arr = ndarray.zeros((total,), ctx=w0.context, dtype=w0.dtype)
        for name, nd in self._slice_weights(arr, num_input, self._num_hidden).items():
            nd[:] = args.pop(name)
        args[self.parameters.name] = arr
        return args

    def __call__(self, inputs, states):
        raise NotImplementedError("FusedRNNCell cannot be stepped. Please use unroll")

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()

        inputs, axis, F, batch_size = _format_sequence(length, inputs, layout, True)
        if axis == 1:
            warnings.warn("NTC layout detected. Consider using "
                          "TNC for FusedRNNCell for faster speed")
            inputs = F.swapaxes(inputs, dim1=0, dim2=1)
        else:
            assert axis == 0, "Unsupported layout %s"%layout
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)

        states = begin_state
        if self._mode == 'lstm':
            states = {'state': states[0], 'state_cell': states[1]} # pylint: disable=redefined-variable-type
        else:
            states = {'state': states[0]}

        if isinstance(inputs, symbol.Symbol):
            parameters = self.parameters.var()
        else:
            parameters = self.parameters.data(inputs.context)

        rnn = F.RNN(data=inputs, parameters=parameters,
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
            outputs, states = rnn[0], [rnn[1]]

        if axis == 1:
            outputs = F.swapaxes(outputs, dim1=0, dim2=1)

        outputs, _, _, _ = _format_sequence(length, outputs, layout, merge_outputs)

        return outputs, states

    def unfuse(self):
        """Unfuse the fused RNN in to a stack of rnn cells.

        Returns
        -------
        cell : SequentialRNNCell
            unfused cell that can be used for stepping, and can run on CPU.
        """
        stack = SequentialRNNCell()
        get_cell = {'rnn_relu': lambda cell_prefix: RNNCell(self._num_hidden,
                                                            activation='relu',
                                                            prefix=cell_prefix),
                    'rnn_tanh': lambda cell_prefix: RNNCell(self._num_hidden,
                                                            activation='tanh',
                                                            prefix=cell_prefix),
                    'lstm': lambda cell_prefix: LSTMCell(self._num_hidden,
                                                         prefix=cell_prefix),
                    'gru': lambda cell_prefix: GRUCell(self._num_hidden,
                                                       prefix=cell_prefix)}[self._mode]
        for i in range(self._num_layers):
            if self._bidirectional:
                stack.add(BidirectionalCell(
                    get_cell('%sl%d_'%(self._prefix, i)),
                    get_cell('%sr%d_'%(self._prefix, i)),
                    output_prefix='%sbi_l%d_'%(self._prefix, i)))
            else:
                stack.add(get_cell('%sl%d_'%(self._prefix, i)))

            if self._dropout > 0 and i != self._num_layers - 1:
                stack.add(DropoutCell(self._dropout, prefix='%s_dropout%d_'%(self._prefix, i)))

        return stack


class SequentialRNNCell(RecurrentCell):
    """Sequantially stacking multiple RNN cells."""
    def __init__(self):
        super(SequentialRNNCell, self).__init__(prefix='', params=None)

    def add(self, cell):
        """Append a cell into the stack.

        Parameters
        ----------
        cell : rnn cell
        """
        self.register_child(cell)

    def state_info(self, batch_size=0):
        return _cells_state_info(self._children, batch_size)

    def begin_state(self, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. ZoneoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        return _cells_begin_state(self._children, **kwargs)

    def unpack_weights(self, args):
        return _cells_unpack_weights(self._children, args)

    def pack_weights(self, args):
        return _cells_pack_weights(self._children, args)

    def __call__(self, inputs, states):
        self._counter += 1
        next_states = []
        p = 0
        for cell in self._children:
            assert not isinstance(cell, BidirectionalCell)
            n = len(cell.state_info())
            state = states[p:p+n]
            p += n
            inputs, state = cell(inputs, state)
            next_states.append(state)
        return inputs, sum(next_states, [])

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()

        inputs, _, F, batch_size = _format_sequence(length, inputs, layout, None)
        num_cells = len(self._children)
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)

        p = 0
        next_states = []
        for i, cell in enumerate(self._children):
            n = len(cell.state_info())
            states = begin_state[p:p+n]
            p += n
            inputs, states = cell.unroll(length, inputs=inputs, begin_state=states, layout=layout,
                                         merge_outputs=None if i < num_cells-1 else merge_outputs)
            next_states.extend(states)

        return inputs, next_states

    def generic_forward(self, *args, **kwargs):
        raise NotImplementedError


class DropoutCell(RecurrentCell):
    """Apply dropout on input.

    Parameters
    ----------
    dropout : float
        percentage of elements to drop out, which
        is 1 - percentage to retain.
    """
    def __init__(self, dropout, prefix=None, params=None):
        super(DropoutCell, self).__init__(prefix, params)
        assert isinstance(dropout, numeric_types), "dropout probability must be a number"
        self.dropout = dropout

    def state_info(self, batch_size=0):
        return []

    def _alias(self):
        return 'dropout'

    def generic_forward(self, F, inputs, states):
        if self.dropout > 0:
            inputs = F.Dropout(data=inputs, p=self.dropout)
        return inputs, states

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()

        inputs, _, F, _ = _format_sequence(length, inputs, layout, merge_outputs)
        if isinstance(inputs, tensor_types):
            return self.generic_forward(F, inputs, begin_state if begin_state else [])
        else:
            return super(DropoutCell, self).unroll(
                length, inputs, begin_state=begin_state, layout=layout,
                merge_outputs=merge_outputs)


class ModifierCell(RecurrentCell):
    """Base class for modifier cells. A modifier
    cell takes a base cell, apply modifications
    on it (e.g. Zoneout), and returns a new cell.

    After applying modifiers the base cell should
    no longer be called directly. The modifer cell
    should be used instead.
    """
    def __init__(self, base_cell):
        super(ModifierCell, self).__init__(prefix=None, params=None)
        base_cell._modified = True
        self.base_cell = base_cell

    @property
    def params(self):
        self._own_params = False
        return self.base_cell.params

    def state_info(self, batch_size=0):
        return self.base_cell.state_info(batch_size)

    def begin_state(self, func=symbol.zeros, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        self.base_cell._modified = False
        begin = self.base_cell.begin_state(func=func, **kwargs)
        self.base_cell._modified = True
        return begin

    def unpack_weights(self, args):
        return self.base_cell.unpack_weights(args)

    def pack_weights(self, args):
        return self.base_cell.pack_weights(args)

    def generic_forward(self, F, inputs, states):
        raise NotImplementedError


class ZoneoutCell(ModifierCell):
    """Apply Zoneout on base cell."""
    def __init__(self, base_cell, zoneout_outputs=0., zoneout_states=0.):
        assert not isinstance(base_cell, FusedRNNCell), \
            "FusedRNNCell doesn't support zoneout. " \
            "Please unfuse first."
        assert not isinstance(base_cell, BidirectionalCell), \
            "BidirectionalCell doesn't support zoneout since it doesn't support step. " \
            "Please add ZoneoutCell to the cells underneath instead."
        assert not isinstance(base_cell, SequentialRNNCell) or not base_cell._bidirectional, \
            "Bidirectional SequentialRNNCell doesn't support zoneout. " \
            "Please add ZoneoutCell to the cells underneath instead."
        super(ZoneoutCell, self).__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self.prev_output = None

    def _alias(self):
        return 'zoneout'

    def reset(self):
        super(ZoneoutCell, self).reset()
        self.prev_output = None

    def generic_forward(self, F, inputs, states):
        cell, p_outputs, p_states = self.base_cell, self.zoneout_outputs, self.zoneout_states
        next_output, next_states = cell(inputs, states)
        mask = (lambda p, like: F.Dropout(F.ones_like(like), p=p))

        prev_output = self.prev_output
        if prev_output is None:
            prev_output = F.zeros_like(next_output)

        output = (F.where(mask(p_outputs, next_output), next_output, prev_output)
                  if p_outputs != 0. else next_output)
        states = ([F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in
                   zip(next_states, states)] if p_states != 0. else next_states)

        self.prev_output = output

        return output, states


class ResidualCell(ModifierCell):
    """
    Adds residual connection as described in Wu et al, 2016
    (https://arxiv.org/abs/1609.08144).
    Output of the cell is output of the base cell plus input.
    """

    def __init__(self, base_cell):
        super(ResidualCell, self).__init__(base_cell)

    def generic_forward(self, F, inputs, states):
        output, states = self.base_cell(inputs, states)
        output = F.elemwise_add(output, inputs, name="%s_plus_residual" % output.name)
        return output, states

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()

        self.base_cell._modified = False
        outputs, states = self.base_cell.unroll(length, inputs=inputs, begin_state=begin_state,
                                                layout=layout, merge_outputs=merge_outputs)
        self.base_cell._modified = True

        merge_outputs = isinstance(outputs, tensor_types) if merge_outputs is None else \
                        merge_outputs
        inputs, _, F, _ = _format_sequence(length, inputs, layout, merge_outputs)
        if merge_outputs:
            outputs = F.elemwise_add(outputs, inputs)
        else:
            outputs = [F.elemwise_add(i, j) for i, j in zip(outputs, inputs)]

        return outputs, states


class BidirectionalCell(RecurrentCell):
    """Bidirectional RNN cell.

    Parameters
    ----------
    l_cell : RecurrentCell
        cell for forward unrolling
    r_cell : RecurrentCell
        cell for backward unrolling
    """
    def __init__(self, l_cell, r_cell, output_prefix='bi_'):
        super(BidirectionalCell, self).__init__(prefix='', params=None)
        self.register_child(l_cell)
        self.register_child(r_cell)
        self._output_prefix = output_prefix

    def unpack_weights(self, args):
        return _cells_unpack_weights(self._children, args)

    def pack_weights(self, args):
        return _cells_pack_weights(self._children, args)

    def __call__(self, inputs, states):
        raise NotImplementedError("Bidirectional cannot be stepped. Please use unroll")

    def state_info(self, batch_size=0):
        return _cells_state_info(self._children, batch_size)

    def begin_state(self, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        return _cells_begin_state(self._children, **kwargs)

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()

        inputs, axis, F, batch_size = _format_sequence(length, inputs, layout, False)
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)

        states = begin_state
        l_cell, r_cell = self._children
        l_outputs, l_states = l_cell.unroll(length, inputs=inputs,
                                            begin_state=states[:len(l_cell.state_info(batch_size))],
                                            layout=layout, merge_outputs=merge_outputs)
        r_outputs, r_states = r_cell.unroll(length,
                                            inputs=list(reversed(inputs)),
                                            begin_state=states[len(l_cell.state_info(batch_size)):],
                                            layout=layout, merge_outputs=merge_outputs)

        if merge_outputs is None:
            merge_outputs = (isinstance(l_outputs, tensor_types)
                             and isinstance(r_outputs, tensor_types))
            l_outputs, _, _, _ = _format_sequence(None, l_outputs, layout, merge_outputs)
            r_outputs, _, _, _ = _format_sequence(None, r_outputs, layout, merge_outputs)

        if merge_outputs:
            r_outputs = F.reverse(r_outputs, axis=axis)
            outputs = F.concat(l_outputs, r_outputs, dim=2, name='%sout'%self._output_prefix)
        else:
            outputs = [F.concat(l_o, r_o, dim=1, name='%st%d'%(self._output_prefix, i))
                       for i, (l_o, r_o) in enumerate(zip(l_outputs, reversed(r_outputs)))]

        states = [l_states, r_states]
        return outputs, states
