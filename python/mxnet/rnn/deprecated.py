# coding: utf-8
# pylint: disable=no-member, invalid-name, protected-access, no-self-use
# pylint: disable=too-many-branches, too-many-arguments, no-self-use
# pylint: disable=too-many-lines
"""Definition of various recurrent neural network cells."""
from __future__ import print_function

import warnings

from .. import symbol, ndarray
from ..base import string_types


def _normalize_sequence(length, inputs, layout, merge, in_layout=None):
    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, symbol.Symbol):
        if merge is False:
            assert len(inputs.list_outputs()) == 1, \
                "unroll doesn't allow grouped symbol as input. Please convert " \
                "to list with list(inputs) first or let unroll handle splitting."
            inputs = list(symbol.split(inputs, axis=in_axis, num_outputs=length,
                                       squeeze_axis=1))
    else:
        assert length is None or len(inputs) == length
        if merge is True:
            inputs = [symbol.expand_dims(i, axis=axis) for i in inputs]
            inputs = symbol.Concat(*inputs, dim=axis)
            in_axis = axis

    if isinstance(inputs, symbol.Symbol) and axis != in_axis:
        inputs = symbol.swapaxes(inputs, dim0=axis, dim1=in_axis)

    return inputs, axis


class RNNParams(object):
    """Container for holding variables.
    Used by RNN cells for parameter sharing between cells.

    Parameters
    ----------
    prefix : str
        Names of all variables created by this container will
        be prepended with prefix.
    """
    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def get(self, name, **kwargs):
        """Get the variable given a name if one exists or create a new one if missing.

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
        warnings.warn(
            "RNNParams and BaseRNNCell has been deprecated. "
            "Please use ParameterDict and RecurrentCell instead. "
            "The new interface supports both symbolic and imperative API.")
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
        """Unroll the RNN for one time step.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : list of sym.Variable
            RNN state from previous step or the output of begin_state().

        Returns
        -------
        output : Symbol
            Symbol corresponding to the output from the RNN when unrolling
            for a single time step.
        states : nested list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of begin_state().
            This can be used as input state to the next time step
            of this RNN.

        See Also
        --------
        begin_state: This function can provide the states for the first time step.
        unroll: This function unrolls an RNN for a given number of (>=1) time steps.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """Parameters of this cell"""
        self._own_params = False
        return self._params

    @property
    def state_info(self):
        """shape and layout information of states"""
        raise NotImplementedError()

    @property
    def state_shape(self):
        """shape(s) of states"""
        return [ele['shape'] for ele in self.state_info]

    @property
    def _gate_names(self):
        """name(s) of gates"""
        return ()

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
        for info in self.state_info:
            self._init_counter += 1
            if info is None:
                state = func(name='%sbegin_state_%d'%(self._prefix, self._init_counter),
                             **kwargs)
            else:
                kwargs.update(info)
                state = func(name='%sbegin_state_%d'%(self._prefix, self._init_counter),
                             **kwargs)
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
            if inputs is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If inputs is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol
            input states. Created by begin_state()
            or output state of another cell. Created
            from begin_state() if None.
        layout : str
            layout of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool
            If False, return outputs as a list of Symbols.
            If True, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.
            If None, output whatever is faster

        Returns
        -------
        outputs : list of Symbol
            output symbols.
        states : Symbol or nested list of Symbol
            has the same structure as begin_state()
        """
        self.reset()

        inputs, _ = _normalize_sequence(length, inputs, layout, False)
        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        outputs = []
        for i in range(length):
            output, states = self(inputs[i], states)
            outputs.append(output)

        outputs, _ = _normalize_sequence(length, outputs, layout, merge_outputs)

        return outputs, states

    #pylint: disable=no-self-use
    def _get_activation(self, inputs, activation, **kwargs):
        """Get activation function. Convert if is string"""
        if isinstance(activation, string_types):
            return symbol.Activation(inputs, act_type=activation, **kwargs)
        else:
            return activation(inputs, **kwargs)
