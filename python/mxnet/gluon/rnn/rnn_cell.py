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
"""Definition of various recurrent neural network cells."""
__all__ = ['RecurrentCell', 'HybridRecurrentCell',
           'RNNCell', 'LSTMCell', 'GRUCell',
           'SequentialRNNCell', 'HybridSequentialRNNCell', 'DropoutCell',
           'ModifierCell', 'ZoneoutCell', 'ResidualCell',
           'BidirectionalCell', 'VariationalDropoutCell', 'LSTMPCell']

from ... import np, npx, cpu
from ...util import use_np
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..parameter import Parameter
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU


def _cells_state_info(cells, batch_size):
    return sum([c().state_info(batch_size) for c in cells], [])

def _cells_begin_state(cells, **kwargs):
    return sum([c().begin_state(**kwargs) for c in cells], [])

def _get_begin_state(cell, begin_state, inputs, batch_size):
    if begin_state is None:
        device = inputs.device if isinstance(inputs, tensor_types) else inputs[0].device
        with device:
            begin_state = cell.begin_state(func=np.zeros, batch_size=batch_size)
    return begin_state

def _format_sequence(length, inputs, layout, merge, in_layout=None):
    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    batch_axis = layout.find('N')
    batch_size = 0
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, np.ndarray):
        batch_size = inputs.shape[batch_axis]
        if merge is False:
            assert length is None or length == inputs.shape[in_axis]
            inputs = _as_list(npx.slice_channel(inputs, axis=in_axis,
                                                num_outputs=inputs.shape[in_axis],
                                                squeeze_axis=1))
    else:
        assert isinstance(inputs, (list, tuple)), \
            "Only support MXNet numpy ndarray or list of MXNet numpy ndarrays as inputs"
        assert length is None or len(inputs) == length
        batch_size = inputs[0].shape[0]
        if merge is True:
            inputs = np.stack(inputs, axis=axis)
            in_axis = axis

    if isinstance(inputs, np.ndarray) and axis != in_axis:
        inputs = np.swapaxes(inputs, axis, in_axis)

    return inputs, axis, batch_size

def _mask_sequence_variable_length(data, length, valid_length, time_axis, merge):
    assert valid_length is not None
    if not isinstance(data, tensor_types):
        data = np.stack(data, axis=time_axis)
    outputs = npx.sequence_mask(data, sequence_length=valid_length, use_sequence_length=True,
                                axis=time_axis)
    if not merge:
        outputs = _as_list(npx.slice_channel(outputs, num_outputs=length, axis=time_axis,
                                             squeeze_axis=True))
    return outputs

def _reverse_sequences(sequences, unroll_step, valid_length=None):
    if valid_length is None:
        reversed_sequences = list(reversed(sequences))
    else:
        reversed_sequences = npx.sequence_reverse(np.stack(sequences, axis=0),
                                                  sequence_length=valid_length,
                                                  use_sequence_length=True)
        if unroll_step > 1:
            reversed_sequences = npx.slice_channel(reversed_sequences, axis=0,
                                                   num_outputs=unroll_step, squeeze_axis=True)
        else:
            reversed_sequences = [reversed_sequences[0]]

    return reversed_sequences


@use_np
class RecurrentCell(Block):
    """Abstract base class for RNN cells

    """
    def __init__(self):
        super(RecurrentCell, self).__init__()
        self._modified = False
        self.reset()

    def reset(self):
        """Reset before re-using the cell for another graph."""
        self._init_counter = -1
        self._counter = -1
        for cell in self._children.values():
            cell().reset()

    def state_info(self, batch_size=0):
        """shape and layout information of states"""
        raise NotImplementedError()

    def begin_state(self, batch_size=0, func=np.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        func : callable, default symbol.zeros
            Function for creating initial state.

            For Symbol API, func can be `symbol.zeros`, `symbol.uniform`,
            `symbol.var etc`. Use `symbol.var` if you want to directly
            feed input as states.

            For NDArray API, func can be `ndarray.zeros`, `ndarray.ones`, etc.
        batch_size: int, default 0
            Only required for NDArray API. Size of the batch ('N' in layout)
            dimension of input.

        **kwargs :
            Additional keyword arguments passed to func. For example
            `mean`, `std`, `dtype`, etc.

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
            if info is not None:
                info.update(kwargs)
            else:
                info = kwargs
            state = func(shape=info.pop("shape", ()),
                         device=info.pop("device", cpu()),
                         dtype=info.pop("dtype", "float32"))
            states.append(state)
        return states

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        """Unrolls an RNN cell across time steps.

        Parameters
        ----------
        length : int
            Number of steps to unroll.
        inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if `layout` is 'NTC',
            or (length, batch_size, ...) if `layout` is 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol, optional
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if `None`.
        layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool, optional
            If `False`, returns outputs as a list of Symbols.
            If `True`, concatenates output across time steps
            and returns a single symbol with shape
            (batch_size, length, ...) if layout is 'NTC',
            or (length, batch_size, ...) if layout is 'TNC'.
            If `None`, output whatever is faster.
        valid_length : Symbol, NDArray or None
            `valid_length` specifies the length of the sequences in the batch without padding.
            This option is especially useful for building sequence-to-sequence models where
            the input and output sequences would potentially be padded.
            If `valid_length` is None, all sequences are assumed to have the same length.
            If `valid_length` is a Symbol or NDArray, it should have shape (batch_size,).
            The ith element will be the length of the ith sequence in the batch.
            The last valid state will be return and the padded outputs will be masked with 0.
            Note that `valid_length` must be smaller or equal to `length`.

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
        """
        # pylint: disable=too-many-locals
        self.reset()

        inputs, axis, batch_size = _format_sequence(length, inputs, layout, False)
        begin_state = _get_begin_state(self, begin_state, inputs, batch_size)

        states = begin_state
        outputs = []
        all_states = []
        for i in range(length):
            output, states = self(inputs[i], states)
            outputs.append(output)
            if valid_length is not None:
                all_states.append(states)
        if valid_length is not None:
            states = [npx.sequence_last(np.stack(ele_list, axis=0),
                                        sequence_length=valid_length,
                                        use_sequence_length=True,
                                        axis=0)
                      for ele_list in zip(*all_states)]
            outputs = _mask_sequence_variable_length(outputs, length, valid_length, axis, True)
        outputs, _, _ = _format_sequence(length, outputs, layout, merge_outputs)

        return outputs, states

    #pylint: disable=no-self-use
    def _get_activation(self, inputs, activation, **kwargs):
        """Get activation function. Convert if is string"""
        func = {'tanh': np.tanh,
                'relu': npx.relu,
                'sigmoid': npx.sigmoid,
                'softsign': npx.softsign}.get(activation)
        if func:
            return func(inputs, **kwargs)
        elif isinstance(activation, string_types):
            return npx.activation(inputs, act_type=activation, **kwargs)
        elif isinstance(activation, LeakyReLU):
            return npx.leaky_relu(inputs, act_type='leaky', slope=activation._alpha, **kwargs)
        return activation(inputs, **kwargs)

    def forward(self, inputs, states):
        """Unrolls the recurrent cell for one time step.

        Parameters
        ----------
        inputs : sym.Variable
            Input symbol, 2D, of shape (batch_size * num_units).
        states : list of sym.Variable
            RNN state from previous step or the output of begin_state().

        Returns
        -------
        output : Symbol
            Symbol corresponding to the output from the RNN when unrolling
            for a single time step.
        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
            This can be used as an input state to the next time step
            of this RNN.

        See Also
        --------
        begin_state: This function can provide the states for the first time step.
        unroll: This function unrolls an RNN for a given number of (>=1) time steps.
        """
        # pylint: disable= arguments-differ
        self._counter += 1
        return super(RecurrentCell, self).forward(inputs, states)

@use_np
class HybridRecurrentCell(RecurrentCell, HybridBlock):
    """HybridRecurrentCell supports hybridize."""
    def __init__(self):
        super(HybridRecurrentCell, self).__init__()

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError


@use_np
class RNNCell(HybridRecurrentCell):
    r"""Elman RNN recurrent neural network cell.

    Each call computes the following function:

    .. math::

        h_t = \tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first layer.
    If nonlinearity='relu', then `ReLU` is used instead of `tanh`.

    Parameters
    ----------
    hidden_size : int
        Number of units in output symbol
    activation : str or Symbol, default 'tanh'
        Type of activation function.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'zeros'
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer, default 'zeros'
        Initializer for the bias vector.
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.


    Inputs:
        - **data**: input tensor with shape `(batch_size, input_size)`.
        - **states**: a list of one initial recurrent state tensor with shape
          `(batch_size, num_hidden)`.

    Outputs:
        - **out**: output tensor with shape `(batch_size, num_hidden)`.
        - **next_states**: a list of one output recurrent state tensor with the
          same shape as `states`.
    """
    def __init__(self, hidden_size, activation='tanh',
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0):
        super(RNNCell, self).__init__()
        self._hidden_size = hidden_size
        self._activation = activation
        self._input_size = input_size
        self.i2h_weight = Parameter('i2h_weight', shape=(hidden_size, input_size),
                                    init=i2h_weight_initializer,
                                    allow_deferred_init=True)
        self.h2h_weight = Parameter('h2h_weight', shape=(hidden_size, hidden_size),
                                    init=h2h_weight_initializer,
                                    allow_deferred_init=True)
        self.i2h_bias = Parameter('i2h_bias', shape=(hidden_size,),
                                  init=i2h_bias_initializer,
                                  allow_deferred_init=True)
        self.h2h_bias = Parameter('h2h_bias', shape=(hidden_size,),
                                  init=h2h_bias_initializer,
                                  allow_deferred_init=True)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._hidden_size), '__layout__': 'NC'}]

    def _alias(self):
        return 'rnn'

    def __repr__(self):
        s = '{name}({mapping}'
        if hasattr(self, '_activation'):
            s += ', {_activation}'
        s += ')'
        shape = self.i2h_weight.shape
        mapping = '{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0])
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

    def forward(self, inputs, states):
        device = inputs.device
        i2h = npx.fully_connected(inputs, weight=self.i2h_weight.data(device),
                                  bias=self.i2h_bias.data(device),
                                  num_hidden=self._hidden_size,
                                  no_bias=False)
        h2h = npx.fully_connected(states[0].to_device(device),
                                  weight=self.h2h_weight.data(device),
                                  bias=self.h2h_bias.data(device),
                                  num_hidden=self._hidden_size,
                                  no_bias=False)
        i2h_plus_h2h = i2h + h2h
        output = self._get_activation(i2h_plus_h2h, self._activation)

        return output, [output]

    def infer_shape(self, i, x, is_bidirect):
        if i == 0:
            self.i2h_weight.shape = (self._hidden_size, x.shape[x.ndim-1])
        else:
            nh = self._hidden_size
            if is_bidirect:
                nh *= 2
            self.i2h_weight.shape = (self._hidden_size, nh)


@use_np
class LSTMCell(HybridRecurrentCell):
    r"""Long-Short Term Memory (LSTM) network cell.

    Each call computes the following function:

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
    hidden_size : int
        Number of units in output symbol.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'zeros'
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer, default 'zeros'
        Initializer for the bias vector.
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.
    activation : str, default 'tanh'
        Activation type to use. See nd/symbol Activation
        for supported types.
    recurrent_activation : str, default 'sigmoid'
        Activation type to use for the recurrent step. See nd/symbol Activation
        for supported types.

    Inputs:
        - **data**: input tensor with shape `(batch_size, input_size)`.
        - **states**: a list of two initial recurrent state tensors. Each has shape
          `(batch_size, num_hidden)`.

    Outputs:
        - **out**: output tensor with shape `(batch_size, num_hidden)`.
        - **next_states**: a list of two output recurrent state tensors. Each has
          the same shape as `states`.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0, activation='tanh', recurrent_activation='sigmoid'):
        super(LSTMCell, self).__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self.i2h_weight = Parameter('i2h_weight', shape=(4*hidden_size, input_size),
                                    init=i2h_weight_initializer,
                                    allow_deferred_init=True)
        self.h2h_weight = Parameter('h2h_weight', shape=(4*hidden_size, hidden_size),
                                    init=h2h_weight_initializer,
                                    allow_deferred_init=True)
        self.i2h_bias = Parameter('i2h_bias', shape=(4*hidden_size,),
                                  init=i2h_bias_initializer,
                                  allow_deferred_init=True)
        self.h2h_bias = Parameter('h2h_bias', shape=(4*hidden_size,),
                                  init=h2h_bias_initializer,
                                  allow_deferred_init=True)
        self._activation = activation
        self._recurrent_activation = recurrent_activation


    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._hidden_size), '__layout__': 'NC'},
                {'shape': (batch_size, self._hidden_size), '__layout__': 'NC'}]

    def _alias(self):
        return 'lstm'

    def __repr__(self):
        s = '{name}({mapping})'
        shape = self.i2h_weight.shape
        mapping = '{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0])
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

    def forward(self, inputs, states):
        # pylint: disable=too-many-locals
        device = inputs.device
        i2h = npx.fully_connected(inputs, weight=self.i2h_weight.data(device),
                                  bias=self.i2h_bias.data(device),
                                  num_hidden=self._hidden_size*4, no_bias=False)
        h2h = npx.fully_connected(states[0].to_device(device),
                                  weight=self.h2h_weight.data(device),
                                  bias=self.h2h_bias.data(device),
                                  num_hidden=self._hidden_size*4, no_bias=False)
        gates = i2h + h2h
        slice_gates = npx.slice_channel(gates, num_outputs=4)
        in_gate = self._get_activation(slice_gates[0], self._recurrent_activation)
        forget_gate = self._get_activation(slice_gates[1], self._recurrent_activation)
        in_transform = self._get_activation(slice_gates[2], self._activation)
        out_gate = self._get_activation(slice_gates[3], self._recurrent_activation)
        next_c = np.multiply(forget_gate, states[1].to_device(device)) + \
                 np.multiply(in_gate, in_transform)
        next_h = np.multiply(out_gate, npx.activation(next_c, act_type=self._activation))

        return next_h, [next_h, next_c]

    def infer_shape(self, i, x, is_bidirect):
        if i == 0:
            self.i2h_weight.shape = (4*self._hidden_size, x.shape[x.ndim-1])
        else:
            nh = self._hidden_size
            if is_bidirect:
                nh *= 2
            self.i2h_weight.shape = (4*self._hidden_size, nh)

@use_np
class GRUCell(HybridRecurrentCell):
    r"""Gated Rectified Unit (GRU) network cell.
    Note: this is an implementation of the cuDNN version of GRUs
    (slight modification compared to Cho et al. 2014; the reset gate :math:`r_t`
    is applied after matrix multiplication).

    Each call computes the following function:

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
    hidden_size : int
        Number of units in output symbol.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'zeros'
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer, default 'zeros'
        Initializer for the bias vector.
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.
    activation : str, default 'tanh'
        Activation type to use. See nd/symbol Activation
        for supported types.
    recurrent_activation : str, default 'sigmoid'
        Activation type to use for the recurrent step. See nd/symbol Activation
        for supported types.


    Inputs:
        - **data**: input tensor with shape `(batch_size, input_size)`.
        - **states**: a list of one initial recurrent state tensor with shape
          `(batch_size, num_hidden)`.

    Outputs:
        - **out**: output tensor with shape `(batch_size, num_hidden)`.
        - **next_states**: a list of one output recurrent state tensor with the
          same shape as `states`.
    """
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0, activation='tanh', recurrent_activation='sigmoid'):
        super(GRUCell, self).__init__()
        self._hidden_size = hidden_size
        self._input_size = input_size
        self.i2h_weight = Parameter('i2h_weight', shape=(3*hidden_size, input_size),
                                    init=i2h_weight_initializer,
                                    allow_deferred_init=True)
        self.h2h_weight = Parameter('h2h_weight', shape=(3*hidden_size, hidden_size),
                                    init=h2h_weight_initializer,
                                    allow_deferred_init=True)
        self.i2h_bias = Parameter('i2h_bias', shape=(3*hidden_size,),
                                  init=i2h_bias_initializer,
                                  allow_deferred_init=True)
        self.h2h_bias = Parameter('h2h_bias', shape=(3*hidden_size,),
                                  init=h2h_bias_initializer,
                                  allow_deferred_init=True)
        self._activation = activation
        self._recurrent_activation = recurrent_activation

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._hidden_size), '__layout__': 'NC'}]

    def _alias(self):
        return 'gru'

    def __repr__(self):
        s = '{name}({mapping})'
        shape = self.i2h_weight.shape
        mapping = '{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0])
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

    def forward(self, inputs, states):
        # pylint: disable=too-many-locals
        device = inputs.device
        prev_state_h = states[0].to_device(device)
        i2h = npx.fully_connected(inputs,
                                  weight=self.i2h_weight.data(device),
                                  bias=self.i2h_bias.data(device),
                                  num_hidden=self._hidden_size * 3,
                                  no_bias=False)
        h2h = npx.fully_connected(prev_state_h,
                                  weight=self.h2h_weight.data(device),
                                  bias=self.h2h_bias.data(device),
                                  num_hidden=self._hidden_size * 3,
                                  no_bias=False)

        i2h_r, i2h_z, i2h = npx.slice_channel(i2h, num_outputs=3)
        h2h_r, h2h_z, h2h = npx.slice_channel(h2h, num_outputs=3)

        reset_gate = self._get_activation(i2h_r + h2h_r,
                                          self._recurrent_activation)
        update_gate = self._get_activation(i2h_z + h2h_z,
                                           self._recurrent_activation)
        next_h_tmp = self._get_activation(i2h + np.multiply(reset_gate, h2h),
                                          self._activation)
        ones = np.ones(update_gate.shape)
        next_h = np.multiply((ones - update_gate), next_h_tmp) + np.multiply(update_gate, prev_state_h)

        return next_h, [next_h]

    def infer_shape(self, i, x, is_bidirect):
        if i == 0:
            self.i2h_weight.shape = (3*self._hidden_size, x.shape[x.ndim-1])
        else:
            nh = self._hidden_size
            if is_bidirect:
                nh *= 2
            self.i2h_weight.shape = (3*self._hidden_size, nh)

@use_np
class SequentialRNNCell(RecurrentCell):
    """Sequentially stacking multiple RNN cells."""
    def __init__(self):
        super(SequentialRNNCell, self).__init__()
        self._layers = []

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        return s.format(name=self.__class__.__name__,
                        modstr='\n'.join(['({i}): {m}'.format(i=i, m=_indent(m().__repr__(), 2))
                                          for i, m in self._children.items()]))

    def add(self, cell):
        """Appends a cell into the stack.

        Parameters
        ----------
        cell : RecurrentCell
            The cell to add.
        """
        self._layers.append(cell)
        self.register_child(cell)

    def state_info(self, batch_size=0):
        return _cells_state_info(self._children.values(), batch_size)

    def begin_state(self, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. ZoneoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        return _cells_begin_state(self._children.values(), **kwargs)

    def __call__(self, inputs, states):
        self._counter += 1
        next_states = []
        p = 0
        assert all(not isinstance(cell(), BidirectionalCell) for cell in self._children.values())
        for cell in self._children.values():
            assert not isinstance(cell(), BidirectionalCell)
            n = len(cell().state_info())
            state = states[p:p+n]
            p += n
            inputs, state = cell()(inputs, state)
            next_states.append(state)
        return inputs, sum(next_states, [])

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        # pylint: disable=too-many-locals
        self.reset()

        inputs, _, batch_size = _format_sequence(length, inputs, layout, None)
        num_cells = len(self._children)
        begin_state = _get_begin_state(self, begin_state, inputs, batch_size)

        p = 0
        next_states = []
        for i, cell in enumerate(self._children.values()):
            n = len(cell().state_info())
            states = begin_state[p:p+n]
            p += n
            inputs, states = cell().unroll(length, inputs=inputs, begin_state=states,
                                           layout=layout,
                                           merge_outputs=None if i < num_cells-1 else merge_outputs,
                                           valid_length=valid_length)
            next_states.extend(states)

        return inputs, next_states

    def __getitem__(self, i):
        return self._children[str(i)]()

    def __len__(self):
        return len(self._children)

    def forward(self, *args, **kwargs):
        # pylint: disable=missing-docstring
        raise NotImplementedError

    def infer_shape(self, _, x, is_bidirect):
        for i, child in enumerate(self._layers):
            child.infer_shape(i, x, is_bidirect)


@use_np
class HybridSequentialRNNCell(HybridRecurrentCell):
    """Sequentially stacking multiple HybridRNN cells."""
    def __init__(self):
        super(HybridSequentialRNNCell, self).__init__()
        self._layers = []

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        return s.format(name=self.__class__.__name__,
                        modstr='\n'.join(['({i}): {m}'.format(i=i, m=_indent(m().__repr__(), 2))
                                          for i, m in self._children.items()]))

    def add(self, cell):
        """Appends a cell into the stack.

        Parameters
        ----------
        cell : RecurrentCell
            The cell to add.
        """
        self._layers.append(cell)
        self.register_child(cell)

    def state_info(self, batch_size=0):
        return _cells_state_info(self._children.values(), batch_size)

    def begin_state(self, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. ZoneoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        return _cells_begin_state(self._children.values(), **kwargs)

    def __call__(self, inputs, states):
        self._counter += 1
        next_states = []
        p = 0
        assert all(not isinstance(cell(), BidirectionalCell) for cell in self._children.values())
        for cell in self._children.values():
            n = len(cell().state_info())
            state = states[p:p+n]
            p += n
            inputs, state = cell()(inputs, state)
            next_states.append(state)
        return inputs, sum(next_states, [])

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        self.reset()

        inputs, _, batch_size = _format_sequence(length, inputs, layout, None)
        num_cells = len(self._children)
        begin_state = _get_begin_state(self, begin_state, inputs, batch_size)

        p = 0
        next_states = []
        for i, cell in enumerate(self._children.values()):
            n = len(cell().state_info())
            states = begin_state[p:p+n]
            p += n
            inputs, states = cell().unroll(length, inputs=inputs, begin_state=states,
                                           layout=layout,
                                           merge_outputs=None if i < num_cells-1 else merge_outputs,
                                           valid_length=valid_length)
            next_states.extend(states)

        return inputs, next_states

    def __getitem__(self, i):
        return self._children[str(i)]()

    def __len__(self):
        return len(self._children)

    def forward(self, inputs, states):
        return self.__call__(inputs, states)

    # pylint: disable=unused-argument
    def infer_shape(self, _, x, is_bidirect):
        for i, child in enumerate(self._layers):
            child.infer_shape(i, x, False)


@use_np
class DropoutCell(HybridRecurrentCell):
    """Applies dropout on input.

    Parameters
    ----------
    rate : float
        Percentage of elements to drop out, which
        is 1 - percentage to retain.
    axes : tuple of int, default ()
        The axes on which dropout mask is shared. If empty, regular dropout is applied.


    Inputs:
        - **data**: input tensor with shape `(batch_size, size)`.
        - **states**: a list of recurrent state tensors.

    Outputs:
        - **out**: output tensor with shape `(batch_size, size)`.
        - **next_states**: returns input `states` directly.
    """
    def __init__(self, rate, axes=()):
        super(DropoutCell, self).__init__()
        assert isinstance(rate, numeric_types), "rate must be a number"
        self._rate = rate
        self._axes = axes

    def __repr__(self):
        s = '{name}(rate={_rate}, axes={_axes})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)

    def state_info(self, batch_size=0):
        return []

    def _alias(self):
        return 'dropout'

    def forward(self, inputs, states):
        if self._rate > 0:
            inputs = npx.dropout(data=inputs, p=self._rate, axes=self._axes)
        return inputs, states

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        self.reset()

        inputs, _, _ = _format_sequence(length, inputs, layout, merge_outputs)
        if isinstance(inputs, tensor_types):
            return self.forward(inputs, begin_state if begin_state else [])
        return super(DropoutCell, self).unroll(
            length, inputs, begin_state=begin_state, layout=layout,
            merge_outputs=merge_outputs, valid_length=None)


@use_np
class ModifierCell(HybridRecurrentCell):
    """Base class for modifier cells. A modifier
    cell takes a base cell, apply modifications
    on it (e.g. Zoneout), and returns a new cell.

    After applying modifiers the base cell should
    no longer be called directly. The modifier cell
    should be used instead.
    """
    def __init__(self, base_cell):
        assert not base_cell._modified, \
            f"Cell {base_cell.name} is already modified. One cell cannot be modified twice"
        base_cell._modified = True
        super(ModifierCell, self).__init__()
        self.base_cell = base_cell

    @property
    def params(self):
        return self.base_cell.params

    def state_info(self, batch_size=0):
        return self.base_cell.state_info(batch_size)

    def begin_state(self, func=np.zeros, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        self.base_cell._modified = False
        begin = self.base_cell.begin_state(func=func, **kwargs)
        self.base_cell._modified = True
        return begin

    def forward(self, inputs, states):
        raise NotImplementedError

    def __repr__(self):
        s = '{name}({base_cell})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


@use_np
class ZoneoutCell(ModifierCell):
    """Applies Zoneout on base cell."""
    def __init__(self, base_cell, zoneout_outputs=0., zoneout_states=0.):
        assert not isinstance(base_cell, BidirectionalCell), \
            "BidirectionalCell doesn't support zoneout since it doesn't support step. " \
            "Please add ZoneoutCell to the cells underneath instead."
        assert not isinstance(base_cell, SequentialRNNCell) or not base_cell._bidirectional, \
            "Bidirectional SequentialRNNCell doesn't support zoneout. " \
            "Please add ZoneoutCell to the cells underneath instead."
        super(ZoneoutCell, self).__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self._prev_output = None

    def __repr__(self):
        s = '{name}(p_out={zoneout_outputs}, p_state={zoneout_states}, {base_cell})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)

    def _alias(self):
        return 'zoneout'

    def reset(self):
        super(ZoneoutCell, self).reset()
        self._prev_output = None

    def forward(self, inputs, states):
        device = inputs.device
        cell, p_outputs, p_states = self.base_cell, self.zoneout_outputs, self.zoneout_states
        next_output, next_states = cell(inputs, states)
        mask = (lambda p, like: npx.dropout(np.ones(like.shape), p=p))

        prev_output = self._prev_output
        if prev_output is None:
            prev_output = np.zeros(next_output.shape)

        output = (np.where(mask(p_outputs, next_output), next_output, prev_output)
                  if p_outputs != 0. else next_output)
        states = ([np.where(mask(p_states, new_s), new_s, old_s.to_device(device)) for new_s, old_s in
                   zip(next_states, states)] if p_states != 0. else next_states)

        self._prev_output = output

        return output, states

    def infer_shape(self, i, x, is_bidirect):
        self.base_cell.infer_shape(i, x, is_bidirect)

@use_np
class ResidualCell(ModifierCell):
    """
    Adds residual connection as described in Wu et al, 2016
    (https://arxiv.org/abs/1609.08144).
    Output of the cell is output of the base cell plus input.
    """

    def __init__(self, base_cell):
        # pylint: disable=useless-super-delegation
        super(ResidualCell, self).__init__(base_cell)

    def forward(self, inputs, states):
        output, states = self.base_cell(inputs, states)
        output = output + inputs
        return output, states

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        self.reset()

        self.base_cell._modified = False
        outputs, states = self.base_cell.unroll(length, inputs=inputs, begin_state=begin_state,
                                                layout=layout, merge_outputs=merge_outputs,
                                                valid_length=valid_length)
        self.base_cell._modified = True

        merge_outputs = isinstance(outputs, tensor_types) if merge_outputs is None else \
                        merge_outputs
        inputs, axis, _ = _format_sequence(length, inputs, layout, merge_outputs)
        if valid_length is not None:
            # mask the padded inputs to zero
            inputs = _mask_sequence_variable_length(inputs, length, valid_length, axis,
                                                    merge_outputs)
        if merge_outputs:
            outputs = outputs + inputs
        else:
            outputs = [i + j for i, j in zip(outputs, inputs)]

        return outputs, states

    def infer_shape(self, i, x, is_bidirect):
        self.base_cell.infer_shape(i, x, is_bidirect)


@use_np
class BidirectionalCell(HybridRecurrentCell):
    """Bidirectional RNN cell.

    Parameters
    ----------
    l_cell : RecurrentCell
        Cell for forward unrolling
    r_cell : RecurrentCell
        Cell for backward unrolling
    """
    def __init__(self, l_cell, r_cell):
        super(BidirectionalCell, self).__init__()
        self.l_cell = l_cell
        self.r_cell = r_cell

    def __call__(self, inputs, states):
        raise NotImplementedError("Bidirectional cannot be stepped. Please use unroll")

    def __repr__(self):
        s = '{name}(forward={l_cell}, backward={r_cell})'
        return s.format(name=self.__class__.__name__,
                        l_cell=self._children['l_cell'](),
                        r_cell=self._children['r_cell']())

    def state_info(self, batch_size=0):
        return _cells_state_info(self._children.values(), batch_size)

    def begin_state(self, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        return _cells_begin_state(self._children.values(), **kwargs)

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        # pylint: disable=too-many-locals
        self.reset()

        inputs, axis, batch_size = _format_sequence(length, inputs, layout, False)
        reversed_inputs = list(_reverse_sequences(inputs, length, valid_length))
        begin_state = _get_begin_state(self, begin_state, inputs, batch_size)

        states = begin_state
        l_cell, r_cell = [c() for c in self._children.values()]
        l_outputs, l_states = l_cell.unroll(length, inputs=inputs,
                                            begin_state=states[:len(l_cell.state_info(batch_size))],
                                            layout=layout, merge_outputs=merge_outputs,
                                            valid_length=valid_length)
        r_outputs, r_states = r_cell.unroll(length,
                                            inputs=reversed_inputs,
                                            begin_state=states[len(l_cell.state_info(batch_size)):],
                                            layout=layout, merge_outputs=False,
                                            valid_length=valid_length)
        reversed_r_outputs = _reverse_sequences(r_outputs, length, valid_length)

        if merge_outputs is None:
            merge_outputs = isinstance(l_outputs, tensor_types)
            l_outputs, _, _ = _format_sequence(None, l_outputs, layout, merge_outputs)
            reversed_r_outputs, _, _ = _format_sequence(None, reversed_r_outputs, layout,
                                                        merge_outputs)

        if merge_outputs:
            reversed_r_outputs = np.stack(reversed_r_outputs, axis=axis)
            outputs = np.concatenate([l_outputs, reversed_r_outputs], axis=2)

        else:
            outputs = [np.concatenate([l_o, r_o], axis=1)
                       for i, (l_o, r_o) in enumerate(zip(l_outputs, reversed_r_outputs))]
        if valid_length is not None:
            outputs = _mask_sequence_variable_length(outputs, length, valid_length, axis,
                                                     merge_outputs)
        states = l_states + r_states
        return outputs, states

    #pylint: disable=W0613
    def infer_shape(self, i, x, is_bidirect):
        l_cell, r_cell = [c() for c in self._children.values()]
        l_cell.infer_shape(i, x, True)
        r_cell.infer_shape(i, x, True)

@use_np
class VariationalDropoutCell(ModifierCell):
    """
    Applies Variational Dropout on base cell.
    https://arxiv.org/pdf/1512.05287.pdf

    Variational dropout uses the same dropout mask across time-steps. It can be applied to RNN
    inputs, outputs, and states. The masks for them are not shared.

    The dropout mask is initialized when stepping forward for the first time and will remain
    the same until .reset() is called. Thus, if using the cell and stepping manually without calling
    .unroll(), the .reset() should be called after each sequence.

    Parameters
    ----------
    base_cell : RecurrentCell
        The cell on which to perform variational dropout.
    drop_inputs : float, default 0.
        The dropout rate for inputs. Won't apply dropout if it equals 0.
    drop_states : float, default 0.
        The dropout rate for state inputs on the first state channel.
        Won't apply dropout if it equals 0.
    drop_outputs : float, default 0.
        The dropout rate for outputs. Won't apply dropout if it equals 0.
    """
    def __init__(self, base_cell, drop_inputs=0., drop_states=0., drop_outputs=0.):
        assert not drop_states or not isinstance(base_cell, BidirectionalCell), \
            "BidirectionalCell doesn't support variational state dropout. " \
            "Please add VariationalDropoutCell to the cells underneath instead."
        assert not drop_states \
               or not isinstance(base_cell, SequentialRNNCell) or not base_cell._bidirectional, \
            "Bidirectional SequentialRNNCell doesn't support variational state dropout. " \
            "Please add VariationalDropoutCell to the cells underneath instead."
        super(VariationalDropoutCell, self).__init__(base_cell)
        self.drop_inputs = drop_inputs
        self.drop_states = drop_states
        self.drop_outputs = drop_outputs
        self.drop_inputs_mask = None
        self.drop_states_mask = None
        self.drop_outputs_mask = None

    def _alias(self):
        return 'vardrop'

    def reset(self):
        super(VariationalDropoutCell, self).reset()
        self.drop_inputs_mask = None
        self.drop_states_mask = None
        self.drop_outputs_mask = None

    def _initialize_input_masks(self, inputs, states):
        if self.drop_states and self.drop_states_mask is None:
            self.drop_states_mask = npx.dropout(np.ones(states[0].shape),
                                                p=self.drop_states)

        if self.drop_inputs and self.drop_inputs_mask is None:
            self.drop_inputs_mask = npx.dropout(np.ones(inputs.shape),
                                                p=self.drop_inputs)

    def _initialize_output_mask(self, output):
        if self.drop_outputs and self.drop_outputs_mask is None:
            self.drop_outputs_mask = npx.dropout(np.ones(output.shape),
                                                 p=self.drop_outputs)


    def forward(self, inputs, states):
        device = inputs.device
        cell = self.base_cell
        self._initialize_input_masks(inputs, states)

        if self.drop_states:
            states = list(states)
            # state dropout only needs to be applied on h, which is always the first state.
            states[0] = states[0].to_device(device) * self.drop_states_mask

        if self.drop_inputs:
            inputs = inputs * self.drop_inputs_mask

        next_output, next_states = cell(inputs, states)

        self._initialize_output_mask(next_output)
        if self.drop_outputs:
            next_output = next_output * self.drop_outputs_mask

        return next_output, next_states

    def __repr__(self):
        s = '{name}(p_out = {drop_outputs}, p_state = {drop_states})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        """Unrolls an RNN cell across time steps.

        Parameters
        ----------
        length : int
            Number of steps to unroll.
        inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if `layout` is 'NTC',
            or (length, batch_size, ...) if `layout` is 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol, optional
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if `None`.
        layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool, optional
            If `False`, returns outputs as a list of Symbols.
            If `True`, concatenates output across time steps
            and returns a single symbol with shape
            (batch_size, length, ...) if layout is 'NTC',
            or (length, batch_size, ...) if layout is 'TNC'.
            If `None`, output whatever is faster.
        valid_length : Symbol, NDArray or None
            `valid_length` specifies the length of the sequences in the batch without padding.
            This option is especially useful for building sequence-to-sequence models where
            the input and output sequences would potentially be padded.
            If `valid_length` is None, all sequences are assumed to have the same length.
            If `valid_length` is a Symbol or NDArray, it should have shape (batch_size,).
            The ith element will be the length of the ith sequence in the batch.
            The last valid state will be return and the padded outputs will be masked with 0.
            Note that `valid_length` must be smaller or equal to `length`.

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
        """

        # Dropout on inputs and outputs can be performed on the whole sequence
        # only when state dropout is not present.
        if self.drop_states:
            return super(VariationalDropoutCell, self).unroll(length, inputs, begin_state,
                                                              layout, merge_outputs,
                                                              valid_length=valid_length)

        self.reset()

        inputs, axis, batch_size = _format_sequence(length, inputs, layout, True)
        states = _get_begin_state(self, begin_state, inputs, batch_size)

        if self.drop_inputs:
            inputs = npx.dropout(inputs, p=self.drop_inputs, axes=(axis,))

        outputs, states = self.base_cell.unroll(length, inputs, states, layout, merge_outputs=True,
                                                valid_length=valid_length)
        if self.drop_outputs:
            outputs = npx.dropout(outputs, p=self.drop_outputs, axes=(axis,))
        merge_outputs = isinstance(outputs, tensor_types) if merge_outputs is None else \
            merge_outputs
        outputs, _, _ = _format_sequence(length, outputs, layout, merge_outputs)
        if valid_length is not None:
            outputs = _mask_sequence_variable_length(outputs, length, valid_length, axis,
                                                     merge_outputs)
        return outputs, states

    def infer_shape(self, i, x, is_bidirect):
        self.base_cell.infer_shape(i, x, is_bidirect)

@use_np
class LSTMPCell(HybridRecurrentCell):
    r"""Long-Short Term Memory Projected (LSTMP) network cell.
    (https://arxiv.org/abs/1402.1128)

    Each call computes the following function:

    .. math::
        \begin{array}{ll}
        i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{ri} r_{(t-1)} + b_{ri}) \\
        f_t = sigmoid(W_{if} x_t + b_{if} + W_{rf} r_{(t-1)} + b_{rf}) \\
        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{rc} r_{(t-1)} + b_{rg}) \\
        o_t = sigmoid(W_{io} x_t + b_{io} + W_{ro} r_{(t-1)} + b_{ro}) \\
        c_t = f_t * c_{(t-1)} + i_t * g_t \\
        h_t = o_t * \tanh(c_t) \\
        r_t = W_{hr} h_t
        \end{array}

    where :math:`r_t` is the projected recurrent activation at time `t`,
    :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the
    cell state at time `t`, :math:`x_t` is the input at time `t`, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell, and
    out gates, respectively.

    Parameters
    ----------

    hidden_size : int
        Number of units in cell state symbol.
    projection_size : int
        Number of units in output symbol.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the hidden state.
    h2r_weight_initializer : str or Initializer
        Initializer for the projection weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'lstmbias'
        Initializer for the bias vector. By default, bias for the forget
        gate is initialized to 1 while all other biases are initialized
        to zero.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    Inputs:
        - **data**: input tensor with shape `(batch_size, input_size)`.
        - **states**: a list of two initial recurrent state tensors, with shape
          `(batch_size, projection_size)` and `(batch_size, hidden_size)` respectively.
    Outputs:
        - **out**: output tensor with shape `(batch_size, num_hidden)`.
        - **next_states**: a list of two output recurrent state tensors. Each has
          the same shape as `states`.
    """
    def __init__(self, hidden_size, projection_size,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 h2r_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0):
        super(LSTMPCell, self).__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._projection_size = projection_size
        self.i2h_weight = Parameter('i2h_weight', shape=(4*hidden_size, input_size),
                                    init=i2h_weight_initializer,
                                    allow_deferred_init=True)
        self.h2h_weight = Parameter('h2h_weight', shape=(4*hidden_size, projection_size),
                                    init=h2h_weight_initializer,
                                    allow_deferred_init=True)
        self.h2r_weight = Parameter('h2r_weight', shape=(projection_size, hidden_size),
                                    init=h2r_weight_initializer,
                                    allow_deferred_init=True)
        self.i2h_bias = Parameter('i2h_bias', shape=(4*hidden_size,),
                                  init=i2h_bias_initializer,
                                  allow_deferred_init=True)
        self.h2h_bias = Parameter('h2h_bias', shape=(4*hidden_size,),
                                  init=h2h_bias_initializer,
                                  allow_deferred_init=True)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._projection_size), '__layout__': 'NC'},
                {'shape': (batch_size, self._hidden_size), '__layout__': 'NC'}]

    def _alias(self):
        return 'lstmp'

    def __repr__(self):
        s = '{name}({mapping})'
        shape = self.i2h_weight.shape
        proj_shape = self.h2r_weight.shape
        mapping = '{0} -> {1} -> {2}'.format(shape[1] if shape[1] else None,
                                             shape[0], proj_shape[0])
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

    # pylint: disable= arguments-differ
    def forward(self, inputs, states):
        device = inputs.device
        i2h = npx.fully_connected(inputs, weight=self.i2h_weight.data(device),
                                  bias=self.i2h_bias.data(device),
                                  num_hidden=self._hidden_size*4, no_bias=False)
        h2h = npx.fully_connected(states[0].to_device(device),
                                  weight=self.h2h_weight.data(device),
                                  bias=self.h2h_bias.data(device),
                                  num_hidden=self._hidden_size*4, no_bias=False)
        gates = i2h + h2h
        slice_gates = npx.slice_channel(gates, num_outputs=4)
        in_gate = npx.activation(slice_gates[0], act_type="sigmoid")
        forget_gate = npx.activation(slice_gates[1], act_type="sigmoid")
        in_transform = npx.activation(slice_gates[2], act_type="tanh")
        out_gate = npx.activation(slice_gates[3], act_type="sigmoid")
        next_c = forget_gate * states[1].to_device(device) + in_gate * in_transform
        hidden = np.multiply(out_gate, npx.activation(next_c, act_type="tanh"))
        next_r = npx.fully_connected(hidden, num_hidden=self._projection_size,
                                     weight=self.h2r_weight.data(device), no_bias=True)

        return next_r, [next_r, next_c]

    def infer_shape(self, i, x, is_bidirect):
        if i == 0:
            self.i2h_weight.shape = (4*self._hidden_size, x.shape[x.ndim-1])
        else:
            nh = self._projection_size
            if is_bidirect:
                nh *= 2
            self.i2h_weight.shape = (4*self._hidden_size, nh)


def dynamic_unroll(cell, inputs, begin_state, drop_inputs=0, drop_outputs=0,
                   layout='TNC', valid_length=None):
    """Unrolls an RNN cell across time steps.

    Currently, 'TNC' is a preferred layout. unroll on the input of this layout
    runs much faster.

    Parameters
    ----------
    cell : an object whose base class is RNNCell.
        The RNN cell to run on the input sequence.
    inputs : Symbol
        It should have shape (batch_size, length, ...) if `layout` is 'NTC',
        or (length, batch_size, ...) if `layout` is 'TNC'.
    begin_state : nested list of Symbol
        The initial states of the RNN sequence.
    drop_inputs : float, default 0.
        The dropout rate for inputs. Won't apply dropout if it equals 0.
    drop_outputs : float, default 0.
        The dropout rate for outputs. Won't apply dropout if it equals 0.
    layout : str, optional
        `layout` of input symbol. Only used if inputs
        is a single Symbol.
    valid_length : Symbol, NDArray or None
        `valid_length` specifies the length of the sequences in the batch without padding.
        This option is especially useful for building sequence-to-sequence models where
        the input and output sequences would potentially be padded.
        If `valid_length` is None, all sequences are assumed to have the same length.
        If `valid_length` is a Symbol or NDArray, it should have shape (batch_size,).
        The ith element will be the length of the ith sequence in the batch.
        The last valid state will be return and the padded outputs will be masked with 0.
        Note that `valid_length` must be smaller or equal to `length`.

    Returns
    -------
    outputs : Symbol
        the output of the RNN from this unrolling.

    states : list of Symbol
        The new state of this RNN after this unrolling.
        The type of this symbol is same as the output of `begin_state`.

    Examples
    --------
    >>> seq_len = 3
    >>> batch_size = 2
    >>> input_size = 5
    >>> cell = mx.gluon.rnn.LSTMCell(input_size)
    >>> cell.initialize(device=mx.cpu())
    >>> rnn_data = mx.np.normal(loc=0, scale=1, shape=(seq_len, batch_size, input_size))
    >>> state_shape = (batch_size, input_size)
    >>> states = [mx.np.normal(loc=0, scale=1, shape=state_shape) for i in range(2)]
    >>> valid_length = mx.np.array([2, 3])
    >>> output, states = mx.gluon.rnn.rnn_cell.dynamic_unroll(cell, rnn_data, states,
    ...                                                       valid_length=valid_length,
    ...                                                       layout='TNC')
    >>> print(output)
    [[[ 0.00767238  0.00023103  0.03973929 -0.00925503 -0.05660512]
      [ 0.00881535  0.05428379 -0.02493718 -0.01834097  0.02189514]]
     [[-0.00676967  0.01447039  0.01287002 -0.00574152 -0.05734247]
      [ 0.01568508  0.02650866 -0.04270559 -0.04328435  0.00904011]]
     [[ 0.          0.          0.          0.          0.        ]
      [ 0.01055336  0.02734251 -0.03153727 -0.03742751 -0.01378113]]]
     <NDArray 3x2x5 @cpu(0)>
    """

    # Merge is always True, so we don't need length.
    inputs, axis, _ = _format_sequence(0, inputs, layout, True)
    if axis != 0:
        axes = list(range(len(layout)))
        tmp = axes[0]
        axes[0] = axes[axis]
        axes[axis] = tmp
        inputs = np.transpose(inputs, axes=axes)
    states = begin_state

    if drop_inputs:
        inputs = npx.dropout(inputs, p=drop_inputs, axes=(axis,))

    if valid_length is None:
        outputs, states = npx.foreach(cell, inputs, states + [valid_length])
    else:
        zeros = []
        for s in states:
            zeros.append(np.zeros(s.shape))
        states = list(_as_list(states))
        states.append(np.zeros((1)))
        class loop_body(HybridBlock):
            """Loop body for foreach operator"""
            def __init__(self, cell):
                super(loop_body, self).__init__()
                self.cell = cell

            def forward(self, inputs, states):
                valid_len = states.pop()
                cell_states = states[:-1]
                iter_no = states[-1]
                out, new_states = self.cell(inputs, cell_states)
                for i, state in enumerate(cell_states):
                    cond = npx.broadcast_greater(valid_len, iter_no)
                    cond_broad = np.broadcast_to(cond, new_states[i].T.shape).T
                    new_states[i] = np.where(cond_broad, new_states[i], state)
                new_states.append(iter_no + 1)
                new_states.append(valid_len)
                return out, new_states
        body = loop_body(cell)
        outputs, states = npx.foreach(body, inputs, states + [valid_length])
        states.pop()
    if drop_outputs:
        outputs = npx.dropout(outputs, p=drop_outputs, axes=(axis,))
    if valid_length is not None:
        if axis != 0:
            outputs = np.transpose(outputs, axes)
        outputs = npx.sequence_mask(outputs, sequence_length=valid_length,
                                    use_sequence_length=True, axis=axis)
        # the last state is the iteration number. We don't need it.
        return outputs, states[:-1]
    else:
        if axis != 0:
            outputs = np.transpose(outputs, axes)
        return outputs, states
