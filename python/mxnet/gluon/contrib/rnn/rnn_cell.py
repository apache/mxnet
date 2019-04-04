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
"""Definition of various recurrent neural network cells."""
__all__ = ['VariationalDropoutCell', 'LSTMPCell']

from ...rnn import BidirectionalCell, SequentialRNNCell, ModifierCell, HybridRecurrentCell
from ...rnn.rnn_cell import _format_sequence, _get_begin_state, _mask_sequence_variable_length
from ... import tensor_types

class VariationalDropoutCell(ModifierCell):
    """
    Applies Variational Dropout on base cell.
    (https://arxiv.org/pdf/1512.05287.pdf, \
     https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf).

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

    def _initialize_input_masks(self, F, inputs, states):
        if self.drop_states and self.drop_states_mask is None:
            self.drop_states_mask = F.Dropout(F.ones_like(states[0]),
                                              p=self.drop_states)

        if self.drop_inputs and self.drop_inputs_mask is None:
            self.drop_inputs_mask = F.Dropout(F.ones_like(inputs),
                                              p=self.drop_inputs)

    def _initialize_output_mask(self, F, output):
        if self.drop_outputs and self.drop_outputs_mask is None:
            self.drop_outputs_mask = F.Dropout(F.ones_like(output),
                                               p=self.drop_outputs)


    def hybrid_forward(self, F, inputs, states):
        cell = self.base_cell
        self._initialize_input_masks(F, inputs, states)

        if self.drop_states:
            states = list(states)
            # state dropout only needs to be applied on h, which is always the first state.
            states[0] = states[0] * self.drop_states_mask

        if self.drop_inputs:
            inputs = inputs * self.drop_inputs_mask

        next_output, next_states = cell(inputs, states)

        self._initialize_output_mask(F, next_output)
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

        inputs, axis, F, batch_size = _format_sequence(length, inputs, layout, True)
        states = _get_begin_state(self, F, begin_state, inputs, batch_size)

        if self.drop_inputs:
            inputs = F.Dropout(inputs, p=self.drop_inputs, axes=(axis,))

        outputs, states = self.base_cell.unroll(length, inputs, states, layout, merge_outputs=True,
                                                valid_length=valid_length)
        if self.drop_outputs:
            outputs = F.Dropout(outputs, p=self.drop_outputs, axes=(axis,))
        merge_outputs = isinstance(outputs, tensor_types) if merge_outputs is None else \
            merge_outputs
        outputs, _, _, _ = _format_sequence(length, outputs, layout, merge_outputs)
        if valid_length is not None:
            outputs = _mask_sequence_variable_length(F, outputs, length, valid_length, axis,
                                                     merge_outputs)
        return outputs, states


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
    prefix : str, default ``'lstmp_``'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
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
                 input_size=0, prefix=None, params=None):
        super(LSTMPCell, self).__init__(prefix=prefix, params=params)

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._projection_size = projection_size
        self.i2h_weight = self.params.get('i2h_weight', shape=(4*hidden_size, input_size),
                                          init=i2h_weight_initializer,
                                          allow_deferred_init=True)
        self.h2h_weight = self.params.get('h2h_weight', shape=(4*hidden_size, projection_size),
                                          init=h2h_weight_initializer,
                                          allow_deferred_init=True)
        self.h2r_weight = self.params.get('h2r_weight', shape=(projection_size, hidden_size),
                                          init=h2r_weight_initializer,
                                          allow_deferred_init=True)
        self.i2h_bias = self.params.get('i2h_bias', shape=(4*hidden_size,),
                                        init=i2h_bias_initializer,
                                        allow_deferred_init=True)
        self.h2h_bias = self.params.get('h2h_bias', shape=(4*hidden_size,),
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
    def hybrid_forward(self, F, inputs, states, i2h_weight,
                       h2h_weight, h2r_weight, i2h_bias, h2h_bias):
        prefix = 't%d_'%self._counter
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._hidden_size*4, name=prefix+'i2h')
        h2h = F.FullyConnected(data=states[0], weight=h2h_weight, bias=h2h_bias,
                               num_hidden=self._hidden_size*4, name=prefix+'h2h')
        gates = i2h + h2h
        slice_gates = F.SliceChannel(gates, num_outputs=4, name=prefix+'slice')
        in_gate = F.Activation(slice_gates[0], act_type="sigmoid", name=prefix+'i')
        forget_gate = F.Activation(slice_gates[1], act_type="sigmoid", name=prefix+'f')
        in_transform = F.Activation(slice_gates[2], act_type="tanh", name=prefix+'c')
        out_gate = F.Activation(slice_gates[3], act_type="sigmoid", name=prefix+'o')
        next_c = F._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                   name=prefix+'state')
        hidden = F._internal._mul(out_gate, F.Activation(next_c, act_type="tanh"),
                                  name=prefix+'hidden')
        next_r = F.FullyConnected(data=hidden, num_hidden=self._projection_size,
                                  weight=h2r_weight, no_bias=True, name=prefix+'out')

        return next_r, [next_r, next_c]
    # pylint: enable= arguments-differ
