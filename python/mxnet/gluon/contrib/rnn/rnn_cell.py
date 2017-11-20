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
__all__ = ['VariationalDropoutCell']

from ...rnn import BidirectionalCell, SequentialRNNCell, ModifierCell
from ...rnn.rnn_cell import _format_sequence, _get_begin_state


class VariationalDropoutCell(ModifierCell):
    """
    Applies Variational Dropout on base cell.
    (https://arxiv.org/pdf/1512.05287.pdf,
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

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
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
                                                              layout, merge_outputs)

        self.reset()

        inputs, axis, F, batch_size = _format_sequence(length, inputs, layout, True)
        states = _get_begin_state(self, F, begin_state, inputs, batch_size)

        if self.drop_inputs:
            first_input = inputs.slice_axis(axis, 0, 1).split(1, axis=axis, squeeze_axis=True)
            self._initialize_input_masks(F, first_input, states)
            inputs = F.broadcast_mul(inputs, self.drop_inputs_mask.expand_dims(axis=axis))

        outputs, states = self.base_cell.unroll(length, inputs, states, layout, merge_outputs=True)
        if self.drop_outputs:
            first_output = outputs.slice_axis(axis, 0, 1).split(1, axis=axis, squeeze_axis=True)
            self._initialize_output_mask(F, first_output)
            outputs = F.broadcast_mul(outputs, self.drop_outputs_mask.expand_dims(axis=axis))

        outputs, _, _, _ = _format_sequence(length, outputs, layout, merge_outputs)

        return outputs, states
