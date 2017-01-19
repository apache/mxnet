# coding: utf-8
"""Functions for constructing recurrent neural networks."""

from . import rnn_cell

from .. import symbol
from .. import ndarray


def rnn_unroll(cell, length, inputs=None, begin_state=None, input_prefix=''):
    if inputs is None:
        inputs = [symbol.Variable('%st%d_data'%(input_prefix, i)) for i in range(length)]
    if begin_state is None:
        begin_state = cell.begin_state()

    states = begin_state
    outputs = []
    for i in range(length):
        output, states = cell(inputs[i], states)
        outputs.append(output)

    return outputs, states
