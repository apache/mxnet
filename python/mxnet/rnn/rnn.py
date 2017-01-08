# coding: utf-8
"""Functions for constructing recurrent neural networks."""

from . import rnn_cell

from .. import symbol
from .. import ndarray


def rnn_unroll(cell, length, inputs=None, begin_state=None, params=None, prefix=''):
    if inputs is None:
        inputs = [symbol.Variable('%st%d_data'%(prefix, i)) for i in range(length)]
    if params is None:
        params = rnn_cell.RNNParams()
    if begin_state is None:
        begin_state = cell.begin_state(prefix)

    states = begin_state
    outputs = []
    for i in range(length):
        output, states = cell(inputs[i], states, params, prefix=prefix)
        outputs.append(output)

    return outputs, states, params
