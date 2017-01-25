# coding: utf-8
"""Functions for constructing recurrent neural networks."""
from .. import symbol


def rnn_unroll(cell, length, inputs=None, begin_state=None, input_prefix=''):
    """Unroll an RNN cell across time steps.

    Parameters
    ----------
    length : int
        number of steps to unroll
    inputs : list of Symbol
        input symbols, 2D, batch_size * num_units.
        Placeholder ariables are automatically
        created if None
    begin_state : nested list of Symbol
        input states. Can be from cell.begin_state()
        or output state of another cell. Created
        from cell.begin_state() if None.
    input_prefix : str
        prefix for automatically created input
        placehodlers.
    """
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
