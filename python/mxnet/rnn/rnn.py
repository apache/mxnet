# coding: utf-8
# pylint: disable=too-many-arguments, no-member
"""Functions for constructing recurrent neural networks."""
from .. import symbol


def rnn_unroll(cell, length, inputs=None, begin_state=None, input_prefix='', layout='NTC'):
    """Unroll an RNN cell across time steps.

    Parameters
    ----------
    cell : children of BaseRNNCell
        the cell to be unrolled.
    length : int
        number of steps to unroll
    inputs : Symbol, list of Symbol, or None
        if inputs is a single Symbol (usually the output
        of Embedding symbol), it should have shape
        (batch_size, length, ...) if layout == 'NTC',
        or (length, batch_size, ...) if layout == 'TNC'.

        If inputs is a grouped symbol or a list of
        symbols (usually output of SliceChannel or previous
        unroll), they should all have shape (batch_size, ...).

        if inputs is None, Placeholder ariables are
        automatically created.
    begin_state : nested list of Symbol
        input states. Created by cell.begin_state()
        or output state of another cell. Created
        from cell.begin_state() if None.
    input_prefix : str
        prefix for automatically created input
        placehodlers.
    layout : str
        layout of input symbol. Only used if inputs
        is a single Symbol.
    """
    if inputs is None:
        inputs = [symbol.Variable('%st%d_data'%(input_prefix, i)) for i in range(length)]
    elif isinstance(inputs, symbol.Symbol):
        if len(inputs.list_outputs()) != length:
            assert len(inputs.list_outputs()) == 1
            axis = layout.find('T')
            inputs = symbol.SliceChannel(inputs, axis=axis, num_outputs=length, squeeze_axis=1)
    else:
        assert len(inputs) == length
    if begin_state is None:
        begin_state = cell.begin_state()

    states = begin_state
    outputs = []
    for i in range(length):
        output, states = cell(inputs[i], states)
        outputs.append(output)

    return outputs, states
