# RNN Cell API

```eval_rst
.. currentmodule:: mxnet.rnn
```

## Overview

The ``rnn`` module includes the recurrent neural network (RNN) cell APIs, a suite of tools for building the symbolic graph of RNN.

## The `rnn` module

### Cell interfaces

```eval_rst
.. autosummary::
    :nosignatures:

    BaseRNNCell.__call__
    BaseRNNCell.unroll
    BaseRNNCell.reset
    BaseRNNCell.begin_state
    BaseRNNCell.unpack_weights
    BaseRNNCell.pack_weights
```

The cell API operates on symbols and returns output symbols based on the type of RNN. Take
Long-short term memory (LSTM) as an example:

```python
>>> import mxnet as mx
>>> batch_size = 32
>>> input_dim = 100
>>> embed_dim = 25
>>> step_input = mx.symbol.Variable('step_data')
>>> in_shape = {'step_data': (batch_size,)}
>>> # First we embed our raw input data to be used as LSTM's input.
... embedded_step = mx.symbol.Embedding(data=step_input, \
...                                    input_dim=input_dim, \
...                                    output_dim=embed_dim)
>>> # Then we create an LSTM cell.
... lstm_cell = mx.rnn.LSTMCell(num_hidden=50)
>>> # Initialize its hidden and memory states.
... begin_state = lstm_cell.begin_state()
```

We define the following utility function to more easily examine the shape of output symbols.
```python
>>> # Define utility function for checking shape.
... def print_shape(args, in_shapes, output=None):
...     def sym_shape(symbol):
...         symbol= symbol.get_internals()
...         arg, out, aux = symbol.infer_shape(**{k:in_shapes[k] for k in
...             in_shapes.iterkeys() if k in symbol.list_arguments()})
...         if not output or 'arg' in output:
...             print('\n'.join(map(str, zip(symbol.list_arguments(), arg))))
...         if not output or 'out' in output:
...             print('\n'.join(map(str, zip(symbol.list_outputs(), out)[-1])))
...         if not output or 'aux' in output:
...             print('\n'.join(map(str, zip(symbol.list_auxiliary_states(), aux))))
...     if isinstance(args, mx.symbol.Symbol):
...         sym_shape(args)
...     elif isinstance(args, list):
...         print('Input is a list of length {0}'.format(len(args)))
...         for arg in args:
...             if isinstance(arg, list):
...                 print('Element is a list of length {0}'.format(len(arg)))
...                 for a in arg:
...                     sym_shape(a)
...             else:
...                 sym_shape(arg)
...
```

The LSTM cell, like several other RNN cells, are callable. Calling the cell transforms the input once based on LSTM definition. See this [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) for a great introduction to LSTM and other RNN.
```python
>>> # Call the cell to get the output of one time step for a batch.
... output, states = lstm_cell(embedded_step, begin_state)
>>> # Shape of LSTM output, which is often called the "hidden state".
>>> print_shape(output, in_shape, ['out'])
lstm_t0_out_output
(32L, 50L)
>>> # Shapes of states that will be carried over to the next step,
... # which includes both the "hidden state" and the "cell state".
... print_shape(states, in_shape, ['out'])
Input is a list of length 2
lstm_t0_out_output
(32L, 50L)
lstm_t0_state_output
(32L, 50L)
```

Most of the time, we need to process not just one step, but rather a sequence of
many steps. For this, we need to unroll the LSTM.
```python
>>> # Embed a sequence of length 5.
... sequence_length = 5
>>> seq_input = mx.symbol.Variable('seq_data')
>>> in_shape = {'seq_data': (sequence_length, batch_size)}
>>> embedded_seq = mx.symbol.Embedding(data=seq_input, \
...                                    input_dim=input_dim, \
...                                    output_dim=embed_dim)
```
```eval_rst
.. note:: Remember to reset the cell when unrolling/stepping for a new sequence.
```
```python
>>> lstm_cell.reset()
>>> outputs, states = lstm_cell.unroll(length=sequence_length, \
...                                    inputs=embedded_seq, \
...                                    layout='TNC', \
...                                    merge_outputs=True)
>>> # Notice that merge_outputs was set to True. This will merge the outputs,
... # to a single symbol.
... print_shape(outputs, in_shape, ['out'])
concat0_output
(5L, 32L, 50L)
>>> # The hidden state and cell state from the last time step is returned.
... print_shape(states, in_shape, ['out'])
Input is a list of length 2
lstm_t4_out_output
(32L, 50L)
lstm_t4_state_output
(32L, 50L)
>>> # If merge_outputs is set to False, a list of symbols of each time step
... # is returned.
... lstm_cell.reset()
>>> outputs, states = lstm_cell.unroll(length=sequence_length, \
...                                    inputs=embedded_seq, \
...                                    layout='TNC', \
...                                    merge_outputs=False)
>>> print_shape(outputs, in_shape, ['out'])
Input is a list of length 5
lstm_t0_out_output
(32L, 50L)
lstm_t1_out_output
(32L, 50L)
lstm_t2_out_output
(32L, 50L)
lstm_t3_out_output
(32L, 50L)
lstm_t4_out_output
(32L, 50L)
```

```eval_rst
.. note:: Loading and saving models that are built with RNN cells API requires using
    ``mx.rnn.load_rnn_checkpoint``, ``mx.rnn.save_rnn_checkpoint``, and ``mx.rnn.do_rnn_checkpoint``.
    The list of all the used cells should be provided as the first argument to those functions.
```

### Basic RNN cells

``rnn`` module supports the following RNN cell types.

```eval_rst
.. autosummary::
    :nosignatures:

    LSTMCell
    GRUCell
    RNNCell
```

### Modifier cells

```eval_rst
.. autosummary::
    :nosignatures:

    BidirectionalCell
    DropoutCell
    ZoneoutCell
```

A modifier cell takes in one or more cells and transforms the output of those cells.
`BidirectionalCell` is one example. It takes two cells for forward unroll and backward unroll
respectively. After unrolling, the outputs of the forward and backward pass are concatenated.
```python
>>> # Bidirectional cell takes two RNN cells, for forward and backward pass respectively.
... bi_cell = mx.rnn.BidirectionalCell(
...                 mx.rnn.LSTMCell(num_hidden=50),
...                 mx.rnn.GRUCell(num_hidden=75))
>>> outputs, states = bi_cell.unroll(length=sequence_length, \
...                                  inputs=embedded_seq, \
...                                  layout='TNC', \
...                                  merge_outputs=True)
>>> # The output feature is the concatenation of the forward and backward pass.
... # Thus, the number of output dimensions is the sum of the dimensions of the two cells.
... print_shape(outputs, in_shape, ['out'])
bi_out_output
(5L, 32L, 125L)
>>> # The states of the BidirectionalCell is a list of two lists, corresponding to the
... # states of the forward and backward cells respectively. Note that LSTM has two state
... # channels, whereas GRU has only one state channel.
>>> print_shape(states, in_shape, ['out'])
Input is a list of length 2
Element is a list of length 2
lstm_t4_out_output
(32L, 50L)
lstm_t4_state_output
(32L, 50L)
Element is a list of length 1
gru_t4_out_output
(32L, 75L)
```
```eval_rst
.. note:: BidirectionalCell cannot be called or stepped, because the backward unroll requires the output of
    future steps, and thus the whole sequence is required.
```

Dropout and zoneout are effective regularization techniques that can be applied on RNN. ``rnn``
module provides `DropoutCell` and `ZoneoutCell` for regularization on the output and recurrent
states of RNN. `ZoneoutCell` takes one RNN cell in the constructor, and are unrolled like
other cells.
```python
>>> zoneout_cell = mx.rnn.ZoneoutCell(lstm_cell, zoneout_states=0.5)
>>> outputs, states = zoneout_cell.unroll(length=sequence_length, \
...                                       inputs=embedded_seq, \
...                                       layout='TNC', \
...                                       merge_outputs=True)
```
`DropoutCell` performs dropout on the output of the sequence input. It can be used in a stacked
multi-layer RNN setting, which we will cover next.

### Multi-layer cells

```eval_rst
.. autosummary::
    :nosignatures:

    SequentialRNNCell
    SequentialRNNCell.add
```

The ``SequentialRNNCell`` allows stacking multiple layers of RNN cells to improve the expressiveness
and performance of the model. Cells can be added to a ``SequentialRNNCell`` in order, from bottom to
top. When unrolling, the output of a lower-level cell is automatically passed to the next
higher-level.

```python
>>> stacked_rnn_cells = mx.rnn.SequentialRNNCell()
>>> stacked_rnn_cells.add(mx.rnn.BidirectionalCell(
...                         mx.rnn.LSTMCell(num_hidden=50),
...                         mx.rnn.LSTMCell(num_hidden=50)))
>>> # Dropout the output of the BidirectionalCell with a retention probability of 0.5.
... stacked_rnn_cells.add(mx.rnn.DropoutCell(0.5))
>>> stacked_rnn_cells.add(mx.rnn.LSTMCell(num_hidden=50))
>>> outputs, states = stacked_rnn_cells.unroll(length=sequence_length, \
...                                            inputs=embedded_seq, \
...                                            layout='TNC', \
...                                            merge_outputs=True)
>>> # The output of SequentialRNNCell is the same as that of the last layer.
... print_shape(outputs, in_shape, ['out'])
concat6_output
(5L, 32L, 50L)
>>> # The states of the SequentialRNNCell is a list of lists corresponding to the
... # states of each of the added cells respectively.
... print_shape(states, in_shape, ['out'])
Input is a list of length 4
Element is a list of length 2
lstm_t4_out_output
(32L, 50L)
lstm_t4_state_output
(32L, 50L)
Element is a list of length 2
lstm_t4_out_output
(32L, 50L)
lstm_t4_state_output
(32L, 50L)
lstm_t4_out_output
(32L, 50L)
lstm_t4_state_output
(32L, 50L)
```

### Fused RNN cell

```eval_rst
.. autosummary::
    :nosignatures:

    FusedRNNCell
    FusedRNNCell.unfuse
```

The computation of RNN for an input sequence consists of many GEMM and point-wise operations with
temporal dependencies dependencies. This could make the computation memory-bound especially on GPU,
resulting in longer wall-time. By combining the computation of many small matrices into that of
larger ones and streaming the computation whenever possible, the ratio of computation to memory I/O
can be increased, which results in better performance on GPU. Such optimization technique is called
"fusing", which
[this post](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/)
talks in more details.

The ``rnn`` module includes a ``FusedRNNCell``, which provides the optimized fused implementation.
This cell also offers functionalities similar to the modifier cells, such as bidirectional and
dropout support.

```python
>>> fused_lstm_cell = mx.rnn.FusedRNNCell(num_hidden=50, \
...                                       num_layers=3, \
...                                       mode='lstm', \
...                                       bidirectional=True, \
...                                       dropout=0.5)
>>> outputs, _ = fused_lstm_cell.unroll(length=sequence_length, \
...                                     inputs=embedded_seq, \
...                                     layout='TNC', \
...                                     merge_outputs=True)
>>> print_shape(outputs, in_shape, ['out'])
lstm_rnn_output
(5L, 32L, 100L)
```
```eval_rst
.. note:: ``FusedRNNCell`` supports GPU-only. It cannot be called or stepped.
.. note:: When `dropout` is set to non-zero in ``FusedRNNCell``, the dropout is applied to the
    output of all layers except the last layer. If there is only one layer in the ``FusedRNNCell``, the
    dropout rate is ignored.
.. note:: Similar to ``BidirectionalCell``, when `bidirectional` flag is set to `True`, the output
    of ``FusedRNNCell`` is twice the size specified by `num_hidden`.
```

The ``unfuse()`` method can be used to convert the ``FusedRNNCell`` into an equivalent
and CPU-compatible ``SequentialRNNCell`` that mirrors the settings of the ``FusedRNNCell``.
```python
>>> unfused_lstm_cell = fused_lstm_cell.unfuse()
>>> unfused_outputs, _ = unfused_lstm_cell.unroll(length=sequence_length, \
...                                               inputs=embedded_seq, \
...                                               layout='TNC', \
...                                               merge_outputs=True)
>>> print_shape(unfused_outputs, in_shape, ['out'])
lstm_bi_l2_out_output
(5L, 32L, 100L)
```

### RNN checkpoint methods and parameters

```eval_rst
.. autosummary::
    :nosignatures:

    save_rnn_checkpoint
    load_rnn_checkpoint
    do_rnn_checkpoint
```
```eval_rst
.. autosummary::
    :nosignatures:

    RNNParams
    RNNParams.get
```

The model parameters from the training with fused cell can be used for inference with unfused cell,
and vice versa. As the parameters of fused and unfused cells are organized differently, they need to
be converted first.
```python
>>> # FusedRNNCell's parameters are merged.
... print_shape(outputs, in_shape, ['arg'])
('seq_data', (5L, 32L))
('embedding1_weight', (100L, 25L))
('lstm_parameters', (152400L,))
>>> # The equivalent SequentialRNNCell's parameters are separate.
... print_shape(unfused_outputs, in_shape, ['arg'])
('seq_data', (5L, 32L))
('embedding1_weight', (100L, 25L))
('lstm_l0_i2h_weight', (200L, 25L))
('lstm_l0_i2h_bias', (200L,))
('lstm_l0_h2h_weight', (200L, 50L))
('lstm_l0_h2h_bias', (200L,))
('lstm_r0_i2h_weight', (200L, 25L))
('lstm_r0_i2h_bias', (200L,))
('lstm_r0_h2h_weight', (200L, 50L))
('lstm_r0_h2h_bias', (200L,))
('lstm_l1_i2h_weight', (200L, 100L))
('lstm_l1_i2h_bias', (200L,))
('lstm_l1_h2h_weight', (200L, 50L))
('lstm_l1_h2h_bias', (200L,))
('lstm_r1_i2h_weight', (200L, 100L))
('lstm_r1_i2h_bias', (200L,))
('lstm_r1_h2h_weight', (200L, 50L))
('lstm_r1_h2h_bias', (200L,))
('lstm_l2_i2h_weight', (200L, 100L))
('lstm_l2_i2h_bias', (200L,))
('lstm_l2_h2h_weight', (200L, 50L))
('lstm_l2_h2h_bias', (200L,))
('lstm_r2_i2h_weight', (200L, 100L))
('lstm_r2_i2h_bias', (200L,))
('lstm_r2_h2h_weight', (200L, 50L))
('lstm_r2_h2h_bias', (200L,))
```

All cells in the ``rnn`` module has ``unpack_weights()`` for converting ``FusedRNNCell`` parameters
to the unfused format and ``pack_weights()`` for fusing the parameters. The RNN specific
checkpointing methods (``load_rnn_checkpoint, save_rnn_checkpoint, do_rnn_checkpoint``) handle the
conversion transparently based on the provided cells.

### I/O utilities

```eval_rst
.. autosummary::
    :nosignatures:

    BucketSentenceIter
    encode_sentences
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. autoclass:: mxnet.rnn.BaseRNNCell
    :members:

    .. automethod:: __call__
.. autoclass:: mxnet.rnn.LSTMCell
    :members:
.. autoclass:: mxnet.rnn.GRUCell
    :members:
.. autoclass:: mxnet.rnn.RNNCell
    :members:
.. autoclass:: mxnet.rnn.FusedRNNCell
    :members:
.. autoclass:: mxnet.rnn.SequentialRNNCell
    :members:
.. autoclass:: mxnet.rnn.BidirectionalCell
    :members:
.. autoclass:: mxnet.rnn.DropoutCell
    :members:
.. autoclass:: mxnet.rnn.ZoneoutCell
    :members:
.. autoclass:: mxnet.rnn.RNNParams
    :members:


.. autoclass:: mxnet.rnn.BucketSentenceIter
    :members:
.. automethod:: mxnet.rnn.encode_sentences

.. automethod:: mxnet.rnn.save_rnn_checkpoint

.. automethod:: mxnet.rnn.load_rnn_checkpoint

.. automethod:: mxnet.rnn.do_rnn_checkpoint

```

<script>auto_index("api-reference");</script>
