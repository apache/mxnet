# RNN Cell API

```eval_rst
.. currentmodule:: mxnet.rnn
```

```eval_rst
.. warning:: This package is currently experimental and may change in the near future.
```

## Overview

The `rnn` module includes the recurrent neural network (RNN) cell APIs, a suite of tools for building an RNN's symbolic graph.
```eval_rst
.. note:: The `rnn` module offers higher-level interface while `symbol.RNN` is a lower-level interface. The cell APIs in `rnn` module are easier to use in most cases.
```

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

When working with the cell API, the precise input and output symbols
depend on the type of RNN you are using. Take Long Short-Term Memory (LSTM) for example:

```python
import mxnet as mx
# Shape of 'step_data' is (batch_size,).
step_input = mx.symbol.Variable('step_data')

# First we embed our raw input data to be used as LSTM's input.
embedded_step = mx.symbol.Embedding(data=step_input, \
                                    input_dim=input_dim, \
                                    output_dim=embed_dim)

# Then we create an LSTM cell.
lstm_cell = mx.rnn.LSTMCell(num_hidden=50)
# Initialize its hidden and memory states.
# 'begin_state' method takes an initialization function, and uses 'zeros' by default.
begin_state = lstm_cell.begin_state()
```

The LSTM cell and other non-fused RNN cells are callable. Calling the cell updates it's state once. This transformation depends on both the current input and the previous states. See this [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) for a great introduction to LSTM and other RNN.
```python
# Call the cell to get the output of one time step for a batch.
output, states = lstm_cell(embedded_step, begin_state)

# 'output' is lstm_t0_out_output of shape (batch_size, hidden_dim).

# 'states' has the recurrent states that will be carried over to the next step,
# which includes both the "hidden state" and the "cell state":
# Both 'lstm_t0_out_output' and 'lstm_t0_state_output' have shape (batch_size, hidden_dim).
```

Most of the time our goal is to process a sequence of many steps. For this, we need to unroll the LSTM according to the sequence length.
```python
# Embed a sequence. 'seq_data' has the shape of (batch_size, sequence_length).
seq_input = mx.symbol.Variable('seq_data')
embedded_seq = mx.symbol.Embedding(data=seq_input, \
                                   input_dim=input_dim, \
                                   output_dim=embed_dim)
```
```eval_rst
.. note:: Remember to reset the cell when unrolling/stepping for a new sequence by calling `lstm_cell.reset()`.
```
```python
# Note that when unrolling, if 'merge_outputs' is set to True, the 'outputs' is merged into a single symbol
# In the layout, 'N' represents batch size, 'T' represents sequence length, and 'C' represents the
# number of dimensions in hidden states.
outputs, states = lstm_cell.unroll(length=sequence_length, \
                                   inputs=embedded_seq, \
                                   layout='NTC', \
                                   merge_outputs=True)
# 'outputs' is concat0_output of shape (batch_size, sequence_length, hidden_dim).
# The hidden state and cell state from the final time step is returned:
# Both 'lstm_t4_out_output' and 'lstm_t4_state_output' have shape (batch_size, hidden_dim).

# If merge_outputs is set to False, a list of symbols for each of the time steps is returned.
outputs, states = lstm_cell.unroll(length=sequence_length, \
                                   inputs=embedded_seq, \
                                   layout='NTC', \
                                   merge_outputs=False)
# In this case, 'outputs' is a list of symbols. Each symbol is of shape (batch_size, hidden_dim).
```

```eval_rst
.. note:: Loading and saving models that are built with RNN cells API requires using
    `mx.rnn.load_rnn_checkpoint`, `mx.rnn.save_rnn_checkpoint`, and `mx.rnn.do_rnn_checkpoint`.
    The list of all the used cells should be provided as the first argument to those functions.
```

### Basic RNN cells

`rnn` module supports the following RNN cell types.

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
    ResidualCell
```

A modifier cell takes in one or more cells and transforms the output of those cells.
`BidirectionalCell` is one example. It takes two cells for forward unroll and backward unroll
respectively. After unrolling, the outputs of the forward and backward pass are concatenated.
```python
# Bidirectional cell takes two RNN cells, for forward and backward pass respectively.
# Having different types of cells for forward and backward unrolling is allowed.
bi_cell = mx.rnn.BidirectionalCell(
                 mx.rnn.LSTMCell(num_hidden=50),
                 mx.rnn.GRUCell(num_hidden=75))
outputs, states = bi_cell.unroll(length=sequence_length, \
                                 inputs=embedded_seq, \
                                 merge_outputs=True)
# The output feature is the concatenation of the forward and backward pass.
# Thus, the number of output dimensions is the sum of the dimensions of the two cells.
# 'outputs' is the symbol 'bi_out_output' of shape (batch_size, sequence_length, 125L)

# The states of the BidirectionalCell is a list of two lists, corresponding to the
# states of the forward and backward cells respectively.
```
```eval_rst
.. note:: BidirectionalCell cannot be called or stepped, because the backward unroll requires the output of
    future steps, and thus the whole sequence is required.
```

Dropout and zoneout are popular regularization techniques that can be applied to RNN. `rnn`
module provides `DropoutCell` and `ZoneoutCell` for regularization on the output and recurrent
states of RNN. `ZoneoutCell` takes one RNN cell in the constructor, and supports unrolling like
other cells.
```python
zoneout_cell = mx.rnn.ZoneoutCell(lstm_cell, zoneout_states=0.5)
outputs, states = zoneout_cell.unroll(length=sequence_length, \
                                      inputs=embedded_seq, \
                                      merge_outputs=True)
```
`DropoutCell` performs dropout on the input sequence. It can be used in a stacked
multi-layer RNN setting, which we will cover next.

Residual connection is a useful technique for training deep neural models because it helps the
propagation of gradients by shortening the paths.  `ResidualCell` provides such functionality for
RNN models.
```python
residual_cell = mx.rnn.ResidualCell(lstm_cell)
outputs, states = residual_cell.unroll(length=sequence_length, \
                                       inputs=embedded_seq, \
                                       merge_outputs=True)
```
The `outputs` are the element-wise sum of both the input and the output of the LSTM cell.

### Multi-layer cells

```eval_rst
.. autosummary::
    :nosignatures:

    SequentialRNNCell
    SequentialRNNCell.add
```

The `SequentialRNNCell` allows stacking multiple layers of RNN cells to improve the expressiveness
and performance of the model. Cells can be added to a `SequentialRNNCell` in order, from bottom to
top. When unrolling, the output of a lower-level cell is automatically passed to the cell above.

```python
stacked_rnn_cells = mx.rnn.SequentialRNNCell()
stacked_rnn_cells.add(mx.rnn.BidirectionalCell(
                          mx.rnn.LSTMCell(num_hidden=50),
                          mx.rnn.LSTMCell(num_hidden=50)))

# Dropout the output of the bottom layer BidirectionalCell with a retention probability of 0.5.
stacked_rnn_cells.add(mx.rnn.DropoutCell(0.5))

stacked_rnn_cells.add(mx.rnn.LSTMCell(num_hidden=50))
outputs, states = stacked_rnn_cells.unroll(length=sequence_length, \
                                           inputs=embedded_seq, \
                                           merge_outputs=True)

# The output of SequentialRNNCell is the same as that of the last layer.
# In this case 'outputs' is the symbol 'concat6_output' of shape (batch_size, sequence_length, hidden_dim)
# The states of the SequentialRNNCell is a list of lists, with each list
# corresponding to the states of each of the added cells respectively.
```

### Fused RNN cell

```eval_rst
.. autosummary::
    :nosignatures:

    FusedRNNCell
    FusedRNNCell.unfuse
```

The computation of an RNN for an input sequence consists of many GEMM and point-wise operations with
temporal dependencies dependencies. This could make the computation memory-bound especially on GPU,
resulting in longer wall-time. By combining the computation of many small matrices into that of
larger ones and streaming the computation whenever possible, the ratio of computation to memory I/O
can be increased, which results in better performance on GPU. Such optimization technique is called
"fusing".
[This post](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/)
talks in greater detail.

The `rnn` module includes a `FusedRNNCell`, which provides the optimized fused implementation.
The FusedRNNCell supports bidirectional RNNs and dropout.

```python
fused_lstm_cell = mx.rnn.FusedRNNCell(num_hidden=50, \
                                      num_layers=3, \
                                      mode='lstm', \
                                      bidirectional=True, \
                                      dropout=0.5)
outputs, _ = fused_lstm_cell.unroll(length=sequence_length, \
                                    inputs=embedded_seq, \
                                    merge_outputs=True)
# The 'outputs' is the symbol 'lstm_rnn_output' that has the shape
# (batch_size, sequence_length, forward_backward_concat_dim)
```
```eval_rst
.. note:: `FusedRNNCell` supports GPU-only. It cannot be called or stepped.
.. note:: When `dropout` is set to non-zero in `FusedRNNCell`, the dropout is applied to the
    output of all layers except the last layer. If there is only one layer in the `FusedRNNCell`, the
    dropout rate is ignored.
.. note:: Similar to `BidirectionalCell`, when `bidirectional` flag is set to `True`, the output
    of `FusedRNNCell` is twice the size specified by `num_hidden`.
```

When training a deep, complex model *on multiple GPUs* it's recommended to stack
fused RNN cells (one layer per cell) together instead of one with all layers.
The reason is that fused RNN cells don't set gradients to be ready until the
computation for the entire layer is completed. Breaking a multi-layer fused RNN
cell into several one-layer ones allows gradients to be processed ealier. This
reduces communication overhead, especially with multiple GPUs.

The `unfuse()` method can be used to convert the `FusedRNNCell` into an equivalent
and CPU-compatible `SequentialRNNCell` that mirrors the settings of the `FusedRNNCell`.
```python
unfused_lstm_cell = fused_lstm_cell.unfuse()
unfused_outputs, _ = unfused_lstm_cell.unroll(length=sequence_length, \
                                              inputs=embedded_seq, \
                                              merge_outputs=True)
# The 'outputs' is the symbol 'lstm_bi_l2_out_output' that has the shape
# (batch_size, sequence_length, forward_backward_concat_dim)
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
be converted first. `FusedRNNCell`'s parameters are merged and flattened. In the fused example above,
the mode has `lstm_parameters` of shape `(total_num_params,)`, whereas the
equivalent SequentialRNNCell's parameters are separate:
```python
'lstm_l0_i2h_weight': (out_dim, embed_dim)
'lstm_l0_i2h_bias': (out_dim,)
'lstm_l0_h2h_weight': (out_dim, hidden_dim)
'lstm_l0_h2h_bias': (out_dim,)
'lstm_r0_i2h_weight': (out_dim, embed_dim)
...
```

All cells in the `rnn` module support the method `unpack_weights()` for converting `FusedRNNCell`
parameters to the unfused format and `pack_weights()` for fusing the parameters. The RNN-specific
checkpointing methods (`load_rnn_checkpoint, save_rnn_checkpoint, do_rnn_checkpoint`) handle the
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
.. autoclass:: mxnet.rnn.ResidualCell
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
