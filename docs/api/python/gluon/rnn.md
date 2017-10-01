# Gluon Recurrent Neural Network API

## Overview

This document lists the recurrent neural network API in Gluon:

```eval_rst
.. currentmodule:: mxnet.gluon.rnn
```

### Recurrent Layers

Recurrent layers can be used in `Sequential` with other regular neural network layers. For example,
to construct a sequence labeling model where a prediction is made for each time-step:

```python
>>> model = mx.gluon.nn.Sequential()
>>> model.add(mx.gluon.nn.Embedding(30, 10))
>>> model.add(mx.gluon.rnn.LSTM(20))
>>> model.add(mx.gluon.nn.Dense(5, flatten=False))
>>> model.initialize()
>>> model(mx.nd.ones((2,3,5)))
```

```eval_rst
.. autosummary::
    :nosignatures:

    RNN
    LSTM
    GRU
```

### Recurrent Cells

Recurrent cells exposes the intermediate recurrent states and allows for explicit stepping and
unrolling, and thus provides more flexibility.

```eval_rst
.. autosummary::
    :nosignatures:

    RNNCell
    LSTMCell
    GRUCell
    RecurrentCell
    SequentialRNNCell
    BidirectionalCell
    DropoutCell
    ZoneoutCell
    ResidualCell
```


## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.gluon.rnn
    :members:
    :imported-members:
```

<script>auto_index("api-reference");</script>
