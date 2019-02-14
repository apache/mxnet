<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Gluon Recurrent Neural Network API

## Overview

This document lists the recurrent neural network API in Gluon:

```eval_rst
.. currentmodule:: mxnet.gluon.rnn
```

### Recurrent Layers

Recurrent layers can be used in `Sequential` with other regular neural network layers.
For example, to construct a sequence labeling model where a prediction is made for each
time-step:

```python
model = mx.gluon.nn.Sequential()
with model.name_scope():
    model.add(mx.gluon.nn.Embedding(30, 10))
    model.add(mx.gluon.rnn.LSTM(20))
    model.add(mx.gluon.nn.Dense(5, flatten=False))
model.initialize()
model(mx.nd.ones((2,3)))
```

```eval_rst
.. autosummary::
    :nosignatures:

    RNN
    LSTM
    GRU
```

### Recurrent Cells

Recurrent cells allows fine-grained control when defining recurrent models. User
can explicit step and unroll to construct complex networks. It provides more
flexibility but is slower than recurrent layers. Recurrent cells can be stacked
with `SequentialRNNCell`:

```python
model = mx.gluon.rnn.SequentialRNNCell()
with model.name_scope():
    model.add(mx.gluon.rnn.LSTMCell(20))
    model.add(mx.gluon.rnn.LSTMCell(20))
states = model.begin_state(batch_size=32)
inputs = mx.nd.random.uniform(shape=(5, 32, 10))
outputs = []
for i in range(5):
    output, states = model(inputs[i], states)
    outputs.append(output)
```

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

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.gluon.rnn
    :members:
    :imported-members:
```

<script>auto_index("api-reference");</script>
