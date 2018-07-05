# Gluon Package


```eval_rst
.. currentmodule:: mxnet.gluon
```

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

## Overview

The Gluon package is a high-level interface for MXNet designed to be easy to use, while keeping most of the flexibility of a low level API. Gluon supports both imperative and symbolic programming, making it easy to train complex models imperatively in Python and then deploy with a symbolic graph in C++ and Scala.

Based on the the [Gluon API specification](https://github.com/gluon-api/gluon-api), the Gluon API in Apache MXNet provides a clear, concise, and simple API for deep learning. It makes it easy to prototype, build, and train deep learning models without sacrificing training speed.

**Advantages**

1. Simple, Easy-to-Understand Code: Gluon offers a full set of plug-and-play neural network building blocks, including predefined layers, optimizers, and initializers.
2. Flexible, Imperative Structure: Gluon does not require the neural network model to be rigidly defined, but rather brings the training algorithm and model closer together to provide flexibility in the development process.
3. Dynamic Graphs: Gluon enables developers to define neural network models that are dynamic, meaning they can be built on the fly, with any structure, and using any of Python’s native control flow.
4. High Performance: Gluon provides all of the above benefits without impacting the training speed that the underlying engine provides. 

**Examples**

*Simple, Easy-to-Understand Code*

Use plug-and-play neural network building blocks, including predefined layers, optimizers, and initializers:

```
net = gluon.nn.Sequential()
# When instantiated, Sequential stores a chain of neural network layers. 
# Once presented with data, Sequential executes each layer in turn, using 
# the output of one layer as the input for the next
with net.name_scope():
    net.add(gluon.nn.Dense(256, activation="relu")) # 1st layer (256 nodes)
    net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    net.add(gluon.nn.Dense(num_outputs))
```

*Flexible, Imperative Structure*

Prototype, build, and train neural networks in fully imperative manner using the MXNet autograd package and the Gluon trainer method:

```
epochs = 10

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        with autograd.record():
            output = net(data) # the forward iteration
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])
```

*Dynamic Graphs*

Build neural networks on the fly for use cases where neural networks must change in size and shape during model training:

```
def forward(self, F, inputs, tree):
    children_outputs = [self.forward(F, inputs, child)
                        for child in tree.children]
    #Recursively builds the neural network based on each input sentence’s
    #syntactic structure during the model definition and training process
    ...
```

*High Performance*

Easily cache the neural network to achieve high performance by defining your neural network with *HybridSequential* and calling the *hybridize* method:

```
net = nn.HybridSequential()
with net.name_scope():
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(128, activation="relu"))
    net.add(nn.Dense(2))
    
net.hybridize()
```


## Contents

```eval_rst
.. toctree::
   :maxdepth: 1

   nn.md
   rnn.md
   loss.md
   data.md
   model_zoo.md
   contrib.md
```


## Parameter

```eval_rst
.. autosummary::
    :nosignatures:

    Parameter
    Constant
    ParameterDict
```


## Containers

```eval_rst
.. autosummary::
    :nosignatures:

    Block
    HybridBlock
    SymbolBlock
    nn.Sequential
    nn.HybridSequential
```


## Trainer

```eval_rst
.. currentmodule:: mxnet.gluon

.. autosummary::
    :nosignatures:

    Trainer
```

## Utilities

```eval_rst
.. currentmodule:: mxnet.gluon.utils
```


```eval_rst
.. autosummary::
    :nosignatures:

    split_data
    split_and_load
    clip_global_norm
```


## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.gluon
    :members:
    :imported-members:
    :special-members:

.. autoclass:: mxnet.gluon.nn.Sequential
    :members:
.. autoclass:: mxnet.gluon.nn.HybridSequential
    :members:

.. automodule:: mxnet.gluon.utils
    :members:
```

<script>auto_index("api-reference");</script>
