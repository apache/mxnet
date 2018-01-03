# Gluon Package


```eval_rst
.. currentmodule:: mxnet.gluon
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

## Overview

Gluon package is a high-level interface for MXNet designed to be easy to use while
keeping most of the flexibility of low level API. Gluon supports both imperative
and symbolic programming, making it easy to train complex models imperatively
in Python and then deploy with symbolic graph in C++ and Scala.

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

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

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
