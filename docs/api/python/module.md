# Module API

```eval_rst
.. currentmodule:: mxnet.module
```

## Overview

The module API, defined in the `module` (or simply `mod`) package, provides an
intermediate and high-level interface for performing computation with a
`Symbol`. One can roughly think a module is a machine which can execute a
program defined by a `Symbol`.

The class `module.Module` is a commonly used module, which accepts a `Symbol` as
the input:

```python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2  = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=10)
out  = mx.symbol.SoftmaxOutput(fc2, name = 'softmax')
mod = mx.mod.Module(out)  # create a module by given a Symbol
```

Assume there is a valid MXNet data iterator `data`. We can initialize the
module:

```python
mod.bind(data_shapes=data.provide_data,
         label_shapes=data.provide_label)  # create memory by given input shapes
mod.init_params()  # initial parameters with the default random initializer
```

Now the module is able to compute. We can call high-level API to train and
predict:

```python
mod.fit(data, num_epoch=10, ...)  # train
mod.predict(new_data)  # predict on new data
```

or use intermediate APIs to perform step-by-step computations

```python
mod.forward(data_batch)  # forward on the provided data batch
mod.backward()  # backward to calculate the gradients
mod.update()  # update parameters using the default optimizer
```

A detailed tutorial is available at [http://mxnet.io/tutorials/python/module.html](http://mxnet.io/tutorials/python/module.html).


```eval_rst

.. note:: ``module`` is used to replace ``model``, which has been deprecated.
```

The `module` package provides several modules:

```eval_rst
.. autosummary::
    :nosignatures:

    BaseModule
    Module
    SequentialModule
    BucketingModule
    PythonModule
    PythonLossModule
```

We summarize the interface for each class in the following sections.

## The `BaseModule` class

The `BaseModule` is the base class for all other module classes. It defines the
interface each module class should provide.

### Initialize memory

```eval_rst
.. autosummary::
    :nosignatures:

    BaseModule.bind
```

### Get and set parameters

```eval_rst
.. autosummary::
    :nosignatures:

    BaseModule.init_params
    BaseModule.set_params
    BaseModule.get_params
    BaseModule.save_params
    BaseModule.load_params
```

### Train and predict

```eval_rst
.. autosummary::
    :nosignatures:

    BaseModule.fit
    BaseModule.score
    BaseModule.iter_predict
    BaseModule.predict
```

### Forward and backward

```eval_rst
.. autosummary::
    :nosignatures:

    BaseModule.forward
    BaseModule.backward
    BaseModule.forward_backward
```

### Update parameters

```eval_rst
.. autosummary::
    :nosignatures:

    BaseModule.init_optimizer
    BaseModule.update
    BaseModule.update_metric
```

### Input and output

```eval_rst
.. autosummary::
    :nosignatures:

    BaseModule.data_names
    BaseModule.output_names
    BaseModule.data_shapes
    BaseModule.label_shapes
    BaseModule.output_shapes
    BaseModule.get_outputs
    BaseModule.get_input_grads
```

### Others

```eval_rst
.. autosummary::
    :nosignatures:

    BaseModule.get_states
    BaseModule.set_states
    BaseModule.install_monitor
    BaseModule.symbol
```


## Other build-in modules

Besides the basic interface defined in `BaseModule`, each module class supports
additional functionality. We summarize them in this section.

### Class `Module`

```eval_rst
.. autosummary::
    :nosignatures:

    Module.load
    Module.save_checkpoint
    Module.reshape
    Module.borrow_optimizer
    Module.save_optimizer_states
    Module.load_optimizer_states
```

### Class `BucketModule`

```eval_rst
.. autosummary::
    :nosignatures:

    BucketModule.switch_bucket
```

### Class `SequentialModule`

```eval_rst
.. autosummary::
    :nosignatures:

    SequentialModule.add
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. autoclass:: mxnet.module.BaseModule
    :members:
.. autoclass:: mxnet.module.Module
    :members:
.. autoclass:: mxnet.module.BucketingModule
    :members:
.. autoclass:: mxnet.module.SequentialModule
    :members:
.. autoclass:: mxnet.module.PythonModule
    :members:
.. autoclass:: mxnet.module.PythonLossModule
    :members:
```

<script>auto_index("api-reference");</script>
