MXNet Python Model API
======================
The model API in mxnet as not really an API.
It is a thin wrapper build on top of [ndarray](ndarray.md) and [symbolic](symbol.md)
modules to make neural network training easy.

* [Train a Model](#overloaded-operators) introduces operator overloading of symbols
* [Serialization](#serialization) introduces how to save and load symbols.
* [Multiple Outputs](#multiple-outputs) introduces how to configure multiple outputs
* [API Reference](#api-reference) gives reference to all functions.
* [Symbol Object Document](#mxnet.symbol.Symbol) gives API reference to the Symbol Object.


Train a Model
-------------
To train a model, you can follow two steps, first a configuration using symbol,
then call ```model.Feedforward.create``` to create a model for you.
The following example creates a two layer neural networks.

```python
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
softmax = mx.symbol.Softmax(fc2, name = 'sm')

model = mx.model.FeedForward.create(
     softmax,
     X=data_set,
     num_round=num_round,
     learning_rate=0.01)
```

You can also use scikit-learn style construct and fit function to create a model.
For more information, you can refer to [Model API Reference](#model-api-reference).

Save the Model
--------------
It is important to save your work after the job done.
To save the model, you can directly pickle it if you like the pythonic way.
We also provide a save and load function.

```python
# save a model to mymodel-symbol.json and mymodel-0100.params
prefix = 'mymodel'
model.save(prefix, 100)

# load model back
model_loaded = mx.model.FeedForward.load(prefix, 100)
```
The advantage of this save and load function is they are language agnostic,
and you should be able to save and load directly into cloud storage such as S3 and HDFS.

Periodically Checkpoint
-----------------------
It is also helpful to periodically checkpoint your model after each iteration.
To do so, you can simply add a checkpoint callback to the function.
The training process will automatically checkpoint to the specified place after
each iteration.

```python
prefix='models/chkpt'
model = mx.model.FeedForward.create(
     softmax,
     X=data_set,
     iter_end_callback=mx.model.do_checkpoint(prefix),
     num_round=num_round,
     learning_rate=0.01)
```
You can load the model checkpoint later using ```Feedforward.load```.

Use Multiple Devices
--------------------
Simply set ```ctx``` to be the list of devices you like to train on.

```python
devices = [mx.gpu(i) for i in range(num_device)]
model = mx.model.FeedForward.create(
     softmax,
     X=dataset,
     ctx=devices,
     ...)
```

Initializer API Reference
-------------------------

```eval_rst
.. automodule:: mxnet.initializer
    :members:
```

Evaluation Metric API Reference
-------------------------------

```eval_rst
.. automodule:: mxnet.metric
    :members:
```

Optimizer API Reference
-----------------------

```eval_rst
.. automodule:: mxnet.optimizer
    :members:
```

Model API Reference
-------------------

```eval_rst
.. automodule:: mxnet.model
    :members:
```
