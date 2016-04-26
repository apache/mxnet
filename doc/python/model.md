MXNet Python Model API
======================
The model API in mxnet is not really an API.
It is a thin wrapper build on top of [ndarray](ndarray.md) and [symbolic](symbol.md)
modules to make neural network training easy.

* [Train a Model](#train-a-model) introduces basic training.
* [Save the Model](#save-the-model)
* [Periodically Checkpoint](#periodically-checkpoint)
* [Initializer API Reference](#initializer-api-reference)
* [Evaluation Metric API Reference](#initializer-api-reference)
* [Optimizer API Reference](#optimizer-api-reference)

Train a Model
-------------
To train a model, you can follow two steps, first a configuration using symbol,
then call ```model.FeedForward.create``` to create a model for you.
The following example creates a two layer neural network.

```python
# configure a two layer neuralnetwork
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type='relu')
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=64)
softmax = mx.symbol.SoftmaxOutput(fc2, name='sm')
# create a model
model = mx.model.FeedForward.create(
     softmax,
     X=data_set,
     num_epoch=num_epoch,
     learning_rate=0.01)
```
You can also use scikit-learn style construct and fit function to create a model.
```python
# create a model using sklearn-style two step way
model = mx.model.FeedForward(
     softmax,
     num_epoch=num_epoch,
     learning_rate=0.01)

mode.fit(X=data_set)
```
For more information, you can refer to [Model API Reference](#model-api-reference).

Save the Model
--------------
It is important to save your work after the job done.
To save the model, you can directly pickle it in a pythonic way.
We also provide a save and load function.

```python
# save a model to mymodel-symbol.json and mymodel-0100.params
prefix = 'mymodel'
iteration = 100
model.save(prefix, iteration)

# load model back
model_loaded = mx.model.FeedForward.load(prefix, iteration)
```
The advantage of this save and load function is they are language agnostic,
and you should be able to save and load directly into cloud storage such as S3 and HDFS.

Periodically Checkpoint
-----------------------
It is also helpful to periodically checkpoint your model after each iteration.
To do so, you can simply add a checkpoint callback ```do_checkpoint(path)``` to the function.
The training process will automatically checkpoint to the specified place after
each iteration.

```python
prefix='models/chkpt'
model = mx.model.FeedForward.create(
     softmax,
     X=data_set,
     iter_end_callback=mx.callback.do_checkpoint(prefix),
     ...)
```
You can load the model checkpoint later using ```FeedForward.load```.

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
The training will be done in a data parallel way on the GPUs you specified.

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
