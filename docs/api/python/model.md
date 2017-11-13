# Model API

The model API provides a simplified way to train neural networks using common best practices.
It's a thin wrapper built on top of the [ndarray](../python/ndarray/ndarray.md) and [symbolic](../python/symbol/symbol.md)
modules that make neural network training easy.

Topics:

* [Train a Model](#train-a-model)
* [Save the Model](#save-the-model)
* [Periodic Checkpoint](#periodic-checkpointing)
* [Initializer API Reference](#initializer-api-reference)
* [Evaluation Metric API Reference](#evaluation-metric-api-reference)
* [Optimizer API Reference](#optimizer-api-reference)
* [Model API Reference](#model-api-reference)

## Train the Model

To train a model, perform two steps: configure the model using the symbol parameter,
then call ```model.Feedforward.create``` to create the model.
The following example creates a two-layer neural network.

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
You can also use the `scikit-learn-style` construct and `fit` function to create a model.

```python
    # create a model using sklearn-style two-step way
    model = mx.model.FeedForward(
         softmax,
         num_epoch=num_epoch,
         learning_rate=0.01)

    model.fit(X=data_set)
```
For more information, see [Model API Reference](#model-api-reference).

## Save the Model

After the job is done, save your work.
To save the model, you can directly pickle it with Python.
We also provide `save` and `load` functions.

```python
    # save a model to mymodel-symbol.json and mymodel-0100.params
    prefix = 'mymodel'
    iteration = 100
    model.save(prefix, iteration)

    # load model back
    model_loaded = mx.model.FeedForward.load(prefix, iteration)
```
The advantage of these two `save` and `load` functions are that they are language agnostic.
You should be able to save and load directly into cloud storage, such as Amazon S3 and HDFS.

##  Periodic Checkpointing

We recommend checkpointing your model after each iteration.
To do this, add a checkpoint callback ```do_checkpoint(path)``` to the function.
The training process automatically checkpoints the specified location after
each iteration.

```python
    prefix='models/chkpt'
    model = mx.model.FeedForward.create(
         softmax,
         X=data_set,
         iter_end_callback=mx.callback.do_checkpoint(prefix),
         ...)
```
You can load the model checkpoint later using ```Feedforward.load```.

## Use Multiple Devices

Set ```ctx``` to the list of devices that you want to train on.

```python
    devices = [mx.gpu(i) for i in range(num_device)]
    model = mx.model.FeedForward.create(
         softmax,
         X=dataset,
         ctx=devices,
         ...)
```
Training occurs in parallel on the GPUs that you specify.

```eval_rst
    .. raw:: html

        <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>
```


## Initializer API Reference


```eval_rst
    .. automodule:: mxnet.initializer
        :members:

    .. raw:: html

        <script>auto_index("initializer-api-reference");</script>
```

## Evaluation Metric API Reference


```eval_rst
    .. automodule:: mxnet.metric
        :members:

    .. raw:: html

        <script>auto_index("evaluation-metric-api-reference");</script>
```

## Optimizer API Reference


```eval_rst
    .. automodule:: mxnet.optimizer
        :members:

    .. raw:: html

        <script>auto_index("optimizer-api-reference");</script>
```

## Model API Reference


```eval_rst
    .. automodule:: mxnet.model
        :members:

    .. raw:: html

        <script>auto_index("model-api-reference");</script>
```

## Next Steps
* See [Symbolic API](../python/symbol/symbol.md) for operations on NDArrays that assemble neural networks from layers.
* See [IO Data Loading API](../python/io/io.md) for parsing and loading data.
* See [NDArray API](../python/ndarray/ndarray.md) for vector/matrix/tensor operations.
* See [KVStore API](../python/kvstore/kvstore.md) for multi-GPU and multi-host distributed training.
