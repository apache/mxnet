# Module Interface 
The module API provides an intermediate- and high-level interface for performing computation with neural networks in MXNet. A *module* is an instance of subclasses of `BaseModule`. The most widely used module class is simply called `Module`, which wraps a `Symbol` and one or more `Executors`. For a full list of functions, see  `BaseModule`. 
Each subclass of modules might have some extra interface functions. In this topic, we provide some examples of common use cases. All of the module APIs are in the `mxnet.module` namespace, simply called `mxnet.mod`.

## Preparing a Module for Computation

To construct a module, refer to the constructors for the specific module class. For example, the `Module` class takes a `Symbol` as input:

```python
    import mxnet as mx

    # construct a simple MLP
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
    out  = mx.symbol.SoftmaxOutput(fc3, name = 'softmax')
 
    # construct the module
    mod = mx.mod.Module(out)
```

Also specify the `data_names` and `label_names` of your `Symbol`. We'll skip those parameters because our `Symbol` follows naming conventions, so the default behavior (data named as `data`, and label named as `softmax_label`) is okay. `context`, which by default is the CPU, is another important parameter. You can specify a GPU context or even a list of GPU contexts if you need data parallelization.

Before you can compute with a module, you need to call `bind()` to allocate the device memory and `init_params()` or `set_params()` to initialize the parameters.

```python
    mod.bind(data_shapes=train_dataiter.provide_data,
         label_shapes=train_dataiter.provide_label)
    mod.init_params()
```

Now you can compute with the module using functions like `forward()`, `backward()`, etc. If you simply want to fit a module, you don't need to call `bind()` and `init_params()` explicitly, because the `fit()` function automatically calls them if they are needed.

## Training, Predicting, and Evaluating

Modules provide high-level APIs for training, predicting, and evaluating. To fit a module, call the `fit()` function with some `DataIter`s:

```python
    mod = mx.mod.Module(softmax)
    mod.fit(train_dataiter, eval_data=eval_dataiter,
            optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
            num_epoch=n_epoch)
```

The interface is very similar to the old `FeedForward` class. You can pass in batch-end callbacks and epoch-end callbacks. To predict with a module, call `predict()` with a `DataIter`:

```python
    mod.predict(val_dataiter)
```

The module collects and returns all of the prediction results. For more details about the format of the return values, see the documentation for the `predict()` function. 

When prediction results might be too large to fit in memory, use the `iter_predict` API:

```python
    for preds, i_batch, batch in mod.iter_predict    (val_dataiter):
        pred_label = preds[0].asnumpy().argmax(axis=1)
        label = batch.label[0].asnumpy().astype('int32')
        # do something...
```

If you need to evaluate on a test set and don't need the prediction output, call the `score()` function with a `DataIter` and an `EvalMetric`:

```python
    mod.score(val_dataiter, metric)
```

This runs predictions on each batch in the provided `DataIter` and computes the evaluation score using the provided `EvalMetric`. The evaluation results are stored in `metric` so that you can query later.

## Saving and Loading Module Parameters

To save the module parameters in each training epoch, use a `checkpoint` callback:

```python
    model_prefix = 'mymodel'
    checkpoint = mx.callback.do_checkpoint(model_prefix)

    mod.fit(..., epoch_end_callback=checkpoint)
```

To load the saved module parameters, call the `load_checkpoint` function:

```python
    sym, arg_params, aux_params = \
        mx.model.load_checkpoint(model_prefix, n_epoch_load)

    # assign parameters
    mod.set_params(arg_params, aux_params)
```

To resume training from a saved checkpoint, instead of calling `set_params()`, directly call `fit()`, passing the loaded parameters, so that `fit()` knows to start from those parameters instead of initializing randomly:

```python
    mod.fit(..., arg_params=arg_params, aux_params=aux_params,
        begin_epoch=n_epoch_load)
```

Pass in `begin_epoch` so that `fit()` knows to resume from a saved epoch.


# Module Interface API


```eval_rst
    .. raw:: html

        <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>
```

## BaseModule Interface API

```eval_rst
    .. automodule:: mxnet.module.base_module
        :members:

    .. raw:: html

        <script>auto_index("basemodule-interface-api");</script>
```

## Built-in Modules API


```eval_rst
    .. automodule:: mxnet.module.module
    :members:

    .. raw:: html

        <script>auto_index("built-in-modules-api");</script>
```

```eval_rst
    .. automodule:: mxnet.module.bucketing_module
    :members:

    .. raw:: html

        <script>auto_index("mxnet.module.bucketing_module");</script>
```

```eval_rst
    .. automodule:: mxnet.module.sequential_module
        :members:

    .. raw:: html

        <script>auto_index("mxnet.module.sequential_module");</script>
```

## Writing Modules in Python


```eval_rst
    .. automodule:: mxnet.module.python_module
        :members:

    .. raw:: html

        <script>auto_index("writing-modules-in-python");</script>
```

## Next Steps
* See [Model API](model.md) for an alternative simple high-level interface for training neural networks.
* See [Symbolic API](symbol.md) for operations on NDArrays that assemble neural networks from layers.
* See [IO Data Loading API](io.md) for parsing and loading data.
* See [NDArray API](ndarray.md) for vector/matrix/tensor operations.
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.
