Module Interface HowTo
======================

The module API provide intermediate-level and high-level interface for computation with neural networks in MXNet. A "module" is an instance of subclasses of `BaseModule`. The most widely used module class is simply called `Module`, which wraps a `Symbol` and one or more `Executor`s. Please refer to the API doc for `BaseModule` below for a full list of functions available. Each specific subclass of modules might have some extra interface functions. We provide here some examples of common use cases. All the module APIs live in the namespace of `mxnet.module` or simply `mxnet.mod`.

Preparing a module for computation
----------------------------------

To construct a module, refer to the constructors of the specific module class. For example, the `Module` class takes a `Symbol` as input,

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

You also need to specify the `data_names` and `label_names` of your `Symbol`. Here we skip those parameters because our `Symbol` follows a conventional way of naming, so the default behavior (data named as `data`, and label named as `softmax_label`) is OK. Another important parameter is `context`, which by default is the CPU. You can specify a GPU context or even a list of GPU contexts if data-parallelization is needed.

Before one can compute with a module, we need to call `bind()` to allocate the device memory, and `init_params()` or `set_params()` to initialize the parameters.

```python
mod.bind(data_shapes=train_dataiter.provide_data,
         label_shapes=train_dataiter.provide_label)
mod.init_params()
```

Now you can compute with the module via functions like `forward()`, `backward()`, etc. If you simply want to fit a module, you do not need to call `bind()` and `init_params()` explicitly, as the `fit()` function will call them automatically if needed.

Training, Predicting, and Evaluating
------------------------------------

Modules provide high-level APIs for training, predicting and evaluating. To fit a module, simply call the `fit()` function with some `DataIter`s:

```python
mod = mx.mod.Module(softmax)
mod.fit(train_dataiter, eval_data=val_dataiter,
        optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
        num_epoch=n_epoch)
```

The interface is very similar to the old `FeedForward` class. You can pass in batch-end callbacks as well as epoch-end callbacks. To predict with a module, simply call `predict()` with a `DataIter`:

```python
mod.predict(val_dataiter)
```

It will collect and return all the prediction results. Please refer to the doc of `predict()` for more details about the format of the return values. Another convenient API for prediction in the case where the prediction results might be too large to fit in the memory is `iter_predict`:

```python
for preds, i_batch, batch in mod.iter_predict(val_dataiter):
    pred_label = preds[0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype('int32')
    # do something...
```

If you do not need the prediction outputs, but just need to evaluate on a test set, you can call the `score()` function with a `DataIter` and a `EvalMetric`:

```python
mod.score(val_dataiter, metric)
```

It will run predictions on each batch in the provided `DataIter` and compute the evaluation score using the provided `EvalMetric`. The evaluation results will be stored in `metric` so that you can query later on.

Saving and Loading Module Parameters
------------------------------------

You can save the module parameters in each training epoch by using a `checkpoint` callback.

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

Or if you just want to resume training from a saved checkpoint, instead of calling `set_params()`, you can directly call `fit()`, passing the loaded parameters, so that `fit()` knows to start from those parameters instead of initializing from random.

```python
mod.fit(..., arg_params=arg_params, aux_params=aux_params,
        begin_epoch=n_epoch_load)
```

Note we also pass in `begin_epoch` so that `fit()` knows we are resuming from a previous saved epoch.


Module Interface API
====================

```eval_rst
.. raw:: html

    <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>
```

The BaseModule Interface
------------------------

```eval_rst
.. automodule:: mxnet.module.base_module
    :members:

.. raw:: html

    <script>auto_index("mxnet.module.base_module");</script>
```

The Built-in Modules
--------------------

```eval_rst
.. automodule:: mxnet.module.module
    :members:

.. raw:: html

    <script>auto_index("mxnet.module.module");</script>
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

Writing Modules in Python
-------------------------

```eval_rst
.. automodule:: mxnet.module.python_module
    :members:

.. raw:: html

    <script>auto_index("mxnet.module.python_module");</script>
```
