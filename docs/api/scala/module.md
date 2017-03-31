# Module API
The module API provides an intermediate and high-level interface for performing computation with neural networks in MXNet. A *module* is an instance of subclasses of the `BaseModule`. The most widely used module class is called `Module`. Module wraps a `Symbol` and one or more `Executors`. For a full list of functions, see `BaseModule`.
A subclass of modules might have extra interface functions. This topic provides some examples of common use cases. All of the module APIs are in the `Module` namespace.

## Preparing a Module for Computation

To construct a module, refer to the constructors for the module class. For example, the `Module` class accepts a `Symbol` as input:

```scala
    import ml.dmlc.mxnet._
    import ml.dmlc.mxnet.module.{FitParams, Module}

    // construct a simple MLP
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")(data)(Map("num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")(fc1)(Map("act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")(act1)(Map("num_hidden" -> 64))
    val act2 = Symbol.Activation(name = "relu2")(fc2)(Map("act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")(act2)(Map("num_hidden" -> 10))
    val out = Symbol.SoftmaxOutput(name = "softmax")(fc3)()

    // construct the module
    val mod = new Module(out)
```

By default, `context` is the CPU. If you need data parallelization, you can specify a GPU context or an array of GPU contexts.

Before you can compute with a module, you need to call `bind()` to allocate the device memory and `initParams()` or `SetParams()` to initialize the parameters.
If you simply want to fit a module, you don't need to call `bind()` and `initParams()` explicitly, because the fit() function automatically calls them if they are needed.

```scala
    mod.bind(dataShapes = train_dataiter.provideData, labelShapes = Some(train_dataiter.provideLabel))
    mod.initParams()
```

Now you can compute with the module using functions like `forward()`, `backward()`, etc.

## Training, Predicting, and Evaluating

Modules provide high-level APIs for training, predicting, and evaluating. To fit a module, call the `fit()` function with some `DataIter`s:

```scala
    import ml.dmlc.mxnet.optimizer.SGD
    val mod = new Module(softmax)

    mod.fit(train_dataiter, evalData = scala.Option(eval_dataiter), \
    numEpoch = n_epoch, fitParams = new FitParams()\
    .setOptimizer(new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f)))
```

The interface is very similar to the old `FeedForward` class. You can pass in batch-end callbacks using `setBatchEndCallback` and epoch-end callbacks using `setEpochEndCallback`. You can also set parameters using methods like `setOptimizer` and `setEvalMetric`. To learn more about the `FitParams()`, see the [API page](http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.module.FitParams). To predict with a module, call `predict()` with a `DataIter`:

```scala
    mod.predict(val_dataiter)
```

The module collects and returns all of the prediction results. For more details about the format of the return values, see the documentation for the [`predict()` function](http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.module.BaseModule).

When prediction results might be too large to fit in memory, use the `predictEveryBatch` API:

```scala
    val preds = mod.predictEveryBatch(val_dataiter)
    val_dataiter.reset()
    var i = 0
    while (val_dataiter.hasNext) {
       val batch = val_dataiter.next()
       val predLabel: Array[Int] = NDArray.argmax_channel(preds(i)(0)).toArray.map(_.toInt)
       val label = batch.label(0).toArray.map(_.toInt)
       //do something...
       i += 1
    }
```

If you need to evaluate on a test set and don't need the prediction output, call the `score()` function with a `DataIter` and an `EvalMetric`:

```scala
    mod.score(val_dataiter, metric)
```

This runs predictions on each batch in the provided `DataIter` and computes the evaluation score using the provided `EvalMetric`. The evaluation results are stored in `metric` so that you can query later.

## Saving and Loading Module Parameters

To save the module parameters in each training epoch, use a `checkpoint` callback:

```scala
    val modelPrefix: String = "mymodel"

    for (epoch <- 0 until 5) {
      while(train_dataiter.hasNext){  
          // forward backward pass
         //do something...
       }
        val checkpoint = mod.saveCheckpoint(modelPrefix, epoch, saveOptStates = true)

    }
```

To load the saved module parameters, call the `loadCheckpoint` function:

```scala
    val mod = Module.loadCheckpoint(modelPrefix, loadModelEpoch, loadOptimizerStates = true)
```

To initialize parameters, Bind the symbols to construct executors first with `bind` method. Then, initialize the parameters and auxiliary states by calling `initParams()` method.

```scala
    mod.bind(dataShapes = train_dataiter.provideData, labelShapes = Some(train_dataiter.provideLabel))
    mod.initParams()
```

To get current parameters, use `getParams` method.

```scala
    val (argParams, auxParams) = mod.getParams
```

To assign parameter and aux state values, use `setParams` method.

```scala
    mod.setParams(argParams, auxParams)
```

To resume training from a saved checkpoint, instead of calling `setParams()`, directly call `fit()`, passing the loaded parameters, so that `fit()` knows to start from those parameters instead of initializing randomly:

```scala
    mod.fit(..., fitParams=new FitParams().setArgParams(argParams).\
    setAuxParams(auxParams).setBeginEpoch(beginEpoch))
```

Create an object of the `FitParams()` class, and then use it to call the `setBeginEpoch()` method to pass `beginEpoch` so that `fit()` knows to resume from a saved epoch.

## Next Steps
* See [Model API](model.md) for an alternative simple high-level interface for training neural networks.
* See [Symbolic API](symbol.md) for operations on NDArrays that assemble neural networks from layers.
* See [IO Data Loading API](io.md) for parsing and loading data.
* See [NDArray API](ndarray.md) for vector/matrix/tensor operations.
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.
