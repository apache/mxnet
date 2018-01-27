# MXNet Scala Model API

The model API provides a simplified way to train neural networks using common best practices.
It's a thin wrapper built on top of the [ndarray](ndarray.md) and [symbolic](symbol.md)
modules that make neural network training easy.

Topics:

* [Train a Model](#train-a-model)
* [Save the Model](#save-the-model)
* [Periodic Checkpoint](#periodic-checkpointing)
* [Multiple Devices](#use-multiple-devices)
* [Model API Reference](#http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.Model)

## Train the Model

To train a model, perform two steps: configure the model using the symbol parameter,
then call ```model.Feedforward.create``` to create the model.
The following example creates a two-layer neural network.

```scala
    // configure a two layer neuralnetwork
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> act1, "num_hidden" -> 64))
    val softmax = Symbol.SoftmaxOutput(name = "sm")()(Map("data" -> fc2))

    // Construct the FeedForward model and fit on the input training data
    val model = FeedForward.newBuilder(softmax)
      .setContext(Context.cpu())
      .setNumEpoch(num_epoch)
      .setOptimizer(new SGD(learningRate = 0.01f, momentum = 0.9f, wd = 0.0001f))
      .setTrainData(trainDataIter)
      .setEvalData(valDataIter)
      .build()
```
You can also use the `scikit-learn-style` construct and `fit` function to create a model.

```scala
    // create a model using sklearn-style two-step way
    val model = new FeedForward(softmax,
                                numEpoch = numEpochs,
                                argParams = argParams,
                                auxParams = auxParams,
                                beginEpoch = beginEpoch,
                                epochSize = epochSize)

  model.fit(trainData = train)
```
For more information, see [API Reference](http://mxnet.io/api/scala/docs/index.html).

## Save the Model

After the job is done, save your work.
We also provide `save` and `load` functions. You can use the `load` function to load a model checkpoint from a file.

```scala
    // checkpoint the model data into file,
    // save a model to modelPrefix-symbol.json and modelPrefix-0100.params
    val modelPrefix: String = "checkpt"
    val num_epoch = 100
    Model.saveCheckpoint(modelPrefix, epoch + 1, symbol, argParams, auxStates)

    // load model back
    val model_loaded = FeedForward.load(modelPrefix, num_epoch)
```
The advantage of these two `save` and `load` functions is that they are language agnostic.
You should be able to save and load directly into cloud storage, such as Amazon S3 and HDFS.

##  Periodic Checkpointing

We recommend checkpointing your model after each iteration.
To do this, use ```EpochEndCallback``` to add a ```Model.saveCheckpoint(<parameters>)``` checkpoint callback to the function after each iteration .

```scala
    // modelPrefix-symbol.json will be saved for symbol.
    // modelPrefix-epoch.params will be saved for parameters.
    // Checkpoint the model into file. Can specify parameters.
    // For more information, check API doc.
    val modelPrefix: String = "checkpt"
    val checkpoint: EpochEndCallback =
    if (modelPrefix == null) null
    else new EpochEndCallback {
      override def invoke(epoch: Int, symbol: Symbol,
                         argParams: Map[String, NDArray],
                         auxStates: Map[String, NDArray]): Unit = {
       Model.saveCheckpoint(modelPrefix, epoch + 1, symbol, argParams, auxParams)
            }
           }

    // Load model checkpoint from file. Returns symbol, argParams, auxParams.
    val (_, argParams, _) = Model.loadCheckpoint(modelPrefix, num_epoch)

```
You can load the model checkpoint later using ```Model.loadCheckpoint(modelPrefix, num_epoch)```.

## Use Multiple Devices

Set ```ctx``` to the list of devices that you want to train on. You can create a list of devices in any way you want.

```scala
    val devices = Array(Context.gpu(0), Context.gpu(1))

    val model = new FeedForward(ctx = devices,
             symbol = network,
             numEpoch = numEpochs,
             optimizer = optimizer,
             epochSize = epochSize,
             ...)
```
Training occurs in parallel on the GPUs that you specify.

## Next Steps
* See [Symbolic API](symbol.md) for operations on NDArrays that assemble neural networks from layers.
* See [IO Data Loading API](io.md) for parsing and loading data.
* See [NDArray API](ndarray.md) for vector/matrix/tensor operations.
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.
