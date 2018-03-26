# MXNet - Scala API

MXNet supports the Scala programming language. The MXNet Scala package brings flexible and efficient GPU
computing and state-of-art deep learning to Scala. It enables you to write seamless tensor/matrix computation with multiple GPUs in Scala. It also lets you construct and customize the state-of-art deep learning models in Scala, and apply them to tasks, such as image classification and data science challenges.

See the [MXNet Scala API Documentation](https://mxnet.incubator.apache.org/api/scala/docs/index.html#ml.dmlc.mxnet.package) for detailed API information.

## Image Classification with the Infer API








## Tensor and Matrix Computations
You can perform tensor or matrix computation in pure Scala:

```scala
   scala> import ml.dmlc.mxnet._
   import ml.dmlc.mxnet._

   scala> val arr = NDArray.ones(2, 3)
   arr: ml.dmlc.mxnet.NDArray = ml.dmlc.mxnet.NDArray@f5e74790

   scala> arr.shape
   res0: ml.dmlc.mxnet.Shape = (2,3)

   scala> (arr * 2).toArray
   res2: Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)

   scala> (arr * 2).shape
   res3: ml.dmlc.mxnet.Shape = (2,3)
```

## Scala API Reference
* [Module API is a flexible high-level interface for training neural networks.](module.md)
* [Symbolic API performs operations on NDArrays to assemble neural networks from layers.](symbol.md)
* [IO Data Loading API performs parsing and data loading.](io.md)
* [NDArray API performs vector/matrix/tensor operations.](ndarray.md)
* [KVStore API performs multi-GPU and multi-host distributed training.](kvstore.md)
* [Model API is an alternate simple high-level interface for training neural networks.](model.md) **DEPRECATED**

## Resources

* [MXNet Scala API Documentation](https://mxnet.incubator.apache.org/api/scala/docs/index.html#ml.dmlc.mxnet.package)
* [Handwritten Digit Classification in Scala](http://mxnet.incubator.apache.org/tutorials/scala/mnist.html)
* [Developing a Character-level Language Model in Scala](http://mxnet.incubator.apache.org/tutorials/scala/char_lstm.html)
* [Neural Style in Scala on MXNet](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/ml/dmlc/mxnetexamples/neuralstyle/NeuralStyle.scala)
* [More Scala Examples](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/ml/dmlc/mxnetexamples)
