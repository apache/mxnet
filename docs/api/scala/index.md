# MXNet - Scala API

MXNet supports the Scala programming language. The MXNet Scala package brings flexible and efficient GPU
computing and state-of-art deep learning to Scala. It enables you to write seamless tensor/matrix computation with multiple GPUs in Scala. It also lets you construct and customize the state-of-art deep learning models in Scala, and apply them to tasks, such as image classification and data science challenges.

See the [MXNet Scala API Documentation](docs/index.html#org.apache.mxnet.package) for detailed API information.

```eval_rst
.. toctree::
   :maxdepth: 1

   infer.md
   io.md
   kvstore.md
   model.md
   module.md
   ndarray.md
   symbol_in_pictures.md
   symbol.md
```

## Image Classification with the Scala Infer API
The Infer API can be used for single and batch image classification. More information can be found at the following locations:

* [Infer API Overview](infer.html)
* [Infer API Scaladocs](docs/index.html#org.apache.mxnet.infer.package)
* [Single Shot Detector Inference Example](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/objectdetector)
* [Image Classification Example](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier)


## Tensor and Matrix Computations
You can perform tensor or matrix computation in pure Scala:

```scala
   scala> import org.apache.mxnet._
   import org.apache.mxnet._

   scala> val arr = NDArray.ones(2, 3)
   arr: org.apache.mxnet.NDArray = org.apache.mxnet.NDArray@f5e74790

   scala> arr.shape
   res0: org.apache.mxnet.Shape = (2,3)

   scala> (arr * 2).toArray
   res2: Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)

   scala> (arr * 2).shape
   res3: org.apache.mxnet.Shape = (2,3)
```


## Scala API Tutorials
* [Module API is a flexible high-level interface for training neural networks.](module.html)
* [Symbolic API performs operations on NDArrays to assemble neural networks from layers.](symbol.html)
* [IO Data Loading API performs parsing and data loading.](io.html)
* [NDArray API performs vector/matrix/tensor operations.](ndarray.html)
* [KVStore API performs multi-GPU and multi-host distributed training.](kvstore.html)
* [Model API is an alternate simple high-level interface for training neural networks.](model.html) **DEPRECATED**

## Related Resources
* [MXNet Scala API Documentation](docs/index.html#org.apache.mxnet.package)
* [Handwritten Digit Classification in Scala](../../tutorials/scala/mnist.html)
* [Developing a Character-level Language Model in Scala](../../tutorials/scala/char_lstm.html)
* [Neural Style in Scala on MXNet](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/neuralstyle/NeuralStyle.scala)
* [More Scala Examples](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples)
