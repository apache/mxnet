# MXNet - Scala API

See the [MXNet Scala API Documentation](http://mxnet.incubator.apache.org/api/scala/docs/index.html).

MXNet supports the Scala programming language. The MXNet Scala package brings flexible and efficient GPU
computing and state-of-art deep learning to Scala. It enables you to write seamless tensor/matrix computation with multiple GPUs in Scala. It also lets you construct and customize the state-of-art deep learning models in Scala,
  and apply them to tasks, such as image classification and data science challenges.

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

 ## Scala API Reference
 * [Module API is a flexible high-level interface for training neural networks.](module.md)
 * [Model API is an alternate simple high-level interface for training neural networks.](model.md)
 * [Symbolic API performs operations on NDArrays to assemble neural networks from layers.](symbol.md)
 * [IO Data Loading API performs parsing and data loading.](io.md)
 * [NDArray API performs vector/matrix/tensor operations.](ndarray.md)
 * [KVStore API performs multi-GPU and multi-host distributed training.](kvstore.md)


## Resources

* [MXNet Scala API Documentation](http://mxnet.incubator.apache.org/api/scala/docs/index.html)
* [Handwritten Digit Classification in Scala](http://mxnet.incubator.apache.org/tutorials/scala/mnist.html)
* [Neural Style in Scala on MXNet](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/neuralstyle/NeuralStyle.scala)
* [More Scala Examples](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples)
