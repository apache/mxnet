package ml.dmlc.mxnet.examples.multitask

import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.IO
import ml.dmlc.mxnet.DataIter

/**
 * @author Depeng Liang
 */
object Data {

  // return train and val iterators for mnist
  def mnistIterator(dataPath: String, batchSize: Int, inputShape: Shape): (DataIter, DataIter) = {
    val flat = if (inputShape.length == 3) "False" else "True"
    val trainParams = Map(
      "image" -> s"$dataPath/train-images-idx3-ubyte",
      "label" -> s"$dataPath/train-labels-idx1-ubyte",
      "input_shape" -> inputShape.toString(),
      "batch_size" -> s"$batchSize",
      "shuffle" -> "True",
      "flat" -> flat
    )
    val trainDataIter = IO.MNISTIter(trainParams)
    val testParams = Map(
      "image" -> s"$dataPath/t10k-images-idx3-ubyte",
      "label" -> s"$dataPath/t10k-labels-idx1-ubyte",
      "input_shape" -> inputShape.toString(),
      "batch_size" -> s"$batchSize",
      "flat" -> flat
    )
    val testDataIter = IO.MNISTIter(testParams)
    (trainDataIter, testDataIter)
  }
}
