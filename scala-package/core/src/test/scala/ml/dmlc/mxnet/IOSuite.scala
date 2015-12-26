package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}


class IOSuite extends FunSuite with BeforeAndAfterAll {
  test("test MNISTIter") {
    val params = Map(
      "image" -> "/home/hzx/workspace/git/mxnet-scala/mxnet/tests/python/common/data/train-images-idx3-ubyte",
      "label" -> "/home/hzx/workspace/git/mxnet-scala/mxnet/tests/python/common/data/train-labels-idx1-ubyte",
      "data_shape" -> "(784,)",
      "batch_size" -> "100",
      "shuffle" -> "1",
      "flat" -> "1",
      "silent" -> "0",
      "seed" -> "10"
    )
//    println("create MNISTIter")
    val mnist_iter = IO.iterCreateFuncs("MNISTIter")(params)
    mnist_iter.reset()
    mnist_iter.iterNext()
    while(mnist_iter.iterNext()) {
      val data = mnist_iter.getData()
      val label = mnist_iter.getLabel()
      val index = mnist_iter.getIndex()
      val pad = mnist_iter.getPad()
//      println("data: " + data.toArray.mkString(","))
//      println("label: " + label.toArray.mkString(","))
//      println("index: " + index)
//      println("pad: " + pad)
    }
  }

  test("test ImageRecordIter") {

  }
}
