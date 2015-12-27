package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}


class IOSuite extends FunSuite with BeforeAndAfterAll {
  test("test MNISTIter") {
    val params = Map(
      "image" -> "/home/hzx/workspace/git/mxnet-scala/mxnet/tests/python/unittest/data/train-images-idx3-ubyte",
      "label" -> "/home/hzx/workspace/git/mxnet-scala/mxnet/tests/python/unittest/data/train-labels-idx1-ubyte",
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
    while(mnist_iter.iterNext()) {
      val data = mnist_iter.getData()
      val label = mnist_iter.getLabel()
      val index = mnist_iter.getIndex()
      val pad = mnist_iter.getPad()
    }
  }

  test("test ImageRecordIter") {
    val params = Map(
      "path_imgrec" -> "/home/hzx/workspace/git/mxnet-scala/mxnet/tests/python/unittest/data/cifar/train.rec",
      "mean_img" -> "/home/hzx/workspace/git/mxnet-scala/mxnet/tests/python/unittest/data/cifar/cifar10_mean.bin",
      "rand_crop" -> "false",
      "and_mirror" -> "false",
      "shuffle" -> "false",
      "data_shape" -> "(3,28,28)",
      "batch_size" -> "100",
      "preprocess_threads" -> "4",
      "prefetch_buffer" -> "1"
    )
    val img_iter = IO.iterCreateFuncs("ImageRecordIter")(params)
    img_iter.reset()
    while(img_iter.iterNext()) {
      val data = img_iter.getData()
      val label = img_iter.getLabel()
      val index = img_iter.getIndex()
      val pad = img_iter.getPad()
    }
  }
}
