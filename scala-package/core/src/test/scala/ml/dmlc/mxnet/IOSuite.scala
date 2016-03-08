package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import scala.sys.process._


class IOSuite extends FunSuite with BeforeAndAfterAll {
  test("test MNISTIter & MNISTPack") {
    // get data
    "./scripts/get_mnist_data.sh" !

    val params = Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "data_shape" -> "(784,)",
      "batch_size" -> "100",
      "shuffle" -> "1",
      "flat" -> "1",
      "silent" -> "0",
      "seed" -> "10"
    )

    val mnistPack = IO.MNISTPack(params)
    // test DataPack
    val nBatch = 600
    var batchCount = 0
    for(batch <- mnistPack) {
      batchCount += 1
    }
    assert(nBatch === batchCount)

    // test DataIter
    val mnistIter = mnistPack.iterator
    // test provideData
    val provideData = mnistIter.provideData
    val provideLabel = mnistIter.provideLabel
    assert(provideData("data") === Shape(100, 784))
    assert(provideLabel("label") === Shape(100))
    // test_loop
    mnistIter.reset()
    batchCount = 0
    while (mnistIter.hasNext) {
      mnistIter.next()
      batchCount += 1
    }
    // test loop
    assert(nBatch === batchCount)
    // test reset
    mnistIter.reset()
    mnistIter.next()
    val label0 = mnistIter.getLabel().head.toArray
    val data0 = mnistIter.getData().head.toArray
    mnistIter.next()
    mnistIter.next()
    mnistIter.next()
    mnistIter.reset()
    mnistIter.next()
    val label1 = mnistIter.getLabel().head.toArray
    val data1 = mnistIter.getData().head.toArray
    assert(label0 === label1)
    assert(data0 === data1)
  }


  /**
    * default skip this test for saving time
    */
  test("test ImageRecordIter") {
    // get data
    "./scripts/get_cifar_data.sh" !

    val params = Map(
      "path_imgrec" -> "data/cifar/train.rec",
      "mean_img" -> "data/cifar/cifar10_mean.bin",
      "rand_crop" -> "False",
      "and_mirror" -> "False",
      "shuffle" -> "False",
      "data_shape" -> "(3,28,28)",
      "batch_size" -> "100",
      "preprocess_threads" -> "4",
      "prefetch_buffer" -> "1"
    )
    val imgRecIter = IO.ImageRecordIter(params)
    val nBatch = 500
    var batchCount = 0
    // test provideData
    val provideData = imgRecIter.provideData
    val provideLabel = imgRecIter.provideLabel
    assert(provideData("data").toArray === Array(100, 3, 28, 28))
    assert(provideLabel("label").toArray === Array(100))

    imgRecIter.reset()
    while (imgRecIter.hasNext) {
      imgRecIter.next()
      batchCount += 1
    }
    // test loop
    assert(batchCount === nBatch)
    // test reset
    imgRecIter.reset()
    imgRecIter.next()
    val label0 = imgRecIter.getLabel().head.toArray
    val data0 = imgRecIter.getData().head.toArray
    imgRecIter.reset()
    imgRecIter.reset()
    imgRecIter.reset()
    imgRecIter.reset()
    imgRecIter.reset()
    val label1 = imgRecIter.getLabel().head.toArray
    val data1 = imgRecIter.getData().head.toArray
    assert(label0 === label1)
    assert(data0 === data1)
  }
}
