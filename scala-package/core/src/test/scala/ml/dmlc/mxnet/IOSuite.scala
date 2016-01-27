package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import scala.sys.process._


class IOSuite extends FunSuite with BeforeAndAfterAll {
  test("test MNISTIter") {
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

    val mnistIter = IO.createIterator("MNISTIter", params)
    // test_loop
    mnistIter.reset()
    val nBatch = 600
    var batchCount = 0
    while(mnistIter.iterNext()) {
      val batch = mnistIter.next()
      batchCount += 1
    }
    // test loop
    assert(nBatch === batchCount)
    // test reset
    mnistIter.reset()
    mnistIter.iterNext()
    val label0 = mnistIter.getLabel().toArray
    val data0 = mnistIter.getData().toArray
    mnistIter.iterNext()
    mnistIter.iterNext()
    mnistIter.iterNext()
    mnistIter.reset()
    mnistIter.iterNext()
    val label1 = mnistIter.getLabel().toArray
    val data1 = mnistIter.getData().toArray
    assert(label0 === label1)
    assert(data0 === data1)
  }


  /**
    * default skip this for saving time
    */
  test("test ImageRecordIter") {
    //get data
    //"./scripts/get_cifar_data.sh" !

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
    val imgRecIter = IO.createIterator("ImageRecordIter", params)
    val nBatch = 500
    var batchCount = 0
    imgRecIter.reset()
    while(imgRecIter.iterNext()) {
      val batch = imgRecIter.next()
      batchCount += 1
    }
    // test loop
    assert(batchCount === nBatch)
    // test reset
    imgRecIter.reset()
    imgRecIter.iterNext()
    val label0 = imgRecIter.getLabel().toArray
    val data0 = imgRecIter.getData().toArray
    imgRecIter.iterNext()
    imgRecIter.iterNext()
    imgRecIter.iterNext()
    imgRecIter.reset()
    imgRecIter.iterNext()
    val label1 = imgRecIter.getLabel().toArray
    val data1 = imgRecIter.getData().toArray
    assert(label0 === label1)
    assert(data0 === data1)
  }

//  test("test NDarryIter") {
//
//  }
}
