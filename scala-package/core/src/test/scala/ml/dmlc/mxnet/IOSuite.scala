package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import scala.sys.process._


class IOSuite extends FunSuite with BeforeAndAfterAll {
  test("test MNISTIter") {
    //get data
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
    //test_loop
    mnistIter.reset()
    val nBatch = 600
    var batchCount = 0
    while(mnistIter.iterNext()) {
      val batch = mnistIter.next()
      batchCount+=1
    }
    //test loop
    assert(nBatch === batchCount)
    //test reset
    mnistIter.reset()
    mnistIter.iterNext()
    val label0 = mnistIter.getLabel().toArray
    mnistIter.iterNext()
    mnistIter.iterNext()
    mnistIter.iterNext()
    mnistIter.reset()
    mnistIter.iterNext()
    val label1 = mnistIter.getLabel().toArray
    assert(label0 === label1)
  }


  /**
    * not work now
    */
//  test("test ImageRecordIter") {
//    //get data
//    "./scripts/get_cifar_data.sh" !
//
//    val params = Map(
//      "path_imgrec" -> "data/cifar/train.rec",
//      "mean_img" -> "data/cifar/cifar10_mean.bin",
//      "rand_crop" -> "False",
//      "and_mirror" -> "False",
//      "shuffle" -> "False",
//      "data_shape" -> "(3,28,28)",
//      "batch_size" -> "100",
//      "preprocess_threads" -> "4",
//      "prefetch_buffer" -> "1"
//    )
//    val img_iter = IO.createIterator("ImageRecordIter", params)
//    img_iter.reset()
//    while(img_iter.iterNext()) {
//      val batch = img_iter.next()
//    }
//  }

//  test("test NDarryIter") {
//
//  }
}
