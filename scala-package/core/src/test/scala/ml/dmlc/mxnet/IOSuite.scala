/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.mxnet

import ml.dmlc.mxnet.io.{NDArrayIter, ResizeIter, PrefetchingIter}
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import scala.sys.process._


class IOSuite extends FunSuite with BeforeAndAfterAll {

  private var tu = new TestUtil

  test("test MNISTIter & MNISTPack") {
    // get data
    "./scripts/get_mnist_data.sh" !

    val params = Map(
      "image" -> tu.dataFile("train-images-idx3-ubyte"),
      "label" -> tu.dataFile("train-labels-idx1-ubyte"),
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
      "path_imgrec" -> tu.dataFile("cifar/train.rec"),
      "mean_img" -> tu.dataFile("cifar/cifar10_mean.bin"),
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

  test("test ResizeIter") {
    // get data
    "./scripts/get_mnist_data.sh" !

    val params = Map(
      "image" -> tu.dataFile("train-images-idx3-ubyte"),
      "label" -> tu.dataFile("train-labels-idx1-ubyte"),
      "data_shape" -> "(784,)",
      "batch_size" -> "100",
      "shuffle" -> "1",
      "flat" -> "1",
      "silent" -> "0",
      "seed" -> "10"
    )

    val mnistIter = IO.MNISTIter(params)
    val nBatch = 400
    var batchCount = 0
    val resizeIter = new ResizeIter(mnistIter, nBatch, false)

    while(resizeIter.hasNext) {
      resizeIter.next()
      batchCount += 1
    }

    assert(batchCount === nBatch)

    batchCount = 0
    resizeIter.reset()
    while(resizeIter.hasNext) {
      resizeIter.next()
      batchCount += 1
    }

    assert(batchCount === nBatch)
  }

  test("test PrefetchIter") {
    // get data
    "./scripts/get_mnist_data.sh" !

    val params = Map(
      "image" -> tu.dataFile("train-images-idx3-ubyte"),
      "label" -> tu.dataFile("train-labels-idx1-ubyte"),
      "data_shape" -> "(784,)",
      "batch_size" -> "100",
      "shuffle" -> "1",
      "flat" -> "1",
      "silent" -> "0",
      "seed" -> "10"
    )

    val mnistPack1 = IO.MNISTPack(params)
    val mnistPack2 = IO.MNISTPack(params)

    val nBatch = 600
    var batchCount = 0

    val mnistIter1 = mnistPack1.iterator
    val mnistIter2 = mnistPack2.iterator

    var prefetchIter = new PrefetchingIter(
        IndexedSeq(mnistIter1, mnistIter2),
        IndexedSeq(Map("data" -> "data1"), Map("data" -> "data2")),
        IndexedSeq(Map("label" -> "label1"), Map("label" -> "label2"))
    )

    // test loop
    while(prefetchIter.hasNext) {
      prefetchIter.next()
      batchCount += 1
    }
    assert(nBatch === batchCount)

    // test provideData
    val provideData = prefetchIter.provideData
    val provideLabel = prefetchIter.provideLabel
    assert(provideData("data1") === Shape(100, 784))
    assert(provideData("data2") === Shape(100, 784))
    assert(provideLabel("label1") === Shape(100))
    assert(provideLabel("label2") === Shape(100))

    // test reset
    prefetchIter.reset()
    prefetchIter.next()
    val label0 = prefetchIter.getLabel().head.toArray
    val data0 = prefetchIter.getData().head.toArray
    prefetchIter.next()
    prefetchIter.next()
    prefetchIter.next()
    prefetchIter.reset()
    prefetchIter.next()
    val label1 = prefetchIter.getLabel().head.toArray
    val data1 = prefetchIter.getData().head.toArray
    assert(label0 === label1)
    assert(data0 === data1)

    prefetchIter.dispose()
  }

  test("test NDArrayIter") {
    val shape0 = Shape(Array(1000, 2, 2))
    val data = IndexedSeq(NDArray.ones(shape0), NDArray.zeros(shape0))
    val shape1 = Shape(Array(1000, 1))
    val label = IndexedSeq(NDArray.ones(shape1))
    val batchData0 = NDArray.ones(Shape(Array(128, 2, 2)))
    val batchData1 = NDArray.zeros(Shape(Array(128, 2, 2)))
    val batchLabel = NDArray.ones(Shape(Array(128, 1)))

    // test pad
    val dataIter0 = new NDArrayIter(data, label, 128, false, "pad")
    var batchCount = 0
    val nBatch0 = 8
    while(dataIter0.hasNext) {
      val tBatch = dataIter0.next()
      batchCount += 1

      assert(tBatch.data(0).toArray === batchData0.toArray)
      assert(tBatch.data(1).toArray === batchData1.toArray)
      assert(tBatch.label(0).toArray === batchLabel.toArray)
    }

    assert(batchCount === nBatch0)

    // test discard
    val dataIter1 = new NDArrayIter(data, label, 128, false, "discard")
    val nBatch1 = 7
    batchCount = 0
    while(dataIter1.hasNext) {
      val tBatch = dataIter1.next()
      batchCount += 1

      assert(tBatch.data(0).toArray === batchData0.toArray)
      assert(tBatch.data(1).toArray === batchData1.toArray)
      assert(tBatch.label(0).toArray === batchLabel.toArray)
    }

    assert(batchCount === nBatch1)

    // test empty label (for prediction)
    val dataIter2 = new NDArrayIter(data = data, dataBatchSize = 128, lastBatchHandle = "discard")
    batchCount = 0
    while(dataIter2.hasNext) {
      val tBatch = dataIter2.next()
      batchCount += 1

      assert(tBatch.data(0).toArray === batchData0.toArray)
      assert(tBatch.data(1).toArray === batchData1.toArray)
    }

    assert(batchCount === nBatch1)
    assert(dataIter2.initLabel == IndexedSeq.empty)
  }
}
