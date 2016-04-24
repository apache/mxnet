package ml.dmlc.mxnet.spark.utils

import java.io.{BufferedWriter, FileWriter, File}

import ml.dmlc.mxnet.IO

object DataSourceTools {
  val dataDir = "/Users/lewis/Workspace/source-codes/forks/mxnet/data/"
  val outputTrain = "/Users/lewis/Workspace/source-codes/forks/mxnet/data/spark/train.txt"
  val outputVal = "/Users/lewis/Workspace/source-codes/forks/mxnet/data/spark/val.txt"

  def binary2svm(data: String, label: String, output: String): Unit = {
    val params = Map(
      "image" -> (dataDir + data),
      "label" -> (dataDir + label),
      "data_shape" -> "(784,)",
      "batch_size" -> "100",
      "shuffle" -> "1",
      "flat" -> "1",
      "silent" -> "0",
      "seed" -> "10"
    )
    val mnistIter = IO.MNISTIter(params)
    val file = new File(output)
    val bw = new BufferedWriter(new FileWriter(new File(output)))
    while (mnistIter.hasNext) {
      val dataBatch = mnistIter.next()
      require(dataBatch.data.size == 1)
      require(dataBatch.label.size == 1)
      val data = dataBatch.data(0)
      val label = dataBatch.label(0)
      println(data.shape)
      val dataArr = data.toArray
      val labelArr = label.toArray
      for (i <- 0 until labelArr.size) {
        bw.append(labelArr(i) + " " + dataArr.slice(i * 784, (i + 1) * 784).mkString(","))
        bw.newLine()
      }
      dataBatch.dispose()
    }
    bw.close()
  }

  def main(args: Array[String]): Unit = {
    binary2svm("train-images-idx3-ubyte", "train-labels-idx1-ubyte", outputTrain)
    binary2svm("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", outputVal)
  }
}
