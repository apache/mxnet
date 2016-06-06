package ml.dmlc.mxnet.spark

import java.io.File

import ml.dmlc.mxnet.{Context, Shape, Symbol}
import org.apache.spark.SparkFiles

/**
 * MXNet on Spark training arguments
 * @author Yizhi Liu
 */
private[mxnet] class MXNetParams extends Serializable {
  // training batch size
  var batchSize: Int = 128
  // dimension of input data
  var dimension: Shape = null
  // number of training epochs
  var numEpoch: Int = 10

  // network architecture
  private var network: String = null
  def setNetwork(net: Symbol): Unit = {
    network = net.toJson
  }
  def getNetwork: Symbol = {
    if (network == null) {
      null
    } else {
      Symbol.loadJson(network)
    }
  }

  // executor running context
  var context: Array[Context] = Context.cpu()

  var numWorker: Int = 1
  var numServer: Int = 1

  var dataName: String = "data"
  var labelName: String = "label"

  // jars on executors for running mxnet application
  var jars: Array[String] = null
  def runtimeClasspath: String = {
    jars.map(jar => SparkFiles.get(new File(jar).getName)).mkString(":")
  }

  // java binary
  var javabin: String = "java"
}
