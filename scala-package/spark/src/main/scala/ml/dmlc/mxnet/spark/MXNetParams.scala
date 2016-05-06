package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.{Context, Shape, Symbol}

/**
 * MXNet on Spark training arguments
 * @author Yizhi Liu
 */
private[mxnet] class MXNetParams extends Serializable {
  // training batch size
  var batchSize: Int = 128
  // dimension of input data
  var dimension: Shape = null

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
  var context: Context = Context.cpu()

  var numWorker: Int = 1
  var numServer: Int = 1

  var labelName: String = "label"

  // java classpath on executors for running mxnet application
  // TODO: upload to a shared storage from driver
  var classpath: String = null
  // java binary
  var javabin: String = "java"
}
