package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.NDArray

/**
 * A wrapper for serialize & deserialize [[ml.dmlc.mxnet.NDArray]] in spark job
 * @author Yizhi Liu
 */
class MXNDArray(@transient private var ndArray: NDArray) extends Serializable {
  require(ndArray != null)
  private val arrayBytes: Array[Byte] = ndArray.serialize()

  def get: NDArray = {
    if (ndArray == null) {
      ndArray = NDArray.deserialize(arrayBytes)
    }
    ndArray
  }
}

object MXNDArray {
  def apply(ndArray: NDArray): MXNDArray = new MXNDArray(ndArray)
}
