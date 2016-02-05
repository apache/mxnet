package ml.dmlc.mxnet

object CheckUtils {
  def reldiff(a: NDArray, b: NDArray): Float = {
    val diff = NDArray.sum(NDArray.abs(a - b)).toScalar
    val norm = NDArray.sum(NDArray.abs(a)).toScalar
    diff / norm
  }
}
