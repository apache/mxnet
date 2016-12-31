package ml.dmlc.mxnet

object CheckUtils {
  def reldiff(a: NDArray, b: NDArray): Float = {
    val diff = NDArray.sum(NDArray.abs(a - b)).toScalar
    val norm = NDArray.sum(NDArray.abs(a)).toScalar
    diff / norm
  }

  def reldiff(a: Array[Float], b: Array[Float]): Float = {
    val diff =
      (a zip b).map { case (aElem, bElem) => Math.abs(aElem - bElem) }.sum
    val norm: Float = a.reduce(Math.abs(_) + Math.abs(_))
    diff / norm
  }
}
