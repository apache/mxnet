package ml.dmlc.mxnet

object DType extends Enumeration {
  type DType = Value
  val Float32 = Value(0)
  val Float64 = Value(1)
  val Float16 = Value(2)
  val UInt8 = Value(3)
  val Int32 = Value(4)
  private[mxnet] def numOfBytes(dtype: DType): Int = {
    dtype match {
      case DType.UInt8 => 1
      case DType.Int32 => 4
      case DType.Float16 => 2
      case DType.Float32 => 4
      case DType.Float64 => 8
    }
  }
}
