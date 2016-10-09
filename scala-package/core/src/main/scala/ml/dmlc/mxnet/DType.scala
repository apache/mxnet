package ml.dmlc.mxnet

object DType extends Enumeration {
  type DType = Value
  val Float32 = Value(0)
  val Float64 = Value(1)
  val Float16 = Value(2)
  val UInt8 = Value(3)
  val Int32 = Value(4)
}
