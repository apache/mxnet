package ml.dmlc.mxnet.init

import scala.collection.mutable.ListBuffer

class LibInfo {
  @native def mxSymbolListAtomicSymbolCreators(symbolList: ListBuffer[Long]): Int
}
