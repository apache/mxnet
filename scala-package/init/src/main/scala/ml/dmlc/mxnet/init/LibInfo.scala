package ml.dmlc.mxnet.init

import ml.dmlc.mxnet.init.Base._

import scala.collection.mutable.ListBuffer

class LibInfo {
  @native def mxSymbolListAtomicSymbolCreators(symbolList: ListBuffer[SymbolHandle]): Int
  @native def mxSymbolGetAtomicSymbolInfo(handle: SymbolHandle,
                                          name: RefString,
                                          desc: RefString,
                                          numArgs: RefInt,
                                          argNames: ListBuffer[String],
                                          argTypes: ListBuffer[String],
                                          argDescs: ListBuffer[String],
                                          keyVarNumArgs: RefString): Int
}
