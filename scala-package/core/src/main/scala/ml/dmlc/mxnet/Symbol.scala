package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

object Symbol {
  private val logger = LoggerFactory.getLogger(classOf[Symbol])
  private val functions: Map[String, SymbolFunction] = initSymbolModule()

  // List and add all the atomic symbol functions to current module.
  private def initSymbolModule(): Map[String, SymbolFunction] = {
    val symbolList = ListBuffer.empty[SymbolHandle]
    checkCall(_LIB.mxSymbolListAtomicSymbolCreators(symbolList))
    symbolList.map(makeAtomicSymbolFunction).toMap
  }

  // Create an atomic symbol function by handle and funciton name.
  private def makeAtomicSymbolFunction(handle: SymbolHandle): (String, SymbolFunction) = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val numArgs = new MXUintRef
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    checkCall(_LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs))
    val paramStr = ctypes2docstring(argNames, argTypes, argDescs)
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
    logger.debug("Atomic Symbol function defination:\n{}", docStr)
    (name.value, new SymbolFunction(handle, keyVarNumArgs.value))
  }
}

class Symbol(private[mxnet] val handle: SymbolHandle) {
  def +(other: Symbol): Symbol = ???
  def +(other: Int): Symbol = ???
  def +(other: Float): Symbol = ???
  def +(other: Double): Symbol = ???

  /**
   * List all the arguments in the symbol.
   * @return Array of all the arguments.
   */
  def listArguments(): Array[String] = ???

  /**
   * List all auxiliary states in the symbool.
   * @return The names of the auxiliary states.
   * Notes
   * -----
   * Auxiliary states are special states of symbols that do not corresponds to an argument,
   * and do not have gradient. But still be useful for the specific operations.
   * A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   * Most operators do not have Auxiliary states.
   */
  def listAuxiliaryStates(): Array[String] = ???
}

case class SymbolFunction(handle: SymbolHandle, keyVarNumArgs: String)
