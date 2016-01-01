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

  // Create an atomic symbol function by handle and function name.
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

  /**
   * Activation Operator of Neural Net.
   * The parameters listed below can be passed in as keyword arguments.
   * @param name Name of the resulting symbol.
   *             // TODO
   * @return the resulting symbol
   */
  def creator(operator: String,
              name: String,
              attr: Map[String, String],
              paramKwargs: Map[String, String],
              symbols: Symbol*): Symbol = {
    val function = functions(operator)
    require(function != null, s"invalid operator name $operator")

    val addkeyVarNumArgs =
      (function.keyVarNumArgs != null) && !paramKwargs.contains(function.keyVarNumArgs)

    val paramKeys: Array[String] = (
        if (addkeyVarNumArgs) Array[String](function.keyVarNumArgs)
        else Array.empty[String]
      ) ++ paramKwargs.keys
    val paramVals: Array[String] = (
        if (addkeyVarNumArgs) Array[String](symbols.length.toString)
        else Array.empty[String]
      ) ++ paramKwargs.values

    // create atomic symbol
    val symHandle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateAtomicSymbol(
      function.handle, paramKeys, paramVals, symHandle))

    val s = new Symbol(symHandle.value)
    val attrAll = AttrScope.current.get(attr)
    s.setAttr(attrAll)
    val hint = operator.toLowerCase
    val managedName = NameManager.current.get(name, hint)
    /* TODO
    s._compose(*args, name = name, **symbol_kwargs)
    */
    s
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
   * List all auxiliary states in the symbol.
   * @return The names of the auxiliary states.
   * Notes
   * -----
   * Auxiliary states are special states of symbols that do not corresponds to an argument,
   * and do not have gradient. But still be useful for the specific operations.
   * A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   * Most operators do not have Auxiliary states.
   */
  def listAuxiliaryStates(): Array[String] = ???

  // Set the attribute of the symbol.
  private def setAttr(attr: Map[String, String]): Unit = {
    attr.foreach { case (key, value) =>
      checkCall(_LIB.mxSymbolSetAttr(handle, key, value))
    }
  }
}

case class SymbolFunction(handle: SymbolHandle, keyVarNumArgs: String)
