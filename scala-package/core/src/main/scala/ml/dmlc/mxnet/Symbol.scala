package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
 * Symbolic configuration API of mxnet. <br />
 * <b>
 * WARNING: it is your responsibility to clear this object through dispose().
 * NEVER rely on the GC strategy
 * </b>
 * @author Yizhi Liu
 */
// scalastyle:off finalize
class Symbol private(private[mxnet] val handle: SymbolHandle) {
  private val logger: Logger = LoggerFactory.getLogger(classOf[Symbol])
  private var disposed = false

  override protected def finalize(): Unit = {
    dispose()
  }

  /**
   * Release the native memory.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (!disposed) {
      _LIB.mxSymbolFree(handle)
      disposed = true
    }
  }

  def +(other: Symbol): Symbol = Symbol.createFromListedSymbols("_Plus")(Array(this, other))
  def +[@specialized(Int, Float, Double) V](other: V): Symbol = {
    Symbol.createFromListedSymbols("_PlusScalar")(Array(this), Map("scalar" -> other.toString))
  }

  def -(other: Symbol): Symbol = Symbol.createFromListedSymbols("_Minus")(Array(this, other))
  def -[@specialized(Int, Float, Double) V](other: V): Symbol = {
    Symbol.createFromListedSymbols("_MinusScalar")(Array(this), Map("scalar" -> other.toString))
  }

  def *(other: Symbol): Symbol = Symbol.createFromListedSymbols("_Mul")(Array(this, other))
  def *[@specialized(Int, Float, Double) V](other: V): Symbol = {
    Symbol.createFromListedSymbols("_MulScalar")(Array(this), Map("scalar" -> other.toString))
  }

  def /(other: Symbol): Symbol = Symbol.createFromListedSymbols("_Div")(Array(this, other))
  def /[@specialized(Int, Float, Double) V](other: V): Symbol = {
    Symbol.createFromListedSymbols("_DivScalar")(Array(this), Map("scalar" -> other.toString))
  }

  override def clone(): Symbol = {
    val clonedHandle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCopy(handle, clonedHandle))
    new Symbol(clonedHandle.value)
  }

  def get(index: Int): Symbol = {
    val newHandle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolGetOutput(handle, index, newHandle))
    new Symbol(handle = newHandle.value)
  }

  def get(name: String): Symbol = {
    var index: Int = -1
    for ((output, i) <- listOutputs().view.zipWithIndex) {
      if (output == name) {
        require(index == -1, s"There are multiple outputs with name $name")
        index = i
      }
    }
    require(index >= 0, s"Cannot find output that matches name $name")
    get(index)
  }

  /**
   * Get a new grouped symbol whose output contains all the internal outputs of this symbol.
   * @return The internal of the symbol.
   */
  def getInternals(): Symbol = {
    val newHandle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolGetInternals(handle, newHandle))
    new Symbol(handle = newHandle.value)
  }

  /**
   * List all the arguments in the symbol.
   * @return Array of all the arguments.
   */
  def listArguments(): Seq[String] = {
    val arr = ArrayBuffer.empty[String]
    checkCall(_LIB.mxSymbolListArguments(handle, arr))
    arr
  }

  /**
   * List all outputs in the symbol.
   * @return : List of all the outputs.
   */
  def listOutputs(): Seq[String] = {
    val arr = ArrayBuffer.empty[String]
    checkCall(_LIB.mxSymbolListOutputs(handle, arr))
    arr
  }

  /**
   * List all auxiliary states in the symbol.
   * @return The names of the auxiliary states.
   * @note
   * Auxiliary states are special states of symbols that do not corresponds to an argument,
   * and do not have gradient. But still be useful for the specific operations.
   * A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   * Most operators do not have Auxiliary states.
   */
  def listAuxiliaryStates(): Seq[String] = {
    val sarr = ArrayBuffer.empty[String]
    checkCall(_LIB.mxSymbolListAuxiliaryStates(handle, sarr))
    sarr
  }

  /**
   * Infer the type of outputs and arguments of given known types of arguments.
   * Tuple of Nones is returned if there is not enough information passed in.
   * An error will be raised if there is inconsistency found in the known types passed in.
   * @param args Provide type of arguments in a positional way. Unknown type can be marked as null
   * @return
   * argTypes : list of numpy.dtype or None
   *            List of types of arguments.
   *            The order is in the same order as list_arguments()
   * outTypes : list of numpy.dtype or None
   *            List of types of outputs.
   *            The order is in the same order as list_outputs()
   * auxTypes : list of numpy.dtype or None
   *            List of types of outputs.
   *            The order is in the same order as list_auxiliary()
   */
  def inferType(args: Class[_ >: Float with Int with Double]*)
    : (Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]]) = {
    val sdata: Array[Int] = args.map(NDArray.DTYPE_NATIVE_TO_MX.getOrElse(_, -1)).toArray
    inferType(null, sdata)
  }

  /**
   * Infer the type of outputs and arguments of given known types of arguments.
   * Tuple of Nones is returned if there is not enough information passed in.
   * An error will be raised if there is inconsistency found in the known types passed in.
   * @param kwargs Provide keyword arguments of known types.
   * @return
   * argTypes : list of numpy.dtype or None
   *            List of types of arguments.
   *            The order is in the same order as list_arguments()
   * outTypes : list of numpy.dtype or None
   *            List of types of outputs.
   *            The order is in the same order as list_outputs()
   * auxTypes : list of numpy.dtype or None
   *            List of types of outputs.
   *            The order is in the same order as list_auxiliary()
   */
  def inferType(kwargs: Map[String, Class[_ >: Float with Int with Double]])
    : (Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]]) = {
    val filteredArgs = kwargs.filter { case (key, value) =>
      NDArray.DTYPE_NATIVE_TO_MX.contains(value)
    }
    val keys = filteredArgs.keys.toArray
    val sdata = filteredArgs.values.map(NDArray.DTYPE_NATIVE_TO_MX(_)).toArray
    inferType(keys, sdata)
  }

  private def inferType(keys: Array[String], values: Array[Int])
    : (Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]]) = {
    val argTypeData = ListBuffer.empty[Int]
    val outTypeData = ListBuffer.empty[Int]
    val auxTypeData = ListBuffer.empty[Int]
    val complete = new RefInt
    checkCall(_LIB.mxSymbolInferType(
      handle, keys, values, argTypeData, outTypeData, auxTypeData, complete))
    if (complete.value != 0) {
      (argTypeData.map(NDArray.DTYPE_MX_TO_NATIVE),
        outTypeData.map(NDArray.DTYPE_MX_TO_NATIVE),
        auxTypeData.map(NDArray.DTYPE_MX_TO_NATIVE))
    } else {
      (null, null, null)
    }
  }

  /**
   * Infer the shape of outputs and arguments of given known shapes of arguments.
   * User can either pass in the known shapes in positional way or keyword argument way.
   * Tuple of Nones is returned if there is not enough information passed in.
   * An error will be raised if there is inconsistency found in the known shapes passed in.
   * @param args Provide shape of arguments in a positional way.
   *             Unknown shape can be marked as None
   * @return
   * argShapes List of shapes of arguments. The order is in the same order as list_arguments()
   * outShapes List of shapes of outputs. The order is in the same order as list_outputs()
   * auxShapes List of shapes of outputs. The order is in the same order as list_auxiliary()
   */
  def inferShape(args: Shape*): (Seq[Shape], Seq[Shape], Seq[Shape]) = {
    val keys: Array[String] = null
    val indPtr = ArrayBuffer(0)
    val sdata = ArrayBuffer.empty[Int]
    args.foreach { shape =>
      if (shape != null) {
        sdata ++= shape.toVector
        indPtr += sdata.size
      }
    }
    inferShape(keys, indPtr.toArray, sdata.toArray)
  }

  /**
   * Infer the shape of outputs and arguments of given known shapes of arguments.
   * User can either pass in the known shapes in positional way or keyword argument way.
   * Tuple of Nones is returned if there is not enough information passed in.
   * An error will be raised if there is inconsistency found in the known shapes passed in.
   * @param kwargs Provide keyword arguments of known shapes.
   * @return
   * argShapes List of shapes of arguments. The order is in the same order as list_arguments()
   * outShapes List of shapes of outputs. The order is in the same order as list_outputs()
   * auxShapes List of shapes of outputs. The order is in the same order as list_auxiliary()
   */
  def inferShape(kwargs: Map[String, Shape]): (Seq[Shape], Seq[Shape], Seq[Shape]) = {
    val keys = ArrayBuffer.empty[String]
    val indPtr = ArrayBuffer(0)
    val sdata = ArrayBuffer.empty[Int]
    kwargs.foreach { case (key, shape) =>
      keys += key
      sdata ++= shape.toVector
      indPtr += sdata.size
    }
    inferShape(keys.toArray, indPtr.toArray, sdata.toArray)
  }

  def inferShape(keys: Array[String], indPtr: Array[Int], values: Array[Int])
    : (Seq[Shape], Seq[Shape], Seq[Shape]) = {
    val argShapeData = ListBuffer.empty[Array[Int]]
    val outShapeData = ListBuffer.empty[Array[Int]]
    val auxShapeData = ListBuffer.empty[Array[Int]]
    val complete = new RefInt

    checkCall(_LIB.mxSymbolInferShape(handle, indPtr.size - 1, keys, indPtr, values,
      argShapeData, outShapeData, auxShapeData, complete))
    if (complete.value != 0) {
      (argShapeData.map(s => Shape(s)),
       outShapeData.map(s => Shape(s)),
       auxShapeData.map(s => Shape(s)))
    } else {
      (null, null, null)
    }
  }

  /**
   * Get attribute string from the symbol, this function only works for non-grouped symbol.
   * @param key  The key to get attribute from.
   * @return value The attribute value of the key, returns None if attribute do not exist.
   */
  def attr(key: String): Option[String] = {
    val ret = new RefString
    val success = new RefInt
    checkCall(_LIB.mxSymbolGetAttr(handle, key, ret, success))
    if (success.value != 0) {
      Option(ret.value)
    } else {
      None
    }
  }

  /**
   * Invoke symbol as function on inputs.
   * @param name resulting symbol name
   * @param symbols provide named symbols
   * @return the resulting symbol
   */
  def apply(name: String, symbols: Map[String, Symbol]): Symbol = {
    val s = clone()
    s.compose(name, symbols)
    s
  }

  /**
   * Get a debug string.
   * @return Debug string of the symbol.
   */
  def debugStr: String = {
    val str = new RefString
    checkCall(_LIB.mxSymbolPrint(handle, str))
    str.value
  }

  // Set the attribute of the symbol.
  private def setAttr(attr: Map[String, String]): Unit = {
    attr.foreach { case (key, value) =>
      checkCall(_LIB.mxSymbolSetAttr(handle, key, value))
    }
  }

  /**
   * Save symbol into file.
   * You can also use pickle to do the job if you only work on python.
   * The advantage of load/save is the file is language agnostic.
   * This means the file saved using save can be loaded by other language binding of mxnet.
   * You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)
   *
   * @param fname The name of the file
   *        - s3://my-bucket/path/my-s3-symbol
   *        - hdfs://my-bucket/path/my-hdfs-symbol
   *        - /path-to/my-local-symbol
   * @see Symbol.load : Used to load symbol from file.
   */
  def save(fname: String): Unit = {
    checkCall(_LIB.mxSymbolSaveToFile(this.handle, fname))
  }

  /**
   * Compose symbol on inputs.
   * This call mutates the current symbol.
   * @param name resulting symbol name
   * @param symbols provide positional arguments
   * @return the resulting symbol
   */
  private def compose(name: String, symbols: Array[Symbol]): Unit = {
    val args = symbols.map(_.handle)
    checkCall(_LIB.mxSymbolCompose(handle, name, null, args))
  }

  private def compose(name: String, symbols: Map[String, Symbol]): Unit = {
    val keys = symbols.keys.toArray
    val args = symbols.values.map(_.handle).toArray
    checkCall(_LIB.mxSymbolCompose(handle, name, keys, args))
  }

  /**
   * Bind current symbol to get an executor, allocate all the ndarrays needed.
   * Allows specifying data types.
   * This function will ask user to pass in ndarray of position
   * they like to bind to, and it will automatically allocate the ndarray
   * for arguments and auxiliary states that user did not specify explicitly.
   *
   * @param ctx The device context the generated executor to run on.
   * @param gradReq {'write', 'add', 'null'}, or list of str or dict of str to str, optional
   *                Specifies how we should update the gradient to the args_grad.
   *                - 'write' means everytime gradient is write to specified args_grad NDArray.
   *                - 'add' means everytime gradient is add to the specified NDArray.
   *                - 'null' means no action is taken, the gradient may not be calculated.
   * @param typeDict Input type dictionary, name->dtype
   * @param shapeDict Input shape dictionary, name->shape
   * @return The generated Executor
   */
  def simpleBind(ctx: Context, gradReq: String = "write",
                 shapeDict: Map[String, Shape],
                 typeDict: Map[String, Class[_ >: Float with Int with Double]] = null): Executor = {
    val types =
      if (typeDict == null) listArguments().map((_, classOf[Float])).toMap
      else typeDict
    val (argShapes, _, auxShapes) = inferShape(shapeDict)
    val (argTypes, _, auxTypes) = inferType(types)
    require(argShapes != null && argTypes != null, "Input node is not complete")
    // alloc space
    val argNDArrays = (argShapes zip argTypes) map { case (shape, t) =>
      // TODO: NDArray dtype
      NDArray.zeros(shape, ctx)
    }
    val gradNDArrays =
      if (gradReq != "null") {
        (((listArguments() zip argShapes) zip argTypes) flatMap { case ((name, shape), t) =>
          if (!(name.endsWith("data") || name.endsWith("label"))) {
            // TODO: NDArray dtype
            Map(name -> NDArray.zeros(shape, ctx))
          } else {
            Map.empty[String, NDArray]
          }
        }).toMap
      } else {
        null
      }
    val auxNDArrays = (auxShapes zip auxTypes) map { case (shape, t) =>
      // TODO: NDArray dtype
      NDArray.zeros(shape, ctx)
    }
    bind(ctx, argNDArrays, gradNDArrays, gradReq, auxNDArrays, null, null)
  }

  /**
   * Bind current symbol to get an executor.
   *
   * @param ctx Context The device context the generated executor to run on.
   * @param args Input arguments to the symbol.
   *             - If type is list of NDArray, the position is in the same order of list_arguments.
   *             - If type is dict of str to NDArray, then it maps the name of arguments
   *               to the corresponding NDArray.
   *             - In either case, all the arguments must be provided.
   * @param argsGrad When specified, args_grad provide NDArrays to hold
   *                 the result of gradient value in backward.
   *                 - If type is list of NDArray,
   *                   the position is in the same order of list_arguments.
   *                 - If type is dict of str to NDArray, then it maps the name of arguments
   *                   to the corresponding NDArray.
   *                 - When the type is dict of str to NDArray, users only need to provide the dict
   *                   for needed argument gradient.
   *                   Only the specified argument gradient will be calculated.
   * @param gradReq {'write', 'add', 'null'}, or list of str or dict of str to str, optional
   *                Specifies how we should update the gradient to the args_grad.
   *                - 'write' means everytime gradient is write to specified args_grad NDArray.
   *                - 'add' means everytime gradient is add to the specified NDArray.
   *                - 'null' means no action is taken, the gradient may not be calculated.
   * @param auxStates Input auxiliary states to the symbol, only need to specify when
   *                  list_auxiliary_states is not empty.
   *                  - If type is list of NDArray,
   *                    the position is in the same order of listAuxiliaryStates
   *                  - If type is dict of str to NDArray, then it maps the name of auxiliary_states
   *                    to the corresponding NDArray,
   *                  - In either case, all the auxiliary_states need to be provided.
   * @param group2ctx The dict mapping the ``ctx_group`` attribute to the context assignment.
   * @param sharedExec Executor to share memory with.
   *                 - This is intended for runtime reshaping, variable length sequences, etc.
   *                 - The returned executor shares state with shared_exec,
   *                   and should not be used in parallel with it.
   * @return The generated Executor
   * @note
   * Auxiliary states are special states of symbols that do not corresponds to an argument,
   * and do not have gradient. But still be useful for the specific operations.
   * A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   * Most operators do not have auxiliary states and this parameter can be safely ignored.
   *
   * User can give up gradient by using a dict in args_grad and only specify
   * gradient they interested in.
   */
  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradReq: String, auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradReq: String, auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradReq: String, auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradReq: String, auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradReq: String, auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradReq: String, auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradReq: String, auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradReq: String, auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradsReq: Seq[String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradsReq: Seq[String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Seq[String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Seq[String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradsReq: Seq[String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradsReq: Seq[String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Seq[String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Seq[String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradsReq: Map[String, String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradsReq: Map[String, String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Map[String, String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Map[String, String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradsReq: Map[String, String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradsReq: Map[String, String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Map[String, String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Map[String, String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray]): Executor = {
    bind(ctx, args, argsGrad, "write", Nil, null, null)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray]): Executor = {
    bind(ctx, args, argsGrad, "write", Nil, null, null)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray]): Executor = {
    bind(ctx, args, argsGrad, "write", Nil, null, null)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray]): Executor = {
    bind(ctx, args, argsGrad, "write", Nil, null, null)
  }

  def bind(ctx: Context, args: Seq[NDArray]): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, null,
               Seq.fill(symbolArguments.size)("write"), Nil, null, null)
  }

  def bind(ctx: Context, args: Map[String, NDArray]): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, null,
      Seq.fill(symbolArguments.size)("write"), Nil, null, null)
  }

  private def bindHelper(ctx: Context, symbolArguments: Seq[String],
                         args: Iterable[_], argsGrad: Iterable[_],
                         gradsReq: Iterable[_], auxStates: Iterable[_],
                         group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    require(args != null && !args.isInstanceOf[Set[_]])
    require(argsGrad == null || !argsGrad.isInstanceOf[Set[_]])
    require(auxStates == null || !auxStates.isInstanceOf[Set[_]])
    require(gradsReq != null && !gradsReq.isInstanceOf[Set[_]])

    val (argsHandle, argsNDArray) =
      if (args.isInstanceOf[Seq[_]]) {
        Symbol.getNDArrayInputs("args", args.asInstanceOf[Seq[NDArray]],
                                symbolArguments, allowMissing = false)
      } else {
        Symbol.getNDArrayInputs("args", args.asInstanceOf[Map[String, NDArray]],
                                symbolArguments, allowMissing = false)
      }

    // setup args gradient
    val (argsGradHandle, argsGradNDArray) =
      if (argsGrad == null) {
        (Array.fill[NDArrayHandle](args.size)(0L), null)
      } else if (argsGrad.isInstanceOf[Seq[_]]) {
        Symbol.getNDArrayInputs("args_grad", argsGrad.asInstanceOf[Seq[NDArray]],
                                symbolArguments, allowMissing = true)
      } else {
        Symbol.getNDArrayInputs("args_grad", argsGrad.asInstanceOf[Map[String, NDArray]],
                                symbolArguments, allowMissing = true)
      }

    val (auxArgsHandle, auxStatesNDArray) =
      if (auxStates == null) {
        Symbol.getNDArrayInputs("aux_states", Nil, listAuxiliaryStates(), allowMissing = false)
      } else if (auxStates.isInstanceOf[Seq[_]]) {
        Symbol.getNDArrayInputs("aux_states", auxStates.asInstanceOf[Seq[NDArray]],
                                listAuxiliaryStates(), allowMissing = false)
      } else {
        Symbol.getNDArrayInputs("aux_states", auxStates.asInstanceOf[Map[String, NDArray]],
                                listAuxiliaryStates(), allowMissing = false)
      }

    // setup requirements
    val reqsArray =
      if (gradsReq.isInstanceOf[Seq[_]]) {
        gradsReq.asInstanceOf[Seq[String]].map { req =>
          require(Symbol.bindReqMap.contains(req), s"grad_req must be in ${Symbol.bindReqMap}")
          Symbol.bindReqMap(req)
        }.toArray
      } else {
        val gradsReqMap = gradsReq.asInstanceOf[Map[String, String]]
        symbolArguments.map { req =>
          val value = gradsReqMap.getOrElse(req, "null")
          require(Symbol.bindReqMap.contains(value), s"grad_req must be in ${Symbol.bindReqMap}")
          Symbol.bindReqMap(value)
        }.toArray
      }

    val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]

    if (group2ctx != null) {
      group2ctx.foreach { case (key, value) =>
        ctxMapKeys += key
        ctxMapDevTypes += value.deviceTypeid
        ctxMapDevIDs += value.deviceId
      }
    }

    val execHandle = new ExecutorHandleRef
    val sharedHadle = if (sharedExec != null) sharedExec.handle else 0L
    checkCall(_LIB.mxExecutorBindEX(handle,
                                   ctx.deviceTypeid,
                                   ctx.deviceId,
                                   ctxMapKeys.size,
                                   ctxMapKeys.toArray,
                                   ctxMapDevTypes.toArray,
                                   ctxMapDevIDs.toArray,
                                   args.size,
                                   argsHandle,
                                   argsGradHandle,
                                   reqsArray,
                                   auxArgsHandle,
                                   sharedHadle,
                                   execHandle))
    val executor = new Executor(execHandle.value, this.clone())
    executor.argArrays = argsNDArray
    executor.gradArrays = argsGradNDArray
    executor.auxArrays = auxStatesNDArray
    executor._ctx = new Context(ctx.deviceType, ctx.deviceId)
    executor._gradsReq = gradsReq
    executor._group2ctx =
      if (group2ctx == null) null
      else group2ctx.map { case (key, value) =>
        (key -> new Context(value.deviceType, value.deviceId))
      }.toMap
    executor
  }

  /**
   * Save symbol into a JSON string.
   * See Also
   * symbol.loadJson : Used to load symbol from JSON string.
   */
  def toJson: String = {
    val jsonStr = new RefString
    checkCall(_LIB.mxSymbolSaveToJSON(handle, jsonStr))
    jsonStr.value
  }
}

// scalastyle:on finalize
@AddSymbolFunctions
object Symbol {
  private type SymbolCreateNamedFunc = Map[String, Any] => Symbol
  private val logger = LoggerFactory.getLogger(classOf[Symbol])
  private val functions: Map[String, SymbolFunction] = initSymbolModule()
  private val bindReqMap = Map("null" -> 0, "write" -> 1, "add" -> 3)

  def pow(sym1: Symbol, sym2: Symbol): Symbol = {
    Symbol.createFromListedSymbols("_Power")(Array(sym1, sym2))
  }

  def pow[@specialized(Int, Float, Double) V](sym: Symbol, number: V): Symbol = {
    Symbol.createFromListedSymbols("_PowerScalar")(Array(sym), Map("scalar" -> number.toString))
  }

  def pow[@specialized(Int, Float, Double) V](number: V, sym: Symbol): Symbol = {
    Symbol.createFromListedSymbols("_RPowerScalar")(Array(sym), Map("scalar" -> number.toString))
  }

  def max(left: Symbol, right: Symbol): Symbol = {
    createFromListedSymbols("_Maximum")(Array(left, right))
  }

  def max[@specialized(Int, Float, Double) V](left: Symbol, right: V): Symbol = {
    createFromListedSymbols("_MaximumScalar")(Array(left), Map("scalar" -> right.toString))
  }

  def max[@specialized(Int, Float, Double) V](left: V, right: Symbol): Symbol = {
    createFromListedSymbols("_MaximumScalar")(Array(right), Map("scalar" -> left.toString))
  }

  def min(left: Symbol, right: Symbol): Symbol = {
    createFromListedSymbols("_Minimum")(Array(left, right))
  }

  def min[@specialized(Int, Float, Double) V](left: Symbol, right: V): Symbol = {
    createFromListedSymbols("_MinimumScalar")(Array(left), Map("scalar" -> right.toString))
  }

  def min[@specialized(Int, Float, Double) V](left: V, right: Symbol): Symbol = {
    createFromListedSymbols("_MinimumScalar")(Array(right), Map("scalar" -> left.toString))
  }

  /**
   * Create a symbolic variable with specified name.
   * @param name Name of the variable.
   * @param attr Additional attributes to set on the variable.
   * @return The created variable symbol.
   */
  def Variable(name: String, attr: Map[String, String] = null): Symbol = {
    val handle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateVariable(name, handle))
    val sym = new Symbol(handle.value)
    sym.setAttr(AttrScope.current.get(Option(attr)))
    sym
  }

  /**
   * Create a symbol that groups symbols together.
   * @param symbols List of symbols to be grouped.
   * @return The created group symbol.
   */
  def Group(symbols: Symbol*): Symbol = {
    val ihandles = symbols.map(_.handle).toArray
    val handle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateGroup(ihandles, handle))
    new Symbol(handle.value)
  }

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

  // Used by SymbolMacro
  private def createSymbolGeneral(operator: String, name: String, attr: Map[String, String],
      symbols: Seq[Symbol], kwargs: Map[String, Any]): Symbol = {
    val symbolKwargs: Map[String, Symbol] =
      if (kwargs == null || kwargs.isEmpty) {
        Map.empty[String, Symbol]
      } else {
        kwargs.filter { case (key, value) =>
          value.isInstanceOf[Symbol]
        }.map { case (key, value) =>
          (key, value.asInstanceOf[Symbol])
        }
      }
    val strKwargs: Map[String, String] =
      if (kwargs == null || kwargs.isEmpty) {
        Map.empty[String, String]
      } else {
        kwargs.filter { case (key, value) =>
          !value.isInstanceOf[Symbol]
        }.map { case (key, value) =>
          (key, value.toString)
        }
      }
    require(symbols.isEmpty || symbolKwargs.isEmpty, String.format(
      "%s can only accept input Symbols either as positional or keyword arguments, not both",
      operator))
    if (symbols.isEmpty) {
      createFromNamedSymbols(operator, name, attr)(symbolKwargs, strKwargs)
    } else {
      createFromListedSymbols(operator, name, attr)(symbols.toArray, strKwargs)
    }
  }

  /**
   * Activation Operator of Neural Net.
   * The parameters listed below can be passed in as keyword arguments.
   * @param symbols Symbol parameters passed to create the resulting symbol
   * @param paramKwargs Key-value parameters passed to create the resulting symbol
   * @param attr Attributes set to the resulting symbol
   * @return the resulting symbol
   */
  def createFromListedSymbols(
      operator: String, name: String = null, attr: Map[String, String] = null)(
      symbols: Array[Symbol], paramKwargs: Map[String, String] = null): Symbol = {
    val function = functions(operator)
    require(function != null, s"invalid operator name $operator")

    val params = if (paramKwargs == null) Map.empty[String, String] else paramKwargs
    val addkeyVarNumArgs = (function.keyVarNumArgs != null
      && !function.keyVarNumArgs.isEmpty
      && !params.contains(function.keyVarNumArgs))

    val paramKeys: Array[String] = (
        if (addkeyVarNumArgs) Array[String](function.keyVarNumArgs)
        else Array.empty[String]
      ) ++ params.keys
    val paramVals: Array[String] = (
        if (addkeyVarNumArgs) Array[String](symbols.length.toString)
        else Array.empty[String]
      ) ++ params.values

    // create atomic symbol
    val symHandle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateAtomicSymbol(
      function.handle, paramKeys, paramVals, symHandle))

    val s = new Symbol(symHandle.value)
    val attrAll = AttrScope.current.get(Option(attr))
    s.setAttr(attrAll)
    val hint = operator.toLowerCase
    val managedName = NameManager.current.get(Option(name), hint)
    s.compose(managedName, symbols)
    s
  }

  /**
   * Activation Operator of Neural Net.
   * The parameters listed below can be passed in as keyword arguments.
   * @param symbols Named symbol parameters passed to create the resulting symbol
   * @param paramKwargs Key-value parameters passed to create the resulting symbol
   * @param attr Attributes set to the resulting symbol
   * @return the resulting symbol
   */
  def createFromNamedSymbols(
      operator: String, name: String = null, attr: Map[String, String] = null)(
      symbols: Map[String, Symbol], paramKwargs: Map[String, String] = null): Symbol = {
    val function = functions(operator)
    require(function != null, s"invalid operator name $operator")
    require(function.keyVarNumArgs == null || function.keyVarNumArgs.isEmpty,
      s"[$operator] support variable length of Symbol arguments.\n" +
      "Please pass all the input Symbols via positional arguments instead of keyword arguments.")

    val paramKeys =
      if (paramKwargs == null) Array.empty[String]
      else paramKwargs.keys.toArray
    val paramVals =
      if (paramKwargs == null) Array.empty[String]
      else paramKwargs.values.toArray
    val symHandle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateAtomicSymbol(
      function.handle, paramKeys, paramVals, symHandle))

    val s = new Symbol(symHandle.value)
    val attrAll = AttrScope.current.get(Option(attr))
    s.setAttr(attrAll)
    val hint = operator.toLowerCase
    val managedName = NameManager.current.get(Option(name), hint)
    s.compose(managedName, symbols)
    s
  }

  // a more friendly interface for creating symbols
  // all values except symbols in kwargs will be cast to String using its toString() method
  @Deprecated
  def createFromNamedSymbolsNoCheck(
      operator: String, name: String = null, attr: Map[String, String] = null)(
      kwargs: Map[String, Any]): Symbol = {
    val symbolArgs = kwargs.filter { case (key, value) =>
      value.isInstanceOf[Symbol]
    }.map { case (key, value) =>
      (key, value.asInstanceOf[Symbol])
    }
    val strArgs = kwargs.filter { case (key, value) =>
      !value.isInstanceOf[Symbol]
    }.map { case (key, value) =>
      (key, value.toString)
    }
    createFromNamedSymbols(operator, name, attr)(symbolArgs, strArgs)
  }

  // a more friendly interface for creating symbols
  // all values except symbols in kwargs will be cast to String using its toString() method
  @Deprecated
  def createFromListedSymbolsNoCheck(
      operator: String, name: String = null, attr: Map[String, String] = null)(
      symbols: Array[Symbol], kwargs: Map[String, Any] = null): Symbol = {
    val args =
      if (kwargs == null) null
      else kwargs.map { case (key, value) => (key, value.toString) }
    createFromListedSymbols(operator, name, attr)(symbols, args)
  }

  /**
   * Helper function to get ndarray lists handles from various inputs.
   * @param argKey The name of argument, used for error message.
   * @param args list of NDArray or dict of str to NDArray
   *             Input arguments to the symbols.
   *             If type is list of NDArray, the position is in the same order of arg_names.
   *             If type is dict of str to NDArray, then it maps the name of arguments
   *             to the corresponding NDArray
   * @param argNames List of argument names.
   * @param allowMissing Whether missing argument is allowed.
   *                     When allowed, the missing handle will be set to None(null)
   * @return The positional list of NDArrayHandles generated from input.
   */
  private def getNDArrayInputs(argKey: String, args: Seq[NDArray], argNames: Seq[String],
                               allowMissing: Boolean): (Array[NDArrayHandle], Array[NDArray]) = {
    require(args.length == argNames.length, s"Length of $argKey do not match number of arguments")
    val argHandles = args.map(_.handle)
    (argHandles.toArray, args.toArray)
  }

  private def getNDArrayInputs(argKey: String, args: Map[String, NDArray], argNames: Seq[String],
                               allowMissing: Boolean): (Array[NDArrayHandle], Array[NDArray]) = {
    val argArrays = ArrayBuffer.empty[NDArray]
    val argHandles = ArrayBuffer.empty[NDArrayHandle]
    argNames.foreach { name =>
      args.get(name) match {
        case narr: Some[NDArray] =>
          argArrays += narr.get
          argHandles += narr.get.handle
        case None =>
          require(allowMissing, s"Must specify all the arguments in $argKey")
          argArrays += null
          argHandles += 0L
      }
    }
    (argHandles.toArray, argArrays.toArray)
  }

  /**
   * Load symbol from a JSON file.
   *
   * You can also use pickle to do the job if you only work on python.
   * The advantage of load/save is the file is language agnostic.
   * This means the file saved using save can be loaded by other language binding of mxnet.
   * You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)
   *
   * @param fname The name of the file, examples:
   *        - `s3://my-bucket/path/my-s3-symbol`
   *        - `hdfs://my-bucket/path/my-hdfs-symbol`
   *        - `/path-to/my-local-symbol`
   * @return The loaded symbol.
   * @see Symbol.save : Used to save symbol into file.
   */
  def load(fname: String): Symbol = {
    val handle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateFromFile(fname, handle))
    new Symbol(handle.value)
  }

  /**
   * Load symbol from json string.
   * @param json A json string.
   * @return The loaded symbol.
   * @see Symbol.tojson : Used to save symbol into json string.
   */
  def loadJson(json: String): Symbol = {
    val handle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateFromJSON(json, handle))
    new Symbol(handle.value)
  }
}

private case class SymbolFunction(handle: SymbolHandle, keyVarNumArgs: String)

object SymbolConversions {
  implicit def int2Scalar(x: Int): SymbolConversions[Int] = new SymbolConversions(x)
  implicit def double2Scalar(x: Double): SymbolConversions[Double] = new SymbolConversions(x)
  implicit def float2Scalar(x: Float): SymbolConversions[Float] = new SymbolConversions(x)
}

class SymbolConversions[@specialized(Int, Float, Double) V](val value: V) {
  def +(other: Symbol): Symbol = {
    other + value
  }

  def -(other: Symbol): Symbol = {
    Symbol.createFromListedSymbols("_RMinusScalar")(
      Array(other), Map("scalar" -> value.toString))
  }

  def *(other: Symbol): Symbol = {
    other + value
  }

  def /(other: Symbol): Symbol = {
    Symbol.createFromListedSymbols("_RDivScalar")(
      Array(other), Map("scalar" -> value.toString))
  }
}
