package ml.dmlc.mxnet

import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Helper class to manage multiple executors for data parallelism.
 * @author Yizhi Liu
 * @param symbol output symbol
 * @param ctx devices to run on
 * @param paramNames Name of all trainable parameters of the network.
 * @param argNames Name of all arguments of the network.
 * @param auxNames Name of all auxiliary states of the network.
 * @param trainData Training data iterator.
 * @param workLoadList The list of work load for different devices, in the same order as ctx
 * @param logger When not specified, default logger will be used.
 */
class DataParallelExecutorManager(symbol: Symbol,
                                  ctx: Array[Context],
                                  paramNames: Seq[String],
                                  argNames: Seq[String],
                                  private val auxNames: Seq[String],
                                  trainData: DataIter,
                                  private var workLoadList: Seq[Float] = null,
                                  logger: Logger = DataParallelExecutorManager.logger) {
  // preparation
  private val numDevice = ctx.length
  logger.info(s"Start training with [${ctx.mkString(",")}]")

  // make sure the architecture is valid
  ExecutorManager.checkArguments(symbol)

  if (workLoadList == null) {
    workLoadList = Seq.fill(numDevice)(1f)
  }
  require(workLoadList.size == numDevice, "Invalid settings for work load.")

  private val slices = ExecutorManager.splitInputSlice(trainData.batchSize, workLoadList)

  private val trainExecs =
    ctx.zipWithIndex.map { case (context, i) =>
      val dataShapes =
        (trainData.provideData ++ trainData.provideLabel).map { case (name: String, shape: Shape) =>
          (name, Shape(slices(i)._2 - slices(i)._1) ++ shape.drop(1))
        }
      symbol.simpleBind(context, "write", shapeDict = dataShapes)
    }

  // data structure
  private val dataNames = trainData.provideData.map(_._1).toArray
  private val labelNames = trainData.provideLabel.map(_._1).toArray

  private val dataArrays =
    dataNames.map { name =>
      trainExecs.zipWithIndex.map { case (exec, i) =>
        val slice = slices(i)
        (slice._1, slice._2, exec.argDict(name))
      }
    }
  private val labelArrays =
    labelNames.map { name =>
      trainExecs.zipWithIndex.map { case (exec, i) =>
        val slice = slices(i)
        (slice._1, slice._2, exec.argDict(name))
      }
    }

  private val paramIdx = (0 until argNames.length).filter { i =>
    paramNames.contains(argNames(i))
  }
  private[mxnet] val _paramNames = paramIdx.map(argNames(_))
  private[mxnet] val paramArrays = paramIdx.map { i =>
    trainExecs.map(_.argArrays(i))
  }.toArray
  private[mxnet] val gradArrays = paramIdx.map { i =>
    trainExecs.map(_.gradArrays(i))
  }.toArray

  private val auxArrays = (0 until auxNames.length).map { i =>
    trainExecs.map(_.auxArrays(i))
  }.toArray
  private val batchSize = trainData.batchSize
  private val outputShapes: Array[Shape] = trainExecs(0).outputs.map { x: NDArray =>
    Shape(batchSize) ++ x.shape.drop(1)
  }
  private[mxnet] val cpuOutputArrays = outputShapes.map(NDArray.zeros(_))

  /**
   * Release the related executors.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    trainExecs.foreach(_.dispose())
  }

  // Install monitor on all executors
  def installMonitor(monitor: Monitor): Unit = {
    trainExecs.foreach(monitor.install)
  }

  /**
   * Set parameter and aux values
   * @param argParams source parameter arrays
   * @param auxParams source aux arrays
   */
  def setParams(argParams: Map[String, NDArray], auxParams: Map[String, NDArray]): Unit = {
    trainExecs.foreach(_.copyParamsFrom(argParams, auxParams))
  }

  /**
   * Copy data from each executor to `arg_params` and `aux_params`
   * @param argParams target parameter arrays
   * @param auxParams target aux arrays
   * @note This function will inplace update the NDArrays in arg_params and aux_params.
   */
  def copyTo(argParams: Map[String, NDArray], auxParams: Map[String, NDArray]): Unit = {
    for ((name, block) <- _paramNames zip paramArrays) {
      val weight = block.map(_.copyTo(Context.cpu())).reduce(_ + _) / block.length
      weight.copyTo(argParams(name))
    }
    for ((name, block) <- auxNames zip auxArrays) {
      val weight = block.map(_.copyTo(Context.cpu())).reduce(_ + _) / block.length
      weight.copyTo(auxParams(name))
    }
  }

  // load data and labels into arrays
  def loadDataBatch(dataBatch: DataBatch): Unit = {
    ExecutorManager.loadDataMulti(dataBatch, dataArrays)
    ExecutorManager.loadLabelMulti(dataBatch, labelArrays)
  }

  // Perform a forward pass on each executor
  def forward(isTrain: Boolean = false): Unit = {
    for ((texec, islice) <- trainExecs zip slices) {
      texec.forward(isTrain)
      for ((cpuOut, devOut) <- cpuOutputArrays zip texec.outputs) {
        devOut.copyTo(cpuOut.slice(islice))
      }
    }
  }

  // Perform a backward pass on each executor
  def backward(): Unit = {
    trainExecs.foreach(_.backward())
  }
}

object DataParallelExecutorManager {
  val logger: Logger = LoggerFactory.getLogger(classOf[DataParallelExecutorManager])
}

class ExecutorManager
object ExecutorManager {
  /**
   * Get input slice from the input shape.
   * @param batchSize The number of samples in a mini-batch.
   * @param workLoadList The list of work load for different devices, in the same order as ctx
   * @return The split slices to get a specific slice.
   * @throws IllegalArgumentException
   *         If there are two many splits such that some slice can be empty.
   */
  private[mxnet] def splitInputSlice(batchSize: Int,
                                     workLoadList: Seq[Float]): Array[(Int, Int)] = {
    val totalWorkLoad = workLoadList.sum
    val batchNumList = workLoadList.map(workLoad =>
      math.round(workLoad * batchSize / totalWorkLoad)).toArray
    val batchNumSum = batchNumList.sum
    if (batchNumSum < batchSize) {
      batchNumList(batchNumList.length-1) += batchSize - batchNumSum
    }

    val slices = ArrayBuffer.empty[(Int, Int)]
    var end = 0
    batchNumList.foreach(batchNum => {
      val begin = math.min(end, batchSize)
      end = math.min(begin + batchNum, batchSize)
      require(begin < end, "Too many slices such that some splits are empty")
      slices.append((begin, end))
    })
    slices.toArray
  }

  /**
   * Check the argument names of symbol.
   * This function checks the duplication of arguments in Symbol.
   * The check is done for feedforward net for now.
   * @param symbol The network configuration
   */
  private[mxnet] def checkArguments(symbol: Symbol): Unit = {
    val argNames = symbol.listArguments()
    require(argNames.toSet.size == argNames.length,
      "Find duplicated argument name," +
        "please make the weight name non-duplicated(using name arguments)," +
        s"arguments are $argNames")

    val auxNames = symbol.listAuxiliaryStates()
    require(auxNames.toSet.size == auxNames.length,
      "Find duplicated auxiliary param name," +
        "please make the weight name non-duplicated(using name arguments)," +
        s"arguments are $auxNames")
  }

  // Load a list of arrays into a list of arrays
  private[mxnet] def loadGeneral(data: Seq[NDArray], targets: Seq[NDArray]): Unit = {
    (data zip targets).foreach { case (dSrc, dTarget) =>
      dSrc.copyTo(dTarget)
    }
  }

  // Load a list of arrays into a list of arrays specified by slices
  private[mxnet] def loadGeneralMulti(data: Seq[NDArray],
                                      targets: Seq[Array[(Int, Int, NDArray)]]): Unit = {
    for ((src, dTargets) <- data zip targets) {
      for ((start, end, dst) <- dTargets) {
        val sliced = src.slice(start, end)
        sliced.copyTo(dst)
        sliced.dispose()
      }
    }
  }

  // Load data into sliced arrays
  private[mxnet] def loadDataMulti(batch: DataBatch,
                                   targets: Seq[Array[(Int, Int, NDArray)]]): Unit = {
    loadGeneralMulti(batch.data, targets)
  }

  private[mxnet] def loadData(batch: DataBatch, targets: Seq[NDArray]): Unit = {
    loadGeneral(batch.data, targets)
  }

  // Load label into sliced arrays
  private[mxnet] def loadLabelMulti(batch: DataBatch,
                                    targets: Seq[Array[(Int, Int, NDArray)]]): Unit = {
    loadGeneralMulti(batch.label, targets)
  }

  private[mxnet] def loadLabel(batch: DataBatch, targets: Seq[NDArray]): Unit = {
    loadGeneral(batch.label, targets)
  }

  // bind executor for bucketing, potentially sharing data with an existing executor.
  private def bindExec(sym: Symbol, ctx: Context, inputShapes: Map[String, Shape],
                       paramNames: Set[String], needGrad: Boolean = false,
                       grads: Set[String] = null, baseExec: Executor = null,
                       sharedDataArrays: mutable.Map[String, NDArray] = null,
                       inputTypes: Map[String, Class[_ >: Float with Int with Double]] = null) = {
    val (argShape, _, auxShape) = sym.inferShape(inputShapes)
    require(argShape != null)
    val inputTypesUpdate =
      if (inputTypes == null) {
        inputShapes.map { case (key, _) => (key, classOf[Float]) }
      } else {
        inputTypes
      }
    val (argTypes, _, auxTypes) = sym.inferType(inputTypesUpdate)
    require(argTypes != null)

    val argArrays = ArrayBuffer.empty[NDArray]
    val gradArrays: mutable.Map[String, NDArray] =
      if (needGrad) mutable.HashMap.empty[String, NDArray] else null

    val argNames = sym.listArguments()

    val gradSet: Set[String] =
      if (!needGrad) {
        Set.empty[String]
      } else if (grads == null) {
        argNames.toSet -- inputShapes.keySet
      } else {
        grads
      }

    val gradReq = argNames.map { name =>
      if (gradSet.contains(name)) name -> "write"
      else name -> "null"
    }(collection.breakOut): Map[String, String]

    // create or borrow arguments and gradients
    argNames.zipWithIndex.foreach { case (name, i) =>
      if (!paramNames.contains(name)) {
        // data or label
        val argArr =
          if (sharedDataArrays != null && sharedDataArrays.contains(name)) {
            val arr = sharedDataArrays(name)
            if (arr.shape.product >= argShape(i).product) {
              // good, we can share this memory
              // TODO: assert(argTypes(i) == argArr.dtype)
              arr.reshape(argShape(i))
            } else {
              DataParallelExecutorManager.logger.warn(
                s"bucketing: data $name has a shape ${argShape(i)}," +
                  s"which is larger than already allocated shape ${arr.shape}." +
                  "Need to re-allocate.Consider putting default_bucket_key" +
                  "to be the bucket taking the largest input for better memory sharing.")
              val zeros = NDArray.zeros(argShape(i), ctx) // TODO: dtype = arg_types[i])
              // replace existing shared array because the new one is bigger
              sharedDataArrays.put(name, zeros)
              // dispose the replaced array
              arr.dispose()
              zeros
            }
          } else {
            val zeros = NDArray.zeros(argShape(i), ctx) // TODO: dtype = arg_types[i])
            if (sharedDataArrays != null) {
              sharedDataArrays.put(name, zeros)
            }
            zeros
          }
        argArrays.append(argArr)
      } else {
        // model parameter
        val argArr =
          if (baseExec == null) {
            if (gradSet.contains(name)) {
              val gradArr = NDArray.zeros(argShape(i), ctx) // TODO: dtype = arg_types[i])
              gradArrays.put(name, gradArr)
            }
            NDArray.zeros(argShape(i), ctx) // TODO: dtype = arg_types[i])
          } else {
            val arr = baseExec.argDict(name)
            require(arr.shape == argShape(i))
            // TODO: require(argArr.dtype == argTypes(i))
            if (gradSet.contains(name)) {
              gradArrays.put(name, baseExec.gradDict(name))
            }
            arr
          }
        argArrays.append(argArr)
      }
    }
    // create or borrow aux variables
    val auxArrays =
      if (baseExec == null) {
        (auxShape zip auxTypes) map { case (s, t) =>
          NDArray.zeros(s, ctx) // TODO: dtype = t
        }
      } else {
        baseExec.auxArrays.zipWithIndex.map { case (a, i) =>
          require(auxShape(i) == a.shape)
          // require(auxTypes(i) == a.dtype)
          a
        }.toSeq
      }
    sym.bind(ctx = ctx, args = argArrays.toSeq, argsGrad = gradArrays.toMap, gradsReq = gradReq,
      auxStates = auxArrays, group2ctx = null, sharedExec = baseExec)
  }
}
