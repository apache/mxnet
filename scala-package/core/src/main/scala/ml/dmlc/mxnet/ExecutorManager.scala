/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.mxnet

import ml.dmlc.mxnet.DType.DType
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.immutable.ListMap
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
 * @param symGen symbol generator for bucketing
 */
private[mxnet] class DataParallelExecutorManager(private val symbol: Symbol,
                                  private val ctx: Array[Context],
                                  private[mxnet] val paramNames: IndexedSeq[String],
                                  private[mxnet] val argNames: IndexedSeq[String],
                                  private[mxnet] val auxNames: IndexedSeq[String],
                                  trainData: DataIter,
                                  private var workLoadList: Seq[Float] = null,
                                  private val symGen: SymbolGenerator = null) {
  // preparation
  private val numDevice = ctx.length
  DataParallelExecutorManager.logger.info(s"Start training with [${ctx.mkString(",")}]")

  // make sure the architecture is valid
  ExecutorManager.checkArguments(symbol)

  if (workLoadList == null) {
    workLoadList = Seq.fill(numDevice)(1f)
  }
  require(workLoadList.size == numDevice, "Invalid settings for work load.")

  private val slices = ExecutorManager.splitInputSlice(trainData.batchSize, workLoadList)

  private val paramNameSet = paramNames.toSet

  private val execGrp = new DataParallelExecutorGroup(
    symbol, argNames, paramNameSet, ctx, slices, trainData)
  private var currExecGrp: DataParallelExecutorGroup = null // this is set when data is loaded

  private val execGrpBucket: mutable.Map[AnyRef, DataParallelExecutorGroup]
    = mutable.HashMap.empty[AnyRef, DataParallelExecutorGroup]
  if (symGen != null) {
    execGrpBucket.put(trainData.defaultBucketKey, execGrp)
  }

  // shared parameter arrays
  def paramArrays: IndexedSeq[Array[NDArray]] = {
    // param arrays should be shared by all executor groups
    execGrp.paramArrays
  }

  // shared gradient arrays
  def gradArrays: IndexedSeq[Array[NDArray]] = {
    // grad arrays should be shared by all executor groups
    execGrp.gradArrays
  }

  // shared aux states
  def auxArrays: IndexedSeq[Array[NDArray]] = {
    // aux arrays are also shared by all executor groups
    execGrp.auxArrays
  }

  /**
   * Release all the executor groups.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    execGrp.dispose()
    execGrpBucket.values.foreach(_.dispose())
  }

  // Install monitor on all executors
  def installMonitor(monitor: Monitor): Unit = {
    require(symGen == null, "Monitoring is not implemented for bucketing")
    execGrp.trainExecs.foreach(monitor.install)
  }

  /**
   * Set parameter and aux values
   * @param argParams source parameter arrays
   * @param auxParams source aux arrays
   */
  def setParams(argParams: Map[String, NDArray], auxParams: Map[String, NDArray]): Unit = {
    execGrp.trainExecs.foreach(_.copyParamsFrom(argParams, auxParams))
  }

  /**
   * Copy data from each executor to `arg_params` and `aux_params`
   * @param argParams target parameter arrays
   * @param auxParams target aux arrays
   * @note This function will inplace update the NDArrays in arg_params and aux_params.
   */
  def copyTo(argParams: Map[String, NDArray], auxParams: Map[String, NDArray]): Unit = {
    for ((name, block) <- paramNames zip paramArrays) {
      val weight = block.map(_.copyTo(Context.cpu())).reduce(_ + _) / block.length
      val typedWeight = weight.asType(argParams(name).dtype)
      typedWeight.copyTo(argParams(name))
      typedWeight.dispose()
    }
    for ((name, block) <- auxNames zip auxArrays) {
      val weight = block.map(_.copyTo(Context.cpu())).reduce(_ + _) / block.length
      val typedWeight = weight.asType(auxParams(name).dtype)
      typedWeight.copyTo(auxParams(name))
      typedWeight.dispose()
    }
  }

  // load data and labels into arrays
  def loadDataBatch(dataBatch: DataBatch): Unit = {
    currExecGrp =
      if (symGen != null) {
        val key = dataBatch.bucketKey
        require(key != null, "bucketKey must not be null for bucketing io")
        if (!execGrpBucket.contains(key)) {
          // create new bucket entry
          val sym = symGen.generate(key)
          val grp = new DataParallelExecutorGroup(sym, argNames, paramNameSet,
            ctx, slices, dataBatch, sharedGroup = execGrp)
          execGrpBucket.put(key, grp)
        }
        execGrpBucket(key)
      } else {
        execGrp
      }
    currExecGrp.loadDataBatch(dataBatch)
  }

  // run forward on the current executor
  def forward(isTrain: Boolean = false): Unit = {
    currExecGrp.forward(isTrain = isTrain)
  }

  // run backward on the current executor
  def backward(): Unit = {
    currExecGrp.backward()
  }

  // update metric with the current executor
  def updateMetric(metric: EvalMetric, labels: IndexedSeq[NDArray]): Unit = {
    currExecGrp.updateMetric(metric, labels)
  }
}

private object DataParallelExecutorManager {
  val logger: Logger = LoggerFactory.getLogger(classOf[DataParallelExecutorManager])
}

private[mxnet] object ExecutorManager {
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
      require(dSrc.shape == dTarget.shape,
        s"src shape ${dSrc.shape} mismatch dst shape ${dTarget.shape}")
      dSrc.copyTo(dTarget)
    }
  }

  // Load a list of arrays into a list of arrays specified by slices
  private[mxnet] def loadGeneralMulti(data: Seq[NDArray],
                                      targets: Seq[Array[(Int, Int, NDArray)]]): Unit = {
    for ((src, dTargets) <- data zip targets) {
      for ((start, end, dst) <- dTargets) {
        val sliced = src.slice(start, end)
        require(sliced.shape == dst.shape,
          s"src shape ${sliced.shape} mismatch dst shape ${dst.shape}")
        sliced.copyTo(dst)
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
  private[mxnet] def bindExec(sym: Symbol, ctx: Context, inputShapes: Map[String, Shape],
      paramNames: Set[String], needGrad: Boolean = false,
      grads: Set[String] = null, baseExec: Executor = null,
      sharedDataArrays: mutable.Map[String, NDArray] = null,
      inputTypes: ListMap[String, DType] = null) = {
    val (argShape, _, auxShape) = sym.inferShape(inputShapes)
    require(argShape != null)
    val inputTypesUpdate =
      if (inputTypes == null) {
        inputShapes.map { case (key, _) => (key, Base.MX_REAL_TYPE) }
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
              require(argTypes(i) == arr.dtype)
              arr.reshape(argShape(i))
            } else {
              DataParallelExecutorManager.logger.warn(
                s"bucketing: data $name has a shape ${argShape(i)}," +
                  s"which is larger than already allocated shape ${arr.shape}." +
                  "Need to re-allocate.Consider putting default_bucket_key" +
                  "to be the bucket taking the largest input for better memory sharing.")
              val zeros = NDArray.zeros(argShape(i), ctx, dtype = argTypes(i))
              // replace existing shared array because the new one is bigger
              sharedDataArrays.put(name, zeros)
              // TODO: shall we dispose the replaced array here?
              // arr.dispose()
              zeros
            }
          } else {
            val zeros = NDArray.zeros(argShape(i), ctx, dtype = argTypes(i))
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
              val gradArr = NDArray.zeros(argShape(i), ctx, dtype = argTypes(i))
              gradArrays.put(name, gradArr)
            }
            NDArray.zeros(argShape(i), ctx, dtype = argTypes(i))
          } else {
            val arr = baseExec.argDict(name)
            require(arr.shape == argShape(i))
            require(arr.dtype == argTypes(i))
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
          NDArray.zeros(s, ctx, dtype = t)
        }
      } else {
        baseExec.auxArrays.zipWithIndex.map { case (a, i) =>
          require(auxShape(i) == a.shape)
          require(auxTypes(i) == a.dtype)
          a
        }.toSeq
      }
    sym.bind(ctx = ctx, args = argArrays.toSeq, argsGrad = gradArrays.toMap, gradsReq = gradReq,
      auxStates = auxArrays, group2ctx = null, sharedExec = baseExec)
  }
}

/**
 * A group of executors living on different devices, for data parallel.
 * @param sym The network configuration.
 * @param argNames Equals `sym.list_arguments()`
 * @param paramNames Names of all trainable parameters.
 * @param ctx List of devices for training (data parallel)
 * @param slices Describes how the data parallel splits data into different devices.
 * @param providedData training data shapes
 * @param providedLabel training label shapes
 * @param sharedGroup: DataParallelExecutorGroup
 *                   An existing executor group, if to share parameters with it.
 *
 */
private class DataParallelExecutorGroup private(sym: Symbol,
                                argNames: IndexedSeq[String], paramNames: Set[String],
                                ctx: Array[Context], private val slices: Array[(Int, Int)],
                                providedData: Map[String, Shape],
                                providedLabel: Map[String, Shape],
                                sharedGroup: DataParallelExecutorGroup)  {
  // make sure the architecture is valid
  ExecutorManager.checkArguments(sym)

  private[mxnet] val sharedDataArrays: Array[mutable.Map[String, NDArray]] =
    if (sharedGroup == null) {
      ctx.map(_ => mutable.HashMap.empty[String, NDArray])
    } else {
      sharedGroup.sharedDataArrays
    }

  private[mxnet] val dataNames = providedData.map { case (k, _) => k }.toList
  private[mxnet] val labelNames = providedLabel.map { case (k, _) => k }.toList
  private[mxnet] val auxNames = sym.listAuxiliaryStates()
  private[mxnet] val paramIdx = argNames.zipWithIndex
    .filter { case (name, i) => paramNames.contains(name) }
    .map { case (name, i) => i }
  private[mxnet] val paramNamesComb = paramIdx.map(i => argNames(i)).toSet

  private[mxnet] val trainExecs: Array[Executor] =
    ctx.zipWithIndex.map { case (ctxi, i) =>
      val dataShapes =
        (providedData ++ providedLabel) map { case (name, shape) =>
          name -> (Shape(slices(i)._2 - slices(i)._1) ++ shape.slice(1, shape.length))
        }
      val sharedExec: Executor = if (sharedGroup == null) null else sharedGroup.trainExecs(i)
      ExecutorManager.bindExec(sym, ctxi, dataShapes, paramNamesComb,
        needGrad = true, baseExec = sharedExec,
        sharedDataArrays = sharedDataArrays(i))
    }

  // data structure
  private[mxnet] val dataArrays =
    dataNames.map(name =>
      trainExecs.zipWithIndex.map { case (e, i) =>
        (slices(i)._1, slices(i)._2, e.argDict(name))
      }
    ).toIndexedSeq
  private[mxnet] val labelArrays =
    labelNames.map(name =>
      trainExecs.zipWithIndex.map { case (e, i) =>
        (slices(i)._1, slices(i)._2, e.argDict(name))
      }
    ).toIndexedSeq
  private[mxnet] val paramArrays = paramIdx.map(i =>
    trainExecs.map(e => e.argArrays(i))
  ).toIndexedSeq
  private[mxnet] val gradArrays = paramIdx.map(i =>
    trainExecs.map(e => e.gradArrays(i))
  ).toIndexedSeq
  private[mxnet] val auxArrays = (0 until auxNames.length).map(i =>
    trainExecs.map(e => e.auxArrays(i))
  )

  /**
   * A group of executors living on different devices, for data parallel
   * @param sym The network configuration.
   * @param argNames Equals `sym.list_arguments()`
   * @param paramNames Names of all trainable parameters.
   * @param ctx List of devices for training (data parallel)
   * @param slices Describes how the data parallel splits data into different devices.
   * @param trainData The dataset for training.
   *                  Loading of actual data is not necessarily needed at this stage.
   * @param sharedGroup: DataParallelExecutorGroup
   *                   An existing executor group, if to share parameters with it.
   *
   */
  def this(sym: Symbol,
      argNames: IndexedSeq[String], paramNames: Set[String],
      ctx: Array[Context], slices: Array[(Int, Int)],
      trainData: DataIter,
      sharedGroup: DataParallelExecutorGroup) {
    this(sym, argNames, paramNames, ctx, slices,
      trainData.provideData, trainData.provideLabel, sharedGroup)
  }

  def this(sym: Symbol,
           argNames: IndexedSeq[String], paramNames: Set[String],
           ctx: Array[Context], slices: Array[(Int, Int)],
           trainData: DataIter) {
    this(sym, argNames, paramNames, ctx, slices,
      trainData.provideData, trainData.provideLabel, null)
  }

  /**
   * A group of executors living on different devices, for data parallel
   * @param sym The network configuration.
   * @param argNames Equals `sym.list_arguments()`
   * @param paramNames Names of all trainable parameters.
   * @param ctx List of devices for training (data parallel)
   * @param slices Describes how the data parallel splits data into different devices.
   * @param trainData The dataset for training.
   *                  Loading of actual data is not necessarily needed at this stage.
   * @param sharedGroup: DataParallelExecutorGroup
   *                   An existing executor group, if to share parameters with it.
   *
   */
  def this(sym: Symbol,
      argNames: IndexedSeq[String], paramNames: Set[String],
      ctx: Array[Context], slices: Array[(Int, Int)],
      trainData: DataBatch,
      sharedGroup: DataParallelExecutorGroup) {
    this(sym, argNames, paramNames, ctx, slices,
      trainData.provideData, trainData.provideLabel, sharedGroup)
  }

  def this(sym: Symbol,
           argNames: IndexedSeq[String], paramNames: Set[String],
           ctx: Array[Context], slices: Array[(Int, Int)],
           trainData: DataBatch) {
    this(sym, argNames, paramNames, ctx, slices,
      trainData.provideData, trainData.provideLabel, null)
  }

  // load data and labels into arrays
  def loadDataBatch(dataBatch: DataBatch): Unit = {
    ExecutorManager.loadDataMulti(dataBatch, dataArrays)
    ExecutorManager.loadLabelMulti(dataBatch, labelArrays)
  }

  // Perform a forward pass on each executor
  def forward(isTrain: Boolean = false): Unit = {
    trainExecs.foreach(_.forward(isTrain = isTrain))
  }

  // Perform a backward pass on each executor
  def backward(): Unit = {
    trainExecs.foreach(_.backward())
  }

  // Update evaluation metric with label and current outputs
  def updateMetric(metric: EvalMetric, labels: IndexedSeq[NDArray]): Unit = {
    (trainExecs zip slices).foreach { case (texec, islice) =>
      val labelsSlice = labels.map(_.slice(islice))
      metric.update(labelsSlice, texec.outputs)
    }
  }

  /**
   * Release the related executors.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    trainExecs.foreach(_.dispose())
  }
}


