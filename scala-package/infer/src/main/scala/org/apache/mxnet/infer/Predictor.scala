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

package org.apache.mxnet.infer

import org.apache.mxnet.MX_PRIMITIVES.MX_PRIMITIVE_TYPE
import org.apache.mxnet.io.NDArrayIter
import org.apache.mxnet._
import org.apache.mxnet.module.Module

import scala.collection.mutable.ListBuffer
import scala.util.Try
import org.slf4j.LoggerFactory


/**
 * Base Trait for MXNet Predictor classes.
 */
private[infer] trait PredictBase {

  /**
    * Converts indexed sequences of 1-D array to NDArrays.
    * This method will take input as IndexedSeq one dimensional arrays and creates the
    * NDArray needed for inference. The array will be reshaped based on the input descriptors.
    * @tparam T The Scala equivalent of the DType used for the input array and return value
    * @param input An Indexed Sequence of a one-dimensional array of datatype
    *              Float or Double
    *              An IndexedSequence is needed when the model has more than one input.
    * @return      Indexed sequence array of outputs
    */
  def predict[@specialized (Base.MX_PRIMITIVES) T](input: IndexedSeq[Array[T]])
  : IndexedSeq[Array[T]]

  /**
    * Predict using NDArray as input.
    * <p>
    * This method is useful when the input is a batch of data
    * or when multiple operations on the input have to performed.
    * Note: User is responsible for managing allocation/deallocation of NDArrays.
    * @param input             IndexedSequence NDArrays.
    * @return                  Output of predictions as NDArrays.
    */
  def predictWithNDArray(input: IndexedSeq[NDArray]): IndexedSeq[NDArray]

  /**
    * Get model output shapes.
    * @return   model output shapes.
    */
  def outputShapes: IndexedSeq[(String, Shape)]
}

/**
 * Implementation of prediction routines.
 *
 * @param modelPathPrefix     Path prefix from where to load the model artifacts.
 *                            These include the symbol, parameters, and synset.txt
 *                            Example: file://model-dir/resnet-152 (containing
 *                            resnet-152-symbol.json, resnet-152-0000.params, and synset.txt).
 * @param inputDescriptors    Descriptors defining the input node names, shape,
 *                            layout and type parameters
 *                            <p>Note: If the input Descriptors is missing batchSize
 *                            ('N' in layout), a batchSize of 1 is assumed for the model.
 * @param contexts            Device contexts on which you want to run inference; defaults to CPU
 * @param epoch               Model epoch to load; defaults to 0

 */
class Predictor(modelPathPrefix: String,
                protected val inputDescriptors: IndexedSeq[DataDesc],
                protected val contexts: Array[Context] = Context.cpu(),
                protected val epoch: Option[Int] = Some(0))
                extends PredictBase {

  private val logger = LoggerFactory.getLogger(classOf[Predictor])

  /*
    By setting -Dmxnet.disableShapeCheck=true would disable the data Shape
    Check of the predictor. Some model may allow different lens of the data
    such as Seq2Seq, however there maybe risk of crashes if the lens beyond
    the acceptable range of the model
   */
  private val traceProperty = "mxnet.disableShapeCheck"
  private lazy val shapeCheckDisabled = {
    val value = Try(System.getProperty(traceProperty).toBoolean).getOrElse(false)
    if (value) {
      logger.warn("Shape check is disabled (property {} is set)", traceProperty)
    }
    value
  }

  require(inputDescriptors.head.layout.size != 0, "layout size should not be zero")

  protected[infer] var batchIndex = inputDescriptors(0).layout.indexOf('N')
  protected[infer] var batchSize = if (batchIndex != -1) inputDescriptors(0).shape(batchIndex)
    else 1

  protected[infer] var iDescriptors = inputDescriptors

  inputDescriptors.foreach((f: DataDesc) => require(f.layout.indexOf('N') == batchIndex,
    "batch size should be in the same index for all inputs"))

  if (batchIndex != -1) {
    inputDescriptors.foreach((f: DataDesc) => require(f.shape(batchIndex) == batchSize,
      "batch size should be same for all inputs"))
  } else {
    // Note: this is assuming that the input needs a batch
    logger.warn("InputDescriptor does not have batchSize, using 1 as the default batchSize")
    iDescriptors = inputDescriptors.map((f: DataDesc) => new DataDesc(f.name,
      Shape(1 +: f.shape.toVector), f.dtype, 'N' +: f.layout))
    batchIndex = 1
  }

  protected[infer] val mxNetHandler = MXNetHandler()

  protected[infer] val mod = loadModule()

  override def outputShapes: IndexedSeq[(String, Shape)] = mod.outputShapes

  /**
   * Takes input as IndexedSeq one dimensional arrays and creates the NDArray needed for inference
   * The array will be reshaped based on the input descriptors.
   *
   * @param input:            An IndexedSequence of a one-dimensional array
    *                         of data type Float or Double.
                              An IndexedSequence is needed when the model has more than one input.
   * @return                  Indexed sequence array of outputs
   */
  override def predict[@specialized (Base.MX_PRIMITIVES) T](input: IndexedSeq[Array[T]])
  : IndexedSeq[Array[T]] = {
    require(input.length == inputDescriptors.length,
      s"number of inputs provided: ${input.length} does not match number of inputs " +
        s"in inputDescriptors: ${inputDescriptors.length}")

    for((i, d) <- input.zip(inputDescriptors)) {
      require(i.length == d.shape.product / batchSize,
        s"number of elements:${i.length} in the input does not match the shape:" +
          s"${d.shape.toString()}")
    }

    // Infer the dtype of input and call relevant method
    val result = input(0)(0) match {
      case d: Double => predictImpl(input.asInstanceOf[IndexedSeq[Array[Double]]])
      case _ => predictImpl(input.asInstanceOf[IndexedSeq[Array[Float]]])
    }

    result.asInstanceOf[IndexedSeq[Array[T]]]
  }

  private def predictImpl[B, A <: MX_PRIMITIVE_TYPE]
  (input: IndexedSeq[Array[B]])(implicit ev: B => A)
  : IndexedSeq[Array[B]] = {

    var inputND: ListBuffer[NDArray] = ListBuffer.empty[NDArray]

    for((i, d) <- input.zip(inputDescriptors)) {
      val shape = d.shape.toVector.patch(from = batchIndex, patch = Vector(1), replaced = 1)
      if (d.dtype == DType.Float64) {
        inputND += mxNetHandler.execute(NDArray.array(i.asInstanceOf[Array[Double]], Shape(shape)))
      }
      else {
        inputND += mxNetHandler.execute(NDArray.array(i.asInstanceOf[Array[Float]], Shape(shape)))
      }
    }

    // rebind with batchsize 1
    if (batchSize != 1) {
      val desc = iDescriptors.map((f : DataDesc) => new DataDesc(f.name,
        Shape(f.shape.toVector.patch(batchIndex, Vector(1), 1)), f.dtype, f.layout) )
      mxNetHandler.execute(mod.bind(desc, forceRebind = true,
        forTraining = false))
    }

    val resultND = mxNetHandler.execute(mod.predict(new NDArrayIter(
      inputND.toIndexedSeq, dataBatchSize = 1)))

    val result =
      resultND.map((f : NDArray) => if (f.dtype == DType.Float64) f.toFloat64Array else f.toArray)

    mxNetHandler.execute(inputND.foreach(_.dispose))
    mxNetHandler.execute(resultND.foreach(_.dispose))

    // rebind to batchSize
    if (batchSize != 1) {
      mxNetHandler.execute(mod.bind(inputDescriptors, forTraining = false, forceRebind = true))
    }

    result.asInstanceOf[IndexedSeq[Array[B]]]
  }



  /**
   * Predict using NDArray as input
   * This method is useful when the input is a batch of data
   * Note: User is responsible for managing allocation/deallocation of input/output NDArrays.
   *
   * @param inputBatch        IndexedSequence NDArrays
   * @return                  Output of predictions as NDArrays
   */
  override def predictWithNDArray(inputBatch: IndexedSeq[NDArray]): IndexedSeq[NDArray] = {

    require(inputBatch.length == inputDescriptors.length,
      s"number of inputs provided: ${inputBatch.length} do not match number " +
        s"of inputs in inputDescriptors: ${inputDescriptors.length}")

    // Shape validation, remove this when backend throws better error messages.
    for((i, d) <- inputBatch.zip(iDescriptors)) {
       require(inputBatch(0).shape(batchIndex) == i.shape(batchIndex),
         "All inputs should be of same batch size")
      if (!shapeCheckDisabled) {
        require(i.shape.drop(batchIndex + 1) == d.shape.drop(batchIndex + 1),
          s"Input Data Shape: ${i.shape} should match the inputDescriptor " +
            s"shape: ${d.shape} except batchSize")
      }
    }

    val inputBatchSize = inputBatch(0).shape(batchIndex)

    // rebind with the new batchSize
    if (batchSize != inputBatchSize) {
      logger.info(s"Latency increased due to batchSize mismatch $batchSize vs $inputBatchSize")
      val desc = inputBatch.zip(iDescriptors).map(f => new DataDesc(f._2.name,
        f._1.shape, f._2.dtype, f._2.layout))
      mxNetHandler.execute(mod.bind(desc, forceRebind = true,
        forTraining = false))
    }

    val resultND = mxNetHandler.execute(mod.predict(new NDArrayIter(
      inputBatch, dataBatchSize = inputBatchSize)))

    if (batchSize != inputBatchSize) {
      mxNetHandler.execute(mod.bind(iDescriptors, forceRebind = true,
        forTraining = false))
    }
    resultND
  }

  /**
    * Creates the module backing the Predictor with the same path, epoch, contexts, and inputs
    * @return The Module
    */
  private[infer] def loadModule(): Module = {
    val mod = mxNetHandler.execute(Module.loadCheckpoint(modelPathPrefix, epoch.get,
      contexts = contexts, dataNames = inputDescriptors.map(desc => desc.name)))
    mxNetHandler.execute(mod.bind(inputDescriptors, forTraining = false))
    mod
  }
}
