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

import org.apache.mxnet.io.NDArrayIter
import org.apache.mxnet.{Context, DataDesc, NDArray, Shape}
import org.apache.mxnet.module.Module

import scala.collection.mutable.ListBuffer
import org.slf4j.LoggerFactory

/**
 * Base Trait for MXNet Predictor classes.
 */
private[infer] trait PredictBase {

  /**
   * Converts indexed sequences of 1-D array to NDArrays.
   * <p>
   * This method will take input as IndexedSeq one dimensional arrays and creates the
   * NDArray needed for inference. The array will be reshaped based on the input descriptors.
   * @param input:            An IndexedSequence of a one-dimensional array.
                              An IndexedSequence is needed when the model has more than one input.
   * @return                  Indexed sequence array of outputs
   */
  def predict(input: IndexedSeq[Array[Float]]): IndexedSeq[Array[Float]]

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

  /**
   * Takes input as IndexedSeq one dimensional arrays and creates the NDArray needed for inference
   * The array will be reshaped based on the input descriptors.
   *
   * @param input:            An IndexedSequence of a one-dimensional array.
                              An IndexedSequence is needed when the model has more than one input.
   * @return                  Indexed sequence array of outputs
   */
  override def predict(input: IndexedSeq[Array[Float]])
  : IndexedSeq[Array[Float]] = {

    require(input.length == inputDescriptors.length, "number of inputs provided: %d" +
      " does not match number of inputs in inputDescriptors: %d".format(input.length,
        inputDescriptors.length))

    for((i, d) <- input.zip(inputDescriptors)) {
      require (i.length == d.shape.product/batchSize, "number of elements:" +
        " %d in the input does not match the shape:%s".format( i.length, d.shape.toString()))
    }
    var inputND: ListBuffer[NDArray] = ListBuffer.empty[NDArray]

    for((i, d) <- input.zip(inputDescriptors)) {
      val shape = d.shape.toVector.patch(from = batchIndex, patch = Vector(1), replaced = 1)

      inputND += mxNetHandler.execute(NDArray.array(i, Shape(shape)))
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

    val result = resultND.map((f : NDArray) => f.toArray)

    mxNetHandler.execute(inputND.foreach(_.dispose))
    mxNetHandler.execute(resultND.foreach(_.dispose))

    // rebind to batchSize
    if (batchSize != 1) {
      mxNetHandler.execute(mod.bind(inputDescriptors, forTraining = false, forceRebind = true))
    }

    result
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

    require(inputBatch.length == inputDescriptors.length, "number of inputs provided: %d" +
      " do not match number of inputs in inputDescriptors: %d".format(inputBatch.length,
        inputDescriptors.length))

    // Shape validation, remove this when backend throws better error messages.
    for((i, d) <- inputBatch.zip(iDescriptors)) {
       require(inputBatch(0).shape(batchIndex) == i.shape(batchIndex),
         "All inputs should be of same batch size")
      require(i.shape.drop(batchIndex + 1) == d.shape.drop(batchIndex + 1),
        "Input Data Shape: %s should match the inputDescriptor shape: %s except batchSize".format(
          i.shape.toString, d.shape.toString))
    }

    val inputBatchSize = inputBatch(0).shape(batchIndex)

    // rebind with the new batchSize
    if (batchSize != inputBatchSize) {
      val desc = iDescriptors.map((f : DataDesc) => new DataDesc(f.name,
        Shape(f.shape.toVector.patch(batchIndex, Vector(inputBatchSize), 1)), f.dtype, f.layout) )
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

  private[infer] def loadModule(): Module = {
    val mod = mxNetHandler.execute(Module.loadCheckpoint(modelPathPrefix, epoch.get,
      contexts = contexts))
    mxNetHandler.execute(mod.bind(inputDescriptors, forTraining = false))
    mod
  }
}
