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

package ml.dmlc.mxnet.spark.transformer

import java.util.UUID

import ml.dmlc.mxnet.{Context, Shape, Symbol}
import ml.dmlc.mxnet.spark.{MXNetModel, MXNetParams}
import org.apache.spark.ml.{Estimator, PredictionModel, Predictor}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.SparkContext
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}

/**
  * MXNet Training On Spark
  *
  * @author Derek Miller
  */
class MXNet extends Predictor[Vector, MXNet, MXNetModelWrap] {

  private val logger: Logger = LoggerFactory.getLogger(classOf[MXNet])
  private val params: MXNetParams = new MXNetParams
  private var _featuresCol: String = _
  private var _labelCol: String = _

  override val uid : String = UUID.randomUUID().toString

  override def train(dataset: DataFrame) : MXNetModelWrap = {
    val lps = dataset.select(_featuresCol, _labelCol).rdd
      .map(row => new LabeledPoint(row.getAs[Double](_labelCol), row.getAs[Vector](_featuresCol)))
    val mxNet = new ml.dmlc.mxnet.spark.MXNet()
      .setBatchSize(params.batchSize)
      .setLabelName(params.labelName)
      .setContext(params.context)
      .setDimension(params.dimension)
      .setNetwork(params.getNetwork)
      .setNumEpoch(params.numEpoch)
      .setNumServer(params.numServer)
      .setNumWorker(params.numWorker)
      .setLabelName(params.labelName)
      .setExecutorJars(params.jars.mkString(","))
    val fitted = mxNet.fit(lps)
    new MXNetModelWrap(lps.sparkContext, fitted, uid)
  }

  override def transformSchema(schema: StructType) : StructType = null

  override def copy(extra: ParamMap) : Estimator[MXNetModelWrap] = null

  override def setFeaturesCol(inputCol: String) : MXNet = {
    this._featuresCol = inputCol
    this
  }

  override def setLabelCol(outputCol: String) : MXNet = {
    this._labelCol = outputCol
    this
  }

  def setBatchSize(batchSize: Int): this.type = {
    params.batchSize = batchSize
    this
  }

  def setNumEpoch(numEpoch: Int): this.type = {
    params.numEpoch = numEpoch
    this
  }

  def setDimension(dimension: Shape): this.type = {
    params.dimension = dimension
    this
  }

  def setNetwork(network: Symbol): this.type = {
    params.setNetwork(network)
    this
  }

  def setContext(ctx: Array[Context]): this.type = {
    params.context = ctx
    this
  }

  def setNumWorker(numWorker: Int): this.type = {
    params.numWorker = numWorker
    this
  }

  def setNumServer(numServer: Int): this.type = {
    params.numServer = numServer
    this
  }

  def setDataName(name: String): this.type = {
    params.dataName = name
    this
  }

  def setLabelName(name: String): this.type = {
    params.labelName = name
    this
  }

  /**
    * The application (including parameter scheduler & servers)
    * will exist if it hasn't received heart beat for over timeout seconds
    * @param timeout timeout in seconds (default 300)
    */
  def setTimeout(timeout: Int): this.type = {
    params.timeout = timeout
    this
  }

  /**
    * These jars are required by the KVStores at runtime.
    * They will be uploaded and distributed to each node automatically
    * @param jars jars required by the KVStore at runtime.
    */
  def setExecutorJars(jars: String): this.type = {
    params.jars = jars.split(",|:")
    this
  }

  def setJava(java: String): this.type = {
    params.javabin = java
    this
  }

}

class MXNetModelWrap(sc: SparkContext, mxNet: MXNetModel, uuid: String)
  extends PredictionModel[Vector, MXNetModel] with Serializable with MLWritable {

  override def copy(extra: ParamMap): MXNetModel = {
    copyValues(new MXNetModelWrap(sc, mxNet, uuid)).setParent(parent)
  }

  override val uid: String = uuid

  override def predict(features: Vector) : Double = {
    val probArrays = mxNet.predict(features)
    val prob = probArrays(0)
    val arr = prob.get.toArray
    arr.max
  }

  protected[MXNetModelWrap] class MXNetModelWriter(instance: MXNetModelWrap) extends MLWriter {
    override protected def saveImpl(path: String): Unit = {
      mxNet.save(sc, path)
    }
  }

  override def write: MLWriter = new MXNetModelWriter(this)

  object MXNetModelWrap extends MLReadable[MXNetModel] {
    override def read: MLReader[MXNetModel] = new MXNetModelReader
    override def load(path: String): MXNetModel = super.load(path)
    private class MXNetModelReader extends MLReader[MXNetModel] {
      override def load(path: String): MXNetModel = MXNetModel.load(sc, path)
    }
  }

}

