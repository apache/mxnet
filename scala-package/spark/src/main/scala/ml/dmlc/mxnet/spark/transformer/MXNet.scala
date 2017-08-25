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

import ml.dmlc.mxnet.spark.{MXNetModel, MXNetParams}
import ml.dmlc.mxnet.{Context, Shape, Symbol}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame
import org.slf4j.{Logger, LoggerFactory}


class MXNet extends Predictor[Vector, MXNet, MXNetModelWrap] {

  private val logger: Logger = LoggerFactory.getLogger(classOf[MXNet])
  private val p: MXNetParams = new MXNetParams
  private var _featuresCol: String = _
  private var _labelCol: String = _

  override val uid = UUID.randomUUID().toString

  override def train(dataset: DataFrame) : MXNetModelWrap = {
    val lps = dataset.select(getFeaturesCol, getLabelCol).rdd
      .map(row => new LabeledPoint(row.getAs[Double](getLabelCol),
        row.getAs[Vector](getFeaturesCol)))
    val mxNet = new ml.dmlc.mxnet.spark.MXNet()
      .setBatchSize(p.batchSize)
      .setLabelName(p.labelName)
      .setContext(p.context)
      .setDimension(p.dimension)
      .setNetwork(p.getNetwork)
      .setNumEpoch(p.numEpoch)
      .setNumServer(p.numServer)
      .setNumWorker(p.numWorker)
      .setExecutorJars(p.jars.mkString(","))
    val fitted = mxNet.fit(lps)
    new MXNetModelWrap(lps.sparkContext, fitted, uid)
  }

  override def copy(extra: ParamMap) : MXNet = defaultCopy(extra)

  def setBatchSize(batchSize: Int): this.type = {
    p.batchSize = batchSize
    this
  }

  def setNumEpoch(numEpoch: Int): this.type = {
    p.numEpoch = numEpoch
    this
  }

  def setDimension(dimension: Shape): this.type = {
    p.dimension = dimension
    this
  }

  def setNetwork(network: Symbol): this.type = {
    p.setNetwork(network)
    this
  }

  def setContext(ctx: Array[Context]): this.type = {
    p.context = ctx
    this
  }

  def setNumWorker(numWorker: Int): this.type = {
    p.numWorker = numWorker
    this
  }

  def setNumServer(numServer: Int): this.type = {
    p.numServer = numServer
    this
  }

  def setDataName(name: String): this.type = {
    p.dataName = name
    this
  }

  def setLabelName(name: String): this.type = {
    p.labelName = name
    this
  }

  /**
    * The application (including parameter scheduler & servers)
    * will exist if it hasn't received heart beat for over timeout seconds
    * @param timeout timeout in seconds (default 300)
    */
  def setTimeout(timeout: Int): this.type = {
    p.timeout = timeout
    this
  }

  /**
    * These jars are required by the KVStores at runtime.
    * They will be uploaded and distributed to each node automatically
    * @param jars jars required by the KVStore at runtime.
    */
  def setExecutorJars(jars: String): this.type = {
    p.jars = jars.split(",|:")
    this
  }

  def setJava(java: String): this.type = {
    p.javabin = java
    this
  }

}

class MXNetModelWrap(sc: SparkContext, mxNet: MXNetModel, uuid: String)
  extends PredictionModel[Vector, MXNetModelWrap] with Serializable with MLWritable {

  override def copy(extra: ParamMap): MXNetModelWrap = {
    copyValues(new MXNetModelWrap(sc, mxNet, uuid)).setParent(parent)
  }

  override val uid: String = uuid

  override def predict(features: Vector) : Double = {
    val probArrays = mxNet.predict(features)
    val prob = probArrays(0)
    val arr = prob.get.toArray
    if (arr.length == 1) {
      arr(0)
    } else {
      arr.indexOf(arr.max)
    }

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
