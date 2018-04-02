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

package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.spark.io.LabeledPointIter

import org.slf4j.{Logger, LoggerFactory}

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

class MXNet extends Serializable {

  class MXNetControllingThread(
      schedulerIP: String,
      schedulerPort: Int,
      sparkContext: SparkContext,
      triggerOfComponent: (String, Int, SparkContext) => Unit) extends Thread {
    override def run() {
      triggerOfComponent(schedulerIP, schedulerPort, sparkContext)
    }
  }

  private val logger: Logger = LoggerFactory.getLogger(classOf[MXNet])
  private val params: MXNetParams = new MXNetParams

  @transient private var psServerThread: MXNetControllingThread = _
  @transient private var psSchedulerThread: MXNetControllingThread = _

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

  private def startPSServers(
      schedulerIP: String,
      schedulerPort: Int,
      sc: SparkContext) = {
    def startPSServersInner(
        schedulerIP: String,
        schedulerPort: Int,
        sc: SparkContext): Unit = {
      sc.parallelize(1 to params.numServer, params.numServer).foreachPartition { p =>
          logger.info("Starting server ...")
          val server = new ParameterServer(params.runtimeClasspath,
            role = "server",
            rootUri = schedulerIP, rootPort = schedulerPort,
            numServer = params.numServer,
            numWorker = params.numWorker,
            timeout = params.timeout,
            java = params.javabin)
          val exitCode = server.startProcess()
          require(exitCode == 0, s"ps server process quit with exit code $exitCode")
        }
    }
    psServerThread = new MXNetControllingThread(schedulerIP, schedulerPort, sc, startPSServersInner)
    psServerThread.start()
  }

  private def startPSScheduler(
      schedulerIP: String,
      schedulerPort: Int,
      sc: SparkContext) = {
    def startPSSchedulerInner(
        schedulerIP: String,
        schedulerPort: Int,
        sc: SparkContext): Unit = {
      // TODO: check ip & port available
      logger.info("Starting scheduler on {}:{}", schedulerIP, schedulerPort)
      val scheduler = new ParameterServer(params.runtimeClasspath, role = "scheduler",
        rootUri = schedulerIP, rootPort = schedulerPort,
        numServer = params.numServer, numWorker = params.numWorker,
        timeout = params.timeout, java = params.javabin)
      val exitCode = scheduler.startProcess()
      require(exitCode == 0, s"Failed to start ps scheduler process with exit code $exitCode")
    }
    psSchedulerThread = new MXNetControllingThread(schedulerIP, schedulerPort, sc,
      startPSSchedulerInner)
    psSchedulerThread.start()
  }

  private def setFeedForwardModel(
      optimizer: Optimizer,
      numExamples: Int,
      kv: KVStore,
      inputInPartition: LabeledPointIter): FeedForward = {
    logger.debug("Define model")
    val model = new FeedForward(ctx = params.context,
      symbol = params.getNetwork,
      numEpoch = params.numEpoch,
      optimizer = optimizer,
      initializer = new Xavier(factorType = "in", magnitude = 2.34f),
      argParams = null,
      auxParams = null,
      beginEpoch = 0,
      epochSize = numExamples / params.batchSize / kv.numWorkers)
    logger.info("Start training ...")
    model.fit(trainData = inputInPartition,
      evalData = null,
      evalMetric = new Accuracy(),
      kvStore = kv)
    model
  }

  private def setupKVStore(schedulerIP: String, schedulerPort: Int): KVStore = {
    KVStoreServer.init(ParameterServer.buildEnv(role = "worker",
      rootUri = schedulerIP, rootPort = schedulerPort,
      numServer = params.numServer,
      numWorker = params.numWorker))
    val kv = KVStore.create("dist_async")
    kv.setBarrierBeforeExit(false)
    kv
  }

  private def reclaimResources(dataIter: LabeledPointIter, kv: KVStore): Unit = {
    dataIter.dispose()
    kv.setBarrierBeforeExit(true)
    kv.dispose()
  }

  private def trainModel(
      trainData: RDD[LabeledPoint],
      schedulerIP: String,
      schedulerPort: Int): MXNetModel = {
    val job = trainData.mapPartitions { partition =>
      val dataIter = new LabeledPointIter(
        partition, params.dimension,
        params.batchSize,
        dataName = params.dataName,
        labelName = params.labelName)
      // TODO: more nature way to get the # of examples?
      var numExamples = 0
      while (dataIter.hasNext) {
        val dataBatch = dataIter.next()
        numExamples += dataBatch.label.head.shape(0)
      }
      logger.debug("Number of samples: {}", numExamples)
      dataIter.reset()

      logger.info("Launching worker ...")
      logger.info("Batch {}", params.batchSize)
      // give enough time for ps-lite to detect the dead nodes
      Thread.sleep(20000)
      val kv = setupKVStore(schedulerIP, schedulerPort)
      val optimizer = new SGD(learningRate = 0.01f, momentum = 0.9f, wd = 0.00001f)
      val model = setFeedForwardModel(optimizer, numExamples, kv, dataIter)
      logger.info("Training finished, waiting for other workers ...")
      reclaimResources(dataIter, kv)
      Iterator(new MXNetModel(
        model, params.dimension, params.batchSize,
        dataName = params.dataName, labelName = params.labelName))
    }.cache()
    // force job to run
    job.foreachPartition(() => _)
    job.first()
  }

  def fit(data: RDD[LabeledPoint]): MXNetModel = {
    val sc = data.context
    // distribute native jars
    params.jars.foreach(jar => sc.addFile(jar))
    val trainData = {
      if (params.numWorker != data.partitions.length) {
        logger.info("repartitioning training set to {} partitions", params.numWorker)
        data.repartition(params.numWorker)
      } else {
        data
      }
    }
    val schedulerIP = utils.Network.ipAddress
    val schedulerPort = utils.Network.availablePort
    startPSScheduler(schedulerIP, schedulerPort, sc)
    startPSServers(schedulerIP, schedulerPort, sc)
    val mxModel = trainModel(trainData, schedulerIP, schedulerPort)
    logger.info("Waiting for scheduler ...")
    psSchedulerThread.join()
    psServerThread.join()
    mxModel
  }
}
