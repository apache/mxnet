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

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.DType.DType
import ml.dmlc.mxnet.io.{MXDataPack, MXDataIter}
import org.slf4j.LoggerFactory

import scala.collection.immutable.ListMap
import scala.collection.mutable.ListBuffer

/**
 * IO iterators for loading training & validation data
 */
object IO {
  type IterCreateFunc = (Map[String, String]) => DataIter
  type PackCreateFunc = (Map[String, String]) => DataPack

  private val logger = LoggerFactory.getLogger(classOf[DataIter])
  private val iterCreateFuncs: Map[String, IterCreateFunc] = initIOModule()

  def MNISTIter: IterCreateFunc = iterCreateFuncs("MNISTIter")

  def ImageRecordIter: IterCreateFunc = iterCreateFuncs("ImageRecordIter")

  def CSVIter: IterCreateFunc = iterCreateFuncs("CSVIter")

  def MNISTPack: PackCreateFunc = createMXDataPack("MNISTIter")

  def ImageRecodePack: PackCreateFunc = createMXDataPack("ImageRecordIter")

  def CSVPack: PackCreateFunc = createMXDataPack("CSVIter")


  /**
   * create iterator via iterName and params
   * @param iterName name of iterator; "MNISTIter" or "ImageRecordIter"
   * @param params parameters for create iterator
   * @return created data iterator
   */
  def createIterator(iterName: String, params: Map[String, String]): DataIter = {
    iterCreateFuncs(iterName)(params)
  }

  /**
   * create dataPack for iterator via itername and params
   * @param iterName name of iterator: "MNISTIter" or "ImageRecordIter"
   * @param params parameters for create iterator
   * @return created dataPack
   */
  def createMXDataPack(iterName: String)(params: Map[String, String]): DataPack = {
    new MXDataPack(iterName, params)
  }

  /**
   * initialize all IO creator Functions
   * @return Map from name to iter creator function
   */
  private def initIOModule(): Map[String, IterCreateFunc] = {
    val IterCreators = new ListBuffer[DataIterCreator]
    checkCall(_LIB.mxListDataIters(IterCreators))
    IterCreators.map(makeIOIterator).toMap
  }

  private def makeIOIterator(handle: DataIterCreator): (String, IterCreateFunc) = {
    val name = new RefString
    val desc = new RefString
    val argNames = new ListBuffer[String]
    val argTypes = new ListBuffer[String]
    val argDescs = new ListBuffer[String]
    checkCall(_LIB.mxDataIterGetIterInfo(handle, name, desc, argNames, argTypes, argDescs))
    val paramStr = Base.ctypes2docstring(argNames, argTypes, argDescs)
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
    logger.debug(docStr)
    (name.value, creator(handle))
  }

  /**
   * DataIter creator
   * @param handle native memory ptr for the iterator
   * @param params parameter passed to the iterator
   * @return created DataIter
   */
  private def creator(handle: DataIterCreator)(
    params: Map[String, String]): DataIter = {
    val out = new DataIterHandleRef
    val keys = params.keys.toArray
    val vals = params.values.toArray
    checkCall(_LIB.mxDataIterCreateIter(handle, keys, vals, out))
    val dataName = params.getOrElse("data_name", "data")
    val labelName = params.getOrElse("label_name", "label")
    new MXDataIter(out.value, dataName, labelName)
  }

  // Convert data into canonical form.
  private[mxnet] def initData(data: IndexedSeq[NDArray],
                              allowEmpty: Boolean,
                              defaultName: String): IndexedSeq[(String, NDArray)] = {
    require(data != null)
    require(data != IndexedSeq.empty || allowEmpty)
    if (data == IndexedSeq.empty) {
      IndexedSeq()
    } else if (data.length == 1) {
      IndexedSeq((defaultName, data(0)))
    } else {
      data.zipWithIndex.map(item => {
        (defaultName + "_" + item._2, item._1)
      }).toIndexedSeq
    }
  }
}

/**
 * class batch of data
 */
class DataBatch(val data: IndexedSeq[NDArray],
                val label: IndexedSeq[NDArray],
                val index: IndexedSeq[Long],
                val pad: Int,
                // the key for the bucket that should be used for this batch,
                // for bucketing io only
                val bucketKey: AnyRef = null,
                // use ListMap to indicate the order of data/label loading
                // (must match the order of input data/label)
                private val providedData: ListMap[String, Shape] = null,
                private val providedLabel: ListMap[String, Shape] = null) {
  /**
   * Dispose its data and labels
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (data != null) {
      data.foreach(arr => if (arr != null) arr.dispose())
    }
    if (label != null) {
      label.foreach(arr => if (arr != null) arr.dispose())
    }
  }

  // The name and shape of data
  def provideData: ListMap[String, Shape] = providedData

  // The name and shape of label
  def provideLabel: ListMap[String, Shape] = providedLabel
}

/**
 * DataIter object in mxnet.
 */
abstract class DataIter extends Iterator[DataBatch] {
  /**
   * reset the iterator
   */
  def reset(): Unit

  def batchSize: Int

  /**
   * get next data batch from iterator
   * @return
   */
  @throws(classOf[NoSuchElementException])
  def next(): DataBatch = {
    new DataBatch(getData(), getLabel(), getIndex(), getPad())
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  def getData(): IndexedSeq[NDArray]

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  def getLabel(): IndexedSeq[NDArray]

  /**
   * Get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  def getPad(): Int

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  def getIndex(): IndexedSeq[Long]

  // The name and shape of data provided by this iterator
  def provideData: ListMap[String, Shape]

  // The name and shape of label provided by this iterator
  def provideLabel: ListMap[String, Shape]

  // For bucketing io only
  // The bucket key for the default symbol.
  def defaultBucketKey: AnyRef = null
}

/**
 * pack of DataIter, use as Iterable class
 */
abstract class DataPack() extends Iterable[DataBatch] {
  /**
   * get data iterator
   * @return DataIter
   */
  def iterator: DataIter
}

// Named data desc description contains name, shape, type and other extended attributes.
case class DataDesc(name: String, shape: Shape,
                    dtype: DType = Base.MX_REAL_TYPE, layout: String = "NCHW") {
  override def toString(): String = {
    s"DataDesc[$name,$shape,$dtype,$layout]"
  }
}

object DataDesc {
  /**
   * Get the dimension that corresponds to the batch size.
   * @param layout layout string. For example, "NCHW".
   * @return An axis indicating the batch_size dimension. When data-parallelism is used,
   *         the data will be automatically split and concatenate along the batch_size dimension.
   *         Axis can be -1, which means the whole array will be copied
   *         for each data-parallelism device.
   */
  def getBatchAxis(layout: Option[String]): Int = {
    layout.map(_.indexOf('N')).getOrElse(0)
  }

  implicit def ListMap2Descs(shapes: ListMap[String, Shape]): IndexedSeq[DataDesc] = {
    shapes.map { case (k, s) => new DataDesc(k, s) }.toIndexedSeq
  }
}
