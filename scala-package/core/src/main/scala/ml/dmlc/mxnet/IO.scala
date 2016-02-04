package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.IO._
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

/**
 * IO iterators for loading training & validation data
 * @author Zixuan Huang, Yizhi Liu
 */
object IO {
  type IterCreateFunc = (Map[String, String]) => DataIter
  type PackCreateFunc = (Map[String, String]) => DataPack

  private val logger = LoggerFactory.getLogger(classOf[DataIter])
  private val iterCreateFuncs: Map[String, IterCreateFunc] = _initIOModule()

  def MNISTIter: IterCreateFunc = iterCreateFuncs("MNISTIter")
  def ImageRecordIter: IterCreateFunc = iterCreateFuncs("ImageRecordIter")
  def CSVIter: IterCreateFunc = iterCreateFuncs("CSVIter")

  def MNISTPack: PackCreateFunc = createDataPack("MNISTIter")
  def ImageRecodePack: PackCreateFunc = createDataPack("ImageRecordIter")
  def CSVPack: PackCreateFunc = createDataPack("CSVIter")


  /**
   * create iterator via iterName and params
   * @param iterName name of iterator; "MNISTIter" or "ImageRecordIter"
   * @param params parameters for create iterator
   * @return
   */
  def createIterator(iterName: String, params: Map[String, String]): DataIter = {
    iterCreateFuncs(iterName)(params)
  }

  /**
    * create dataPack for iterator via itername and params
    * @param iterName name of iterator: "MNISTIter" or "ImageRecordIter"
    * @param params parameters for create iterator
    * @return
    */
  def createDataPack(iterName: String)(params: Map[String, String]): DataPack = {
    new MXDataPack(iterName, params)
  }

  /**
   * initi all IO creator Functions
   * @return
   */
  private def _initIOModule(): Map[String, IterCreateFunc] = {
    val IterCreators = new ListBuffer[DataIterCreator]
    checkCall(_LIB.mxListDataIters(IterCreators))
    IterCreators.map(_makeIOIterator).toMap
  }

  private def _makeIOIterator(handle: DataIterCreator): (String, IterCreateFunc) = {
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
   * @param handle
   * @param params
   * @return
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
  private def initData(data: List[NDArray], allowEmpty: Boolean, defaultName: String) = {
    require(data != null || allowEmpty)
    // TODO
  }
}


/**
 * class batch of data
 * @param data
 * @param label
 * @param index
 * @param pad
 */
case class DataBatch(data: IndexedSeq[NDArray],
                     label: IndexedSeq[NDArray],
                     index: IndexedSeq[Long],
                     pad: Int)


/**
 * DataIter object in mxnet.
 */
abstract class DataIter(val batchSize: Int = 0) extends Iterator[DataBatch] {
  /**
   * reset the iterator
   */
  def reset(): Unit

  /**
   * Iterate to next batch
   * @return whether the move is successful
   */
  def iterNext(): Boolean

  /**
   * get next data batch from iterator
   * @return
   */
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
   * get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  def getPad(): Int

  /**
   * the index of current batch
   * @return
   */
  def getIndex(): IndexedSeq[Long]

  // The name and shape of data provided by this iterator
  def provideData: Map[String, Shape]

  // The name and shape of label provided by this iterator
  def provideLabel: Map[String, Shape]
}

abstract class DataPack() extends Iterable[DataBatch] {
  /**
    * get data iterator
    * @return DataIter
    */
  def iterator: DataIter
}


/**
  * DataIter built in MXNet.
  * @param handle the handle to the underlying C++ Data Iterator
  */
// scalastyle:off finalize
class MXDataIter(private[mxnet] val handle: DataIterHandle,
                 private val dataName: String = "data",
                 private val labelName: String = "label") extends DataIter {
  private val logger = LoggerFactory.getLogger(classOf[MXDataIter])

  // use currentBatch to implement hasNext
  // (may be this is not the best way to do this work,
  // fix me if any better way found)
  private var currentBatch: DataBatch = null
  iterNext()
  private val data = currentBatch.data(0)
  private val label = currentBatch.label(0)
  reset()

  // properties
  val _provideData: Map[String, Shape] = Map(dataName -> data.shape)
  val _provideLabel: Map[String, Shape] = Map(labelName -> label.shape)
  override val batchSize = data.shape(0)

  override def finalize(): Unit = {
    checkCall(_LIB.mxDataIterFree(handle))
  }

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    // TODO: self._debug_at_begin = True
    currentBatch = null
    checkCall(_LIB.mxDataIterBeforeFirst(handle))
  }

  override def next(): DataBatch = {
    // TODO
    // if self._debug_skip_load and not self._debug_at_begin:
    //   return DataBatch(data =[self.getdata()], label =[self.getlabel()],
    //                    pad = self.getpad(), index = self.getindex())
    if (currentBatch == null) {
      iterNext()
    }

    if (currentBatch != null) {
      val batch = currentBatch
      currentBatch = null
      batch
    } else {
        // TODO raise StopIteration
        null
    }
  }

  /**
   * Iterate to next batch
   * @return whether the move is successful
   */
  override def iterNext(): Boolean = {
      val next = new RefInt
      checkCall(_LIB.mxDataIterNext(handle, next))
      currentBatch = null
      if (next.value > 0) {
        currentBatch = new DataBatch(data = getData(), label = getLabel(),
                                     index = getIndex(), pad = getPad())
      }
      next.value > 0
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    val out = new NDArrayHandleRef
    checkCall(_LIB.mxDataIterGetData(handle, out))
    IndexedSeq(new NDArray(out.value, writable = false))
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    val out = new NDArrayHandleRef
    checkCall(_LIB.mxDataIterGetLabel(handle, out))
    IndexedSeq(new NDArray(out.value, writable = false))
  }

  /**
   * the index of current batch
   * @return
   */
  override def getIndex(): IndexedSeq[Long] = {
    val outIndex = new ListBuffer[Long]
    val outSize = new RefLong
    checkCall(_LIB.mxDataIterGetIndex(handle, outIndex, outSize))
    outIndex.toIndexedSeq
  }

  /**
   * get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): MXUint = {
    val out = new MXUintRef
    checkCall(_LIB.mxDataIterGetPadNum(handle, out))
    out.value
  }

  // The name and shape of data provided by this iterator
  override def provideData: Map[String, Shape] = _provideData

  // The name and shape of label provided by this iterator
  override def provideLabel: Map[String, Shape] = _provideLabel

  override def hasNext: Boolean = {
    if (currentBatch != null) {
      true
    } else {
      iterNext()
    }
  }
}

// scalastyle:on finalize
class MXDataPack(val iterName: String,
                 val params: Map[String, String]) extends DataPack {
  /**
    * get data iterator
    * @return DataIter
    */
  override def iterator: DataIter = {
    createIterator(iterName, params)
  }
}

/**
  * Base class for prefetching iterators. Takes one or more DataIters
  * (or any class with "reset" and "read" methods) and combine them with
  * prefetching.
  * @param iters list of DataIters
  * @param dataNames
  * @param labelNames
  */
class PrefetchingIter(val iters: List[DataIter],
                      val dataNames: Map[String, String] = null,
                      val labelNames: Map[String, String] = null) extends DataIter {
  /**
   * reset the iterator
   */
  override def reset(): Unit = ???

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = ???

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = ???

  /**
   * the index of current batch
   * @return
   */
  override def getIndex(): IndexedSeq[Long] = ???

  /**
   * Iterate to next batch
   * @return whether the move is successful
   */
  override def iterNext(): Boolean = ???

  /**
   * get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): Int = ???

  // The name and shape of data provided by this iterator
  override def provideData: Map[String, Shape] = ???

  // The name and shape of label provided by this iterator
  override def provideLabel: Map[String, Shape] = ???

  override def hasNext: Boolean = ???
}

/**
 * TODO
 * NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.
 * @param data NDArrayIter supports single or multiple data and label.
 * @param label Same as data, but is not fed to the model during testing.
 * @param batchSize Batch Size
 * @param shuffle Whether to shuffle the data
 * @param lastBatchHandle "pad", "discard" or "roll_over". How to handle the last batch
 * @note
 * This iterator will pad, discard or roll over the last batch if
 * the size of data does not match batch_size. Roll over is intended
 * for training and can cause problems if used for prediction.
 */
class NDArrayIter(data: NDArray, label: NDArray = null,
                  batchSize: Int = 1, shuffle: Boolean = false,
                  lastBatchHandle: String = "pad") extends DataIter(batchSize) {
  /**
   * reset the iterator
   */
  override def reset(): Unit = ???

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = ???

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = ???

  /**
   * the index of current batch
   * @return
   */
  override def getIndex(): IndexedSeq[Long] = ???

  /**
   * Iterate to next batch
   * @return whether the move is successful
   */
  override def iterNext(): Boolean = ???

  /**
   * get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): MXUint = ???

  // The name and shape of data provided by this iterator
  override def provideData: Map[String, Shape] = ???

  // The name and shape of label provided by this iterator
  override def provideLabel: Map[String, Shape] = ???

  override def hasNext: Boolean = ???
}
