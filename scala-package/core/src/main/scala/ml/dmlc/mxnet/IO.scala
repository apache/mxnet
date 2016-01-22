package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

/**
 * IO iterators for loading training & validation data
 * @author Zixuan Huang, Yizhi Liu
 */
object IO {
  type IterCreateFunc = (Map[String, String]) => DataIter

  private val logger = LoggerFactory.getLogger(classOf[DataIter])
  private val iterCreateFuncs: Map[String, IterCreateFunc] = _initIOModule()

  def MNISTIter: IterCreateFunc = iterCreateFuncs("MNISTIter")

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
   *
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
    new MXDataIter(out.value)
  }

  // Convert data into canonical form.
  private def initData(data: NDArray, allowEmpty: Boolean, defaultName: String) = {
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
abstract class DataIter(val batchSize: Int = 0) {
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

/**
  * DataIter built in MXNet.
  * @param handle the handle to the underlying C++ Data Iterator
  */
// scalastyle:off finalize
class MXDataIter(val handle: DataIterHandle) extends DataIter {
  private val logger = LoggerFactory.getLogger(classOf[MXDataIter])

  override def finalize(): Unit = {
    checkCall(_LIB.mxDataIterFree(handle))
  }

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    checkCall(_LIB.mxDataIterBeforeFirst(handle))
  }

  /**
   * Iterate to next batch
   * @return whether the move is successful
   */
  override def iterNext(): Boolean = {
    val next = new RefInt
    checkCall(_LIB.mxDataIterNext(handle, next))
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
  override def provideData: Map[String, Shape] = ???

  // The name and shape of label provided by this iterator
  override def provideLabel: Map[String, Shape] = ???
}
// scalastyle:on finalize

/**
 * TODO
 */
class ArrayDataIter() extends DataIter {
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
}
