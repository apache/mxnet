package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.NDArray
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer


abstract class DataIter (val batchSize: Int = 0) {
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
    * get data of current batch
    * @return the data of current batch
    */
  def getData(): NDArray

  /**
    * Get label of current batch
    * @return the label of current batch
    */
  def getLabel(): NDArray

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
  def getIndex(): List[Long]
}

class MXDataIter(var handle: DataIterHandle) extends DataIter {
  private val logger = LoggerFactory.getLogger(classOf[MXDataIter])

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
    return next.value > 0
  }

  /**
    * get data of current batch
    * @return the data of current batch
    */
  override def getData(): NDArray = {
    val out = new NDArrayHandle
    checkCall(_LIB.mxDataIterGetData(handle, out))
    return new NDArray(out)
  }

  /**
    * Get label of current batch
    * @return the label of current batch
    */
  override def getLabel(): NDArray = {
    val out = new NDArrayHandle
    checkCall(_LIB.mxDataIterGetLabel(handle, out))
    return new NDArray(out)
  }

  /**
    * the index of current batch
    * @return
    */
  override def getIndex(): List[Long] = {
    val outIndex = new ListBuffer[Long]
    val outSize = new RefLong
    checkCall(_LIB.mxDataIterGetIndex(handle, outIndex, outSize))
    return outIndex.toList
  }

  /**
    * get the number of padding examples
    * in current batch
    * @return number of padding examples in current batch
    */
  override def getPad(): MXUint = {
    val out = new MXUintRef
    checkCall(_LIB.mxDataIterGetPadNum(handle, out))
    return out.value
  }
}