package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.NDArray
import org.slf4j.LoggerFactory



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
  def getIndex(): Seq[Int]
}

class MXDataIter(var handle: DataIterHandle) extends DataIter {
  private val logger = LoggerFactory.getLogger(classOf[MXDataIter])

  def reset(): Unit = {
    checkCall(_LIB.mxDataIterBeforeFirst(handle))
  }

  def iterNext(): Boolean = {
    checkCall(_LIB.mxDataIterNext(handle))
    return true
  }
}