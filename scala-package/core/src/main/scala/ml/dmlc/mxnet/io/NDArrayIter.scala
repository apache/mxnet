package ml.dmlc.mxnet.io

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.{DataIter, NDArray, Shape}

/**
 * TODO
 * NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.
 * @param data NDArrayIter supports single or multiple data and label.
 * @param label Same as data, but is not fed to the model during testing.
 * @param dataBatchSize Batch Size
 * @param shuffle Whether to shuffle the data
 * @param lastBatchHandle "pad", "discard" or "roll_over". How to handle the last batch
 * @note
 * This iterator will pad, discard or roll over the last batch if
 * the size of data does not match batch_size. Roll over is intended
 * for training and can cause problems if used for prediction.
 */
class NDArrayIter(data: NDArray, label: NDArray = null,
                  private val dataBatchSize: Int = 1, shuffle: Boolean = false,
                  lastBatchHandle: String = "pad") extends DataIter {
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

  override def batchSize: Int = dataBatchSize
}
