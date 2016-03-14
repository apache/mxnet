package ml.dmlc.mxnet.io

import java.util.NoSuchElementException

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet._
import org.slf4j.LoggerFactory

/**
 * NDArrayIter object in mxnet. Taking NDArray to get dataiter.
 *
 * @author Zixuan Huang
 *
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
class NDArrayIter (data: IndexedSeq[NDArray], label: IndexedSeq[NDArray] = null,
                  private val dataBatchSize: Int = 1, shuffle: Boolean = false,
                  lastBatchHandle: String = "pad") extends DataIter {
  private val logger = LoggerFactory.getLogger(classOf[NDArrayIter])


  private val (_dataList: IndexedSeq[NDArray],
  _labelList: IndexedSeq[NDArray]) = {
    // data should not be null and size > 0
    require(data != null && data.size > 0,
      "data should not be null and data.size should not be zero")

    // shuffle is not supported currently
    require(shuffle == false, "shuffle is not supported currently")

    // discard final part if lastBatchHandle equals discard
    if (lastBatchHandle.equals("discard")) {
      val dataSize = data(0).shape(0)
      require(dataBatchSize <= dataSize,
        "batch_size need to be smaller than data size when not padding.")
      val keepSize = dataSize - dataSize % dataBatchSize
      val dataList = data.map(ndArray => {ndArray.slice(0, keepSize)})
      if (label != null) {
        val labelList = label.map(ndArray => {ndArray.slice(0, keepSize)})
        (dataList, labelList)
      } else {
        (dataList, null)
      }
    } else {
      (data, label)
    }
  }


  val initData: IndexedSeq[(String, NDArray)] = IO.initData(_dataList, false, "data")
  val initLabel: IndexedSeq[(String, NDArray)] = IO.initData(_labelList, true, "label")
  val numData = _dataList(0).shape(0)
  val numSource = initData.size
  var cursor = -dataBatchSize


  private val (_provideData: Map[String, Shape],
  _provideLabel: Map[String, Shape]) = {
    val pData = initData.map(getShape).toMap
    val pLabel = initLabel.map(getShape).toMap
    (pData, pLabel)
  }

  /**
   * get shape via dataBatchSize
   * @param dataItem
   */
  private def getShape(dataItem: (String, NDArray)): (String, Shape) = {
    val len = dataItem._2.shape.size
    val newShape = dataItem._2.shape.slice(1, len)
    (dataItem._1, Shape(Array[Int](dataBatchSize)) ++ newShape)
  }


  /**
   * Igore roll over data and set to start
   */
  def hardReset(): Unit = {
    cursor = -dataBatchSize
  }

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    if (lastBatchHandle.equals("roll_over") && cursor>numData) {
      cursor = -dataBatchSize + (cursor%numData)%dataBatchSize
    } else {
      cursor = -dataBatchSize
    }
  }

  override def hasNext: Boolean = {
    if (cursor + dataBatchSize < numData) {
      true
    } else {
      false
    }
  }

  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (hasNext) {
      cursor += dataBatchSize
      new DataBatch(getData(), getLabel(), getIndex(), getPad())
    } else {
      throw new NoSuchElementException
    }
  }

  /**
   * handle the last batch
   * @param ndArray
   * @return
   */
  private def _padData(ndArray: NDArray): NDArray = {
    val padNum = cursor + dataBatchSize - numData
    val newArray = NDArray.zeros(ndArray.slice(0, dataBatchSize).shape)
    newArray.slice(0, dataBatchSize - padNum).set(ndArray.slice(cursor, numData))
    newArray.slice(dataBatchSize - padNum, dataBatchSize).set(ndArray.slice(0, padNum))
    newArray
  }

  private def _getData(data: IndexedSeq[NDArray]): IndexedSeq[NDArray] = {
    require(cursor < numData, "DataIter needs reset.")
    if (data == null) {
      null
    } else {
      if (cursor + dataBatchSize <= numData) {
        data.map(ndArray => {ndArray.slice(cursor, cursor + dataBatchSize)}).toIndexedSeq
      } else {
        // padding
        data.map(_padData).toIndexedSeq
      }
    }
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    _getData(_dataList)
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    _getData(_labelList)
  }

  /**
   * the index of current batch
   * @return
   */
  override def getIndex(): IndexedSeq[Long] = {
    (cursor.toLong to (cursor + dataBatchSize).toLong).toIndexedSeq
  }

  /**
   * get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): MXUint = {
    if (lastBatchHandle.equals("pad") && cursor + batchSize > numData) {
      cursor + batchSize - numData
    } else {
      0
    }
  }

  // The name and shape of data provided by this iterator
  override def provideData: Map[String, Shape] = _provideData

  // The name and shape of label provided by this iterator
  override def provideLabel: Map[String, Shape] = _provideLabel

  override def batchSize: Int = dataBatchSize
}
