package ml.dmlc.mxnet.io

import java.util.NoSuchElementException

import ml.dmlc.mxnet.{DataBatch, DataIter, NDArray, Shape}
import org.slf4j.LoggerFactory



/**
 * Resize a DataIter to given number of batches per epoch.
 *  May produce incomplete batch in the middle of an epoch due
 *  to padding from internal iterator.
 *
 * @author Zixuan Huang
 *
 * @param dataIter Internal data iterator.
 * @param reSize number of batches per epoch to resize to.
 * @param resetInternal whether to reset internal iterator on ResizeIter.reset
 */
class ResizeIter(val dataIter: DataIter,
                 val reSize: Int,
                 val resetInternal: Boolean = true) extends DataIter {

  private val logger = LoggerFactory.getLogger(classOf[ResizeIter])

  private var currentBatch: DataBatch = null
  private var cur = 0;


  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    cur = 0
    if(resetInternal) {
      dataIter.reset()
    }
  }

  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (currentBatch == null) {
      iterNext()
    }

    if (currentBatch != null) {
      val batch = currentBatch
      currentBatch = null
      batch
    } else {
      throw new NoSuchElementException
    }
  }

  private def iterNext(): Boolean = {
    if (cur == reSize) {
      false
    } else {
      try {
        currentBatch = dataIter.next()
      } catch {
        case ex: NoSuchElementException => {
          dataIter.reset()
          currentBatch = dataIter.next()
        }
      }
      cur+=1
      true
    }
  }

  override def hasNext: Boolean = {
    if (currentBatch != null) {
      true
    } else {
      iterNext()
    }
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    currentBatch.data
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    currentBatch.label
  }

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  override def getIndex(): IndexedSeq[Long] = {
    currentBatch.index
  }

  /**
   * Get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): Int = {
    currentBatch.pad
  }

  override def batchSize: Int = {
    dataIter.batchSize
  }

  // The name and shape of data provided by this iterator
  override def provideData: Map[String, Shape] = {
    dataIter.provideData
  }

  // The name and shape of label provided by this iterator
  override def provideLabel: Map[String, Shape] = {
    dataIter.provideLabel
  }
}
