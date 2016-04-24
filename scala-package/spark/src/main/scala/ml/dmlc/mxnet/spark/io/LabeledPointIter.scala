package ml.dmlc.mxnet.spark.io

import ml.dmlc.mxnet.{DataBatch, NDArray, Shape, DataIter}
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.ArrayBuffer

class LabeledPointIter(
  private val points: Iterator[LabeledPoint],
  private val dimension: Int,
  private val _batchSize: Int) extends DataIter {

  private val cache: ArrayBuffer[DataBatch] = ArrayBuffer.empty[DataBatch]
  private var index: Int = -1

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    index = -1
  }

  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (!hasNext) {
      throw new NoSuchElementException("No more data")
    }
    index += 1
    if (index >= 0 && index < cache.size) {
      cache(index)
    } else {
      val dataBuilder = NDArray.empty(_batchSize, dimension)
      val labelBuilder = NDArray.empty(_batchSize, 1)
      var instNum = 0
      while (instNum < batchSize && points.hasNext) {
        val point = points.next()
        val features = point.features.toArray.map(_.toFloat)
        require(features.length == dimension, s"Dimension mismatch: ${features.length} != $dimension")
        dataBuilder.slice(instNum).set(features)
        labelBuilder.slice(instNum).set(Array(point.label.toFloat))
        instNum += 1
      }
      val (data, label) =
        if (instNum == batchSize) (dataBuilder, labelBuilder)
        else (dataBuilder.slice(0, instNum), labelBuilder.slice(0, instNum))
      val dataBatch = new DataBatch(IndexedSeq(data), IndexedSeq(label), null, 0)
      cache += dataBatch
      dataBatch
    }
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    if (index >= 0 && index < cache.size) {
      cache(index).data
    } else {
      null
    }
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    if (index >= 0 && index < cache.size) {
      cache(index).label
    } else {
      null
    }
  }

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  override def getIndex(): IndexedSeq[Long] = {
    if (index >= 0 && index < cache.size) {
      cache(index).index
    } else {
      null
    }
  }

  // The name and shape of label provided by this iterator
  override def provideLabel: Map[String, Shape] = ???

  // The name and shape of data provided by this iterator
  override def provideData: Map[String, Shape] = ???

  /**
   * Get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): Int = 0

  override def batchSize: Int = _batchSize

  override def hasNext: Boolean = {
    points.hasNext || (index < cache.size - 1 && cache.size > 0)
  }
}
