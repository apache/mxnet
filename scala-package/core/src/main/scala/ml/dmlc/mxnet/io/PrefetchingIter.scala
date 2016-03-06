package ml.dmlc.mxnet.io

import ml.dmlc.mxnet.{DataIter, NDArray, Shape}

/**
 * TODO
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

  // The name and shape of label provided by this iterator
  override def provideLabel: Map[String, Shape] = ???

  /**
   * get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): Int = ???

  // The name and shape of data provided by this iterator
  override def provideData: Map[String, Shape] = ???

  override def hasNext: Boolean = ???

  override def batchSize: Int = ???
}
