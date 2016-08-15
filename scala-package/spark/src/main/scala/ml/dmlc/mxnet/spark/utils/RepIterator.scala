package ml.dmlc.mxnet.spark.utils

import scala.collection.Iterator

/**
 * Repeatable Iterator useful in mapPartitions
 * @author Yuance.Li
 */
class RepIterator[T](iteratorInternal: Iterator[T], repetition: Int = 1) extends Iterator[T] {
  assert(repetition > 0)
  var counter = repetition - 1
  var (currentIter, backupIter) = iteratorInternal.duplicate

  override def hasNext: Boolean = {
    currentIter.hasNext || counter > 0
  }

  override def next(): T = {
    assert(hasNext)
    if(currentIter.hasNext) {
      currentIter.next()
    } else if (counter > 0) {
      counter = counter - 1
      var iterTuple = backupIter.duplicate
      currentIter = iterTuple._1
      backupIter = iterTuple._2
      currentIter.next()
    } else {
      throw new NoSuchElementException("No element in this collection")
    }
  }
}
