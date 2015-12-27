package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

object IO {
  private val logger = LoggerFactory.getLogger(classOf[DataIter])
  type IterCreateFunc = (Map[String, String]) => DataIter
  val iterCreateFuncs: Map[String, IterCreateFunc] = _initIOModule()

  def _initIOModule(): Map[String, IterCreateFunc] = {
    val IterCreators = new ListBuffer[DataIterCreator]
    checkCall(_LIB.mxListDataIters(IterCreators))
    IterCreators.map(_makeIOIterator).toMap
  }

  def _makeIOIterator(handle: DataIterCreator): (String, IterCreateFunc) = {
    val name = new RefString
    val desc = new RefString
    val argNames = new ListBuffer[String]
    val argTypes = new ListBuffer[String]
    val argDescs = new ListBuffer[String]
    checkCall(_LIB.mxDataIterGetIterInfo(handle, name, desc, argNames, argTypes, argDescs))
    val paramStr = Base.ctypes2docstring(argNames, argTypes, argDescs)
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
    logger.debug(docStr)
    return (name.value, creator(handle))
  }

  def creator(handle: DataIterCreator)(
              params: Map[String, String]): DataIter = {
    val out = new DataIterHandle
    val keys = params.keys.toArray
    val vals = params.values.toArray
    checkCall(_LIB.mxDateIterCreateIter(handle, keys, vals, out))
    return new MXDataIter(out)
  }
}

//class for batch of data
class DataBatch(val data: NDArray,
                val label: NDArray,
                val index: List[Long],
                val pad: Int)

/**
  *DataIter object in mxnet.
  */
abstract class DataIter {
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

  /**
    * return next batch of data
    * @return
    */
  def next(): DataBatch = {
      return new DataBatch(getData(), getLabel(), getIndex(), getPad())
  }
}

/**
  * DataIter built in MXNet.
  * @param handle the handle to the underlying C++ Data Iterator
  */
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

/**
  * NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.
  * @param data a list of NDArray
  * @param label a list of NDArray
  * @param batch_size Batch Size
  * @param shuffle Whether to shuffle the data
  * @param last_batch_handle 'pad', 'discard' or 'roll_over',  How to handle the last batch
  */
class NDArrayIter(var data: List[NDArray], var label: List[NDArray],
                  var batch_size: Int, var shuffle: Boolean,
                  var last_batch_handle: String) extends DataIter {
  /**
    * reset the iterator
    */
  override def reset(): Unit = ???

  /**
    * get data of current batch
    * @return the data of current batch
    */
  override def getData(): NDArray = ???

  /**
    * Get label of current batch
    * @return the label of current batch
    */
  override def getLabel(): NDArray = ???

  /**
    * the index of current batch
    * @return
    */
  override def getIndex(): List[Long] = ???

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
}


