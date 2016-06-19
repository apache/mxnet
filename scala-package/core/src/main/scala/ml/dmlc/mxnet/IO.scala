package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.io.{MXDataPack, MXDataIter}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

/**
 * IO iterators for loading training & validation data
 * @author Zixuan Huang, Yizhi Liu
 */
object IO {
  type IterCreateFunc = (Map[String, String]) => DataIter
  type PackCreateFunc = (Map[String, String]) => DataPack

  private val logger = LoggerFactory.getLogger(classOf[DataIter])
  private val iterCreateFuncs: Map[String, IterCreateFunc] = initIOModule()

  def MNISTIter: IterCreateFunc = iterCreateFuncs("MNISTIter")

  def ImageRecordIter: IterCreateFunc = iterCreateFuncs("ImageRecordIter")

  def CSVIter: IterCreateFunc = iterCreateFuncs("CSVIter")

  def MNISTPack: PackCreateFunc = createMXDataPack("MNISTIter")

  def ImageRecodePack: PackCreateFunc = createMXDataPack("ImageRecordIter")

  def CSVPack: PackCreateFunc = createMXDataPack("CSVIter")


  /**
   * create iterator via iterName and params
   * @param iterName name of iterator; "MNISTIter" or "ImageRecordIter"
   * @param params parameters for create iterator
   * @return created data iterator
   */
  def createIterator(iterName: String, params: Map[String, String]): DataIter = {
    iterCreateFuncs(iterName)(params)
  }

  /**
   * create dataPack for iterator via itername and params
   * @param iterName name of iterator: "MNISTIter" or "ImageRecordIter"
   * @param params parameters for create iterator
   * @return created dataPack
   */
  def createMXDataPack(iterName: String)(params: Map[String, String]): DataPack = {
    new MXDataPack(iterName, params)
  }

  /**
   * initialize all IO creator Functions
   * @return Map from name to iter creator function
   */
  private def initIOModule(): Map[String, IterCreateFunc] = {
    val IterCreators = new ListBuffer[DataIterCreator]
    checkCall(_LIB.mxListDataIters(IterCreators))
    IterCreators.map(makeIOIterator).toMap
  }

  private def makeIOIterator(handle: DataIterCreator): (String, IterCreateFunc) = {
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
   * DataIter creator
   * @param handle native memory ptr for the iterator
   * @param params parameter passed to the iterator
   * @return created DataIter
   */
  private def creator(handle: DataIterCreator)(
    params: Map[String, String]): DataIter = {
    val out = new DataIterHandleRef
    val keys = params.keys.toArray
    val vals = params.values.toArray
    checkCall(_LIB.mxDataIterCreateIter(handle, keys, vals, out))
    val dataName = params.getOrElse("data_name", "data")
    val labelName = params.getOrElse("label_name", "label")
    new MXDataIter(out.value, dataName, labelName)
  }

  // Convert data into canonical form.
  private[mxnet] def initData(data: IndexedSeq[NDArray],
                              allowEmpty: Boolean,
                              defaultName: String): IndexedSeq[(String, NDArray)] = {
    require(data != null || allowEmpty)
    if (data == null) {
      IndexedSeq()
    } else if (data.length == 1) {
      IndexedSeq((defaultName, data(0)))
    } else {
      data.zipWithIndex.map(item => {
        (defaultName + "_" + item._2, item._1)
      }).toIndexedSeq
    }
  }
}


/**
 * class batch of data
 */
class DataBatch(val data: IndexedSeq[NDArray],
                val label: IndexedSeq[NDArray],
                val index: IndexedSeq[Long],
                val pad: Int) {
  /**
   * Dispose its data and labels
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (data != null) {
      data.foreach(arr => if (arr != null) arr.dispose())
    }
    if (label != null) {
      label.foreach(arr => if (arr != null) arr.dispose())
    }
  }
}

/**
 * DataIter object in mxnet.
 */
abstract class DataIter extends Iterator[DataBatch] {
  /**
   * reset the iterator
   */
  def reset(): Unit

  def batchSize: Int

  /**
   * get next data batch from iterator
   * @return
   */
  @throws(classOf[NoSuchElementException])
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
   * Get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  def getPad(): Int

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  def getIndex(): IndexedSeq[Long]

  // The name and shape of data provided by this iterator
  def provideData: Map[String, Shape]

  // The name and shape of label provided by this iterator
  def provideLabel: Map[String, Shape]
}

/**
 * pack of DataIter, use as Iterable class
 */
abstract class DataPack() extends Iterable[DataBatch] {
  /**
   * get data iterator
   * @return DataIter
   */
  def iterator: DataIter
}


