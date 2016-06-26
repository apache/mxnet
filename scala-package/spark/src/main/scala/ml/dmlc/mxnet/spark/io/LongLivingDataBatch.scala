package ml.dmlc.mxnet.spark.io

import ml.dmlc.mxnet.{NDArray, DataBatch}

/**
 * Dispose only when 'disposeForce' called
 * @author Yizhi Liu
 */
class LongLivingDataBatch(
  override val data: IndexedSeq[NDArray],
  override val label: IndexedSeq[NDArray],
  override val index: IndexedSeq[Long],
  override val pad: Int) extends DataBatch(data, label, index, pad) {
  override def dispose(): Unit = {}
  def disposeForce(): Unit = super.dispose()
}
