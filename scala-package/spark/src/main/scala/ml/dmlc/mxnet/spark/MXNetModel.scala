package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.spark.io.PointIter
import ml.dmlc.mxnet.{FeedForward, NDArray, Shape}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector

/**
 * Wrapper for [[ml.dmlc.mxnet.Model]] which used in Spark application
 * @author Yizhi Liu
 */
class MXNetModel private[mxnet](
    @transient private var model: FeedForward,
    private val dimension: Shape,
    private val batchSize: Int,
    private val dataName: String = "data",
    private val labelName: String = "label") extends Serializable {
  require(model != null, "try to serialize an empty FeedForward model")
  require(dimension != null, "unknown dimension")
  require(batchSize > 0, s"invalid batchSize: $batchSize")
  val serializedModel = model.serialize()

  /**
   * Get inner model [[FeedForward]]
   * @return the underlying model used to train & predict
   */
  def innerModel: FeedForward = {
    if (model == null) {
      model = FeedForward.deserialize(serializedModel)
    }
    model
  }

  /**
   * Predict a bunch of Vectors
   * @param dataset points
   * @return predicted results.
   */
  def predict(dataset: Iterator[Vector]): Array[MXNDArray] = {
    val dt = new PointIter(dataset, dimension, batchSize, dataName, labelName)
    val results = innerModel.predict(dt)
    results.map(arr => MXNDArray(arr))
  }

  def predict(data: Vector): Array[MXNDArray] = {
    predict(Iterator(data))
  }

  /**
   * Save [[MXNetModel]] as object file
   * @param sc SparkContext
   * @param path output path
   */
  def save(sc: SparkContext, path: String): Unit = {
    sc.parallelize(Seq(this), 1).saveAsObjectFile(path)
  }
}

object MXNetModel {
  /**
   * Load [[MXNetModel]] from path
   * @param sc SparkContext
   * @param path input path
   * @return Loaded [[MXNetModel]]
   */
  def load(sc: SparkContext, path: String): MXNetModel = {
    sc.objectFile[MXNetModel](path, 1).first()
  }
}
