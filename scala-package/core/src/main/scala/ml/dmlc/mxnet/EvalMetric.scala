package ml.dmlc.mxnet

/**
 * Base class of all evaluation metrics
 * @param name Metric name
 *
 * @author Yuan Tang, Yizhi Liu
 */
abstract class EvalMetric(protected val name: String) {

  protected var numInst: Int = 0
  protected var sumMetric: Float = 0.0f

  /**
   * Update the internal evaluation.
   *
   * @param labels The labels of the data
   * @param preds Predicted values.
   */
  def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit

  /**
   * Clear the internal statistics to initial state.
   */
  def reset(): Unit = {
    this.numInst = 0
    this.sumMetric = 0.0f
  }

  /**
   * Get the current evaluation result.
   * @return name, Name of the metric
   *         value, Value of the evaluation
   */
  def get: (String, Float) = {
    (this.name, this.sumMetric / this.numInst)
  }
}


// Classification metrics

/**
 * Calculate accuracy
 */
class Accuracy extends EvalMetric("accuracy") {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.length == preds.length,
      "labels and predictions should have the same length.")

    for ((pred, label) <- preds zip labels) {
      val predLabel = NDArray.argmaxChannel(pred)
      require(label.shape == predLabel.shape,
        s"label ${label.shape} and prediction ${predLabel.shape}" +
        s"should have the same length.")
      for ((labelElem, predElem) <- label.toArray zip predLabel.toArray) {
        if (labelElem == predElem) {
          this.sumMetric += 1
        }
      }
      this.numInst += predLabel.shape(0)
      predLabel.dispose()
    }
  }
}

// Regression metrics

/**
 * Calculate Mean Absolute Error loss
 */
class MAE extends EvalMetric("mae") {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ((label, pred) <- labels zip preds) {
      val labelArr = label.toArray
      val predArr = pred.toArray
      require(labelArr.length == predArr.length)
      this.sumMetric +=
        (labelArr zip predArr).map { case (l, p) => Math.abs(l - p) }.sum / labelArr.length
      this.numInst += 1
    }
  }
}

// Calculate Mean Squared Error loss
class MSE extends EvalMetric("mse") {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ((label, pred) <- labels zip preds) {
      val labelArr = label.toArray
      val predArr = pred.toArray
      require(labelArr.length == predArr.length)
      this.sumMetric +=
        (labelArr zip predArr).map { case (l, p) => (l - p) * (l - p) }.sum / labelArr.length
      this.numInst += 1
    }
  }
}

/**
 * Calculate Root Mean Squred Error loss
 */
class RMSE extends EvalMetric("rmse") {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ((label, pred) <- labels zip preds) {
      val labelArr = label.toArray
      val predArr = pred.toArray
      require(labelArr.length == predArr.length)
      val metric: Double = Math.sqrt(
        (labelArr zip predArr).map { case (l, p) => (l - p) * (l - p) }.sum / labelArr.length)
      this.sumMetric += metric.toFloat
    }
    this.numInst += 1
  }
}


/**
 * Custom evaluation metric that takes a NDArray function.
 * @param fEval Customized evaluation function.
 * @param name The name of the metric
 */
class CustomMetric(private val fEval: (NDArray, NDArray) => Float,
                   override val name: String) extends EvalMetric(name) {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ((label, pred) <- labels zip preds) {
      this.sumMetric += fEval(label, pred)
      this.numInst += 1
    }
  }
}
