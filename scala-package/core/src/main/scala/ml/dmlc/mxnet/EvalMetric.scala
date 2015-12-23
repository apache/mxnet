package ml.dmlc.mxnet

/**
 * Base class of all evaluation metrics
 * @param name Metric name
 *
 * @author Yuan Tang
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
  def update(labels: NDArray, preds: NDArray): Unit

  /**
   * Clear the internal statistics to initial state.
   */
  def reset: Unit = {
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


/**
 * Calculate accuracy
 */
class Accuracy extends EvalMetric("accuracy") {
  def update(labels: NDArray, preds: NDArray): Unit = {

    require(labels.size == preds.size, "labels and predictions should have the same length.")

    (0 to preds.size) foreach (i => {
      val pred: NDArray = preds.slice(i, i)
      val label: NDArray = labels.slice(i, i)

//      require(label.shape(0) < predLabel.shape(0), "Should not have more predict labels than actual labels ")
    })
  }
}


/**
 * Calculate Mean Absolute Error loss
 */
class MAE extends EvalMetric("mae") {
  def update(labels: NDArray, preds: NDArray): Unit = {

    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ( (label, pred) <- (labels.toArray zip preds.toArray)) {

    }
  }
}


/**
 * Calculate Root Mean Squred Error loss
 */
class RMSE extends EvalMetric("rmse") {
  def update(labels: NDArray, preds: NDArray): Unit = {

    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ( (label, pred) <- (labels.toArray zip preds.toArray)) {

    }
  }
}


/**
 * Custom evaluation metric that takes a NDArray function.
 * @param fEval Customized evaluation function.
 * @param name The name of the metric
 */
class CustomMetric(var fEval: () => Unit, override val name: String) extends EvalMetric(name) {
  def update(labels: NDArray, preds: NDArray): Unit = {

    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ( (label, pred) <- (labels.toArray zip preds.toArray)) {

    }
  }
}





