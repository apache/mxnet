package ml.dmlc.mxnet

abstract class EvalMetric(protected val name: String) {

  protected var numInst: Int = 0
  protected var sumMetric: Float = 0.0f

  def update(labels: NDArray, preds: NDArray): Unit

  def reset: Unit = {
    this.numInst = 0
    this.sumMetric = 0.0f
  }

  def get: (String, Float) = {
    (this.name, this.sumMetric / this.numInst)
  }
}

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

class MAE extends EvalMetric("mae") {
  def update(labels: NDArray, preds: NDArray): Unit = {

    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ( (label, pred) <- (labels.toArray zip preds.toArray)) {

    }
  }
}


class RMSE extends EvalMetric("rmse") {
  def update(labels: NDArray, preds: NDArray): Unit = {

    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ( (label, pred) <- (labels.toArray zip preds.toArray)) {

    }
  }
}







