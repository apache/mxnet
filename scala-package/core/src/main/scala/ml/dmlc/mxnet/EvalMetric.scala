/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.mxnet

import scala.collection.mutable.ArrayBuffer

/**
 * Base class of all evaluation metrics
 * @param name Metric name
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
  def get: (Array[String], Array[Float]) = {
    (Array(this.name), Array(this.sumMetric / this.numInst))
  }
}

/**
 * Manage multiple evaluation metrics.
 */
class CompositeEvalMetric extends EvalMetric("composite") {
  private val metrics = ArrayBuffer[EvalMetric]()

  // Add a child metric.
  def add(metric: EvalMetric): Unit = {
    this.metrics += metric
  }

  // Get a child metric.
  def getMetric(index: Int): EvalMetric = {
    require(index < this.metrics.length,
        s"Metric index $index is out of range 0 and ${this.metrics.length}")
    this.metrics(index)
  }

  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    for (metric <- this.metrics) {
      metric.update(labels, preds)
    }
  }

  override def reset(): Unit = {
    for (metric <- this.metrics) {
      metric.reset()
    }
  }

  override def get(): (Array[String], Array[Float]) = {
    val names = ArrayBuffer[String]()
    val results = ArrayBuffer[Float]()
    for (metric <- this.metrics) {
      val (name, result) = metric.get
      names += name(0)
      results += result(0)
    }
    (names.toArray, results.toArray)
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
      val predLabel = NDArray.argmax_channel(pred)
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

/**
 * Calculate top k predictions accuracy
 */
class TopKAccuracy(topK: Int) extends EvalMetric("top_k_accuracy") {
  require(topK > 1, "Please use Accuracy if topK is no more than 1")

  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.length == preds.length,
      "labels and predictions should have the same length.")

    for ((pred, label) <- preds zip labels) {
      val predShape = pred.shape
      val dims = predShape.length
      require(dims <= 2, "Predictions should be no more than 2 dims.")
      val labelArray = label.toArray
      val numSamples = predShape(0)
      if (dims == 1) {
        val predArray = pred.toArray.zipWithIndex.sortBy(_._1).reverse.map(_._2)
        require(predArray.length == labelArray.length)
        this.sumMetric +=
          labelArray.zip(predArray).map { case (l, p) => if (l == p) 1 else 0 }.sum
      } else if (dims == 2) {
        val numclasses = predShape(1)
        val predArray = pred.toArray.grouped(numclasses).map { a =>
          a.zipWithIndex.sortBy(_._1).reverse.map(_._2)
        }.toArray
        require(predArray.length == labelArray.length)
        val topK = Math.max(this.topK, numclasses)
        for (j <- 0 until topK) {
          this.sumMetric +=
            labelArray.zip(predArray.map(_(j))).map { case (l, p) => if (l == p) 1 else 0 }.sum
        }
      }
      this.numInst += numSamples
    }
  }
}

/**
 * Calculate the F1 score of a binary classification problem.
 */
class F1 extends EvalMetric("f1") {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.length == preds.length,
      "labels and predictions should have the same length.")

    for ((pred, label) <- preds zip labels) {
      val predLabel = NDArray.argmax_channel(pred)
      require(label.shape == predLabel.shape,
        s"label ${label.shape} and prediction ${predLabel.shape}" +
        s"should have the same length.")
      val labelArray = label.toArray
      var unique = Array[Float]()
      labelArray.foreach(l => if (!unique.contains(l)) unique = unique :+ l)
      require(unique.length <= 2, "F1 currently only supports binary classification.")

      var truePositives, falsePositives, falseNegatives = 0f
      for ((labelElem, predElem) <- labelArray zip predLabel.toArray) {
        if (predElem == 1 && labelElem == 1) truePositives += 1
        else if (predElem == 1 && labelElem == 0) falsePositives += 1
        else if (predElem == 0 && labelElem == 1) falseNegatives += 1
      }

      val precision = {
        if (truePositives + falsePositives > 0) truePositives / (truePositives + falsePositives)
        else 0f
      }

      val recall = {
        if (truePositives + falseNegatives > 0) truePositives / (truePositives + falseNegatives)
        else 0f
      }

      val f1Score = {
        if (precision + recall > 0) (2 * precision * recall) / (precision + recall)
        else 0f
      }

      this.sumMetric += f1Score
      this.numInst += 1
    }
  }
}

/**
 * Calculate perplexity.
 *
 * @param ignoreLabel
 *          Index of invalid label to ignore when
 *          counting. Usually should be -1. Include
 *          all entries if None.
 * @param axis
 *            The axis from prediction that was used to
 *          compute softmax. Default is -1 which means use the last axis.
 */
class Perplexity(ignoreLabel: Option[Int] = None, axis: Int = -1) extends EvalMetric("Perplexity") {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.length == preds.length,
      "labels and predictions should have the same length.")
    var loss = 0d
    var num = 0
    val probs = ArrayBuffer[NDArray]()

    for ((label, pred) <- labels.zip(preds)) {
      require(label.size == pred.size / pred.shape.toArray.reverse.head,
        s"shape mismatch: ${label.shape} vs. ${pred.shape}")
      val l = label.asInContext(pred.context).asType(DType.Int32).reshape(Shape(label.size))
      val p = NDArray.pick(Map("axis" -> this.axis))(pred, label)
      probs += p.head
    }

    for ((label, prob) <- labels.zip(probs)) {
      val probArr = prob.toArray
      if (this.ignoreLabel != None) {
        val ignore = label.toArray.map(l => if (l == this.ignoreLabel.get) 1 else 0)
        val p = prob.toArray.zip(ignore).map { case (p, i) => p * (1 - i) + i }
        prob.set(p)
        num += p.length - ignore.sum
      } else {
        num += prob.size
      }
      loss += prob.toArray.map(p => -Math.log(Math.max(1e-10f, p))).sum
    }

    this.sumMetric += Math.exp(loss / num).toFloat
    this.numInst += 1
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
class CustomMetric(fEval: (NDArray, NDArray) => Float,
                   name: String) extends EvalMetric(name) {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
    require(labels.size == preds.size, "labels and predictions should have the same length.")

    for ((label, pred) <- labels zip preds) {
      this.sumMetric += fEval(label, pred)
      this.numInst += 1
    }
  }
}
