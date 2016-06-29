package ml.dmlc.mxnet

import scala.collection.mutable

object Optimizer {
  def getUpdater(optimizer: Optimizer): MXKVStoreUpdater = {
    new MXKVStoreUpdater {
      val states = new scala.collection.mutable.HashMap[Int, AnyRef]
      override def update(index: Int, grad: NDArray, weight: NDArray): Unit = {
        val state =
          if (states.contains(index)) {
            states.get(index).get
          } else {
            val newState = optimizer.createState(index, weight)
            states.put(index, newState)
            newState
          }
        optimizer.update(index, weight, grad, state)
      }
      override def dispose(): Unit = {
        states.values.foreach(optimizer.disposeState)
        states.clear()
      }
    }
  }
}

abstract class Optimizer extends Serializable {
  protected var lrScale: mutable.Map[Int, Float] = mutable.HashMap.empty[Int, Float]
  protected var numUpdate: Int = 0
  protected val indexUpdateCount: mutable.Map[Int, Int] = mutable.HashMap.empty[Int, Int]

  protected var specialized: Boolean = false
  protected val weightSet: mutable.Set[Int] = mutable.HashSet.empty[Int]
  protected var rescaleGrad: Float = 1

  /**
   * Update the parameters.
   * @param index An unique integer key used to index the parameters
   * @param weight weight ndarray
   * @param grad grad ndarray
   * @param state NDArray or other objects returned by initState
   *              The auxiliary state used in optimization.
   */
  // TODO: make state a ClassTag
  def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit

  // Create additional optimizer state such as momentum.
  // TODO: make returned state a ClassTag
  def createState(index: Int, weight: NDArray): AnyRef

  // Dispose the state it created
  def disposeState(state: AnyRef): Unit

  // Set individual learning rate scale for parameters
  def setLrScale(lrScale: Map[Int, Float]) {
    this.lrScale = mutable.Map(lrScale.toSeq: _*)
  }

  def setArgNames(argNames: Seq[String]): Unit = {
    if (argNames != null) {
      specialized = true
      var index = 0
      argNames foreach { name =>
        if (!name.endsWith("data") && !name.endsWith("label")) {
          if (name.endsWith("weight")) {
            weightSet.add(index)
          }
          index += 1
        }
      }
    }
  }

  // Set rescaling factor of gradient.
  def setRescaleGrad(rescaleGrad: Float): Unit = {
    this.rescaleGrad = rescaleGrad
  }

  /**
   * update num_update
   * @param index The index will be updated
   */
  protected def updateCount(index: Int): Unit = {
    val count = indexUpdateCount.getOrElseUpdate(index, 0) + 1
    indexUpdateCount.update(index, count)
    numUpdate = Math.max(count, numUpdate)
  }

  protected def getWd(index: Int, wd: Float): Float = {
    if (specialized) {
      if (this.weightSet.contains(index)) {
        wd
      } else {
        0f
      }
    } else {
      wd
    }
  }
}

trait MXKVStoreUpdater {
  /**
   * user-defined updater for the kvstore
   * It's this updater's responsibility to delete recv and local
   * @param key the key
   * @param recv the pushed value on this key
   * @param local the value stored on local on this key
   */
  def update(key: Int, recv: NDArray, local: NDArray): Unit
  def dispose(): Unit
}
