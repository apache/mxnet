package ml.dmlc.mxnet

object Optimizer {
  def getUpdater(optimizer: Optimizer): MXKVStoreUpdater = {
    new MXKVStoreUpdater {
      private val states = new scala.collection.mutable.HashMap[Int, AnyRef]
      override def update(index: Int, grad: NDArray, weight: NDArray, handle: AnyRef): Unit = {
        val state = states.getOrElseUpdate(index, optimizer.createState(index, weight))
        optimizer.update(index, weight, grad, state)
      }
    }
  }
}

abstract class Optimizer extends Serializable {
  /**
   * Update the parameters.
   * @param index An unique integer key used to index the parameters
   * @param weight weight ndarray
   * @param grad grad ndarray
   * @param state NDArray or other objects returned by initState
   *              The auxiliary state used in optimization.
   */
  def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit = ???

  // Create additional optimizer state such as momentum.
  def createState(index: Int, weight: NDArray): AnyRef
}

trait MXKVStoreUpdater {
  /**
   * user-defined updater for the kvstore
   * It's this updater's responsibility to delete recv and local
   * @param key the key
   * @param recv the pushed value on this key
   * @param local the value stored on local on this key
   * @param handle The additional handle to the updater
   */
  def update(key: Int, recv: NDArray, local: NDArray, handle: AnyRef = null): Unit
}
