package ml.dmlc.mxnet

class Optimizer {

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
  def update(key: Int, recv: NDArray, local: NDArray, handle: AnyRef): Unit
  //def update(key: Int): Unit
}
