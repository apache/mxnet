package ml.dmlc.mxnet

object Context {
  val devtype2str = Map(1 -> "cpu", 2 -> "gpu", 3 -> "cpu_pinned")
  val devstr2type = Map("cpu" -> 1, "gpu" -> 2, "cpu_pinned" -> 3)
  val defaultCtx = new Context("cpu", 0)

  def cpu(deviceId: Int = 0): Context = {
    new Context("cpu", deviceId)
  }

  def gpu(deviceId: Int = 0): Context = {
    new Context("gpu", deviceId)
  }
}

/**
 * Constructing a context.
 * @author Yizhi Liu
 * @param deviceTypeName {'cpu', 'gpu'} String representing the device type
 * @param deviceId (default=0) The device id of the device, needed for GPU
 */
class Context(deviceTypeName: String, val deviceId: Int = 0) {
  val deviceTypeid: Int = Context.devstr2type(deviceTypeName)

  def this(context: Context) = {
    this(context.deviceType, context.deviceId)
  }

  /**
   * Return device type of current context.
   * @return device_type
   */
  def deviceType: String = Context.devtype2str(deviceTypeid)

  override def toString: String = {
    deviceType
  }
}
