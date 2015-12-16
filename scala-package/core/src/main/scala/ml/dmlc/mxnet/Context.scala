package ml.dmlc.mxnet

object Context {
  val defaultCtx = new Context("cpu", 0)
}

/**
 * Constructing a context.
 * @param deviceTypeName {'cpu', 'gpu'} String representing the device type
 * @param deviceId (default=0) The device id of the device, needed for GPU
 */
class Context(deviceTypeName: String, val deviceId: Int = 0) {
  private val devtype2str = Map(1 -> "cpu", 2 -> "gpu", 3 -> "cpu_pinned")
  private val devstr2type = Map("cpu" -> 1, "gpu" -> 2, "cpu_pinned" -> 3)

  val deviceTypeid: Int = devstr2type(deviceTypeName)

  def this(context: Context) = {
    this(context.deviceType, context.deviceId)
  }

  /**
   * Return device type of current context.
   * @return device_type
   */
  def deviceType: String = devtype2str(deviceTypeid)
}
