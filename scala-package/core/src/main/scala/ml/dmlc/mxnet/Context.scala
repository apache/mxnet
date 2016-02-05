package ml.dmlc.mxnet

object Context {
  val devtype2str = Map(1 -> "cpu", 2 -> "gpu", 3 -> "cpu_pinned")
  val devstr2type = Map("cpu" -> 1, "gpu" -> 2, "cpu_pinned" -> 3)
  private var _defaultCtx = new Context("cpu", 0)

  def defaultCtx: Context = _defaultCtx

  def cpu(deviceId: Int = 0): Context = {
    new Context("cpu", deviceId)
  }

  def gpu(deviceId: Int = 0): Context = {
    new Context("gpu", deviceId)
  }

  def withScope[T](device: Context)(body: => T): T = {
    val oldDefaultCtx = Context.defaultCtx
    Context._defaultCtx = device
    try {
      body
    } finally {
      Context._defaultCtx = oldDefaultCtx
    }
  }

  implicit def ctx2Array(ctx: Context): Array[Context] = Array(ctx)
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
    s"$deviceType($deviceId)"
  }
}
