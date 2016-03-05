package ml.dmlc.mxnet

import ml.dmlc.mxnet.util.NativeLibraryLoader
import org.slf4j.{LoggerFactory, Logger}

object Base {
  private val logger: Logger = LoggerFactory.getLogger("MXNetJVM")

  // type definitions
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefFloat(val value: Float = 0)
  class RefString(val value: String = null)

  type MXUint = Int
  type MXFloat = Float
  type CPtrAddress = Long

  type NDArrayHandle = CPtrAddress
  type FunctionHandle = CPtrAddress
  type DataIterHandle = CPtrAddress
  type DataIterCreator = CPtrAddress
  type KVStoreHandle = CPtrAddress
  type ExecutorHandle = CPtrAddress
  type SymbolHandle = CPtrAddress

  type MXUintRef = RefInt
  type MXFloatRef = RefFloat
  type NDArrayHandleRef = RefLong
  type FunctionHandleRef = RefLong
  type DataIterHandleRef = RefLong
  type DataIterCreatorRef = RefLong
  type KVStoreHandleRef = RefLong
  type ExecutorHandleRef = RefLong
  type SymbolHandleRef = RefLong

  try {
    try {
      System.loadLibrary("mxnet-scala")
    } catch {
      case e: UnsatisfiedLinkError =>
        NativeLibraryLoader.loadLibrary("mxnet-scala")
    }
  } catch {
    case e: UnsatisfiedLinkError =>
      logger.error("Couldn't find native library mxnet-scala")
      throw e
  }

  val _LIB = new LibInfo
  checkCall(_LIB.nativeLibInit())

  // TODO: shutdown hook won't work on Windows
  Runtime.getRuntime.addShutdownHook(new Thread() {
    override def run(): Unit = {
      notifyShutdown()
    }
  })

  // helper function definitions
  /**
   * Check the return value of C API call
   *
   * This function will raise exception when error occurs.
   * Wrap every API call with this function
   * @param ret return value from API calls
   */
  def checkCall(ret: Int): Unit = {
    if (ret != 0) {
      throw new MXNetError(_LIB.mxGetLastError())
    }
  }

  // Notify MXNet about a shutdown
  private def notifyShutdown(): Unit = {
    checkCall(_LIB.mxNotifyShutdown())
  }

  // Convert ctypes returned doc string information into parameters docstring.
  def ctypes2docstring(
      argNames: Seq[String],
      argTypes: Seq[String],
      argDescs: Seq[String]): String = {

    val params =
      (argNames zip argTypes zip argDescs) map { case ((argName, argType), argDesc) =>
        val desc = if (argDesc.isEmpty) "" else s"\n$argDesc"
        s"$argName : $argType$desc"
      }
    s"Parameters\n----------\n${params.mkString("\n")}\n"
  }
}

class MXNetError(val err: String) extends Exception(err)
