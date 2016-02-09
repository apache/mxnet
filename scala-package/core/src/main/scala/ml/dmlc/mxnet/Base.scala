package ml.dmlc.mxnet

object Base {
  // type definitions
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefFloat(val value: Float = 0)
  class RefString(val value: String = null)

  type MXUint = Int
  type MXFloat = Float
  type CPtrAddress = Long
  // TODO: make it more friendly to java
  type Shape = Vector[Int]

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

  System.loadLibrary("mxnet-scala")
  val _LIB = new LibInfo

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
