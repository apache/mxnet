package ml.dmlc.mxnet.init

object Base {
  tryLoadInitLibrary()
  val _LIB = new LibInfo

  // type definitions
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefFloat(val value: Float = 0)
  class RefString(val value: String = null)

  type CPtrAddress = Long

  type NDArrayHandle = CPtrAddress
  type FunctionHandle = CPtrAddress
  type KVStoreHandle = CPtrAddress
  type ExecutorHandle = CPtrAddress
  type SymbolHandle = CPtrAddress

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadInitLibrary(): Unit = {
    val baseDir = System.getProperty("user.dir") + "/init-native"
    val os = System.getProperty("os.name")
    // ref: http://lopica.sourceforge.net/os.html
    if (os.startsWith("Linux")) {
      System.load(s"$baseDir/linux-x86_64/target/libmxnet-init-scala-linux-x86_64.so")
    } else if (os.startsWith("Mac")) {
      System.load(s"$baseDir/osx-x86_64/target/libmxnet-init-scala-osx-x86_64.jnilib")
    } else {
      // TODO(yizhi) support windows later
      throw new UnsatisfiedLinkError()
    }
  }
}
