package ml.dmlc.mxnet.init

object Base {
  println("Current Dir: " + System.getProperty("user.dir"))
  System.load("/Users/lewis/Workspace/source-codes/forks/mxnet/scala-package/init-native/osx-x86_64-cpu/target/libmxnet-init-scala-osx-x86_64-cpu.jnilib")
  val _LIB = new LibInfo

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadLibraryOS(libname: String): Unit = {
    try {
      System.loadLibrary(libname)
    } catch {
      case e: UnsatisfiedLinkError =>
        val os = System.getProperty("os.name")
        // ref: http://lopica.sourceforge.net/os.html
        if (os.startsWith("Linux")) {
          tryLoadLibraryXPU(libname, "linux-x86_64")
        } else if (os.startsWith("Mac")) {
          tryLoadLibraryXPU(libname, "osx-x86_64")
        } else {
          // TODO(yizhi) support windows later
          throw new UnsatisfiedLinkError()
        }
    }
  }

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadLibraryXPU(libname: String, arch: String): Unit = {
    try {
      // try gpu first
      System.loadLibrary(s"$libname-$arch-gpu")
    } catch {
      case e: UnsatisfiedLinkError =>
        System.loadLibrary(s"$libname-$arch-cpu")
    }
  }

}
