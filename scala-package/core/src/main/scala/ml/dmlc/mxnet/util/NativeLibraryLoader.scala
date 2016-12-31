package ml.dmlc.mxnet.util

import java.io._

import org.slf4j.{Logger, LoggerFactory}

class NativeLibraryLoader
object NativeLibraryLoader {
  private val logger: Logger = LoggerFactory.getLogger(classOf[NativeLibraryLoader])
  private val libPathInJar = "/lib/native/"
  private val _tempDir: File =
    try {
      val tempDir = File.createTempFile("mxnet", "")
      if (!tempDir.delete || !tempDir.mkdir) {
        throw new IOException(s"Couldn't create directory ${tempDir.getAbsolutePath}")
      }

      /*
       * Different cleanup strategies for Windows and Linux.
       * TODO: shutdown hook won't work on Windows
       */
      if (getUnifiedOSName != "Windows") {
        Runtime.getRuntime.addShutdownHook(new Thread() {
          override def run(): Unit = {
            for (f <- tempDir.listFiles()) {
              logger.info("Deleting " + f.getAbsolutePath)
              if (!f.delete()) {
                logger.warn(s"Couldn't delete temporary file ${f.getAbsolutePath}")
              }
            }
            logger.info(s"Deleting ${tempDir.getAbsolutePath}")
            if (!tempDir.delete()) {
              logger.warn(s"Couldn't delete temporary directory ${tempDir.getAbsolutePath}")
            }
          }
        })
        tempDir
      } else {
        throw new RuntimeException("Windows not supported yet.")
      }
    } catch {
      case ex: IOException =>
        logger.error("Couldn't create temporary directory: {}", ex.getMessage)
        null
    }

  /**
   * Find the library as a resource in jar, copy it to a tempfile
   * and load it using System.load(). The name of the library has to be the
   * base name, it is mapped to the corresponding system name using
   * System.mapLibraryName(). e.g., the library "foo" is called "libfoo.so"
   * under Linux and "foo.dll" under Windows, but you just have to pass "foo" to
   * the loadLibrary().
   *
   * @param libname basename of the library
   * @throws UnsatisfiedLinkError if library cannot be founds
   */
  @throws(classOf[UnsatisfiedLinkError])
  def loadLibrary(libname: String) {
    val mappedLibname = System.mapLibraryName(libname)
    val loadLibname: String =
      if (mappedLibname.endsWith("dylib")) {
        logger.info("Replaced .dylib with .jnilib")
        mappedLibname.replace(".dylib", ".jnilib")
      } else {
        mappedLibname
      }
    logger.debug(s"Attempting to load $loadLibname")
    val libFileInJar = libPathInJar + loadLibname
    val is: InputStream = getClass.getResourceAsStream(libFileInJar)
    if (is == null) {
      throw new UnsatisfiedLinkError(s"Couldn't find the resource $loadLibname")
    }
    logger.info(s"Loading $loadLibname from $libPathInJar copying to $libname")
    loadLibraryFromStream(libname, is)
  }

  /**
   * Translate all those Windows to "Windows". ("Windows XP", "Windows Vista", "Windows 7", etc.)
   */
  private def unifyOSName(osname: String): String = {
    if (osname.startsWith("Windows")) {
      "Windows"
    }
    osname
  }

  private def getUnifiedOSName: String = {
    unifyOSName(System.getProperty("os.name"))
  }

  @throws(classOf[IOException])
  private def createTempFile(name: String): File = {
    new File(_tempDir + File.separator + name)
  }

  /**
   * Load a system library from a stream. Copies the library to a temp file
   * and loads from there.
   *
   * @param libname name of the library (just used in constructing the library name)
   * @param is      InputStream pointing to the library
   */
  private def loadLibraryFromStream(libname: String, is: InputStream) {
    try {
      val tempfile: File = createTempFile(libname)
      val os: OutputStream = new FileOutputStream(tempfile)
      logger.debug("tempfile.getPath() = {}", tempfile.getPath)
      val savedTime: Long = System.currentTimeMillis
      val buf: Array[Byte] = new Array[Byte](8192)
      var len: Int = is.read(buf)
      while (len > 0) {
        os.write(buf, 0, len)
        len = is.read(buf)
      }
      os.flush()
      val lock: InputStream = new FileInputStream(tempfile)
      os.close()
      val seconds: Double = (System.currentTimeMillis - savedTime).toDouble / 1e3
      logger.debug(s"Copying took $seconds seconds.")
      logger.debug("Loading library from {}", tempfile.getPath)
      System.load(tempfile.getPath)
      lock.close()
    } catch {
      case io: IOException =>
        logger.error("Could not create the temp file: {}", io.toString)
      case ule: UnsatisfiedLinkError =>
        logger.error("Couldn't load copied link file: {}", ule.toString)
        throw ule
    }
  }
}
