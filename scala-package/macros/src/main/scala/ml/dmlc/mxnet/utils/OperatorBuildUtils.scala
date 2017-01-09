package ml.dmlc.mxnet.utils

private[mxnet] object OperatorBuildUtils {
  // Convert ctypes returned doc string information into parameters docstring.
  def ctypes2docstring(argNames: Seq[String],
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
