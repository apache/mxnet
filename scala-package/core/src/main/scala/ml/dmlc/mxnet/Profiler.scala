package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

/**
 * @author Depeng Liang
 */
object Profiler {

  val mode2Int = Map("symbolic" -> 0, "all" -> 1)
  val state2Int = Map("stop" -> 0, "run" -> 1)

  /**
   * Set up the configure of profiler.
   * @param mode, optional
   *                  Indicting whether to enable the profiler, can
   *                  be "symbolic" or "all". Default is "symbolic".
   * @param fileName, optional
   *                  The name of output trace file. Default is "profile.json".
   */
  def profilerSetConfig(mode: String = "symbolic", fileName: String = "profile.json"): Unit = {
    require(mode2Int.contains(mode))
    checkCall(_LIB.mxSetProfilerConfig(mode2Int(mode), fileName))
  }

  /**
   * Set up the profiler state to record operator.
   * @param state, optional
   *                  Indicting whether to run the profiler, can
   *                  be "stop" or "run". Default is "stop".
   */
  def profilerSetState(state: String = "stop"): Unit = {
    require(state2Int.contains(state))
    checkCall(_LIB.mxSetProfilerState(state2Int(state)))
  }

  /**
   * Dump profile and stop profiler. Use this to save profile
   * in advance in case your program cannot exit normally.
   */
  def dumpProfile(): Unit = {
    checkCall(_LIB.mxDumpProfile())
  }
}
