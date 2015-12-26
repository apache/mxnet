package ml.dmlc.mxnet

/**
 * Created by yuantang on 12/23/15.
 */
abstract class Executor(var argArrays: Array[NDArray]) {
  def forward
  def backward
  def setMonitorCallback(callback: (String, NDArray) => Any)

}
