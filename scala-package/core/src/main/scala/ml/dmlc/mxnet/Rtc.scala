package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

 /**
 * This class allow you to write cuda kernel in Scala
 * and call them with NDArray.
 *
 * @author Depeng Liang
 *
 * @param name String, name of the kernel.
 * @param inputs Array[(String, NDArray)], array of input names and ndarray.
 * @param inputs Array[(String, NDArray)], array of output names and ndarray.
 * @param kernel String, the actual kernel code.
 *      Note that this is only the body of the kernel, i.e.
 *      after { and before }. Rtc will decorate the kernel.
 *      For example, if name = "mykernel" and
 *      inputs = Array(("x", NDArray.zeros(10)))
 *      outputs = Array(("y", NDArray.zeros(10)))
 *      kernel = "y[threadIdx.x] = x[threadIdx.x];",
 *      the kernel that is compile will be:
 *      extern "C" __global__ mykernel(float *x, float *y) {
 *         const int x_ndim = 1;
 *         const int x_dims[] = { 10 };
 *         const int y_ndim = 1;
 *         const int y_dims[] = { 10 };
 *
 *         y[threadIdx.x] = x[threadIdx.x];
 *      }
 */
class Rtc(name: String, inputs: Array[(String, NDArray)],
          outputs: Array[(String, NDArray)], kernel: String) {

  private val rtcHandle = new RtcHandleRef
  private val inputNames = inputs.map(_._1)
  private val outputNames = outputs.map(_._1)
  private val inputNDs = inputs.map(_._2.handle)
  private val outputNDs = outputs.map(_._2.handle)
  checkCall(_LIB.mxRtcCreate(name,
                             inputNames,
                             outputNames,
                             inputNDs,
                             outputNDs,
                             kernel,
                             rtcHandle))

  /**
  * run the kernel.
  * @param ins, array of NDArray
  *            array of input. Can be different NDArray then uses for constructor,
  *            but must have the same shape and in the same order.
  * @param outs, array of NDArray
  *            array of output. Can be different NDArray then uses for constructor,
  *            but must have the same shape and in the same order.
  * @param gridDims, tuple of 3 Int
  *            grid dimension for kernel launch.
  * @param blockDims, tuple of 3 Int
  *            block dimension for kernel launch
  */
  def push(ins: Array[NDArray], outs: Array[NDArray],
    gridDims: (Int, Int, Int), blockDims: (Int, Int, Int)): Unit = {
    checkCall(_LIB.mxRtcPush(rtcHandle.value,
                             ins.map(_.handle),
                             outs.map(_.handle),
                             gridDims._1,
                             gridDims._2,
                             gridDims._3,
                             blockDims._1,
                             blockDims._2,
                             blockDims._3))
}

  /**
   * Free the rtc handle.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    checkCall(_LIB.mxRtcFree(rtcHandle.value))
  }
}
