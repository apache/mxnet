/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base.NDArrayHandle
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
 * Monitor outputs, weights, and gradients for debugging.
 *
 * @param interval Number of batches between printing.
 * @param statFunc A function that computes statistics of tensors.
 *                 Takes a NDArray and returns a NDArray. defaults
 *                 to mean absolute value |x|/size(x).
 */
class Monitor(
    protected val interval: Int,
    protected var statFunc: (NDArray) => NDArray = null) {

  private val logger = LoggerFactory.getLogger(classOf[Monitor])

  if (statFunc == null) {
    statFunc = (x: NDArray) => {
      NDArray.norm(x) / math.sqrt(x.size.toDouble).toFloat
    }
  }

  private var activated: Boolean = false
  private var queue = new mutable.Queue[(Int, String, NDArray)]
  private var step: Int = 0
  private var exes = new mutable.Queue[Executor]

  val statHelper: MXMonitorCallback = new MXMonitorCallback {
    override def invoke(name: String, arr: NDArrayHandle): Unit = {
      // wrapper for executor callback
      if (activated) {
        val array = new NDArray(arr, writable = false)
        val elem = (step, name, statFunc(array))
        queue += elem
      }
    }
  }

  /**
   * Install callback to executor.
   * Supports installing to multiple exes
   * @param exe the Executor (returned by symbol.bind) to install to.
   */
  def install(exe: Executor): Unit = {
    exe.setMonitorCallback(statHelper)
    exes += exe
  }

  /**
   * Start collecting stats for current batch.
   * Call before forward
   */
  def tic(): Unit = {
    if (step % interval == 0) {
      exes.foreach { exe =>
        exe.argArrays.foreach(_.waitToRead())
      }
      queue = new mutable.Queue[(Int, String, NDArray)]
      activated = true
    }
    step += 1
  }

  /**
   * End collecting for current batch and return results.
   * Call after computation of current batch.
   */
  def toc(): mutable.Queue[(Int, String, String)] = {
    if (activated) {
      exes.foreach { exe =>
        exe.argArrays.foreach(_.waitToRead())
      }
      exes.foreach { exe =>
        (exe.symbol.listArguments() zip exe.argArrays).foreach { case (name, array) =>
          val elem = (step, name, statFunc(array))
          queue += elem
        }
      }
      activated = false
      val res = new mutable.Queue[(Int, String, String)]
      queue.foreach { q =>
        val (n, k, v) = q
        if (v.shape == Shape(1)) {
          res += ((n, k, v.toScalar.toString))
        } else {
          res += ((n, k, s"[${v.toArray.mkString(",")}]"))
        }
      }
      queue = new mutable.Queue[(Int, String, NDArray)]
      res
    } else {
      new mutable.Queue[(Int, String, String)]
    }
  }

  /**
   * End collecting and print results
   */
  def tocPrint(): Unit = {
    val res = toc()
    res.foreach { case (n, k, v) =>
      logger.info(s"Batch: $n $k $v")
    }
  }

}

private[mxnet] trait MXMonitorCallback {
  def invoke(name: String, arr: NDArrayHandle): Unit
}
