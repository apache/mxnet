package ml.dmlc.mxnet

import org.slf4j.LoggerFactory
import scala.collection.mutable

/**
 * Monitor outputs, weights, and gradients for debugging.
 *
 * @author Yuan Tang
 *
 * @param interval Number of batches between printing.
 * @param statFunc A function that computes statistics of tensors.
 *                 Takes a NDArray and returns a NDArray. defaults
 *                 to mean absolute value |x|/size(x).
 */
class Monitor(protected val interval: Int, protected var statFunc: (NDArray) => NDArray = null) {

  private val logger = LoggerFactory.getLogger(classOf[Monitor])

  if (statFunc == null) {
    // TODO: more details here
    statFunc = (x: NDArray) => x
  }

  private var activated: Boolean = false
  private var queue =  new mutable.Queue[(Int, String, NDArray)]
  private var step: Int = 0
  private var exes =  new mutable.Queue[Executor]

  protected val statHelper = (name: String, arr: NDArray) => {
    if (activated) {
      // TODO: more details here
      queue ++= List((step, name, statFunc(arr)))
    }
  }


  /**
   * Install callback to executor.
   * Supports installing to multiple exes
   * @param exe the Executor (returned by symbol.bind) to install to.
   */
  def install(exe: Executor) = {
    exe.setMonitorCallback(statHelper)
    exes ++= List(exe)
  }


  /**
   * Start collecting stats for current batch.
   * Call before forward
   */
  def tic = {
    if (step % interval == 0) {
      exes.foreach { exe =>
        exe.argArrays.foreach {arr => arr.waitToRead()}
      }
      queue =  new mutable.Queue[(Int, String, NDArray)]
      activated = true
    }
    step += 1
  }


  /**
   * End collecting for current batch and return results.
   * Call after computation of current batch.
   */
  def toc: mutable.Queue[(Int, String, String)] = {

    if (activated) {
      exes.foreach { exe =>
        exe.argArrays.foreach {arr => arr.waitToRead()}
      }
      exes.foreach { exe =>
        null
        // TODO: need to implement Symbol first
      /*  for name, array in zip(exe._symbol.list_arguments(), exe.arg_arrays):
          self.queue.append((self.step, name, self.stat_func(array)))*/
      }
    } else {
      return new mutable.Queue[(Int, String, String)]
    }

    activated = false

    val res = new mutable.Queue[(Int, String, String)]

    queue.foreach { q =>
      val (n, k, v) = q
      if (v.shape.sameElements(Array(1))) {
        res ++= List((n, k, v.toScalar.toString))
      } else {
        res ++= List((n, k, v.toArray.toString))
      }
    }

    queue = new mutable.Queue[(Int, String, NDArray)]

    return res
  }

  /**
   * End collecting and print results
   */
  def tocPrint = {
    val res = toc
    res.foreach { re =>
      val (n, k, v) = re
      logger.info(s"Batch: ${n} ${k} ${v}")
    }
  }

}
