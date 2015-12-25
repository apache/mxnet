package ml.dmlc.mxnet

import scala.collection.mutable.ArrayBuffer

/**
 * Created by yuantang on 12/23/15.
 */
class Monitor(protected val interval: Int, protected var statFunc: (NDArray) => NDArray = null) {

  if (statFunc == null) {
    // TODO: more details here
    statFunc = (x: NDArray) => x
  }

  protected var activated: Boolean = false
  protected var queue =  ArrayBuffer.empty[(Int, String, NDArray)]
  protected var step: Int = 0
  protected var exes =  ArrayBuffer.empty[Executor]

  protected val statHelper = (name: String, arr: NDArray) => {
    if (activated) {
      // TODO: more details here
      queue.append((step, name, statFunc(arr)))
    }
  }

  def install(exe: Executor) = {
    exe.set_monitor_callback(statHelper)
    exes.append(exe)
  }

  def tic = {
    if (step % interval == 0) {
      exes.foreach {
        exe => exe.argArrays.foreach {arr => arr.waitToRead()}
      }
      queue =  ArrayBuffer.empty[(Int, String, NDArray)]
      activated = true
    }
    step += 1
  }

  def toc = {
    if (activated) {
      exes.foreach {
        exe => exe.argArrays.foreach {arr => arr.waitToRead()}
      }
      exes.foreach {
        _
        // need to implement Symbol first
      /*  for name, array in zip(exe._symbol.list_arguments(), exe.arg_arrays):
          self.queue.append((self.step, name, self.stat_func(array)))*/
      }
    }
  }
  

}
