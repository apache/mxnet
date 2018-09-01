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

package org.apache.mxnet

import org.apache.mxnet.Base.CPtrAddress
import org.apache.mxnet.annotation.Experimental
import org.slf4j.LoggerFactory

import scala.annotation.varargs
import scala.collection.mutable

/**
 *  A collector to store NDArrays.
 *  It provides a scope, NDArrays allocated in the scope can either <br />
 *  - be disposed automatically when the code block finishes, or <br />
 *  - simply be collected for future usage.
 *  <br />
 *  If the return type of scope is <em>NDArray</em> or <em>NDArrayFuncReturn</em>,
 *  the collector is smart enough NOT to collect or dispose the returned NDArray. <br />
 *  However in other cases, it is users' responsibility NOT to leak allocated NDArrays outside,
 *  (e.g., store to a global variable and use later, pass to another thread, etc.) <br />
 *  Usage Example:
 *  <pre>
 *  val a = NDArray.array(Array(-1f, 0f, 1f, 2f, 3f, 4f), shape = Shape(2, 3))
 *  val res = NDArrayCollector.auto().withScope {
 *    (NDArray.relu(a) + a).toArray
 *  }
 *  </pre>
 *  In the case above, the intermediate NDArrays
 *  (created by <em>NDArray.relu</em> and <em>+</em>) will be disposed automatically. <br />
 *  User can also decide to dispose the collected NDArrays later: <br />
 *  <pre>
 *  val collector = NDArrayCollector.manual()
 *  val res = collector.withScope {
 *    (NDArray.relu(a) + a).toArray
 *  }
 *  collector.foreach(_.dispose())
 *  </pre>
 *  For Java users: <br />
 *  <pre>
 *  NDArray a = NDArray.array(new float[]{-1f, 0f, 1f, 2f, 3f, 4f},
 *                            Shape.create(2, 3), Context.cpu(0));
 *  float[] sliced = NDArrayCollector.auto().withScope(
 *    new scala.runtime.AbstractFunction0<float[]>() {
 *    @Override
 *    public float[] apply() {
 *      a.slice(0, 1).toArray();
 *    }
 *  });
 *  </pre>
 */
object NDArrayCollector {
  private val logger = LoggerFactory.getLogger(classOf[NDArrayCollector])

  private val currCollector = new ThreadLocal[NDArrayCollector] {
    override def initialValue = new NDArrayCollector(false, false)
  }

  /**
   * Create a collector which will dispose the collected NDArrays automatically.
   * @return an auto-disposable collector.
   */
  def auto(): NDArrayCollector = new NDArrayCollector(true)

  /**
   * Create a collector allows users to later dispose the collected NDArray manually.
   * @return a manually-disposable collector.
   */
  @Experimental
  def manual(): NDArrayCollector = new NDArrayCollector(false)

  /**
   * Collect the NDArrays into the collector of the current thread.
   * @param ndArray NDArrays need to be collected.
   */
  @varargs def collect(ndArray: NDArray*): Unit = {
    currCollector.get().add(ndArray: _*)
  }
}

class NDArrayCollector private(private val autoDispose: Boolean = true,
                               private val doCollect: Boolean = true) {
  // native ptr (handle) of the NDArray -> NDArray
  // in some rare situation, multiple NDArrays have same native ptr,
  // the Map here is to prevent from disposing more than once.
  private val arrays = mutable.HashMap.empty[CPtrAddress, NDArray]

  private def add(nd: NDArray*): Unit = {
    if (doCollect) nd.foreach(arr => arrays.put(arr.handle, arr))
  }

  /**
   * Clear the collector.
   */
  def clear(): Unit = {
    arrays.clear()
  }

  /**
   * Iterate over the collected NDArrays and apply the user-defined function to each NDArray.
   * @param f the function that is applied for its side-effect to every NDArray.
   *          The result of function <em>f</em> is discarded.
   */
  def foreach(f: NDArray => Unit): Unit = {
    arrays.values.foreach(f(_))
  }

  /**
   * @return how many unique NDArrays are collected.
   */
  def size: Int = arrays.size

  /**
   * Create a code scope, NDArrays allocated within this scope will be collected.
   * The collected NDArrays will be either <br />
   * - disposed automatically when the code block finishes (when using <em>auto</em>) or <br />
   * - stored for later access (when using <em>manual</em>) <br />
   * If the return type of scope is <em>NDArray</em> or <em>NDArrayFuncReturn</em>,
   * it is smart enough NOT to collect or dispose the returned NDArray. <br />
   * However in other cases, it is users' responsibility NOT to leak allocated NDArrays outside.
   * <br />
   * We might switch to try -with-resources statement (by AutoCloseable in Java 1.7+)
   * and deprecate this method later, thus it is marked as Experimental.
   *
   * @param codeBlock code block to be executed within the scope.
   * @tparam T return type of the function <em>codeBlock</em>.
   * @return The result of function <em>codeBlock</em>.
   */
  @Experimental
  def withScope[T](codeBlock: => T): T = {
    val old = NDArrayCollector.currCollector.get()
    NDArrayCollector.currCollector.set(this)
    try {
      val ret = codeBlock
      ret match {
        case ndRet: NDArray =>
          arrays.remove(ndRet.handle)
        case ndarrays: NDArrayFuncReturn =>
          ndarrays.arr.foreach(nd => arrays.remove(nd.handle))
        case _ => // do nothing
      }
      ret
    } finally {
      if (autoDispose) {
        foreach(_.dispose())
        clear()
      }
      NDArrayCollector.currCollector.set(old)
    }
  }
}
