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

package org.apache.mxnet.infer

import java.util.concurrent._

import org.slf4j.LoggerFactory

private[infer] trait MXNetHandler {

  /**
    * Executes a function within a thread-safe executor
    * @param f The function to execute
    * @tparam T The return type of the function
    * @return Returns the result of the function f
    */
  def execute[T](f: => T): T

  val executor: ExecutorService

}

private[infer] object MXNetHandlerType extends Enumeration {

  /**
    * The internal type of the MXNetHandlerType enumeration
    */
  type MXNetHandlerType = Value

  val SingleThreadHandler = Value("MXNetSingleThreadHandler")
  val OneThreadPerModelHandler = Value("MXNetOneThreadPerModelHandler")
}

private[infer] class MXNetThreadPoolHandler(numThreads: Int = 1)
  extends MXNetHandler {

  require(numThreads > 0, s"Invalid numThreads $numThreads")

  private val logger = LoggerFactory.getLogger(classOf[MXNetThreadPoolHandler])
  private var threadCount: Int = 0

  private val threadFactory = new ThreadFactory {
    override def newThread(r: Runnable): Thread = new Thread(r) {
      setName(classOf[MXNetThreadPoolHandler].getCanonicalName
        + "-%d".format(threadCount))
      // setting to daemon threads to exit along with the main threads
      setDaemon(true)
      threadCount += 1
    }
  }

  override val executor: ExecutorService =
    Executors.newFixedThreadPool(numThreads, threadFactory)

  private val creatorThread = executor.submit(new Callable[Thread] {
    override def call(): Thread = Thread.currentThread()
  }).get()

  override def execute[T](f: => T): T = {

    if (Thread.currentThread() eq creatorThread) {
      f
    } else {

      val task = new Callable[T] {
        override def call(): T = {
          logger.debug("threadId: %s".format(Thread.currentThread().getId()))
          f
        }
      }

      val result = executor.submit(task)
      try {
        result.get()
      } catch {
        case e : InterruptedException => throw e
        // unwrap the exception thrown by the task
        case e1: Exception => throw e1.getCause()
      }
    }
  }

}

private[infer] object MXNetSingleThreadHandler extends MXNetThreadPoolHandler(1) {

}

private[infer] object MXNetHandler {

  /**
    * Creates a handler based on the handlerType
    * @return A ThreadPool or Thread Handler
    */
  def apply(): MXNetHandler = {
    if (handlerType == MXNetHandlerType.OneThreadPerModelHandler) {
      new MXNetThreadPoolHandler(1)
    } else {
      MXNetSingleThreadHandler
    }
  }
}
