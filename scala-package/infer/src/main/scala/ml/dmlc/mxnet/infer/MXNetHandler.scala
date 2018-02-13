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

package ml.dmlc.mxnet.infer

import java.util.concurrent._
import org.slf4j.LoggerFactory

private[infer] trait MXNetHandler {

  def execute[T](f: => T): T

  val executor: ExecutorService

}

private[infer] object MXNetHandlerType extends Enumeration {

  type MXNetHandlerType = Value
  val SingleThreadHandler = Value("MXNetSingleThreadHandler")
  val OneThreadPerModelHandler = Value("MXNetOneThreadPerModelHandler")
}

private[infer] class MXNetThreadPoolHandler(numThreads: Option[Int] = Some(1))
  extends MXNetHandler {
  private val logger = LoggerFactory.getLogger(classOf[MXNetThreadPoolHandler])

  private val threadFactory = new ThreadFactory {

    override def newThread(r: Runnable): Thread = new Thread(r) {
      setName(classOf[MXNetThreadPoolHandler].getCanonicalName)
    }
  }

  override val executor: ExecutorService = Executors.newFixedThreadPool(1, threadFactory)

  override def execute[T](f: => T): T = {
    val task = new Callable[T] {
      override def call(): T = {
        logger.info("threadId: %s".format(Thread.currentThread().getId()))
        f
      }
    }
    val result = executor.submit(task)
    try {
      result.get()
    } catch {
      case e: Exception => throw e.getCause()
    }
  }

}

private[infer] object MXNetSingleThreadHandler extends MXNetThreadPoolHandler(Some(1)) {

}

private[infer] object MXNetHandler {

  def apply(): MXNetHandler = {
    if (handlerType == MXNetHandlerType.OneThreadPerModelHandler) {
      new MXNetThreadPoolHandler(Some(1))
    } else {
      MXNetSingleThreadHandler
    }
  }
}
