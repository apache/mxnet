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

import org.slf4j.{Logger, LoggerFactory}
import scala.util.Try
import scala.collection._

private object WarnIfNotDisposed {
  private val traceProperty = "mxnet.traceLeakedObjects"

  private val logger: Logger = LoggerFactory.getLogger(classOf[WarnIfNotDisposed])

  // This set represents the list of classes we've logged a warning about if we're not running
  // in tracing mode. This is used to ensure we only log once.
  // Don't need to synchronize on this set as it's usually used from a single finalizer thread.
  private val classesWarned = mutable.Set.empty[String]

  lazy val tracingEnabled = {
    val value = Try(System.getProperty(traceProperty).toBoolean).getOrElse(false)
    if (value) {
      logger.info("Leaked object tracing is enabled (property {} is set)", traceProperty)
    }
    value
  }
}

// scalastyle:off finalize
protected trait WarnIfNotDisposed {
  import WarnIfNotDisposed.logger
  import WarnIfNotDisposed.traceProperty
  import WarnIfNotDisposed.classesWarned

  protected def isDisposed: Boolean

  protected val creationTrace: Option[Array[StackTraceElement]] = if (tracingEnabled) {
    Some(Thread.currentThread().getStackTrace())
  } else {
    None
  }

  override protected def finalize(): Unit = {
    if (!isDisposed) logDisposeWarning()

    super.finalize()
  }

  // overridable for testing
  protected def tracingEnabled = WarnIfNotDisposed.tracingEnabled

  protected def logDisposeWarning(): Unit = {
    // The ":Any" casts below are working around the Slf4j Java API having overloaded methods that
    // Scala doesn't resolve automatically.
    if (creationTrace.isDefined) {
      logger.warn(
        "LEAK: An instance of {} was not disposed. Creation point of this resource was:\n\t{}",
        getClass(), creationTrace.get.mkString("\n\t"): Any)
    } else {
      // Tracing disabled but we still warn the first time we see a leak to ensure the code author
      // knows. We could warn every time but this can be very noisy.
      val className = getClass().getName()
      if (!classesWarned.contains(className)) {
        logger.warn(
          "LEAK: [one-time warning] An instance of {} was not disposed. " + //
          "Set property {} to true to enable tracing",
          className, traceProperty: Any)

        classesWarned += className
      }
    }
  }
}
// scalastyle:on finalize
