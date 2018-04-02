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

import org.scalatest.{BeforeAndAfterAll, FunSuite}

// scalastyle:off finalize
class Leakable(enableTracing: Boolean = false, markDisposed: Boolean = false)
    extends WarnIfNotDisposed {
  def isDisposed: Boolean = markDisposed
  override protected def tracingEnabled = enableTracing

  var warningWasLogged: Boolean = false
  def getCreationTrace: Option[Array[StackTraceElement]] = creationTrace

  override def finalize(): Unit = super.finalize()
  override protected def logDisposeWarning() = {
    warningWasLogged = true
  }
}
// scalastyle:on finalize

class WarnIfNotDisposedSuite extends FunSuite with BeforeAndAfterAll {
  test("trace collected if tracing enabled") {
    val leakable = new Leakable(enableTracing = true)

    val trace = leakable.getCreationTrace
    assert(trace.isDefined)
    assert(trace.get.exists(el => el.getClassName() == getClass().getName()))
  }

  test("trace not collected if tracing disabled") {
    val leakable = new Leakable(enableTracing = false)
    assert(!leakable.getCreationTrace.isDefined)
  }

  test("no warning logged if object disposed") {
    val notLeaked = new Leakable(markDisposed = true)
    notLeaked.finalize()
    assert(!notLeaked.warningWasLogged)
  }

  test("warning logged if object not disposed") {
    val leaked = new Leakable(markDisposed = false)
    leaked.finalize()
    assert(leaked.warningWasLogged)
  }
}
