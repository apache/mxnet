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

import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}

class NDArrayCollectorSuite extends FunSuite with BeforeAndAfterAll with Matchers {

  test("auto dispose") {
    val a = NDArray.array(Array(-1f, 0f, 1f, 2f, 3f, 4f), shape = Shape(2, 3))
    var b, c: NDArray = null

    val res = NDArrayCollector.auto().withScope {
      b = NDArray.relu(a) // [0, 0, 1, 2, 3, 4]
      c = a + b           // [-1, 0, 2, 4, 6, 8]
      c.slice(0, 1)
    }

    assert(b.isDisposed)
    assert(c.isDisposed)
    assert(!res.isDisposed) // smart enough not to dispose the returned NDArray

    assert(res.toArray === Array(-1f, 0f, 2f))

    res.dispose()
  }

  test("manually dispose") {
    val a = NDArray.array(Array(-1f, 0f, 1f, 2f, 3f, 4f), shape = Shape(2, 3))
    var b, c: NDArray = null

    val collector = NDArrayCollector.manual()
    val res = collector.withScope {
      b = NDArray.relu(a) // [0, 0, 1, 2, 3, 4]
      c = a + b           // [-1, 0, 2, 4, 6, 8]
      c.slice(0, 1)
    }

    assert(res.toArray === Array(-1f, 0f, 2f))

    assert(collector.size === 2) // smart enough not to collect the returned NDArray
    assert(!b.isDisposed)
    assert(!c.isDisposed)
    assert(!res.isDisposed)

    collector.foreach(_.dispose())
    assert(b.isDisposed)
    assert(c.isDisposed)
    assert(!res.isDisposed)

    collector.clear()
    assert(collector.size === 0)

    res.dispose()
  }
}
