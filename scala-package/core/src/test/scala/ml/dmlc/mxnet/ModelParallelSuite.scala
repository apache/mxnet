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

import ml.dmlc.mxnet.CheckUtils._
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class ModelParallelSuite extends FunSuite with BeforeAndAfterAll {
  test("chain") {
    val n = 2
    val data1 = Symbol.Variable("data1")
    val data2 = Symbol.Variable("data2")

    var net: Symbol = null
    new AttrScope(Map("ctx_group" -> "dev1")).withScope {
      net = (data1 + data2) * 3
    }

    new AttrScope(Map("ctx_group" -> "dev2")).withScope {
      net = net + data1
    }

    val shape = Shape(4, 5)
    val (arr, arrGrad) =
      new Context(Context.cpu(0)).withScope {
        val arr = (0 until n).map(_ => NDArray.empty(shape))
        val arrGrad = (0 until n).map(_ => NDArray.empty(shape))
        (arr, arrGrad)
      }

    val exec1 = net.bind(Context.cpu(),
      args = arr,
      argsGrad = arrGrad,
      gradReq = "write",
      auxStates = Nil,
      group2ctx = Map("dev1" -> Context.cpu(0), "dev2" -> Context.cpu(1)),
      sharedExec = null)

    arr(0).set(1f)
    arr(1).set(2f)

    val arr2 = arr.map(_.copyTo(Context.cpu()))
    val arrGrad2 = arrGrad.map(_.copyTo(Context.cpu()))
    val exec2 = net.bind(Context.cpu(), args = arr2, argsGrad = arrGrad2)

    // Show the execution plan that involves copynode
    // scalastyle:off println
    print(exec1.debugStr)
    // scalastyle:on println

    exec1.forward()
    exec2.forward()
    assert(reldiff(exec1.outputs(0).copyTo(Context.cpu()),
        exec2.outputs(0).copyTo(Context.cpu())) < 1e-6f)

    val outGrad = NDArray.ones(shape, Context.cpu(1))
    exec1.backward(Array(outGrad))
    exec2.backward(Array(outGrad.copyTo(Context.cpu())))
    (arrGrad zip arrGrad2) foreach { case (a, b) =>
      assert(reldiff(a, b) < 1e-6f)
    }
  }
}
