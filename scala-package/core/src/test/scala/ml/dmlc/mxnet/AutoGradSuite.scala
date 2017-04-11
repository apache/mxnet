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

import ml.dmlc.mxnet.NDArrayConversions._
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.contrib.AutoGrad._
import org.scalatest.{Matchers, BeforeAndAfterAll, FunSuite}

class AutoGradSuite extends FunSuite with BeforeAndAfterAll with Matchers {

  def autoGradAssert(args: Array[NDArray], func: Array[NDArray] => Array[NDArray],
    gradF: Array[NDArray] => Array[NDArray]): Unit = {
    val gradFunc = gradAndLoss(func)
    val (gradVals, outputs) = gradFunc(args)
    val res = func(args)
    assert(outputs.zip(res).forall(x => x._1 == x._2))
    val gradRes = gradF(args)
    assert(gradVals.length == gradRes.length)
    assert(gradVals.zip(gradRes).forall(x => x._1 == x._2))
  }

  test("test unary func") {
    val x = NDArray.uniform(Map("shape" -> Shape(4, 5)))().get
    val fExp = (x: Array[NDArray]) => x.map(NDArray.exp(_).get)
    val fExpGrad = (x: Array[NDArray]) => x.map(NDArray.exp(_).get)
    autoGradAssert(Array(x), func = fExp, gradF = fExpGrad)
    val fHalf = (x: Array[NDArray]) => x.map(_ / 2f)
    val fHalfGrad = (x: Array[NDArray]) => x.map(e => NDArray.ones(e.shape) * 0.5f)
    autoGradAssert(Array(x), func = fHalf, gradF = fHalfGrad)
    val fSquare = (x: Array[NDArray]) => x.map(NDArray.square(_).get)
    val fSquareGrad = (x: Array[NDArray]) => x.map(_ * 2f)
    autoGradAssert(Array(x), func = fSquare, gradF = fSquareGrad)
  }

  test("test binary func") {
    val x = NDArray.uniform(Map("shape" -> Shape(4, 5)))().get
    val y = NDArray.uniform(Map("shape" -> Shape(4, 5)))().get
    val fAdd = (xy: Array[NDArray]) => Array(xy(0) + xy(1))
    val fAddGrad = (xy: Array[NDArray]) =>
      Array(NDArray.ones(xy(0).shape), NDArray.ones(xy(1).shape))
    autoGradAssert(Array(x, y), func = fAdd, gradF = fAddGrad)
    val fMul = (xy: Array[NDArray]) => Array(xy(0) * xy(1))
    val fMulGrad = (xy: Array[NDArray]) => xy.reverse
    autoGradAssert(Array(x, y), func = fMul, gradF = fMulGrad)
    val fCompose = (xy: Array[NDArray]) => Array(xy(0) + xy(0) * xy(1))
    val fComposeGrad = (xy: Array[NDArray]) => Array(NDArray.ones(xy(0).shape) + xy(1), xy(0))
    autoGradAssert(Array(x, y), func = fCompose, gradF = fComposeGrad)
  }
}
