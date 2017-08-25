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

object CheckUtils {
  def reldiff(a: NDArray, b: NDArray): Float = {
    val diff = NDArray.sum(NDArray.abs(a - b)).toScalar
    val norm = NDArray.sum(NDArray.abs(a)).toScalar
    diff / norm
  }

  def reldiff(a: Array[Float], b: Array[Float]): Float = {
    val diff =
      (a zip b).map { case (aElem, bElem) => Math.abs(aElem - bElem) }.sum
    val norm: Float = a.reduce(Math.abs(_) + Math.abs(_))
    diff / norm
  }
}
