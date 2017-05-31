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

package ml.dmlc.mxnet.rnn

import ml.dmlc.mxnet.DType
import ml.dmlc.mxnet.DType.DType
import ml.dmlc.mxnet.{Shape, Symbol}

class RNNStateInfo(val shape: Shape,
                   val layout: String,
                   val kwargsOpt: Option[Map[String, Any]] = None)

trait RNNStateInitFunction {
  def invoke(name: String, stateInfo: RNNStateInfo = null): Symbol
}

class RNNStateInitFuncZeros extends RNNStateInitFunction {
  override def invoke(name: String, stateInfo: RNNStateInfo): Symbol = {
    val kwargs = stateInfo.kwargsOpt.getOrElse(Map.empty[String, Any])
    val dtype = kwargs.getOrElse("dtype", DType.Float32).asInstanceOf[DType]
    Symbol.zeros(stateInfo.shape, dtype)
  }
}
