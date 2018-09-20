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


package org.apache.mxnetexamples.rnn

import org.apache.mxnet.{Shape, Symbol}

import scala.collection.mutable.ArrayBuffer

object Lstm {

  final case class LSTMState(c: Symbol, h: Symbol)
  final case class LSTMParam(i2hWeight: Symbol, i2hBias: Symbol,
                             h2hWeight: Symbol, h2hBias: Symbol)

  // LSTM Cell symbol
  def lstm(numHidden: Int, inData: Symbol, prevState: LSTMState,
           param: LSTMParam, seqIdx: Int, layerIdx: Int, dropout: Float = 0f): LSTMState = {
    val inDataa = {
      if (dropout > 0f) Symbol.api.Dropout(data = inData, p = dropout)
      else inData
    }
    val i2h = Symbol.api.FullyConnected(data = inDataa, weight = param.i2hWeight,
      bias = param.i2hBias, num_hidden = numHidden * 4, name = s"t${seqIdx}_l${layerIdx}_i2h")
    val h2h = Symbol.api.FullyConnected(data = prevState.h, weight = param.h2hWeight,
      bias = param.h2hBias, num_hidden = numHidden * 4, name = s"t${seqIdx}_l${layerIdx}_h2h")
    val gates = i2h + h2h
    val sliceGates = Symbol.api.SliceChannel(data = gates, num_outputs = 4,
      name = s"t${seqIdx}_l${layerIdx}_slice")
    val ingate = Symbol.api.Activation(data = sliceGates.get(0), act_type = "sigmoid")
    val inTransform = Symbol.api.Activation(data = sliceGates.get(1), act_type = "tanh")
    val forgetGate = Symbol.api.Activation(data = sliceGates.get(2), act_type = "sigmoid")
    val outGate = Symbol.api.Activation(data = sliceGates.get(3), act_type = "sigmoid")
    val nextC = (forgetGate * prevState.c) + (ingate * inTransform)
    val nextH = outGate * Symbol.api.Activation(data = nextC, "tanh")
    LSTMState(c = nextC, h = nextH)
  }

  // we define a new unrolling function here because the original
  // one in lstm.py concats all the labels at the last layer together,
  // making the mini-batch size of the label different from the data.
  // I think the existing data-parallelization code need some modification
  // to allow this situation to work properly
  def lstmUnroll(numLstmLayer: Int, seqLen: Int, inputSize: Int, numHidden: Int,
                 numEmbed: Int, numLabel: Int, dropout: Float = 0f): Symbol = {
    val embedWeight = Symbol.Variable("embed_weight")
    val clsWeight = Symbol.Variable("cls_weight")
    val clsBias = Symbol.Variable("cls_bias")

    val paramCellsBuf = ArrayBuffer[LSTMParam]()
    val lastStatesBuf = ArrayBuffer[LSTMState]()
    for (i <- 0 until numLstmLayer) {
      paramCellsBuf.append(LSTMParam(i2hWeight = Symbol.Variable(s"l${i}_i2h_weight"),
        i2hBias = Symbol.Variable(s"l${i}_i2h_bias"),
        h2hWeight = Symbol.Variable(s"l${i}_h2h_weight"),
        h2hBias = Symbol.Variable(s"l${i}_h2h_bias")))
      lastStatesBuf.append(LSTMState(c = Symbol.Variable(s"l${i}_init_c_beta"),
        h = Symbol.Variable(s"l${i}_init_h_beta")))
    }
    val paramCells = paramCellsBuf.toArray
    val lastStates = lastStatesBuf.toArray
    require(lastStates.length == numLstmLayer)

    // embeding layer
    val data = Symbol.Variable("data")
    var label = Symbol.Variable("softmax_label")
    val embed = Symbol.api.Embedding(data = data, input_dim = inputSize,
      weight = embedWeight, output_dim = numEmbed, name = "embed")
    val wordvec = Symbol.api.SliceChannel(data = embed,
      num_outputs = seqLen, squeeze_axis = true)

    val hiddenAll = ArrayBuffer[Symbol]()
    var dpRatio = 0f
    var hidden: Symbol = null
    for (seqIdx <- 0 until seqLen) {
      hidden = wordvec.get(seqIdx)
      // stack LSTM
      for (i <- 0 until numLstmLayer) {
        if (i == 0) dpRatio = 0f else dpRatio = dropout
        val nextState = lstm(numHidden, inData = hidden,
          prevState = lastStates(i),
          param = paramCells(i),
          seqIdx = seqIdx, layerIdx = i, dropout = dpRatio)
        hidden = nextState.h
        lastStates(i) = nextState
      }
      // decoder
      if (dropout > 0f) hidden = Symbol.api.Dropout(data = hidden, p = dropout)
      hiddenAll.append(hidden)
    }
    val hiddenConcat = Symbol.api.Concat(data = hiddenAll.toArray, num_args = hiddenAll.length,
      dim = 0)
    val pred = Symbol.api.FullyConnected(data = hiddenConcat, num_hidden = numLabel,
      weight = clsWeight, bias = clsBias)
    label = Symbol.api.transpose(data = label)
    label = Symbol.api.Reshape(data = label, target_shape = Shape(0))
    val sm = Symbol.api.SoftmaxOutput(data = pred, label = label, name = "softmax")
    sm
  }

  def lstmInferenceSymbol(numLstmLayer: Int, inputSize: Int, numHidden: Int,
                          numEmbed: Int, numLabel: Int, dropout: Float = 0f): Symbol = {
    val seqIdx = 0
    val embedWeight = Symbol.Variable("embed_weight")
    val clsWeight = Symbol.Variable("cls_weight")
    val clsBias = Symbol.Variable("cls_bias")

    var paramCells = Array[LSTMParam]()
    var lastStates = Array[LSTMState]()
    for (i <- 0 until numLstmLayer) {
      paramCells = paramCells :+ LSTMParam(i2hWeight = Symbol.Variable(s"l${i}_i2h_weight"),
        i2hBias = Symbol.Variable(s"l${i}_i2h_bias"),
        h2hWeight = Symbol.Variable(s"l${i}_h2h_weight"),
        h2hBias = Symbol.Variable(s"l${i}_h2h_bias"))
      lastStates = lastStates :+ LSTMState(c = Symbol.Variable(s"l${i}_init_c_beta"),
        h = Symbol.Variable(s"l${i}_init_h_beta"))
    }
    assert(lastStates.length == numLstmLayer)

    val data = Symbol.Variable("data")

    var hidden = Symbol.api.Embedding(data = data, input_dim = inputSize,
      weight = embedWeight, output_dim = numEmbed, name = "embed")

    var dpRatio = 0f
    // stack LSTM
    for (i <- 0 until numLstmLayer) {
      if (i == 0) dpRatio = 0f else dpRatio = dropout
      val nextState = lstm(numHidden, inData = hidden,
        prevState = lastStates(i),
        param = paramCells(i),
        seqIdx = seqIdx, layerIdx = i, dropout = dpRatio)
      hidden = nextState.h
      lastStates(i) = nextState
    }
    // decoder
    if (dropout > 0f) hidden = Symbol.api.Dropout(data = hidden, p = dropout)
    val fc = Symbol.api.FullyConnected(data = hidden,
      num_hidden = numLabel, weight = clsWeight, bias = clsBias)
    val sm = Symbol.api.SoftmaxOutput(data = fc, name = "softmax")
    var output = Array(sm)
    for (state <- lastStates) {
      output = output :+ state.c
      output = output :+ state.h
    }
    Symbol.Group(output: _*)
  }
}
