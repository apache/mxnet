/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file quantized_rnn-inl.h
 * \brief Common functions for quantized recurrent neural network
 * \author Zixuan Wei
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RNN_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RNN_INL_H_

namespace mxnet {
namespace op {

namespace quantized_rnn {
enum QuantizedRnnInputs { kData, kParams, kState, kStateCell };
enum QuantizedRnnInputMinMax { kDataScale, kDataShift };
enum QuantizedRnnOutputs { kOut, kStateOut, kStateCellOut };
}  // namespace quantized_rnn

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RNN_INL_H_
