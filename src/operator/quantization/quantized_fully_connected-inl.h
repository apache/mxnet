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
 * Copyright (c) 2020 by Contributors
 * \file quantized_fully_connect-inl.h
 * \brief implementation of quantized fully connect operator
*/
#ifndef MXNET_OPERATOR_QUANTIZED_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_QUANTIZED_FULLY_CONNECTED_INL_H_

namespace mxnet {
namespace op {

enum QuantizedFCFloatOutType { kInt32_qFC = 0, kFloat32_qFC, kFloat16_qFC };

template<typename Param>
static mshadow::TypeFlag GetQuantizedFCFloatOutType(const Param &param) {
  auto float_out = mshadow::kInt32;
  if (param.float_out == QuantizedFCFloatOutType::kFloat32_qFC) {
    float_out = mshadow::kFloat32;
  } else if (param.float_out == QuantizedFCFloatOutType::kFloat16_qFC) {
    float_out = mshadow::kFloat16;
  } else if (param.float_out != QuantizedFCFloatOutType::kInt32_qFC){
    LOG(FATAL) << "Unsupported float_out in params: " <<param.float_out;
  }
  return float_out;
}

struct QuantizedFullyConnectedParam : public dmlc::Parameter<QuantizedFullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  bool flatten;
  int float_out;

  DMLC_DECLARE_PARAMETER(QuantizedFullyConnectedParam) {
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor.");
    DMLC_DECLARE_FIELD(float_out)
      .add_enum("float16", QuantizedFCFloatOutType::kFloat16_qFC)
      .add_enum("float32", QuantizedFCFloatOutType::kFloat32_qFC)
      .add_enum("none", QuantizedFCFloatOutType::kInt32_qFC)
      .set_default(QuantizedFCFloatOutType::kInt32_qFC)
      .describe("Whether to fuse requantize and dequantize and have float output, "
                "and what kind of float out_type if float out is enabled.");
      
  }
  bool operator==(const QuantizedFullyConnectedParam& other) const {
    return this->num_hidden == other.num_hidden &&
           this->no_bias == other.no_bias &&
           this->flatten == other.flatten &&
           this->float_out == other.float_out;
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZED_FULLY_CONNECTED_INL_H_
