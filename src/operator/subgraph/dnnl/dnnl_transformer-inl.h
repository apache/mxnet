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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_INL_H_

#include "operator/mshadow_op.h"
#include "operator/mxnet_op.h"

namespace mxnet {
namespace op {

struct DNNLSelfAttParam : public dmlc::Parameter<DNNLSelfAttParam> {
  int heads;
  bool quantized;
  dmlc::optional<float> min_calib_range;     // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;     // max float value calculated from calibration dataset
  dmlc::optional<int> enabled_float_output;  // mshadow dtype of a fused amp_cast node

  DMLC_DECLARE_PARAMETER(DNNLSelfAttParam) {
    DMLC_DECLARE_FIELD(heads).describe("Set number of heads.");
    DMLC_DECLARE_FIELD(quantized).set_default(false).describe(
        "Whether it's a quantized self attention matmul operator.");
    DMLC_DECLARE_FIELD(min_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The minimum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized self-attention op to calculate primitive scale.");
    DMLC_DECLARE_FIELD(max_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The maximum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized self-attention op to calculate primitive scale.");
    DNNL_DECLARE_ENABLED_FLOAT_OUTPUT_PARAMETER();
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_INL_H_
