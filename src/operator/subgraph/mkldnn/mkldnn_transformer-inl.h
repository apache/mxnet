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

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_INL_H_

#include "../../mshadow_op.h"
#include "../../mxnet_op.h"


namespace mxnet {
namespace op {

struct MKLDNNSelfAttParam : public dmlc::Parameter<MKLDNNSelfAttParam> {
  int heads;
  bool quantized;
  bool enable_float_output;
  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset
  DMLC_DECLARE_PARAMETER(MKLDNNSelfAttParam) {
    DMLC_DECLARE_FIELD(heads)
    .describe("Set number of heads.");
    DMLC_DECLARE_FIELD(quantized).set_default(false)
    .describe("Whether it's a quantized self attention matmul operator.");
    DMLC_DECLARE_FIELD(enable_float_output).set_default(false)
    .describe("Whether to enable float32 output.");
    DMLC_DECLARE_FIELD(min_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The minimum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized self-attention op to calculate primitive scale.");
    DMLC_DECLARE_FIELD(max_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The maximum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized self-attention op to calculate primitive scale.");
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_INL_H_
