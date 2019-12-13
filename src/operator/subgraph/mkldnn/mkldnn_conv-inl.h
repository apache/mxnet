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

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_CONV_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_CONV_INL_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <utility>
#include <vector>
#include "../../nn/activation-inl.h"
#include "../../nn/batch_norm-inl.h"
#include "../../nn/convolution-inl.h"
#include "../../nn/mkldnn/mkldnn_convolution-inl.h"

namespace mxnet {
namespace op {

struct MKLDNNConvFusionParam {
  MKLDNNConvFullParam full_conv_param;
  std::shared_ptr<BatchNormParam> bn_param;
};

static inline bool IsOutputUInt8(const MKLDNNConvFusionParam& param) {
  bool result = false;
  const auto& mkldnn_param = param.full_conv_param.mkldnn_param;
  auto IsOutputUInt8Helper = [](const MKLDNNPostEltwiseParam &param) {
    return ((param.alg == mkldnn::algorithm::eltwise_relu && param.alpha == 0.f) ||
            param.alg == mkldnn::algorithm::eltwise_logistic ||
            param.alg == mkldnn::algorithm::eltwise_soft_relu ||
            param.alg == mkldnn::algorithm::eltwise_bounded_relu);
  };
  if ((!mkldnn_param.with_sum) && mkldnn_param.with_act) {
    CHECK(param.full_conv_param.act_param.alg != mkldnn::algorithm::undef);
    result = IsOutputUInt8Helper(param.full_conv_param.act_param);
  } else if (mkldnn_param.with_postsum_act) {
    CHECK(param.full_conv_param.postsum_act_param.alg != mkldnn::algorithm::undef);
    result = IsOutputUInt8Helper(param.full_conv_param.postsum_act_param);
  }
  return result;
}

enum MKLDNNConvOpOutputs { kOut, kMin, kMax };

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_CONV_INL_H_
