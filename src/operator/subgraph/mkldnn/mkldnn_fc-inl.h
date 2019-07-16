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

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_INL_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <utility>
#include <vector>
#include "mkldnn.hpp"
#include "../../nn/activation-inl.h"

namespace mxnet {
namespace op {

namespace sg_mkldnn_fc_fusion {
  std::set<std::string> eltwise_name_list = {
    "Activation",
    "square",
    "square_root",
    "exp",
    "abs",
    "clip"
  };
}

static inline bool SupportMKLDNNFCEltwiseFusion(const std::string op_name) {
  auto res = sg_mkldnn_fc_fusion::eltwise_name_list.find(op_name);
  if (res != sg_mkldnn_fc_fusion::eltwise_name_list.end())
    return true;
  else
    return false;
}

static inline mkldnn::algorithm GetMKLDNNEltwiseAlgo(const std::string op_name) {
  switch (op_name) {
    case "square":
      return mkldnn::algorithm::eltwise_square;
    case "square_root":
      return mkldnn::algorithm::eltwise_sqrt;
    case "exp:
      return mkldnn::algorithm::exp;
    case "abs":
      return mkldnn::algorithm::abs;
    default:
      LOG(FATAL) << "Unsupported eltwise fusion op: " << op_name; 
  }
}

static inline bool IsOutputUInt8(const MKLDNNConvFusionParam& param) {
  bool result = false;
  const auto& mkldnn_param = param.full_conv_param.mkldnn_param;
  auto IsOutputUInt8Helper = [](const mkldnn::algorithm& act_alg) {
    return (act_alg == mkldnn::algorithm::eltwise_relu ||
            act_alg == mkldnn::algorithm::eltwise_logistic ||
            act_alg == mkldnn::algorithm::eltwise_soft_relu ||
            act_alg == mkldnn::algorithm::eltwise_bounded_relu);
  };
  if ((!mkldnn_param.with_sum) && mkldnn_param.with_act) {
    CHECK(param.full_conv_param.act_param.alg != mkldnn::algorithm::algorithm_undef);
    result = IsOutputUInt8Helper(param.full_conv_param.act_param.alg);
  } else if (mkldnn_param.with_postsum_act) {
    CHECK(param.full_conv_param.postsum_act_param.alg != mkldnn::algorithm::algorithm_undef);
    result = IsOutputUInt8Helper(param.full_conv_param.postsum_act_param.alg);
  }
  return result;
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_INL_H_
