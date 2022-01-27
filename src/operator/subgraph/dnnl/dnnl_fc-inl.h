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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_FC_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_FC_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <utility>
#include <vector>

#include "operator/nn/dnnl/dnnl_fully_connected-inl.h"
#include "dnnl.hpp"

namespace mxnet {
namespace op {

static inline bool SupportDNNLFCEltwiseFusion(const std::string op_name) {
  if (op_name == "Activation" || op_name == "square" || op_name == "_npi_square" ||
      op_name == "sqrt" || op_name == "_npi_sqrt" || op_name == "exp" || op_name == "_npi_exp" ||
      op_name == "abs" || op_name == "_npi_absolute" || op_name == "clip" ||
      op_name == "LeakyReLU") {
    return true;
  } else {
    return false;
  }
}

static inline dnnl::algorithm GetDNNLEltwiseAlgo(const std::string op_name) {
  if (op_name == "square" || op_name == "_npi_square")
    return dnnl::algorithm::eltwise_square;
  else if (op_name == "sqrt" || op_name == "_npi_sqrt")
    return dnnl::algorithm::eltwise_sqrt;
  else if (op_name == "exp" || op_name == "_npi_exp")
    return dnnl::algorithm::eltwise_exp;
  else if (op_name == "abs" || op_name == "_npi_absolute")
    return dnnl::algorithm::eltwise_abs;
  else
    LOG(FATAL) << "Unsupported eltwise fusion op: " << op_name;

  return dnnl::algorithm::undef;
}

static inline bool IsOutputUint8(const DNNLFCFullParam& full_param) {
  auto alg = full_param.eltwise_param.alg;
  // TODO(ciyong): some alg doesn't support int8 so far.
  if (full_param.dnnl_param.with_eltwise &&
      (alg == dnnl::algorithm::eltwise_relu || alg == dnnl::algorithm::eltwise_logistic ||
       alg == dnnl::algorithm::eltwise_soft_relu || alg == dnnl::algorithm::eltwise_bounded_relu ||
       alg == dnnl::algorithm::eltwise_square || alg == dnnl::algorithm::eltwise_sqrt ||
       alg == dnnl::algorithm::eltwise_exp || alg == dnnl::algorithm::eltwise_abs)) {
    return true;
  }

  return false;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_FC_INL_H_
