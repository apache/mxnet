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
 * \file dnnl_batch_dot-inl.h
 * \author: Bartosz Kuncer, bartosz.kuncer@intel.com
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_DOT_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_DOT_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <numeric>
#include <utility>
#include <vector>

#include "operator/tensor/dot-inl.h"
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

struct DNNLDotParam : public dmlc::Parameter<DNNLDotParam> {
  bool transpose_a;
  bool transpose_b;
  bool quantized;

  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset
  dmlc::optional<int> enabled_float_output;
  DMLC_DECLARE_PARAMETER(DNNLDotParam) {
    DMLC_DECLARE_FIELD(transpose_a)
        .describe("If true then transpose the first input before dot.")
        .set_default(false);
    DMLC_DECLARE_FIELD(transpose_b)
        .describe("If true then transpose the second input before dot.")
        .set_default(false);
    DMLC_DECLARE_FIELD(quantized).set_default(false).describe("enable quantization");
    DMLC_DECLARE_FIELD(min_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The minimum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized convolution op to calculate primitive scale");
    DMLC_DECLARE_FIELD(max_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The maximum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized convolution op to calculate primitive scale");
    DNNL_DECLARE_ENABLED_FLOAT_OUTPUT_PARAMETER();
  }

  bool operator==(const DNNLDotParam& other) const {
    return this->transpose_a == other.transpose_a && this->transpose_b == other.transpose_b &&
           this->quantized == other.quantized && this->min_calib_range == other.min_calib_range &&
           this->max_calib_range == other.max_calib_range;
  }
};

using batch_dot_fwd_t    = dnnl::matmul;
using batch_dot_fwd_pd_t = dnnl::matmul::primitive_desc;

typedef ParamOpSign<DNNLDotParam> BatchDotSignature;

class DNNLBatchDotFwd {
 public:
  static DNNLBatchDotFwd& GetCached(const DNNLDotParam& param,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<NDArray>& outputs);

  DNNLBatchDotFwd(const DNNLDotParam& param,
                  const std::vector<NDArray>& inputs,
                  const std::vector<NDArray>& outputs);

  void Execute(const OpContext& ctx,
               const DNNLDotParam& param,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  std::shared_ptr<batch_dot_fwd_t> fwd;
  std::shared_ptr<batch_dot_fwd_pd_t> fwd_pd;
};

template <bool subgraph = true>
void DNNLBatchDotForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  DNNLDotParam dnnl_param;
  if (!subgraph) {
    const DotParam& param  = nnvm::get<DotParam>(attrs.parsed);
    dnnl_param.transpose_a = param.transpose_a;
    dnnl_param.transpose_b = param.transpose_b;
    dnnl_param.quantized   = false;
  } else {
    dnnl_param = nnvm::get<DNNLDotParam>(attrs.parsed);
  }

  DNNLBatchDotFwd& fwd = DNNLBatchDotFwd::GetCached(dnnl_param, inputs, outputs);
  fwd.Execute(ctx, dnnl_param, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_DOT_INL_H__
