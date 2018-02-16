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
 * Copyright (c) 2015 by Contributors
 * \file lrn-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_NN_LRN_INL_H_
#define MXNET_OPERATOR_NN_LRN_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace lrn_enum {
enum LRNInputs {kData};
enum LRNOutputs {kOut, kTmpNorm};
}  // namespace lrn_enum

struct LRNParam : public dmlc::Parameter<LRNParam> {
  float alpha;
  float beta;
  float knorm;
  uint32_t nsize;
  DMLC_DECLARE_PARAMETER(LRNParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1e-4f)
    .describe("The variance scaling parameter :math:`\alpha` in the LRN expression.");
    DMLC_DECLARE_FIELD(beta).set_default(0.75f)
    .describe("The power parameter :math:`\beta` in the LRN expression.");
    DMLC_DECLARE_FIELD(knorm).set_default(2.0f)
    .describe("The parameter :math:`k` in the LRN expression.");
    DMLC_DECLARE_FIELD(nsize)
    .describe("normalization window width in elements.");
  }
};  // struct LRNParam

template<typename xpu>
void LRNForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const LRNParam& param_ = nnvm::get<LRNParam>(attrs.parsed);
  // TODO(xxx): Test with gradient chceker
  CHECK_EQ(in_data.size(), 1U);
  CHECK_EQ(out_data.size(), 2U);
  // CHECK_EQ(req.size(), 2);
  CHECK_EQ(param_.nsize % 2, 1U) << "LRN only supports odd values for local_size";
  const real_t salpha = param_.alpha / param_.nsize;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4> data = in_data[lrn_enum::kData].get<xpu, 4, real_t>(s);
  Tensor<xpu, 4> out = out_data[lrn_enum::kOut].get<xpu, 4, real_t>(s);
  Tensor<xpu, 4> tmp_norm = out_data[lrn_enum::kTmpNorm].get<xpu, 4, real_t>(s);
  tmp_norm = chpool<red::sum>(F<mshadow_op::square>(data) , param_.nsize) * salpha + param_.knorm;
  Assign(out, req[lrn_enum::kOut], data *  F<mshadow_op::power>(tmp_norm, -param_.beta));
}

template<typename xpu>
void LRNBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                 const TBlob &out_grad, const TBlob &in_data,
                 const TBlob &out_norm, const OpReqType &req,
                 const TBlob &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const LRNParam& param_ = nnvm::get<LRNParam>(attrs.parsed);
  const real_t salpha = param_.alpha / param_.nsize;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4> grad = out_grad.get<xpu, 4, real_t>(s);
  Tensor<xpu, 4> tmp_norm = out_norm.get<xpu, 4, real_t>(s);
  Tensor<xpu, 4> data = in_data.get<xpu, 4, real_t>(s);
  Tensor<xpu, 4> grad_in = in_grad.get<xpu, 4, real_t>(s);
  grad_in = grad * F<mshadow_op::power>(tmp_norm, -param_.beta);
  grad_in += (- 2.0f * param_.beta * salpha) *
      chpool<red::sum>(grad * data *
                       F<mshadow_op::power>(tmp_norm, -param_.beta - 1.0f),
                       param_.nsize)  * data;
}

template<typename xpu>
void LRNCompute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  LRNForward<xpu>(attrs, ctx, inputs, req, outputs);
}

template<typename xpu>
void LRNGradCompute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  LRNBackward<xpu>(attrs, ctx, inputs[0],  // out_grad
                   inputs[1],              // in_data
                   inputs[2],              // out_norm
                   req[lrn_enum::kData], outputs[lrn_enum::kData]);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_LRN_INL_H_
