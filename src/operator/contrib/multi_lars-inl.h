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
 *  Copyright (c) 2019 by Contributors
 * \file multi_lars-inl.h
 * \brief vectorized lars coefficient computed from sums of squared weights and grads
 * \author Clement Fuji Tsang
 */
#ifndef MXNET_OPERATOR_CONTRIB_MULTI_LARS_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTI_LARS_INL_H_

#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/init_op.h"
#include "../tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

struct LARSParam : public dmlc::Parameter<LARSParam> {
  float eta;
  float eps;
  float rescale_grad;
  DMLC_DECLARE_PARAMETER(LARSParam) {
    DMLC_DECLARE_FIELD(eta)
    .describe("LARS eta");
    DMLC_DECLARE_FIELD(eps)
    .describe("LARS eps");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Gradient rescaling factor");
  }
};

struct MultiLARSKernel {
  MSHADOW_XINLINE static void Map(int i, float* out_data, const float* lrs,
                                  const float* weights_sum_sq, const float* grads_sum_sq,
                                  const float* wds, const float eta, const float eps,
                                  const float rescale_grad, const OpReqType req) {
    float w_norm = sqrtf(weights_sum_sq[i]);
    bool is_lars_valid = w_norm > 0. && grads_sum_sq[i] > 0.;
    KERNEL_ASSIGN(out_data[i], req, is_lars_valid ?
        lrs[i] * eta * w_norm / (sqrtf(grads_sum_sq[i]) * rescale_grad + wds[i] * w_norm + eps) :
        lrs[i]);
  }
};

template<typename xpu>
inline void MultiLARS(const nnvm::NodeAttrs& attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  auto param = nnvm::get<LARSParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  if (inputs[0].type_flag_ != mshadow::kFloat32)
    LOG(FATAL) << "MultiLARS only support float";
  Tensor<xpu, 2, float> lrs = inputs[0].FlatTo2D<xpu, float>(s);
  Tensor<xpu, 2, float> weights_sum_sq = inputs[1].FlatTo2D<xpu, float>(s);
  Tensor<xpu, 2, float> grads_sum_sq = inputs[2].FlatTo2D<xpu, float>(s);
  Tensor<xpu, 2, float> wds = inputs[3].FlatTo2D<xpu, float>(s);
  Tensor<xpu, 2, float> out_data = outputs[0].FlatTo2D<xpu, float>(s);
  Kernel<MultiLARSKernel, xpu>::Launch(s, weights_sum_sq.shape_.Size(), out_data.dptr_,
                                       lrs.dptr_, weights_sum_sq.dptr_, grads_sum_sq.dptr_,
                                       wds.dptr_, param.eta, param.eps,
                                       param.rescale_grad, req[0]);
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_CONTRIB_MULTI_LARS_INL_H_
