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
 * Copyright (c) 2018 by Contributors
 * \file gradient_reversal_op-inl.h
 * \brief
 * \author Istvan Fehervari
*/
#ifndef MXNET_OPERATOR_CONTRIB_GRADIENT_REVERSAL_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_GRADIENT_REVERSAL_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct GradientReversalParam : public dmlc::Parameter<GradientReversalParam> {
  float l;
  DMLC_DECLARE_PARAMETER(GradientReversalParam) {
    DMLC_DECLARE_FIELD(l)
      .set_default(1.0)
      .describe("Lambda coefficient of the gradient reversal function.");
  }
};

template<int req>
struct gradient_reversal_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* out_grad,
                                  const DType* in_data, const float l) {
    KERNEL_ASSIGN(in_grad[i], req, out_grad[i] * -l);
  }
};

template<typename xpu>
void GradientReversalOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data = inputs[1];
  const TBlob& in_grad = outputs[0];
  const GradientReversalParam& param = nnvm::get<GradientReversalParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<gradient_reversal_backward<req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),
          in_data.dptr<DType>(), param.l);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_GRADIENT_REVERSAL_OP_INL_H_
