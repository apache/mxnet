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
 * \file np_true_divide-inl.h
 * \brief Function definitions of true_divide operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_INL_H_

#include <vector>
#include "../../common/utils.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

template<typename xpu, typename OP>
void TrueDivideScalarCompute(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp || outputs[0].Size() == 0U) return;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const double alpha = nnvm::get<double>(attrs.parsed);
  if (common::is_float(inputs[0].type_flag_)) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
            s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), DType(alpha));
      });
    });
  } else {
    CHECK_EQ(outputs[0].type_flag_, kFloat32) << "true_divide only supports float32 output "
                                                 "when input's dtype is "
                                              << type_string(inputs[0].type_flag_);
    MXNET_INT_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
            s, inputs[0].Size(), outputs[0].dptr<float>(), inputs[0].dptr<DType>(), DType(alpha));
      });
    });
  }
}

template<typename xpu, typename OP>
void TrueDivideElemwiseCompute(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<TBlob> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp || outputs[0].Size() == 0U) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    if (common::is_float(inputs[0].type_flag_)) {
      MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
        Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(s, outputs[0].Size(),
                                                            outputs[0].dptr<DType>(),
                                                            inputs[0].dptr<DType>(),
                                                            inputs[1].dptr<DType>());
      });
    } else {
      CHECK_EQ(outputs[0].type_flag_, kFloat32) << "true_divide only supports float32 output "
                                                   "when input's dtype is "
                                                << type_string(inputs[0].type_flag_);
      MXNET_INT_TYPE_SWITCH(inputs[0].type_flag_, DType, {
        Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(s, outputs[0].Size(),
                                                            outputs[0].dptr<float>(),
                                                            inputs[0].dptr<DType>(),
                                                            inputs[1].dptr<DType>());
      });
    }
  });
}

template<typename xpu, typename OP>
void TrueDivideBroadcastCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0U) return;
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    TrueDivideElemwiseCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    if (req[0] == kNullOp) return;
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    BROADCAST_NDIM_SWITCH(ndim, NDim, {
      mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
      mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
      mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
      if (common::is_float(inputs[0].type_flag_)) {
        MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, DType, DType, OP>, xpu>::
            template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
                              inputs[0].dptr<DType>(), inputs[1].dptr<DType>(),
                              outputs[0].dptr<DType>());
        });
      } else {
        CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
            << "true_divide only supports float32 output when input's dtype is "
            << type_string(inputs[0].type_flag_);
        MXNET_INT_TYPE_SWITCH(inputs[0].type_flag_, DType, {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, DType, float, OP>, xpu>::
            template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
                              inputs[0].dptr<DType>(), inputs[1].dptr<DType>(),
                              outputs[0].dptr<float>());
        });
      }
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_INL_H_
