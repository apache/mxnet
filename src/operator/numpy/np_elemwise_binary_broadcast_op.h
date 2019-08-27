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
 * \file np_elemwise_binary_broadcast_op.h
 * \brief Function definition of elementwise binary broadcast related operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BINARY_BROADCAST_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BINARY_BROADCAST_OP_H_

#include <vector>
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

/*! \brief Minimum of three */
static MSHADOW_XINLINE size_t minthree(const size_t a, const size_t b, const size_t c) {
  return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

template<typename xpu, typename OP>
static void BitCompute(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  if (req[0] != kNullOp) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 1U);
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      MXNET_INT_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        const size_t size = (minthree(outputs[0].Size(), inputs[0].Size(), inputs[1].Size()) +
                            DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
        if (size != 0) {
          Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(s, size,
                                                              outputs[0].dptr<DType>(),
                                                              inputs[0].dptr<DType>(),
                                                              inputs[1].dptr<DType>());
        }
      });
    });
  }
}

template<typename xpu, typename OP>
void BitBinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0U) return;
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    BitCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    if (req[0] != kNullOp) {
      mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
      MXNET_INT_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
          mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
          mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, DType, OP>, xpu>::
          template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
                            inputs[0].dptr<DType>(), inputs[1].dptr<DType>(),
                            outputs[0].dptr<DType>());
        });
      });
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BINARY_BROADCAST_OP_H_
