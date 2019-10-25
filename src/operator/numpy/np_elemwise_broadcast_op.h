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
 * \file np_elemwise_binary_op.h
 * \brief 
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_

#include <vector>

#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

template<typename xpu, typename LOP, typename ROP>
void MixedBinaryElemwiseCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TBlob& lhs = inputs[0];
  const TBlob& rhs = inputs[1];
  const TBlob& out = outputs[0];

  CHECK((lhs.type_flag_ == mshadow::kBool) || (rhs.type_flag_ == mshadow::kBool))
    << "now supports bool with another type only";

  Stream<xpu> *s = ctx.get_stream<xpu>();

  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      const size_t size = (ElemwiseBinaryOp::minthree(out.Size(), lhs.Size(), rhs.Size())
      + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
      if (size != 0) {
        if (lhs.type_flag_ == kBool) {
          Kernel<mxnet_op::op_with_req<LOP, Req>, xpu>::Launch(
            s, size, out.dptr<DType>(), lhs.dptr<bool>(), rhs.dptr<DType>());
        } else {
          Kernel<mxnet_op::op_with_req<ROP, Req>, xpu>::Launch(
            s, size, out.dptr<DType>(), rhs.dptr<bool>(), lhs.dptr<DType>());
        }
      }
    });
  });
}

template<typename xpu, typename OP, typename LOP, typename ROP>
void MixedBinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TBlob& lhs = inputs[0];
  const TBlob& rhs = inputs[1];
  const TBlob& out = outputs[0];

  if ((out.shape_.Size() == 0U) || (req[0] == kNullOp)) return;

  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(lhs.shape_, rhs.shape_, out.shape_,
                                         &new_lshape, &new_rshape, &new_oshape);


  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  }

  CHECK((lhs.type_flag_ == mshadow::kBool) || (rhs.type_flag_ == mshadow::kBool))
    << "now supports bool with another type only";


  if (!ndim) {
    MixedBinaryElemwiseCompute<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
        mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
        if (lhs.type_flag_ == mshadow::kBool) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, LOP>, xpu>::
          template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
          lhs.dptr<bool>(), rhs.dptr<DType>(), out.dptr<DType>());
        } else {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, ROP>, xpu>::
          template LaunchEx(s, new_oshape.Size(), req[0], rstride, lstride, oshape,
          rhs.dptr<bool>(), lhs.dptr<DType>(), out.dptr<DType>());
        }
      });
    });
  }
}

template<typename xpu, typename LOP, typename ROP>
void MixedBinaryBackwardUseIn(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& lhs = inputs[1];
  const TBlob& rhs = inputs[2];
  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastBackwardUseIn<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
    return;
  }

  LOG(ERROR) << "Binary operation with mixed input data types does not support backward yet...";
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_
