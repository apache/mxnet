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
 * \file np_true_divide-inl.h
 * \brief Function definitions of true_divide operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_INL_H_

#include <vector>
#include "../../common/utils.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../numpy/np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

template <typename xpu, typename OP>
void TrueDivideScalarCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp || outputs[0].Size() == 0U)
    return;
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  Stream<xpu>* s                      = ctx.get_stream<xpu>();
  const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
  const double alpha                  = param.scalar;
  const TBlob& data                   = inputs[0];
  const TBlob& out                    = outputs[0];
  if (out.type_flag_ == data.type_flag_) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<op_with_req<OP, Req>, xpu>::Launch(
            s, data.Size(), out.dptr<DType>(), data.dptr<DType>(), DType(alpha));
      });
    });
  } else {
    CHECK(out.type_flag_ == mshadow::kFloat32 || out.type_flag_ == mshadow::kFloat64)
        << "true_divide only supports float32 and float64"
           " output when input's dtype is "
        << type_string(inputs[0].type_flag_);
    MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, ODType, {
      MXNET_INT_TYPE_SWITCH(inputs[0].type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          Kernel<op_with_req<OP, Req>, xpu>::Launch(
              s, data.Size(), out.dptr<ODType>(), data.dptr<DType>(), static_cast<ODType>(alpha));
        });
      });
    });
  }
}

template <typename xpu>
void TrueDivideElemwiseCompute(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp || outputs[0].Size() == 0U)
    return;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TBlob& lhs = inputs[0];
  const TBlob& rhs = inputs[1];
  const TBlob& out = outputs[0];
  if (lhs.type_flag_ == rhs.type_flag_) {
    // Case when types of the 2 input tensors are the same
    if (common::is_float(lhs.type_flag_)) {
      // If both are the same floats, normal launch
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MSHADOW_REAL_TYPE_SWITCH_EX(lhs.type_flag_, DType, _, {
          Kernel<op_with_req<mshadow_op::true_divide, Req>, xpu>::Launch(
              s, out.Size(), out.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>());
        });
      });
    } else {
      // If both are the same integers, output is float32 or float64
      CHECK_EQ(out.type_flag_, mxnet::common::GetDefaultDtype())
          << "true_divide only supports float32 and float64"
             " output when input's dtype is "
          << type_string(lhs.type_flag_);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MXNET_INT_TYPE_SWITCH(lhs.type_flag_, DType, {
          Kernel<op_with_req<mshadow_op::true_divide, Req>, xpu>::Launch(
              s, out.Size(), out.dptr<float>(), lhs.dptr<DType>(), rhs.dptr<DType>());
        });
      });
    }
  } else {
    // Case when types of the 2 input tensors are different
    if ((common::is_float(lhs.type_flag_)) && (common::is_float(rhs.type_flag_))) {
      // both lhs and rhs are float types, output type is the more precise one
      TBlob temp_tblob;
      if (lhs.type_flag_ == out.type_flag_) {
        MSHADOW_REAL_TYPE_SWITCH(lhs.type_flag_, LType, {
          Tensor<xpu, 1, LType> temp_tensor =
              ctx.requested[0].get_space_typed<xpu, 1, LType>(Shape1(rhs.Size()), s);
          temp_tblob = TBlob(temp_tensor);
        });
        CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob});
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
            Kernel<op_with_req<mshadow_op::true_divide, Req>, xpu>::Launch(
                s, out.Size(), out.dptr<DType>(), lhs.dptr<DType>(), temp_tblob.dptr<DType>());
          });
        });
      } else {
        MSHADOW_REAL_TYPE_SWITCH(rhs.type_flag_, RType, {
          Tensor<xpu, 1, RType> temp_tensor =
              ctx.requested[0].get_space_typed<xpu, 1, RType>(Shape1(lhs.Size()), s);
          temp_tblob = TBlob(temp_tensor);
        });
        CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob});
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
            Kernel<op_with_req<mshadow_op::true_divide, Req>, xpu>::Launch(
                s, out.Size(), out.dptr<DType>(), temp_tblob.dptr<DType>(), rhs.dptr<DType>());
          });
        });
      }
    } else if (common::is_float(lhs.type_flag_) || common::is_float(rhs.type_flag_)) {
      // one is float type, the other is integer type, the output type should be the same as float
      CHECK_EQ(out.type_flag_, common::is_float(lhs.type_flag_) ? lhs.type_flag_ : rhs.type_flag_)
          << "This case out type should be same as the float type";
      if (common::is_float(lhs.type_flag_)) {
        // lhs is the float one
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          MSHADOW_REAL_TYPE_SWITCH(lhs.type_flag_, LType, {
            MXNET_INT_TYPE_SWITCH(rhs.type_flag_, RType, {
              Kernel<op_with_req<mshadow_op::rtrue_divide, Req>, xpu>::Launch(
                  s, out.Size(), out.dptr<LType>(), rhs.dptr<RType>(), lhs.dptr<LType>());
            });
          });
        });
      } else {
        // rhs is the float one
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          MXNET_INT_TYPE_SWITCH(lhs.type_flag_, LType, {
            MSHADOW_REAL_TYPE_SWITCH(rhs.type_flag_, RType, {
              Kernel<op_with_req<mshadow_op::true_divide, Req>, xpu>::Launch(
                  s, out.Size(), out.dptr<RType>(), lhs.dptr<LType>(), rhs.dptr<RType>());
            });
          });
        });
      }
    } else {
      // lhs is integer type, rhs is integer type, output type should be float
      LOG(FATAL) << "not implemented yet...";
    }
  }
}

template <typename xpu>
void TrueDivideBroadcastCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (outputs[0].shape_.Size() == 0U)
    return;
  CHECK_EQ(inputs.size(), 2U);
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(
      inputs[0].shape_, inputs[1].shape_, outputs[0].shape_, &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    TrueDivideElemwiseCompute<xpu>(attrs, ctx, inputs, req, outputs);
  } else {
    if (req[0] == kNullOp)
      return;
    mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
    const TBlob& lhs        = inputs[0];
    const TBlob& rhs        = inputs[1];
    const TBlob& out        = outputs[0];
    BROADCAST_NDIM_SWITCH(ndim, NDim, {
      mshadow::Shape<NDim> oshape  = new_oshape.get<NDim>();
      mshadow::Shape<NDim> lstride = calc_stride(new_lshape.get<NDim>());
      mshadow::Shape<NDim> rstride = calc_stride(new_rshape.get<NDim>());
      if (lhs.type_flag_ == rhs.type_flag_) {
        // When the both inputs have the same data types
        if (common::is_float(lhs.type_flag_)) {
          // If both inputs are the same float types, output is the same float type
          MSHADOW_REAL_TYPE_SWITCH(lhs.type_flag_, DType, {
            Kernel<binary_broadcast_kernel<NDim, mshadow_op::true_divide>, xpu>::template LaunchEx(
                s,
                new_oshape.Size(),
                req[0],
                lstride,
                rstride,
                oshape,
                lhs.dptr<DType>(),
                rhs.dptr<DType>(),
                out.dptr<DType>());
          });
        } else {
          CHECK_EQ(out.type_flag_, mxnet::common::GetDefaultDtype())
              << "true_divide only supports float32 and float64 output when input's dtype is "
              << type_string(lhs.type_flag_);
          MXNET_INT_TYPE_SWITCH(lhs.type_flag_, DType, {
            // If both inputs are the same integer types, output is float type
            Kernel<binary_broadcast_kernel<NDim, mshadow_op::true_divide>, xpu>::template LaunchEx(
                s,
                new_oshape.Size(),
                req[0],
                lstride,
                rstride,
                oshape,
                lhs.dptr<DType>(),
                rhs.dptr<DType>(),
                out.dptr<float>());
          });
        }
      } else {
        if ((common::is_float(lhs.type_flag_)) && (common::is_float(rhs.type_flag_))) {
          // lhs and rhs have different float types, the output is the more precise one
          TBlob temp_tblob;
          if (lhs.type_flag_ == out.type_flag_) {
            MSHADOW_REAL_TYPE_SWITCH(lhs.type_flag_, LType, {
              Tensor<xpu, 1, LType> temp_tensor =
                  ctx.requested[0].get_space_typed<xpu, 1, LType>(Shape1(rhs.Size()), s);
              temp_tblob = TBlob(temp_tensor);
            });
            CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob});
            MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
              Kernel<binary_broadcast_kernel<NDim, mshadow_op::true_divide>,
                     xpu>::template LaunchEx(s,
                                             new_oshape.Size(),
                                             req[0],
                                             lstride,
                                             rstride,
                                             oshape,
                                             lhs.dptr<DType>(),
                                             temp_tblob.dptr<DType>(),
                                             out.dptr<DType>());
            });
          } else {
            MSHADOW_REAL_TYPE_SWITCH(rhs.type_flag_, RType, {
              Tensor<xpu, 1, RType> temp_tensor =
                  ctx.requested[0].get_space_typed<xpu, 1, RType>(Shape1(lhs.Size()), s);
              temp_tblob = TBlob(temp_tensor);
            });
            CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob});
            MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
              Kernel<binary_broadcast_kernel<NDim, mshadow_op::true_divide>,
                     xpu>::template LaunchEx(s,
                                             new_oshape.Size(),
                                             req[0],
                                             lstride,
                                             rstride,
                                             oshape,
                                             temp_tblob.dptr<DType>(),
                                             rhs.dptr<DType>(),
                                             out.dptr<DType>());
            });
          }
        } else if (common::is_float(lhs.type_flag_) || common::is_float(rhs.type_flag_)) {
          // one of lhs and rhs is float, the output is the same type as the float one
          if (common::is_float(lhs.type_flag_)) {
            // lhs is float type, output will be the same float type
            CHECK_EQ(lhs.type_flag_, out.type_flag_)
                << "lhs should have the same type as out, infer type broken?";
            MSHADOW_REAL_TYPE_SWITCH(lhs.type_flag_, LType, {
              MXNET_INT_TYPE_SWITCH(rhs.type_flag_, RType, {
                Kernel<binary_broadcast_kernel<NDim, mshadow_op::rtrue_divide>,
                       xpu>::template LaunchEx(s,
                                               new_oshape.Size(),
                                               req[0],
                                               rstride,
                                               lstride,
                                               oshape,
                                               rhs.dptr<RType>(),
                                               lhs.dptr<LType>(),
                                               out.dptr<LType>());
              });
            });
          } else {
            // rhs is float type, output will be the same float type
            CHECK_EQ(rhs.type_flag_, out.type_flag_)
                << "rhs should have the same type as out, infer type broken?";
            MXNET_INT_TYPE_SWITCH(lhs.type_flag_, LType, {
              MSHADOW_REAL_TYPE_SWITCH(rhs.type_flag_, RType, {
                Kernel<binary_broadcast_kernel<NDim, mshadow_op::true_divide>,
                       xpu>::template LaunchEx(s,
                                               new_oshape.Size(),
                                               req[0],
                                               lstride,
                                               rstride,
                                               oshape,
                                               lhs.dptr<LType>(),
                                               rhs.dptr<RType>(),
                                               out.dptr<RType>());
              });
            });
          }
        } else {
          // lhs and rhs have different integer types, the output is float type
          LOG(FATAL) << "not implemented yet...";
        }
      }
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_INL_H_
