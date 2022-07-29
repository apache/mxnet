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
 * \file np_elemwise_binary_op.h
 * \brief Function definition of elemwise and broadcast operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_

#include <algorithm>
#include <utility>
#include <vector>
#include <string>

#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

inline void PrintErrorMessage(const std::string& op_name, const int dtype1, const int dtype2) {
  LOG(FATAL) << "Operator " << op_name << " does not support combination of "
             << mshadow::dtype_string(dtype1) << " with " << mshadow::dtype_string(dtype2)
             << " yet...";
}

template <typename xpu, typename OP>
void MixedAllRealBinaryElemwiseCompute(const std::string& op_name,
                                       const OpContext& ctx,
                                       const TBlob& lhs,
                                       const TBlob& rhs,
                                       const TBlob& out,
                                       const OpReqType req) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(lhs.type_flag_, out.type_flag_);

  Stream<xpu>* s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    const size_t size = (ElemwiseBinaryOp::minthree(out.Size(), lhs.Size(), rhs.Size()) +
                         DataType<DType>::kLanes - 1) /
                        DataType<DType>::kLanes;
    if (size == 0)
      return;

    switch (lhs.type_flag_) {
      case mshadow::kFloat32: {
        if (rhs.type_flag_ == mshadow::kFloat16) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
                s, size, out.dptr<float>(), rhs.dptr<mshadow::half::half_t>(), lhs.dptr<float>());
          });
        } else if (rhs.type_flag_ == mshadow::kBfloat16) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
                s, size, out.dptr<float>(), rhs.dptr<mshadow::bfloat::bf16_t>(), lhs.dptr<float>());
          });
        } else {
          PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        }
        break;
      }
      case mshadow::kFloat64: {
        if (rhs.type_flag_ == mshadow::kFloat16) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
                s, size, out.dptr<double>(), rhs.dptr<mshadow::half::half_t>(), lhs.dptr<double>());
          });
        } else if (rhs.type_flag_ == mshadow::kFloat32) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
                s, size, out.dptr<double>(), rhs.dptr<float>(), lhs.dptr<double>());
          });
        } else if (rhs.type_flag_ == mshadow::kBfloat16) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(s,
                                                                size,
                                                                out.dptr<double>(),
                                                                rhs.dptr<mshadow::bfloat::bf16_t>(),
                                                                lhs.dptr<double>());
          });
        } else {
          PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        }
        break;
      }
      default: {
        PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        break;
      }
    }
  });
}

template <typename xpu, typename OP>
void MixedIntRealBinaryElemwiseCompute(const OpContext& ctx,
                                       const TBlob& lhs,
                                       const TBlob& rhs,
                                       const TBlob& out,
                                       const OpReqType req) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(lhs.type_flag_, out.type_flag_);

  Stream<xpu>* s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, FType, {
    const size_t size = (ElemwiseBinaryOp::minthree(out.Size(), lhs.Size(), rhs.Size()) +
                         DataType<FType>::kLanes - 1) /
                        DataType<FType>::kLanes;
    if (size == 0)
      return;

    MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(rhs.type_flag_, IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
            s, size, out.dptr<FType>(), rhs.dptr<IType>(), lhs.dptr<FType>());
      });
    });
  });
}

template <typename xpu, typename OP>
void MixedIntBinaryElemwiseCompute(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const TBlob& lhs,
                                   const TBlob& rhs,
                                   const TBlob& out,
                                   const OpReqType req) {
  using namespace mshadow;
  using namespace mxnet_op;

  Stream<xpu>* s = ctx.get_stream<xpu>();
  TBlob temp_tblob;
  if (lhs.type_flag_ == out.type_flag_) {
    MXNET_INT_TYPE_SWITCH_EXT(lhs.type_flag_, LType, {
      Tensor<xpu, 1, LType> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, LType>(Shape1(rhs.Size()), s);
      temp_tblob = TBlob(temp_tensor);
    });
    CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob});
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      MXNET_INT_TYPE_SWITCH_EXT(out.type_flag_, DType, {
        const size_t size = (ElemwiseBinaryOp::minthree(out.Size(), lhs.Size(), temp_tblob.Size()) +
                             DataType<DType>::kLanes - 1) /
                            DataType<DType>::kLanes;
        if (size != 0) {
          Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
              s, size, out.dptr<DType>(), lhs.dptr<DType>(), temp_tblob.dptr<DType>());
        }
      });
    });
  } else if (rhs.type_flag_ == out.type_flag_) {
    MXNET_INT_TYPE_SWITCH_EXT(rhs.type_flag_, RType, {
      Tensor<xpu, 1, RType> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, RType>(Shape1(lhs.Size()), s);
      temp_tblob = TBlob(temp_tensor);
    });
    CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob});
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      MXNET_INT_TYPE_SWITCH_EXT(out.type_flag_, DType, {
        const size_t size = (ElemwiseBinaryOp::minthree(out.Size(), temp_tblob.Size(), rhs.Size()) +
                             DataType<DType>::kLanes - 1) /
                            DataType<DType>::kLanes;
        if (size != 0) {
          Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
              s, size, out.dptr<DType>(), temp_tblob.dptr<DType>(), rhs.dptr<DType>());
        }
      });
    });
  } else {
    TBlob temp_tblob_l;
    TBlob temp_tblob_r;
    MXNET_INT_TYPE_SWITCH_EXT(out.type_flag_, OType, {
      Tensor<xpu, 1, OType> workspace =
          ctx.requested[0].get_space_typed<xpu, 1, OType>(Shape1(lhs.Size() + rhs.Size()), s);
      TBlob temp_tblob = TBlob(workspace);
      temp_tblob_l     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_),
                           lhs.shape_,
                           temp_tblob.dev_mask(),
                           temp_tblob.dev_id());
      temp_tblob_r     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_) + lhs.Size() + 1,
                           rhs.shape_,
                           temp_tblob.dev_mask(),
                           temp_tblob.dev_id());
    });
    CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob_l});
    CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob_r});
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      MXNET_INT_TYPE_SWITCH_EXT(out.type_flag_, DType, {
        const size_t size =
            (ElemwiseBinaryOp::minthree(out.Size(), temp_tblob_l.Size(), temp_tblob_r.Size()) +
             DataType<DType>::kLanes - 1) /
            DataType<DType>::kLanes;
        if (size != 0) {
          Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
              s, size, out.dptr<DType>(), temp_tblob_l.dptr<DType>(), temp_tblob_r.dptr<DType>());
        }
      });
    });
  }
}

template <typename xpu, typename OP, typename LOP, typename ROP>
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
  if ((common::is_float(lhs.type_flag_)) && (common::is_float(rhs.type_flag_))) {
    if (lhs.type_flag_ == out.type_flag_) {
      MixedAllRealBinaryElemwiseCompute<xpu, ROP>(attrs.op->name, ctx, lhs, rhs, out, req[0]);
    } else {
      MixedAllRealBinaryElemwiseCompute<xpu, LOP>(attrs.op->name, ctx, rhs, lhs, out, req[0]);
    }
  } else if (common::is_float(lhs.type_flag_) || common::is_float(rhs.type_flag_)) {
    if (lhs.type_flag_ == out.type_flag_) {
      MixedIntRealBinaryElemwiseCompute<xpu, ROP>(ctx, lhs, rhs, out, req[0]);
    } else {
      MixedIntRealBinaryElemwiseCompute<xpu, LOP>(ctx, rhs, lhs, out, req[0]);
    }
  } else {
    MixedIntBinaryElemwiseCompute<xpu, OP>(attrs, ctx, lhs, rhs, out, req[0]);
  }
}

template <typename xpu, typename OP>
void MixedAllRealBinaryBroadcastCompute(const std::string& op_name,
                                        const OpContext& ctx,
                                        const TBlob& lhs,
                                        const TBlob& rhs,
                                        const TBlob& out,
                                        const OpReqType req,
                                        const int ndim,
                                        const mxnet::TShape& new_oshape,
                                        const mxnet::TShape& new_lshape,
                                        const mxnet::TShape& new_rshape) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(lhs.type_flag_, out.type_flag_);

  Stream<xpu>* s = ctx.get_stream<xpu>();

  BROADCAST_NDIM_SWITCH(ndim, NDim, {
    mshadow::Shape<NDim> oshape  = new_oshape.get<NDim>();
    mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
    mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
    switch (lhs.type_flag_) {
      case mshadow::kFloat32: {
        if (rhs.type_flag_ == mshadow::kFloat16) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::template LaunchEx(
              s,
              new_oshape.Size(),
              req,
              rstride,
              lstride,
              oshape,
              rhs.dptr<mshadow::half::half_t>(),
              lhs.dptr<float>(),
              out.dptr<float>());
        } else if (rhs.type_flag_ == mshadow::kBfloat16) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::template LaunchEx(
              s,
              new_oshape.Size(),
              req,
              rstride,
              lstride,
              oshape,
              rhs.dptr<mshadow::bfloat::bf16_t>(),
              lhs.dptr<float>(),
              out.dptr<float>());
        } else {
          PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        }
        break;
      }
      case mshadow::kFloat64: {
        if (rhs.type_flag_ == mshadow::kFloat16) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::template LaunchEx(
              s,
              new_oshape.Size(),
              req,
              rstride,
              lstride,
              oshape,
              rhs.dptr<mshadow::half::half_t>(),
              lhs.dptr<double>(),
              out.dptr<double>());
        } else if (rhs.type_flag_ == mshadow::kBfloat16) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::template LaunchEx(
              s,
              new_oshape.Size(),
              req,
              rstride,
              lstride,
              oshape,
              rhs.dptr<mshadow::bfloat::bf16_t>(),
              lhs.dptr<double>(),
              out.dptr<double>());
        } else if (rhs.type_flag_ == mshadow::kFloat32) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::template LaunchEx(
              s,
              new_oshape.Size(),
              req,
              rstride,
              lstride,
              oshape,
              rhs.dptr<float>(),
              lhs.dptr<double>(),
              out.dptr<double>());
        } else {
          PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        }
        break;
      }
      default: {
        PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        break;
      }
    }
  });
}

template <typename xpu, typename OP, typename LOP, typename ROP>
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

  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(
      lhs.shape_, rhs.shape_, out.shape_, &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    MixedBinaryElemwiseCompute<xpu, OP, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
    if ((common::is_float(lhs.type_flag_)) && (common::is_float(rhs.type_flag_))) {
      if (lhs.type_flag_ == out.type_flag_) {
        MixedAllRealBinaryBroadcastCompute<xpu, ROP>(
            attrs.op->name, ctx, lhs, rhs, out, req[0], ndim, new_oshape, new_lshape, new_rshape);
      } else {
        MixedAllRealBinaryBroadcastCompute<xpu, LOP>(
            attrs.op->name, ctx, rhs, lhs, out, req[0], ndim, new_oshape, new_rshape, new_lshape);
      }
    } else if (common::is_float(lhs.type_flag_) || common::is_float(rhs.type_flag_)) {
      CHECK(lhs.type_flag_ == out.type_flag_ || rhs.type_flag_ == out.type_flag_)
          << "One of the input type should be the same as the output";
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape  = new_oshape.get<NDim>();
        mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
        mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
        if (lhs.type_flag_ == out.type_flag_) {
          MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, LType, {
            MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(rhs.type_flag_, RType, {
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, ROP>,
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
          MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, RType, {
            MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(lhs.type_flag_, LType, {
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, LOP>,
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
      });
    } else if (!common::is_float(lhs.type_flag_) && !common::is_float(rhs.type_flag_)) {
      TBlob temp_tblob;
      if (lhs.type_flag_ == out.type_flag_) {
        MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(lhs.type_flag_, LType, {
          Tensor<xpu, 1, LType> temp_tensor =
              ctx.requested[0].get_space_typed<xpu, 1, LType>(Shape1(rhs.Size()), s);
          temp_tblob = TBlob(temp_tensor);
        });
        CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob});
        BinaryBroadcastCompute<xpu, OP>(
            attrs, ctx, {lhs, temp_tblob.reshape(rhs.shape_)}, req, outputs);
      } else if (rhs.type_flag_ == out.type_flag_) {
        MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(rhs.type_flag_, RType, {
          Tensor<xpu, 1, RType> temp_tensor =
              ctx.requested[0].get_space_typed<xpu, 1, RType>(Shape1(lhs.Size()), s);
          temp_tblob = TBlob(temp_tensor);
        });
        CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob});
        BinaryBroadcastCompute<xpu, OP>(
            attrs, ctx, {temp_tblob.reshape(lhs.shape_), rhs}, req, outputs);
      } else {
        TBlob temp_tblob_l;
        TBlob temp_tblob_r;
        MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(out.type_flag_, OType, {
          Tensor<xpu, 1, OType> workspace =
              ctx.requested[0].get_space_typed<xpu, 1, OType>(Shape1(lhs.Size() + rhs.Size()), s);
          TBlob temp_tblob = TBlob(workspace);
          temp_tblob_l     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_),
                               lhs.shape_,
                               temp_tblob.dev_mask(),
                               temp_tblob.dev_id());
          temp_tblob_r     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_) + lhs.Size() + 1,
                               rhs.shape_,
                               temp_tblob.dev_mask(),
                               temp_tblob.dev_id());
        });
        CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob_l});
        CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob_r});
        BinaryBroadcastCompute<xpu, OP>(attrs, ctx, {temp_tblob_l, temp_tblob_r}, req, outputs);
      }
    } else {
      PrintErrorMessage(attrs.op->name, lhs.type_flag_, rhs.type_flag_);
    }
  }
}

template <typename xpu, typename OP, typename LOP, typename ROP>
void NumpyBinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
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

  if ((out.shape_.Size() == 0U) || (req[0] == kNullOp))
    return;

  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  }

  MixedBinaryBroadcastCompute<xpu, OP, LOP, ROP>(attrs, ctx, inputs, req, outputs);
}

template <typename xpu, typename OP, typename LOP, typename ROP>
void NumpyBinaryBroadcastComputeWithBool(const nnvm::NodeAttrs& attrs,
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

  if ((out.shape_.Size() == 0U) || (req[0] == kNullOp))
    return;

  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastComputeWithBool<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  }
  if (!common::is_float(lhs.type_flag_) && !common::is_float(rhs.type_flag_)) {
    Stream<xpu>* s = ctx.get_stream<xpu>();
    TBlob temp_tblob;
    if (lhs.type_flag_ == out.type_flag_) {
      MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(lhs.type_flag_, LType, {
        Tensor<xpu, 1, LType> temp_tensor =
            ctx.requested[0].get_space_typed<xpu, 1, LType>(Shape1(rhs.Size()), s);
        temp_tblob = TBlob(temp_tensor);
      });
      CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob});
      BinaryBroadcastCompute<xpu, OP>(
          attrs, ctx, {lhs, temp_tblob.reshape(rhs.shape_)}, req, outputs);
    } else if (rhs.type_flag_ == out.type_flag_) {
      MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(rhs.type_flag_, RType, {
        Tensor<xpu, 1, RType> temp_tensor =
            ctx.requested[0].get_space_typed<xpu, 1, RType>(Shape1(lhs.Size()), s);
        temp_tblob = TBlob(temp_tensor);
      });
      CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob});
      BinaryBroadcastCompute<xpu, OP>(
          attrs, ctx, {temp_tblob.reshape(lhs.shape_), rhs}, req, outputs);
    } else {
      TBlob temp_tblob_l;
      TBlob temp_tblob_r;
      MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(out.type_flag_, OType, {
        Tensor<xpu, 1, OType> workspace =
            ctx.requested[0].get_space_typed<xpu, 1, OType>(Shape1(lhs.Size() + rhs.Size()), s);
        TBlob temp_tblob = TBlob(workspace);
        temp_tblob_l     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_),
                             lhs.shape_,
                             temp_tblob.dev_mask(),
                             temp_tblob.dev_id());
        temp_tblob_r     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_) + lhs.Size() + 1,
                             rhs.shape_,
                             temp_tblob.dev_mask(),
                             temp_tblob.dev_id());
      });
      CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob_l});
      CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob_r});
      BinaryBroadcastCompute<xpu, OP>(attrs, ctx, {temp_tblob_l, temp_tblob_r}, req, outputs);
    }
    return;
  }
  MixedBinaryBroadcastCompute<xpu, OP, LOP, ROP>(attrs, ctx, inputs, req, outputs);
}

template <typename xpu, typename OP>
void NumpyBinaryBroadcastIntComputeWithBool(const nnvm::NodeAttrs& attrs,
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

  if ((out.shape_.Size() == 0U) || (req[0] == kNullOp))
    return;

  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastIntComputeWithBool<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  }
  Stream<xpu>* s = ctx.get_stream<xpu>();
  TBlob temp_tblob;
  if (lhs.type_flag_ == out.type_flag_) {
    MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(lhs.type_flag_, LType, {
      Tensor<xpu, 1, LType> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, LType>(Shape1(rhs.Size()), s);
      temp_tblob = TBlob(temp_tensor);
    });
    CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob});
    BinaryBroadcastIntComputeWithBool<xpu, OP>(
        attrs, ctx, {lhs, temp_tblob.reshape(rhs.shape_)}, req, outputs);
  } else if (rhs.type_flag_ == out.type_flag_) {
    MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(rhs.type_flag_, RType, {
      Tensor<xpu, 1, RType> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, RType>(Shape1(lhs.Size()), s);
      temp_tblob = TBlob(temp_tensor);
    });
    CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob});
    BinaryBroadcastIntComputeWithBool<xpu, OP>(
        attrs, ctx, {temp_tblob.reshape(lhs.shape_), rhs}, req, outputs);
  } else {
    TBlob temp_tblob_l;
    TBlob temp_tblob_r;
    MXNET_INT_TYPE_SWITCH_EXT_WITH_BOOL(out.type_flag_, OType, {
      Tensor<xpu, 1, OType> workspace =
          ctx.requested[0].get_space_typed<xpu, 1, OType>(Shape1(lhs.Size() + rhs.Size()), s);
      TBlob temp_tblob = TBlob(workspace);
      temp_tblob_l     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_),
                           lhs.shape_,
                           temp_tblob.dev_mask(),
                           temp_tblob.dev_id());
      temp_tblob_r     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_) + lhs.Size() + 1,
                           rhs.shape_,
                           temp_tblob.dev_mask(),
                           temp_tblob.dev_id());
    });
    CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob_l});
    CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob_r});
    BinaryBroadcastIntComputeWithBool<xpu, OP>(
        attrs, ctx, {temp_tblob_l, temp_tblob_r}, req, outputs);
  }
  return;
}

template <typename xpu, typename OP>
void NumpyBinaryBroadcastIntCompute(const nnvm::NodeAttrs& attrs,
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

  if ((out.shape_.Size() == 0U) || (req[0] == kNullOp))
    return;

  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastIntCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  }
  Stream<xpu>* s = ctx.get_stream<xpu>();
  TBlob temp_tblob;
  if (lhs.type_flag_ == out.type_flag_) {
    MXNET_INT_TYPE_SWITCH_EXT(lhs.type_flag_, LType, {
      Tensor<xpu, 1, LType> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, LType>(Shape1(rhs.Size()), s);
      temp_tblob = TBlob(temp_tensor);
    });
    CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob});
    BinaryBroadcastIntCompute<xpu, OP>(
        attrs, ctx, {lhs, temp_tblob.reshape(rhs.shape_)}, req, outputs);
  } else if (rhs.type_flag_ == out.type_flag_) {
    MXNET_INT_TYPE_SWITCH_EXT(rhs.type_flag_, RType, {
      Tensor<xpu, 1, RType> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, RType>(Shape1(lhs.Size()), s);
      temp_tblob = TBlob(temp_tensor);
    });
    CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob});
    BinaryBroadcastIntCompute<xpu, OP>(
        attrs, ctx, {temp_tblob.reshape(lhs.shape_), rhs}, req, outputs);
  } else {
    TBlob temp_tblob_l;
    TBlob temp_tblob_r;
    MXNET_INT_TYPE_SWITCH_EXT(out.type_flag_, OType, {
      Tensor<xpu, 1, OType> workspace =
          ctx.requested[0].get_space_typed<xpu, 1, OType>(Shape1(lhs.Size() + rhs.Size()), s);
      TBlob temp_tblob = TBlob(workspace);
      temp_tblob_l     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_),
                           lhs.shape_,
                           temp_tblob.dev_mask(),
                           temp_tblob.dev_id());
      temp_tblob_r     = TBlob(reinterpret_cast<OType*>(temp_tblob.dptr_) + lhs.Size() + 1,
                           rhs.shape_,
                           temp_tblob.dev_mask(),
                           temp_tblob.dev_id());
    });
    CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob_l});
    CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob_r});
    BinaryBroadcastIntCompute<xpu, OP>(attrs, ctx, {temp_tblob_l, temp_tblob_r}, req, outputs);
  }
  return;
}

inline bool NumpyBinaryMixedFloatingType(const nnvm::NodeAttrs& attrs,
                                         std::vector<int>* in_attrs,
                                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int ltype = in_attrs->at(0);
  const int rtype = in_attrs->at(1);

  if (ltype != -1 && rtype != -1 && (ltype != rtype)) {
    // Only when both input types are known and not the same, we enter the mixed-precision mode
    TYPE_ASSIGN_CHECK(*out_attrs, 0, common::type_promotion(ltype, rtype));
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 1, out_attrs->at(0));
  }
  // check if it is float16, float32 or float64. If not, raise error.
  CHECK(common::is_float(in_attrs->at(0))) << "Do not support `int` as input.\n";
  return out_attrs->at(0) != -1;
}

template <typename xpu, typename OP>
void NumpyBinaryMixedFloatingCompute(const nnvm::NodeAttrs& attrs,
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

  if ((out.shape_.Size() == 0U) || (req[0] == kNullOp))
    return;

  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  }
  Stream<xpu>* s = ctx.get_stream<xpu>();
  TBlob temp_tblob;
  if (lhs.type_flag_ == out.type_flag_) {
    MSHADOW_REAL_TYPE_SWITCH(lhs.type_flag_, LType, {
      Tensor<xpu, 1, LType> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, LType>(Shape1(rhs.Size()), s);
      temp_tblob = TBlob(temp_tensor);
    });
    CastCompute<xpu>(attrs, ctx, {rhs}, {kWriteTo}, {temp_tblob});
    BinaryBroadcastCompute<xpu, OP>(
        attrs, ctx, {lhs, temp_tblob.reshape(rhs.shape_)}, req, outputs);
  } else {
    MSHADOW_REAL_TYPE_SWITCH(rhs.type_flag_, RType, {
      Tensor<xpu, 1, RType> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, RType>(Shape1(lhs.Size()), s);
      temp_tblob = TBlob(temp_tensor);
    });
    CastCompute<xpu>(attrs, ctx, {lhs}, {kWriteTo}, {temp_tblob});
    BinaryBroadcastCompute<xpu, OP>(
        attrs, ctx, {temp_tblob.reshape(lhs.shape_), rhs}, req, outputs);
  }
  return;
}

template <typename xpu, typename LOP, typename ROP>
void NumpyBinaryBackwardUseIn(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& lhs = inputs[1];
  const TBlob& rhs = inputs[2];
  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastBackwardUseIn<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
    return;
  }

  const TBlob& ograd = inputs[0];
  const TBlob& lgrad = outputs[0];
  const TBlob& rgrad = outputs[1];

  if (common::is_float(lhs.type_flag_) || common::is_float(rhs.type_flag_)) {
    // If any of the inputs is a float, it's the same type as the output
    // So 2 of the 3 tensors have the same data type
    Stream<xpu>* s = ctx.get_stream<xpu>();
    mxnet::TShape new_lshape, new_rshape, new_oshape;
    using namespace broadcast;
    const bool need_bc =
        BinaryBroadcastShapeCompact(
            lgrad.shape_, rgrad.shape_, ograd.shape_, &new_lshape, &new_rshape, &new_oshape) != 0;

    // Prepare all the temporary memory
    size_t workspace_size_l = 0, workspace_size_r = 0;
    TBlob temp_tblob;  // The TBlob for casted input data
    TBlob temp_igrad;  // The TBlob for casted grad results
    size_t tensor_size = (lgrad.type_flag_ != ograd.type_flag_) ? lgrad.Size() : rgrad.Size();
    Tensor<xpu, 1, char> workspace;

    MSHADOW_TYPE_SWITCH(ograd.type_flag_, OType, {
      if (need_bc) {
        workspace_size_l =
            ReduceWorkspaceSize(s, new_lshape, req[0], new_oshape, new_lshape, new_rshape);
        workspace_size_r =
            ReduceWorkspaceSize(s, new_rshape, req[1], new_oshape, new_lshape, new_rshape);
      }
      size_t workspace_size   = std::max(workspace_size_l, workspace_size_r);
      size_t cast_tensor_size = tensor_size * sizeof(OType);
      // Allocate the temporary memories now
      Tensor<xpu, 1, char> temp_space = ctx.requested[0].get_space_typed<xpu, 1, char>(
          Shape1(workspace_size + cast_tensor_size * 2), s);
      // Tensor for temp_tblob
      Tensor<xpu, 1, OType> temp_tblob_tensor(
          reinterpret_cast<OType*>(temp_space.dptr_), Shape1(tensor_size), s);
      // Tensor for temp_igrad
      Tensor<xpu, 1, OType> temp_igrad_tensor(
          reinterpret_cast<OType*>(temp_space.dptr_) + tensor_size, Shape1(tensor_size), s);
      temp_tblob = TBlob(temp_tblob_tensor)
                       .reshape(((lgrad.type_flag_ != ograd.type_flag_) ? lhs.shape_ : rhs.shape_));
      temp_igrad = TBlob(temp_igrad_tensor)
                       .reshape(((lgrad.type_flag_ != ograd.type_flag_) ? lhs.shape_ : rhs.shape_));
      if (temp_igrad.Size() != 0) {
        Kernel<set_zero, xpu>::Launch(s, temp_igrad.Size(), temp_igrad.dptr<OType>());
      }
      workspace =
          Tensor<xpu, 1, char>(temp_space.dptr_ + 2 * cast_tensor_size, Shape1(workspace_size), s);
    });

    // Cast the input that does not have consistent type to temp_tblob
    CastCompute<xpu>(attrs,
                     ctx,
                     {((lgrad.type_flag_ != ograd.type_flag_) ? lhs : rhs)},
                     {kWriteTo},
                     {temp_tblob});

    if (!need_bc) {
      if (lhs.type_flag_ != ograd.type_flag_) {
        ElemwiseBinaryOp::BackwardUseIn<xpu, LOP, ROP>(
            attrs, ctx, {ograd, temp_tblob, rhs}, {kWriteTo, req[1]}, {temp_igrad, rgrad});
      } else {
        ElemwiseBinaryOp::BackwardUseIn<xpu, LOP, ROP>(
            attrs, ctx, {ograd, lhs, temp_tblob}, {req[0], kWriteTo}, {lgrad, temp_igrad});
      }
    } else {
      if (lhs.type_flag_ != ograd.type_flag_) {
        MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
          BROADCAST_NDIM_SWITCH(new_oshape.ndim(), NDim, {
            BinaryBroadcastBackwardUseInImplWithWorkspace<xpu, NDim, DType, LOP, ROP>(
                ctx,
                {ograd, temp_tblob, rhs},
                {kWriteTo, req[1]},
                {temp_igrad, rgrad},
                workspace,
                new_lshape,
                new_rshape,
                new_oshape);
          });
        });
      } else {
        MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
          BROADCAST_NDIM_SWITCH(new_oshape.ndim(), NDim, {
            BinaryBroadcastBackwardUseInImplWithWorkspace<xpu, NDim, DType, LOP, ROP>(
                ctx,
                {ograd, lhs, temp_tblob},
                {req[0], kWriteTo},
                {lgrad, temp_igrad},
                workspace,
                new_lshape,
                new_rshape,
                new_oshape);
          });
        });
      }
    }

    // If both inputs are floating numbers, cast the igrad to the input that has
    // the different data type
    if (common::is_float(lhs.type_flag_) && common::is_float(rhs.type_flag_)) {
      if (lhs.type_flag_ != ograd.type_flag_) {
        CastCompute<xpu>(attrs, ctx, {temp_igrad}, {req[0]}, {lgrad});
      } else {
        CastCompute<xpu>(attrs, ctx, {temp_igrad}, {req[1]}, {rgrad});
      }
    }
  } else {
    // Case where both inputs are integer types, should not even do
    // backward computation for this case.
    PrintErrorMessage(attrs.op->name, lhs.type_flag_, rhs.type_flag_);
  }
}

#if MXNET_USE_ONEDNN == 1
inline bool NumpyBinaryBroadcastStorageType(const nnvm::NodeAttrs& attrs,
                                            const int dev_mask,
                                            DispatchMode* dispatch_mode,
                                            std::vector<int>* in_attrs,
                                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);

  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

void NumpyDivideBroadcastComputeCPU(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs);

template <typename OP>
void NumpyBinaryOperatorComputeExCPU(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<mxnet::NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<mxnet::NDArray>& outputs) {
  if (SupportDNNLBinary(inputs, outputs)) {
    const dnnl::algorithm alg = DNNLAlgorithm<OP>::value;
    DNNLRun(DNNLBinaryOpForward<alg>, attrs, ctx, inputs, req, outputs);
    return;
  }
  using namespace op::mshadow_op;
  std::vector<mxnet::TBlob> in_data  = {inputs[0].data(), inputs[1].data()};
  std::vector<mxnet::TBlob> out_data = {outputs[0].data()};
  if (std::is_same<OP, plus>::value) {
    NumpyBinaryBroadcastComputeWithBool<cpu, OP, mixed_plus, mixed_plus>(
        attrs, ctx, in_data, req, out_data);
  } else if (std::is_same<OP, minus>::value) {
    NumpyBinaryBroadcastCompute<cpu, OP, mixed_minus, mixed_rminus>(
        attrs, ctx, in_data, req, out_data);
  } else if (std::is_same<OP, mul>::value) {
    NumpyBinaryBroadcastComputeWithBool<cpu, OP, mixed_mul, mixed_mul>(
        attrs, ctx, in_data, req, out_data);
  } else if (std::is_same<OP, div>::value) {
    NumpyDivideBroadcastComputeCPU(attrs, ctx, in_data, req, out_data);
  }
}
#endif  // MXNET_USE_ONEDNN

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(name)                        \
  NNVM_REGISTER_OP(name)                                                      \
      .set_num_inputs(1)                                                      \
      .set_num_outputs(1)                                                     \
      .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)                   \
      .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)       \
      .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarType)        \
      .set_attr<FResourceRequest>(                                            \
          "FResourceRequest",                                                 \
          [](const NodeAttrs& attrs) {                                        \
            return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; \
          })                                                                  \
      .add_argument("data", "NDArray-or-Symbol", "source input")              \
      .add_arguments(NumpyBinaryScalarParam::__FIELDS__())

inline bool NumpyBinaryMixedPrecisionType(const nnvm::NodeAttrs& attrs,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int ltype = in_attrs->at(0);
  const int rtype = in_attrs->at(1);
  if (ltype != -1 && rtype != -1 && (ltype != rtype)) {
    // Only when both input types are known and not the same, we enter the mixed-precision mode
    TYPE_ASSIGN_CHECK(*out_attrs, 0, common::type_promotion(ltype, rtype));
  } else {
    return ElemwiseType<2, 1>(attrs, in_attrs, out_attrs);
  }
  return true;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(name)                                   \
  NNVM_REGISTER_OP(name)                                                                          \
      .set_num_inputs(2)                                                                          \
      .set_num_outputs(1)                                                                         \
      .set_attr<nnvm::FListInputNames>("FListInputNames",                                         \
                                       [](const NodeAttrs& attrs) {                               \
                                         return std::vector<std::string>{"lhs", "rhs"};           \
                                       })                                                         \
      .set_attr<nnvm::FListOutputNames>(                                                          \
          "FListOutputNames",                                                                     \
          [](const NodeAttrs& attrs) { return std::vector<std::string>{"output"}; })              \
      .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)                          \
      .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryMixedPrecisionType)                    \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                           \
                                      [](const NodeAttrs& attrs) {                                \
                                        return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}}; \
                                      })                                                          \
      .set_attr<FResourceRequest>(                                                                \
          "FResourceRequest",                                                                     \
          [](const NodeAttrs& attrs) {                                                            \
            return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};                     \
          })                                                                                      \
      .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")                    \
      .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")

inline bool NumpyBinaryMixedIntPrecisionTypeWithBool(const nnvm::NodeAttrs& attrs,
                                                     std::vector<int>* in_attrs,
                                                     std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int ltype = in_attrs->at(0);
  const int rtype = in_attrs->at(1);
  CHECK(common::is_int(ltype) || ltype == mshadow::kBool)
      << "1st input only supports integer types or bool types.";
  CHECK(common::is_int(rtype) || rtype == mshadow::kBool)
      << "2nd input only supports integer types or bool types.";
  if (ltype != -1 && rtype != -1 && (ltype != rtype)) {
    // Only when both input types are known and not the same, we enter the mixed-precision mode
    TYPE_ASSIGN_CHECK(*out_attrs, 0, common::type_promotion(ltype, rtype));
  } else {
    return ElemwiseType<2, 1>(attrs, in_attrs, out_attrs);
  }
  return true;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_INT_PRECISION_WITH_BOOL(name)                     \
  NNVM_REGISTER_OP(name)                                                                          \
      .set_num_inputs(2)                                                                          \
      .set_num_outputs(1)                                                                         \
      .set_attr<nnvm::FListInputNames>("FListInputNames",                                         \
                                       [](const NodeAttrs& attrs) {                               \
                                         return std::vector<std::string>{"lhs", "rhs"};           \
                                       })                                                         \
      .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)                          \
      .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryMixedIntPrecisionTypeWithBool)         \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                           \
                                      [](const NodeAttrs& attrs) {                                \
                                        return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}}; \
                                      })                                                          \
      .set_attr<FResourceRequest>(                                                                \
          "FResourceRequest",                                                                     \
          [](const NodeAttrs& attrs) {                                                            \
            return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};                     \
          })                                                                                      \
      .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")                    \
      .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")

inline bool NumpyBinaryMixedIntPrecisionType(const nnvm::NodeAttrs& attrs,
                                             std::vector<int>* in_attrs,
                                             std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int ltype = in_attrs->at(0);
  const int rtype = in_attrs->at(1);
  CHECK(common::is_int(ltype)) << "1st input only supports integer types.";
  CHECK(common::is_int(rtype)) << "2nd input only supports integer types.";
  if (ltype != -1 && rtype != -1 && (ltype != rtype)) {
    // Only when both input types are known and not the same, we enter the mixed-precision mode
    TYPE_ASSIGN_CHECK(*out_attrs, 0, common::type_promotion(ltype, rtype));
  } else {
    return ElemwiseType<2, 1>(attrs, in_attrs, out_attrs);
  }
  return true;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_INT_PRECISION(name)                               \
  NNVM_REGISTER_OP(name)                                                                          \
      .set_num_inputs(2)                                                                          \
      .set_num_outputs(1)                                                                         \
      .set_attr<nnvm::FListInputNames>("FListInputNames",                                         \
                                       [](const NodeAttrs& attrs) {                               \
                                         return std::vector<std::string>{"lhs", "rhs"};           \
                                       })                                                         \
      .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)                          \
      .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryMixedIntPrecisionType)                 \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                           \
                                      [](const NodeAttrs& attrs) {                                \
                                        return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}}; \
                                      })                                                          \
      .set_attr<FResourceRequest>(                                                                \
          "FResourceRequest",                                                                     \
          [](const NodeAttrs& attrs) {                                                            \
            return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};                     \
          })                                                                                      \
      .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")                    \
      .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_
