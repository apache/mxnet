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
 * \file np_elemwise_broadcast_op.h
 * \brief Function definition of elemwise and broadcast operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_

#if MXNET_USE_TVM_OP
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP

#include <algorithm>
#include <vector>
#include <string>

#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

inline bool NumpyBinaryScalarType(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return in_attrs->at(0) != -1;
}

inline void PrintErrorMessage(const std::string& op_name, const int dtype1, const int dtype2) {
  LOG(FATAL) << "Operator " << op_name << " does not support combination of "
             << mshadow::dtype_string(dtype1) << " with " << mshadow::dtype_string(dtype2)
             << " yet...";
}

#ifndef _WIN32
template<typename xpu, typename OP>
void MixedAllRealBinaryElemwiseCompute(const std::string& op_name,
                                       const OpContext& ctx,
                                       const TBlob& lhs,
                                       const TBlob& rhs,
                                       const TBlob& out,
                                       const OpReqType req) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(lhs.type_flag_, out.type_flag_);

  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    const size_t size = (ElemwiseBinaryOp::minthree(out.Size(), lhs.Size(), rhs.Size())
      + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
    if (size == 0) return;

    switch (lhs.type_flag_) {
      case mshadow::kFloat32:
      {
        if (rhs.type_flag_ == mshadow::kFloat16) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
              s, size, out.dptr<float>(), rhs.dptr<mshadow::half::half_t>(),
              lhs.dptr<float>());
          });
        } else {
          PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        }
        break;
      }
      case mshadow::kFloat64:
      {
        if (rhs.type_flag_ == mshadow::kFloat16) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
              s, size, out.dptr<double>(), rhs.dptr<mshadow::half::half_t>(),
              lhs.dptr<double>());
          });
        } else if (rhs.type_flag_ == mshadow::kFloat32) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
              s, size, out.dptr<double>(), rhs.dptr<float>(),
              lhs.dptr<double>());
          });
        } else {
          PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        }
        break;
      }
      default:
      {
        PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        break;
      }
    }
  });
}

template<typename xpu, typename OP>
void MixedIntRealBinaryElemwiseCompute(const OpContext& ctx,
                                       const TBlob& lhs,
                                       const TBlob& rhs,
                                       const TBlob& out,
                                       const OpReqType req) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(lhs.type_flag_, out.type_flag_);

  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, FType, {
    const size_t size = (ElemwiseBinaryOp::minthree(out.Size(), lhs.Size(), rhs.Size())
      + DataType<FType>::kLanes - 1) / DataType<FType>::kLanes;
    if (size == 0) return;

    MXNET_INT_TYPE_SWITCH(rhs.type_flag_, IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
          s, size, out.dptr<FType>(), rhs.dptr<IType>(),
          lhs.dptr<FType>());
      });
    });
  });
}

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

  if (common::is_float(lhs.type_flag_) && common::is_float(rhs.type_flag_)) {
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
    PrintErrorMessage(attrs.op->name, lhs.type_flag_, rhs.type_flag_);
  }
}

template<typename xpu, typename OP>
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

  Stream<xpu> *s = ctx.get_stream<xpu>();

  BROADCAST_NDIM_SWITCH(ndim, NDim, {
    mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
    mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
    mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
    switch (lhs.type_flag_) {
      case mshadow::kFloat32:
      {
        if (rhs.type_flag_ == mshadow::kFloat16) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::
          template LaunchEx(s, new_oshape.Size(), req, rstride, lstride, oshape,
          rhs.dptr<mshadow::half::half_t>(), lhs.dptr<float>(), out.dptr<float>());
        } else {
          PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        }
        break;
      }
      case mshadow::kFloat64:
      {
        if (rhs.type_flag_ == mshadow::kFloat16) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::
          template LaunchEx(s, new_oshape.Size(), req, rstride, lstride, oshape,
          rhs.dptr<mshadow::half::half_t>(), lhs.dptr<double>(), out.dptr<double>());
        } else if (rhs.type_flag_ == mshadow::kFloat32) {
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::
          template LaunchEx(s, new_oshape.Size(), req, rstride, lstride, oshape,
          rhs.dptr<float>(), lhs.dptr<double>(), out.dptr<double>());
        } else {
          PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        }
        break;
      }
      default:
      {
        PrintErrorMessage(op_name, lhs.type_flag_, rhs.type_flag_);
        break;
      }
    }
  });
}
#endif

#ifndef _WIN32
template<typename xpu, typename OP, typename LOP, typename ROP>
#else
template<typename xpu, typename OP>
#endif
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

#ifndef _WIN32
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(lhs.shape_, rhs.shape_, out.shape_,
                                         &new_lshape, &new_rshape, &new_oshape);

  if (!ndim) {
    MixedBinaryElemwiseCompute<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    if (common::is_float(lhs.type_flag_) && common::is_float(rhs.type_flag_)) {
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
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
        mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
        if (lhs.type_flag_ == out.type_flag_) {
          MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, LType, {
            MXNET_INT_TYPE_SWITCH(rhs.type_flag_, RType, {
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, ROP>, xpu>::
              template LaunchEx(s, new_oshape.Size(), req[0], rstride, lstride, oshape,
              rhs.dptr<RType>(), lhs.dptr<LType>(), out.dptr<LType>());
            });
          });
        } else {
          MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, RType, {
            MXNET_INT_TYPE_SWITCH(lhs.type_flag_, LType, {
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, LOP>, xpu>::
              template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
              lhs.dptr<LType>(), rhs.dptr<RType>(), out.dptr<RType>());
            });
          });
        }
      });
    } else {
      PrintErrorMessage(attrs.op->name, lhs.type_flag_, rhs.type_flag_);
    }
  }
#else
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (common::is_float(lhs.type_flag_) || common::is_float(rhs.type_flag_)) {
    TBlob temp_tblob;
    // one is float, the other is bool
    CHECK((out.type_flag_ == lhs.type_flag_) || (out.type_flag_ == rhs.type_flag_))
      << "This case out type should be same as the float type";
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
  } else {
    PrintErrorMessage(attrs.op->name, lhs.type_flag_, rhs.type_flag_);
  }
#endif
}

#ifndef _WIN32
template<typename xpu, typename OP, typename LOP, typename ROP>
#else
template<typename xpu, typename OP>
#endif
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

  if ((out.shape_.Size() == 0U) || (req[0] == kNullOp)) return;

  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  }

#ifndef _WIN32
  MixedBinaryBroadcastCompute<xpu, OP, LOP, ROP>(attrs, ctx, inputs, req, outputs);
#else
  MixedBinaryBroadcastCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
#endif
}

#ifndef _WIN32
template<typename xpu, typename OP, typename LOP, typename ROP>
#else
template<typename xpu, typename OP>
#endif
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

  if ((out.shape_.Size() == 0U) || (req[0] == kNullOp)) return;

  if (lhs.type_flag_ == rhs.type_flag_) {
    BinaryBroadcastComputeWithBool<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  }

#ifndef _WIN32
  MixedBinaryBroadcastCompute<xpu, OP, LOP, ROP>(attrs, ctx, inputs, req, outputs);
#else
  MixedBinaryBroadcastCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
#endif
}

template<typename xpu, typename LOP, typename ROP>
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
    Stream<xpu> *s = ctx.get_stream<xpu>();
    mxnet::TShape new_lshape, new_rshape, new_oshape;
    using namespace broadcast;
    const bool need_bc = BinaryBroadcastShapeCompact(lgrad.shape_, rgrad.shape_, ograd.shape_,
                                                     &new_lshape, &new_rshape, &new_oshape) != 0;

    // Prepare all the temporary memory
    size_t workspace_size_l = 0, workspace_size_r = 0;
    TBlob temp_tblob;  // The TBlob for casted input data
    TBlob temp_igrad;  // The TBlob for casted grad results
    size_t tensor_size = (lgrad.type_flag_ != ograd.type_flag_) ? lgrad.Size() : rgrad.Size();
    Tensor<xpu, 1, char> workspace;

    MSHADOW_TYPE_SWITCH(ograd.type_flag_, OType, {
      if (need_bc) {
        BROADCAST_NDIM_SWITCH(new_oshape.ndim(), ndim, {
          workspace_size_l = ReduceWorkspaceSize<ndim, OType>(
            s, new_lshape, req[0], new_oshape, new_lshape, new_rshape);
          workspace_size_r = ReduceWorkspaceSize<ndim, OType>(
            s, new_rshape, req[1], new_oshape, new_lshape, new_rshape);
        });
      }
      size_t workspace_size = std::max(workspace_size_l, workspace_size_r);
      size_t cast_tensor_size = tensor_size * sizeof(OType);
      // Allocate the temporary memories now
      Tensor<xpu, 1, char> temp_space =
        ctx.requested[0].get_space_typed<xpu, 1, char>(
          Shape1(workspace_size + cast_tensor_size * 2), s);
      // Tensor for temp_tblob
      Tensor<xpu, 1, OType> temp_tblob_tensor(
                              reinterpret_cast<OType*>(temp_space.dptr_),
                              Shape1(tensor_size), s);
      // Tensor for temp_igrad
      Tensor<xpu, 1, OType> temp_igrad_tensor(
                              reinterpret_cast<OType*>(temp_space.dptr_) + tensor_size,
                              Shape1(tensor_size), s);
      temp_tblob =
        TBlob(temp_tblob_tensor)
          .reshape(((lgrad.type_flag_ != ograd.type_flag_) ? lhs.shape_ : rhs.shape_));
      temp_igrad =
        TBlob(temp_igrad_tensor)
          .reshape(((lgrad.type_flag_ != ograd.type_flag_) ? lhs.shape_ : rhs.shape_));
      if (temp_igrad.Size() != 0) {
        Kernel<set_zero, xpu>::Launch(s, temp_igrad.Size(), temp_igrad.dptr<OType>());
      }
      workspace =
        Tensor<xpu, 1, char>(temp_space.dptr_ + 2 * cast_tensor_size, Shape1(workspace_size), s);
    });

    // Cast the input that does not have consistent type to temp_tblob
    CastCompute<xpu>(
      attrs, ctx, {((lgrad.type_flag_ != ograd.type_flag_) ? lhs : rhs)}, {kWriteTo}, {temp_tblob});

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
              ctx, {ograd, temp_tblob, rhs}, {kWriteTo, req[1]}, {temp_igrad, rgrad},
              workspace, new_lshape, new_rshape, new_oshape);
          });
        });
      } else {
        MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
          BROADCAST_NDIM_SWITCH(new_oshape.ndim(), NDim, {
            BinaryBroadcastBackwardUseInImplWithWorkspace<xpu, NDim, DType, LOP, ROP>(
              ctx, {ograd, lhs, temp_tblob}, {req[0], kWriteTo}, {lgrad, temp_igrad},
              workspace, new_lshape, new_rshape, new_oshape);
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

inline TBlob PrependAxes(const TBlob& src, const int dst_ndim) {
  CHECK_LE(src.shape_.ndim(), dst_ndim);
  const int src_ndim = src.shape_.ndim();
  if (src_ndim == dst_ndim) return src;
  mxnet::TShape dst_shape(dst_ndim, 1);
  for (int i = dst_ndim - src_ndim; i < dst_ndim; ++i) {
    dst_shape[i] = src.shape_[i - dst_ndim + src_ndim];
  }
  return src.reshape(dst_shape);
}

inline std::string SetAttr(const std::string& name,
                           const std::string& val) {
  return name + '_' + val;
}

inline std::string SetReq(OpReqType req) {
  if (req == kWriteTo)
    return "req_kWriteTo";
  return "req_kAddTo";
}

static constexpr int maxdim = 5;

struct TVMBinaryBroadcastCompute {
  const char* func;
  void operator()(const nnvm::NodeAttrs& attrs,
                  const mxnet::OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
#if MXNET_USE_TVM_OP
    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 1U);
    if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

    // prepare tblobs and TVMArgs
    std::vector<TBlob> tblobs = {inputs[0], inputs[1], outputs[0], outputs[0]};
    std::vector<int> type_codes;
    std::vector<TVMValue> values;

    const int ondim = outputs[0].shape_.ndim();
    const size_t num_args = 4;
    type_codes.resize(num_args);
    values.resize(num_args);
    for (size_t i = 0; i < num_args; ++i) {
      tblobs[i] = PrependAxes(tblobs[i], ondim);
      type_codes[i] = kArrayHandle;
      values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
    }
    std::string funcname = std::string(func);
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      funcname += SetReq(req_type);
    });
    tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], tblobs.size());
    tvm::runtime::TVMOpModule::Get()->CallEx(funcname, ctx, tblobs, tvm_args);
#else
    LOG(FATAL) << "Please add USE_TVM_OP=1 as a compile flag for compiling MXNet source code "
                  "to enable TVM-generated kernels for operator " << func;
#endif  // MXNET_USE_TVM_OP
  }
};

struct TVMBinaryBroadcastScalarCompute {
  const char* func;
  void operator()(const nnvm::NodeAttrs& attrs,
                  const mxnet::OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
#if MXNET_USE_TVM_OP
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

    // prepare tblobs and TVMArgs
    std::vector<TBlob> tblobs = {inputs[0], outputs[0], outputs[0]};
    std::vector<int> type_codes;
    std::vector<TVMValue> values;

    const size_t num_args = 4;  // one input tensor, one scalar param, and one output
    type_codes.resize(num_args);
    values.resize(num_args);

    // input tensor setup
    type_codes[0] = kArrayHandle;
    values[0].v_handle = const_cast<DLTensor*>(&(tblobs[0].dltensor()));

    // scalar param
    type_codes[1] = kDLFloat;
    values[1].v_float64 = nnvm::get<double>(attrs.parsed);

    // output tensor
    type_codes[2] = kArrayHandle;
    values[2].v_handle = const_cast<DLTensor*>(&(tblobs[1].dltensor()));

    // output tensor
    type_codes[3] = kArrayHandle;
    values[3].v_handle = const_cast<DLTensor*>(&(tblobs[1].dltensor()));

    std::string funcname = std::string(func);
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      funcname += SetReq(req_type);
    });

    tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], num_args);
    tvm::runtime::TVMOpModule::Get()->CallEx(funcname, ctx, tblobs, tvm_args);
#else
    LOG(FATAL) << "Please add USE_TVM_OP=1 as a compile flag for compiling MXNet source code "
                  "to enable TVM-generated kernels for operator " << func;
#endif  // MXNET_USE_TVM_OP
  }
};

enum AxisType {
  XReduce,     // operand X's broadcast axis
  YReduce,     // operand Y's broadcast axis
  XYIter       // other axis
};

enum ReductionType {
  ReduceAxis,      // broadcast axis
  IterAxis         // iter axis
};

struct TVMBinaryBroadcastBackwardUseIn {
  const char* func;
  void operator()(const nnvm::NodeAttrs& attrs,
                  const mxnet::OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
    /* Suppose that the two operands are operand X and operand Y, and we are
    * calculating the gradient of operand X. Each axis is labeled as X if
    * it's X’s broadcast axis, Y if its Y’s broadcast axis, and 0 is it’s
    * not a broadcast axis. Also note that no axis can be X and Y’s broadcast
    * axis at the same time. We may compress consecutive X axes and consecutive
    * non-X axes, but this results in the mix of axes labeled as Y and 0. To handle
    * this, for each pair of adjacent axes labeled as Y and 0, a dummy axis (of size 1)
    * with label X is inserted in the middle. After that all consecutive X axes and
    * non-X axes are compressed, with the cost of at most `n - 1` dummy axes,
    * where `n` denotes the maximum dimension of input operands.
    */
#if MXNET_USE_TVM_OP
    CHECK_EQ(inputs.size(), 3U);
    CHECK_EQ(outputs.size(), 2U);
    const int ndim = inputs[0].shape_.ndim();
    const TShape& oshape = inputs[0].shape_;
    TShape ishape[2];
    for (int k = 0; k < 2; ++k) {
      ishape[k] = PrependAxes(inputs[1 + k], ndim).shape_;
    }
    for (int k = 0; k < 2; ++k) {
      // dispatch by broadcast dims
      // seperate outputs[k] iter dim from outputs[1 - k] reduce dim
      const TShape& xs = ishape[k], ys = ishape[1 - k];
      if (xs.Size() == 0U) continue;  // skip zero-size tensor
      // get axis type
      std::vector<AxisType> axis_type(ndim);
      for (int i = 0; i < ndim; ++i) {
        if (oshape[i] != xs[i]) {
          axis_type[i] = XReduce;
        } else if (oshape[i] != ys[i]) {
          axis_type[i] = YReduce;
        } else {
          axis_type[i] = XYIter;
        }
      }
      // get reduction type of x with seperation dims inserted
      std::vector<ReductionType> seperated_type;
      std::vector<AxisType> seperated_axis_type;
      std::vector<int> seperated_shape;
      for (int i = 0; i < ndim; ++i) {
        ReductionType val;
        if (i > 0 && axis_type[i - 1] != XReduce && axis_type[i] != XReduce
            && axis_type[i - 1] != axis_type[i]) {
          seperated_type.push_back(ReduceAxis);
          seperated_axis_type.push_back(XReduce);
          seperated_shape.push_back(1);
        }
        if (axis_type[i] == XReduce) {
          val = ReduceAxis;
        } else {
          val = IterAxis;
        }
        seperated_type.push_back(val);
        seperated_shape.push_back(oshape[i]);
        seperated_axis_type.push_back(axis_type[i]);
      }
      // Sequeeze continuous dims of the same type
      std::vector<AxisType> otype;
      std::vector<int> ov;
      int size = seperated_type.size();
      for (int i = 0; i < size; ++i) {
        if (i > 0 && seperated_type[i] == seperated_type[i - 1]) {
          ov.back() *= seperated_shape[i];
          CHECK_EQ(otype.back(), seperated_axis_type[i]);
        } else {
          ov.push_back(seperated_shape[i]);
          otype.push_back(seperated_axis_type[i]);
        }
      }
      // Padding to maxdim
      for (int i = ov.size(); i < 2 * maxdim - 1; ++i) {
        ov.push_back(1);
        otype.push_back(XReduce);
      }
      // Calculate reduce1st_dim
      int reduce1st_dim = otype[0] == XReduce;
      // Calculate iv, xy, and yv
      std::vector<int> iv, xv, yv;
      for (int i = reduce1st_dim; i < 2 * maxdim - 1; i += 2) {
        iv.push_back(ov[i]);
      }
      for (int i = 0; i < 2 * maxdim - 1; ++i) {
        if (otype[i] == XReduce) {
          xv.push_back(1);
        } else {
          xv.push_back(ov[i]);
        }
      }
      for (int i = 0; i < 2 * maxdim - 1; ++i) {
        if (otype[i] == YReduce) {
          yv.push_back(1);
        } else {
          yv.push_back(ov[i]);
        }
      }

      // Prepare tblobs and TVMArgs
      TShape oshape_tvm(ov.begin(), ov.end());
      TShape xshape_tvm(xv.begin(), xv.end());
      TShape yshape_tvm(yv.begin(), yv.end());
      TShape ishape_tvm(iv.begin(), iv.end());
      std::vector<TBlob> tblobs = {inputs[0].reshape(oshape_tvm),
                                  inputs[1 + k].reshape(xshape_tvm),
                                  inputs[2 - k].reshape(yshape_tvm),
                                  outputs[k].reshape(ishape_tvm),
                                  outputs[k].reshape(ishape_tvm)};
      std::vector<int> type_codes;
      std::vector<TVMValue> values;
      const int num_args = 5;
      type_codes.resize(num_args);
      values.resize(num_args);
      for (size_t i = 0; i < num_args; ++i) {
        type_codes[i] = kArrayHandle;
        values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
      }

      // Set attrs
      std::string funcname = std::string(func);
      funcname += SetAttr("output", std::to_string(k));
      funcname += SetAttr("reduce1st_dim", std::to_string(reduce1st_dim));
      MXNET_ASSIGN_REQ_SWITCH(req[k], req_type, {
        funcname += SetReq(req_type);
      });
      tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], num_args);
      tvm::runtime::TVMOpModule::Get()->CallEx(funcname, ctx, tblobs, tvm_args);
    }
#else
    LOG(FATAL) << "Please add USE_TVM_OP=1 as a compile flag for compiling MXNet source code "
                  "to enable TVM-generated kernels for operator " << func;
#endif  // MXNET_USE_TVM_OP
  }
};

struct TVMBinaryBroadcastBackwardUseNone{
  const char* func;
  void operator()(const nnvm::NodeAttrs& attrs,
                  const mxnet::OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
    /* The backward of broadcast op is basically a reduction on broadcast axes.
    * We label the reduce axes as 1 and other axes as 0, and they form a bit string.
    * Each bit string correponds to a kernel, so the number of kernels is as many as `2^n`
    * To reduce it, the bit string is compressed by combining consecutive 0s or 1s.
    * In this way, the number of bit string (the number of kernels) is reduced to `2 * n`
    * They compressed bit string is stored in `axes`. And `reduce1st_dim` represents the first bit
    * of the compressed bit string. Credit to @junrushao1994 and @yzhliu.
    */
#if MXNET_USE_TVM_OP
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 2U);
    const TShape& oshape = PrependAxes(inputs[0], maxdim).shape_;
    for (int k = 0; k < 2; ++k) {
      // dispatch by backward
      TShape ishape = PrependAxes(outputs[k], maxdim).shape_;
      if (ishape.Size() == 0U) continue;  // skip zero-size tensor
      std::vector<ReductionType> reduction_type;
      for (int i = 0; i < maxdim; ++i) {
        if (oshape[i] != ishape[i]) {
          reduction_type.push_back(ReduceAxis);
        } else {
          reduction_type.push_back(IterAxis);
        }
      }
      // Calculate ov
      std::vector<int> tv;
      for (int i = 0; i < maxdim; ++i) {
        if (i > 0 && reduction_type[i] == reduction_type[i - 1]) {
          tv.back() *= oshape[i];
        } else {
          tv.push_back(oshape[i]);
        }
      }
      // Prepend to maxdim
      std::vector<int> ov(maxdim - tv.size(), 1), iv;
      for (auto const& i : tv) {
        ov.push_back(i);
      }
      // Calculate reduce1st_dim
      int reduce1st_dim = reduction_type[0] == ReduceAxis;
      reduce1st_dim = (reduce1st_dim + maxdim - tv.size()) % 2;

      // Calculate iv
      for (uint32_t i = reduce1st_dim; i < ov.size(); i += 2) {
        iv.push_back(ov[i]);
      }

      // Prepare tblobs and TVMArgs
      TShape oshape_tvm(ov.begin(), ov.end());
      TShape ishape_tvm(iv.begin(), iv.end());
      std::vector<TBlob> tblobs = {inputs[0].reshape(oshape_tvm),
                                  outputs[k].reshape(ishape_tvm),
                                  outputs[k].reshape(ishape_tvm)};
      std::vector<int> type_codes;
      std::vector<TVMValue> values;
      const size_t num_args = 3;
      type_codes.resize(num_args);
      values.resize(num_args);
      for (size_t i = 0; i < num_args; ++i) {
        type_codes[i] = kArrayHandle;
        values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
      }

      std::string funcname = std::string(func);
      funcname += SetAttr("output", std::to_string(k));
      funcname += SetAttr("reduce1st_dim", std::to_string(reduce1st_dim));
      MXNET_ASSIGN_REQ_SWITCH(req[k], req_type, {
        funcname += SetReq(req_type);
      });

      tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], num_args);
      tvm::runtime::TVMOpModule::Get()->CallEx(funcname, ctx, tblobs, tvm_args);
    }
#else
    LOG(FATAL) << "Please add USE_TVM_OP=1 as a compile flag for compiling MXNet source code "
                  "to enable TVM-generated kernels for operator " << func;
#endif  // MXNET_USE_TVM_OP
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_ELEMWISE_BROADCAST_OP_H_
