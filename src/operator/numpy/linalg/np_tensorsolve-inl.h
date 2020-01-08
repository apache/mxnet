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
 * Copyright (c) 2019 by Contributors
 * \file np_tensorsolve-inl.h
 * \brief Placeholder for tensor solve
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_TENSORSOLVE_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_TENSORSOLVE_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../../operator_common.h"
#include "../../mshadow_op.h"
#include "../../tensor/la_op.h"
#include "../../tensor/la_op-inl.h"
#include "../np_tensordot_op-inl.h"
#include "./np_solve-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct TensorsolveParam : public dmlc::Parameter<TensorsolveParam> {
  mxnet::Tuple<int> a_axes;
  DMLC_DECLARE_PARAMETER(TensorsolveParam) {
    DMLC_DECLARE_FIELD(a_axes)
    .set_default(mxnet::Tuple<int>())
    .describe("Tuple of ints, optional. Axes in a to reorder to the right, before inversion.");
  }
};

// Fix negative axes.
inline void FixNegativeAxes(mxnet::Tuple<int> *a_axes_param,
                            const mxnet::TShape& a_shape) {
  if (-1 == a_axes_param->ndim()) { return; }
  const int a_ndim = a_shape.ndim();
  for (auto& i : *a_axes_param) {
    i = (i + a_ndim) % a_ndim;
  }
}

// Get remained axes and axes of a.
inline void GetReorderedAxes(const mxnet::Tuple<int>& a_axes_param,
                             mxnet::Tuple<int> *a_axes_remained,
                             mxnet::Tuple<int> *a_axes,
                             const mxnet::TShape& a_shape) {
  std::vector<int> a_axes_vec;
  for (int i = 0; i < a_shape.ndim(); ++i) {
    a_axes_vec.push_back(i);
  }
  // Get remained axes and axes.
  if (-1 == a_axes_param.ndim()) {
    *a_axes_remained = mxnet::Tuple<int>(a_axes_vec);
    *a_axes = mxnet::Tuple<int>(a_axes_vec);
    return;
  }
  for (const auto& i : a_axes_param) {
    a_axes_vec.erase(std::find(a_axes_vec.begin(), a_axes_vec.end(), i));
  }
  *a_axes_remained = mxnet::Tuple<int>(a_axes_vec);

  a_axes_vec.clear();
  for (const auto& i : *a_axes_remained) {
    a_axes_vec.push_back(i);
  }
  for (const auto& i : a_axes_param) {
    a_axes_vec.push_back(i);
  }
  *a_axes = mxnet::Tuple<int>(a_axes_vec);
}

// Calculate output shape if a and b is tensor
inline mxnet::TShape GetOutShape(const mxnet::TShape& a_shape,
                                 const mxnet::TShape& b_shape) {
  const int a_ndim = a_shape.ndim(), b_ndim = b_shape.ndim();
  const int temp = a_ndim > b_ndim ? b_ndim : b_ndim - a_ndim;
  mxnet::TShape out_shape(a_ndim - temp, -1);
  for (int i = temp; i < a_ndim; ++i) {
    out_shape[i - temp] = a_shape[i];
  }
  return out_shape;
}

// Calculates workspace size of tensorsolve forward.
template<typename xpu>
size_t TensorsolveForwardWorkspaceSize(const Tuple<int>& a_axes_param,
                                       const TBlob& a,
                                       const TBlob& b,
                                       const TBlob& out,
                                       const std::vector<OpReqType>& req) {
  if (kNullOp == req[0]) { return 0U; }

  // Zero-size output, no need to launch kernel
  if (0U == out.shape_.Size()) { return 0U; }

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;
  MSHADOW_SGL_DBL_TYPE_SWITCH(out.type_flag_, DType, {
    if (0U == a_shape.Size() || 0U == b_shape.Size()) {
      // 0-size input
      return 0U;
    } else if (0 == a_shape.ndim() || 0 == b_shape.ndim()) {
      // At least 1 scalar.
      return (a.Size() + b.Size()) * sizeof(DType) + b.Size() * sizeof(int);
    } else {
      // Two tensors of at least 1 dimensions.
      return (2 * a.Size() + b.Size()) * sizeof(DType) + b.Size() * sizeof(int);
    }
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

template<int req>
struct assign_helper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, DType *out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

struct tensorsolve {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& A,
                 const Tensor<xpu, 2, DType>& X,
                 const Tensor<xpu, 1, int>& ipiv,
                 const OpContext& ctx) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    linalg_solve(A, X, ipiv, s);  // ipiv for work_space in Lapacke_#gesv
  }
};

template<typename xpu, typename laop>
void TensorsolveOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;
  const mxnet::TShape out_shape = out.shape_;
  const TensorsolveParam& param = nnvm::get<TensorsolveParam>(attrs.parsed);
  mxnet::Tuple<int> a_axes_param = param.a_axes;
  FixNegativeAxes(&a_axes_param, a_shape);

  size_t workspace_size = TensorsolveForwardWorkspaceSize<xpu>(a_axes_param, a, b, out, req);
  Tensor<xpu, 1, char> workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(
    Shape1(workspace_size), ctx.get_stream<xpu>());

  if (kNullOp == req[0]) { return; }

  // Zero-size output, no need to launch kernel
  if (0U == out.shape_.Size()) { return; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(out.type_flag_, DType, {
    if (0U == a_shape.Size() || 0U == b_shape.Size()) {  // 0-size input
      if (req[0] != kAddTo) {
        Tensor<xpu, 1, DType> out_tensor =
          out.get_with_shape<xpu, 1, DType>(Shape1(out.shape_.Size()), s);
        out_tensor = static_cast<DType>(0);
      }
    } else if (0U == a_shape.ndim() || 0U ==  b_shape.ndim()) {  // At least 1 scalar.
      // Check again
      CHECK_EQ(a_shape.Size(), 1U)
        << "a's and b's dimensions don't match";
      CHECK_EQ(b_shape.Size(), 1U)
        << "a's and b's dimensions don't match";

      DType* a_ptr =
        reinterpret_cast<DType*>(workspace.dptr_);
      DType* b_ptr =
        reinterpret_cast<DType*>(workspace.dptr_+ a.Size() * sizeof(DType));
      int* ipiv_ptr =
        reinterpret_cast<int*>(workspace.dptr_ + (a.Size() + b.Size()) * sizeof(DType));

      // Cast type
      MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, a_shape.Size(), a_ptr, a.dptr<AType>());
      });
      MSHADOW_TYPE_SWITCH(b.type_flag_, BType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, b_shape.Size(), b_ptr, b.dptr<BType>());
      });

      mxnet::TBlob a_tblob(a_ptr, Shape2(1, 1), a.dev_mask(), a.dev_id());
      mxnet::TBlob b_tblob(b_ptr, Shape2(1, 1), b.dev_mask(), b.dev_id());
      mxnet::TBlob ipiv_tblob(ipiv_ptr, Shape1(1), out.dev_mask(), out.dev_id());
      Tensor<xpu, 2, DType> a_tensor = a_tblob.get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> b_tensor = b_tblob.get<xpu, 2, DType>(s);
      Tensor<xpu, 1, int> ipiv_tensor = ipiv_tblob.get<xpu, 1, int>(s);

      // Solve linear equation
      laop::op(a_tensor, b_tensor, ipiv_tensor, ctx);
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          mxnet_op::Kernel<assign_helper<req_type>, xpu>::Launch(
            s, out_shape.Size(), b_tensor.dptr_, out.dptr<DType>());
      });
    } else {
      // Two tensors of at least 1 dimensions.
      Tuple<int> a_axes_remained;
      Tuple<int> a_axes;
      GetReorderedAxes(a_axes_param, &a_axes_remained, &a_axes, a_shape);
      mxnet::TShape a_transpose_shape = GetReorderedShape(a_shape, a_axes);
      const int N = b_shape.Size();

      DType* a_ptr =
        reinterpret_cast<DType*>(workspace.dptr_);
      DType* a_trans_ptr =
        reinterpret_cast<DType*>(workspace.dptr_ + a.Size() * sizeof(DType));
      DType* b_ptr =
        reinterpret_cast<DType*>(workspace.dptr_ + 2 * a.Size() * sizeof(DType));
      int* ipiv_ptr =
        reinterpret_cast<int*>(workspace.dptr_ + (2 * a.Size() + b.Size()) * sizeof(DType));

      // Cast type
      MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, a_shape.Size(), a_ptr, a.dptr<AType>());
      });
      // Cast type
      MSHADOW_TYPE_SWITCH(b.type_flag_, BType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, b_shape.Size(), b_ptr, b.dptr<BType>());
      });

      mxnet::TBlob a_tblob =
        TBlob(a_ptr, a_shape, a.dev_mask(), a.dev_id());
      mxnet::TBlob a_transpose_tblob =
        TBlob(a_trans_ptr, a_transpose_shape, a.dev_mask(), a.dev_id());
      mxnet::TBlob b_tblob =
        TBlob(b_ptr, b_shape, b.dev_mask(), b.dev_id());
      mxnet::TBlob ipiv_tblob =
        TBlob(ipiv_ptr, b_shape, out.dev_mask(), out.dev_id());
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx,
                                    a_tblob,            // src
                                    a_transpose_tblob,  // res
                                    mxnet::TShape(a_axes.begin(), a_axes.end()));

      Tensor<xpu, 2, DType> a_tensor =
        a_tblob.get_with_shape<xpu, 2, DType>(Shape2(N, N), s);
      Tensor<xpu, 1, int> ipiv_tensor =
        ipiv_tblob.get_with_shape<xpu, 1, int>(Shape1(N), s);
      Tensor<xpu, 2, DType> b_tensor =
        b_tblob.get_with_shape<xpu, 2, DType>(Shape2(1, N), s);
      Tensor<xpu, 2, DType> out_tensor =
        out.get_with_shape<xpu, 2, DType>(Shape2(1, N), s);

      a_tblob = a_tblob.reshape(Shape2(N, N));
      a_transpose_tblob = a_transpose_tblob.reshape(Shape2(N, N));
      Tuple<int> a_axes_2D(std::vector<int>{1, 0});
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx,
                                    a_transpose_tblob,  // src
                                    a_tblob,            // res
                                    mxnet::TShape(a_axes_2D.begin(), a_axes_2D.end()));
      // Solve linear equation
      laop::op(a_tensor, b_tensor, ipiv_tensor, ctx);
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<assign_helper<req_type>, xpu>::Launch(
          s, out_shape.Size(), b_tensor.dptr_, out_tensor.dptr_);
      });
    }
  });
}

// Calculates workspace size of tensordot backward.
template<typename xpu>
size_t TensorsolveBackwardWorkspaceSize(const TBlob& out_grad,
                                        const TBlob& a,
                                        const TBlob& b,
                                        const TBlob& x) {
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;
  const mxnet::TShape& x_shape = x.shape_;

  // Zero-size output, no need to launch kernel
  if (0U == a_shape.Size() || 0U == b_shape.Size()) { return 0U; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    int work_space_size = 0;
    if (0U == a_shape.ndim() || 0U == b_shape.ndim()) {
      // At least 1 scalar.
      work_space_size += sizeof(DType) * a_shape.Size();  // for tensorinv(a)
      work_space_size += sizeof(DType) * a_shape.Size();  // for getri work space lu
      work_space_size += sizeof(int) * b_shape.Size();    // for getri work space pivot
    } else {
      // Two tensors of at least 1 dimensions.
      work_space_size += sizeof(DType) * a_shape.Size();  // for tensorinv(a)
      work_space_size += sizeof(DType) * a_shape.Size();  // for getri work space lu
      work_space_size += sizeof(DType) * b_shape.Size();  // for b
      work_space_size += sizeof(DType) * x_shape.Size();  // for x
      work_space_size += sizeof(DType) * a_shape.Size();  // for grad_a
      work_space_size += sizeof(DType) * b_shape.Size();  // for grad_b
      work_space_size += sizeof(int) * b_shape.Size();    // for getri work space pivot
    }
    return work_space_size;
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

// Get original axes for tensor a.
inline void GetOriginAxes(const mxnet::TShape& a_shape,
                          const mxnet::Tuple<int>& a_axes,
                          mxnet::Tuple<int> *a_origin_axes) {
  std::vector<int> a_origin_axes_vec(a_shape.ndim(), -1);
  for (int i = 0; i < a_shape.ndim(); ++i) {
    a_origin_axes_vec[a_axes[i]] = i;
  }
  *a_origin_axes = mxnet::Tuple<int>(a_origin_axes_vec);
}

struct tensorsolve_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dX,
                 const Tensor<xpu, 3, DType>& inv_A,
                 const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& X,
                 const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& dB,
                 const OpContext& ctx) {
    // (1) calcualte dB = trans(tensorinv(A)) * dX
    // (2) calcualte dA = dB * trans(X)
    Stream<xpu> *s = ctx.get_stream<xpu>();
    gemm2::op(inv_A, dX, dB, DType(1), true, false, s);
    gemm2::op(dB, X, dA, DType(-1), false, true, s);
  }
};

template<typename xpu, typename laop>
void TensorsolveBackwardImpl(const Tuple<int>& a_axes_param,
                             const TBlob& out_grad,
                             const TBlob& a,
                             const TBlob& b,
                             const TBlob& x,
                             const TBlob& grad_a,
                             const TBlob& grad_b,
                             const OpContext& ctx,
                             const std::vector<OpReqType>& req,
                             const Tensor<xpu, 1, char>& workspace) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;
  const mxnet::TShape& x_shape = x.shape_;

  if (kNullOp == req[0] && kNullOp == req[1]) { return; }

  // Zero-size output, no need to launch kernel
  if (0U == a_shape.Size() || 0U == b_shape.Size()) { return; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    if (0 == a_shape.ndim() || 0 == b_shape.ndim()) {
      // At least 1 scalar.
      CHECK_EQ(a_shape.Size(), 1U)
        << "a's and b's dimensions don't match";
      CHECK_EQ(b_shape.Size(), 1U)
        << "a's and b's dimensions don't match";

      // Allocate workspace.
      DType *tensorinv_a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      DType *lu_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a_shape.Size() * sizeof(DType));
      int *ipiv_ptr = reinterpret_cast<int*>(workspace.dptr_ + 2 * a_shape.Size() * sizeof(DType));
      TBlob tensorinv_a(tensorinv_a_ptr, a_shape, xpu::kDevMask);
      TBlob lu(lu_ptr, a_shape, xpu::kDevMask);
      TBlob ipiv(ipiv_ptr, b_shape, xpu::kDevMask);

      MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, a_shape.Size(),
          tensorinv_a_ptr,
          a.dptr<AType>());
      });
      // Calculate tensorinv(a)
      Tensor<xpu, 3, DType> tensorinv_a_tensor =
        tensorinv_a.get_with_shape<xpu, 3, DType>(Shape3(1, 1, 1), s);
      Tensor<xpu, 3, DType> lu_tensor =
        lu.get_with_shape<xpu, 3, DType>(Shape3(1, 1, 1), s);
      Tensor<xpu, 2, int> ipiv_tensor =
        ipiv.get_with_shape<xpu, 2, int>(Shape2(1, 1), s);
      batch_inverse(tensorinv_a_tensor, lu_tensor, ipiv_tensor, ctx);

      MSHADOW_TYPE_SWITCH(x.type_flag_, XType, {
        DType temp1 = (*(tensorinv_a_tensor.dptr_)) * (*(out_grad.dptr<DType>()));
        DType temp2 = -temp1 * static_cast<DType>(*x.dptr<XType>());
        ASSIGN_DISPATCH(*grad_b.dptr<DType>(), req[1], temp1);
        ASSIGN_DISPATCH(*grad_a.dptr<DType>(), req[0], temp2);
      });
    } else {
      // Two tensors of at least 1 dimensions.
      const int N = b_shape.Size();
      Tuple<int> a_axes_remained;
      Tuple<int> a_axes;
      Tuple<int> a_origin_axes;
      // Use a_axes to transpose (a_shape) --> (a_reordered_shape).
      GetReorderedAxes(a_axes_param, &a_axes_remained, &a_axes, a_shape);
      // Use a_origin_axes to transpose (a_reordered_shape) --> (a_shape).
      GetOriginAxes(a_shape, a_axes, &a_origin_axes);
      mxnet::TShape reordered_a_shape = GetReorderedShape(a_shape, a_axes);

      // Allocate workspace.
      DType *tensorinv_a_ptr = reinterpret_cast<DType*>(
        workspace.dptr_);
      DType *lu_ptr = reinterpret_cast<DType*>(
        workspace.dptr_ + a_shape.Size() * sizeof(DType));
      DType *b_ptr = reinterpret_cast<DType*>(
        workspace.dptr_ + 2 * a_shape.Size() * sizeof(DType));
      DType *x_ptr = reinterpret_cast<DType*>(
        workspace.dptr_ + (2 * a_shape.Size() + b_shape.Size()) * sizeof(DType));
      DType *grad_a_ptr = reinterpret_cast<DType*>(
        workspace.dptr_ + 2 * (a_shape.Size() + b_shape.Size()) * sizeof(DType));
      DType *grad_b_ptr = reinterpret_cast<DType*>(
        workspace.dptr_ + (3 * a_shape.Size() + 2 * b_shape.Size()) * sizeof(DType));
      int *ipiv_ptr = reinterpret_cast<int*>(
        workspace.dptr_ + 3 * (a_shape.Size() + b_shape.Size()) * sizeof(DType));

      TBlob tensorinv_a_data(tensorinv_a_ptr, a_shape, xpu::kDevMask);
      TBlob lu_data(lu_ptr, a_shape, xpu::kDevMask);
      TBlob b_data(b_ptr, b_shape, xpu::kDevMask);
      TBlob x_data(x_ptr, x_shape, xpu::kDevMask);
      TBlob grad_a_data(grad_a_ptr, reordered_a_shape, xpu::kDevMask);
      TBlob grad_b_data(grad_b_ptr, b_shape, xpu::kDevMask);
      TBlob ipiv_data(ipiv_ptr, b_shape, xpu::kDevMask);
      MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, a_shape.Size(),
          lu_ptr,
          a.dptr<AType>());
      });
      MSHADOW_TYPE_SWITCH(b.type_flag_, BType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, b_shape.Size(),
          b_ptr,
          b.dptr<BType>());
      });
      MSHADOW_TYPE_SWITCH(x.type_flag_, XType, {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, x_shape.Size(),
          x_ptr,
          x.dptr<XType>());
      });
      // Eg: lu_data(2, 3, 2, 15, 4, 5) -> tensorinv_a_data(3, 4, 5, 15, 2, 2)
      tensorinv_a_data = tensorinv_a_data.reshape(reordered_a_shape);
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx,
                                    lu_data,           // src
                                    tensorinv_a_data,  // res
                                    mxnet::TShape(a_axes.begin(), a_axes.end()));

      Tensor<xpu, 3, DType> tensorinv_a_tensor =
        tensorinv_a_data.get_with_shape<xpu, 3, DType>(Shape3(1, N, N), s);
      Tensor<xpu, 3, DType> lu_tensor =
        lu_data.get_with_shape<xpu, 3, DType>(Shape3(1, N, N), s);
      Tensor<xpu, 3, DType> b_tensor =
        b_data.get_with_shape<xpu, 3, DType>(Shape3(1, N, 1), s);
      Tensor<xpu, 3, DType> x_tensor =
        x_data.get_with_shape<xpu, 3, DType>(Shape3(1, N, 1), s);
      Tensor<xpu, 3, DType> grad_a_tensor =
        grad_a_data.get_with_shape<xpu, 3, DType>(Shape3(1, N, N), s);
      Tensor<xpu, 3, DType> grad_b_tensor =
        grad_b_data.get_with_shape<xpu, 3, DType>(Shape3(1, N, 1), s);
      Tensor<xpu, 2, int> ipiv_tensor =
        ipiv_data.get_with_shape<xpu, 2, int>(Shape2(1, N), s);

      // Calculate tensorinv(a).
      batch_inverse(tensorinv_a_tensor, lu_tensor, ipiv_tensor, ctx);
      // No need to transpose tensorinv_a
      // because transpose(tensorinv_a).shape == reordered_a_shape.
      laop::op(out_grad.get_with_shape<xpu, 3, DType>(x_tensor.shape_, s),
               tensorinv_a_tensor,
               b_tensor,
               x_tensor,
               grad_a_tensor,
               grad_b_tensor,
               ctx);
      // Eg: grad_a_src(3, 4, 5, 15, 2, 2) --> lu_data(2, 3, 2, 15, 4, 5)
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx,
                                    grad_a_data,  // src
                                    lu_data,      // res
                                    mxnet::TShape(a_origin_axes.begin(), a_origin_axes.end()));

      MXNET_ASSIGN_REQ_SWITCH(req[1], req_type, {
        mxnet_op::Kernel<assign_helper<req_type>, xpu>::Launch(
          s, b_shape.Size(), grad_b_tensor.dptr_, grad_b.dptr<DType>());
      });
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<assign_helper<req_type>, xpu>::Launch(
          s, a_shape.Size(), lu_tensor.dptr_, grad_a.dptr<DType>());
      });
    }
  });
}

template<typename xpu, typename laop>
void TensorsolveOpBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);

  const TBlob& out_grad = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& x = inputs[3];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;
  const TensorsolveParam& param = nnvm::get<TensorsolveParam>(attrs.parsed);
  mxnet::Tuple<int> a_axes_param = param.a_axes;
  FixNegativeAxes(&a_axes_param, a_shape);

  size_t workspace_size = TensorsolveBackwardWorkspaceSize<xpu>(out_grad, a, b, x);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size),
                                                   ctx.get_stream<xpu>());
  TensorsolveBackwardImpl<xpu, laop>(a_axes_param,
                                     out_grad,
                                     a, b, x,
                                     grad_a, grad_b,
                                     ctx, req,
                                     workspace);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_TENSORSOLVE_INL_H_
