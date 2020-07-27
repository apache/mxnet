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
 * Copyright (c) 2020 by Contributors
 * \file np_matrix_rank-inl.h
 * \brief Placeholder for matrix_rank
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_MATRIX_RANK_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_MATRIX_RANK_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../../operator_common.h"
#include "../../mshadow_op.h"
#include "./np_pinv-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct MatrixRankNoneTolParam : public dmlc::Parameter<MatrixRankNoneTolParam> {
  float finfoEps32;
  double finfoEps64;
  bool hermitian;
  DMLC_DECLARE_PARAMETER(MatrixRankNoneTolParam) {
    DMLC_DECLARE_FIELD(finfoEps32)
    .set_default(0)
    .describe("Machine limits for float32 type");
    DMLC_DECLARE_FIELD(finfoEps64)
    .set_default(0)
    .describe("Machine limits for float64 type");
    DMLC_DECLARE_FIELD(hermitian)
    .set_default(false)
    .describe("If True, M is assumed to be Hermitian (symmetric if real-valued).");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream finfoEps32_s, finfoEps64_s, hermitian_s;
    finfoEps32_s << finfoEps32;
    finfoEps64_s << finfoEps64;
    hermitian_s << hermitian;
    (*dict)["finfoEps32"] = finfoEps32_s.str();
    (*dict)["finfoEps64"] = finfoEps64_s.str();
    (*dict)["hermitian"] = hermitian_s.str();
  }
};

struct MatrixRankParam : public dmlc::Parameter<MatrixRankParam> {
  bool hermitian;
  DMLC_DECLARE_PARAMETER(MatrixRankParam) {
    DMLC_DECLARE_FIELD(hermitian)
    .set_default(false)
    .describe("If True, M is assumed to be Hermitian (symmetric if real-valued).");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream hermitian_s;
    hermitian_s << hermitian;
    (*dict)["hermitian"] = hermitian_s.str();
  }
};

template<int req>
struct VectorRankKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data,
                                  int64_t *out_data, const int& data_size) {
    bool all_nozero = true;
    for (int j = 0; j < data_size; ++j) {
      if (!((in_data[j] > 0 ? in_data[j] : -in_data[j]) > 0)) {
        all_nozero = false;
        break;
      }
    }
    KERNEL_ASSIGN(*out_data, req, static_cast<int64_t>(all_nozero ? 1 : 0));
  }
};

template<int req>
struct MatrixRankNoneTolKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, int64_t *out_data,
                                  const int& nrow, const int& ncol, const double& finfoEps,
                                  const int& data_size, const int& batch_size) {
    if (i < batch_size) {
      DType max_singular_value = 0;
      for (int j = 0; j < data_size; ++j) {
        DType sv = in_data[j + i * data_size];
        max_singular_value = sv > max_singular_value ? sv : max_singular_value;
      }
      double tol = (nrow > ncol ? nrow : ncol) * static_cast<double>(max_singular_value) * finfoEps;
      int64_t rank_num = 0;
      for (int j = 0; j < data_size; ++j) {
        rank_num += in_data[j + i * data_size] > tol ? 1 : 0;
      }
      KERNEL_ASSIGN(out_data[i], req, rank_num);
    }
  }
};

template<int req>
struct MatrixRankKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, int64_t *out_data,
                                  const int& data_size, const int& batch_size) {
    if (i < batch_size) {
      int64_t rank_num = 0;
      for (int j = 0; j < data_size; ++j) {
        rank_num += in_data[j + i * data_size] > 0 ? 1 : 0;
      }
      KERNEL_ASSIGN(out_data[i], req, rank_num);
    }
  }
};

struct SVDWrapper {
  template<typename xpu, typename DType>
  static void op(const TBlob& a, const TBlob& s,
                 const TBlob& u, const mxnet::TShape& ut_shape,
                 const TBlob& v, const mxnet::TShape& vt_shape,
                 const TBlob& work, const OpContext& ctx) {
    Stream<xpu> *s_xpu = ctx.get_stream<xpu>();
    const mxnet::TShape& a_shape = a.shape_;
    const mxnet::TShape& ut_axis = GetTransAxis(u.shape_);
    const int a_ndim = a.ndim();
    const int nrow = a_shape[a_ndim - 2];
    const int ncol = a_shape[a_ndim - 1];
    if (nrow > ncol) {
      const_cast<TBlob&>(u) = u.reshape(ut_shape);
      const_cast<TBlob&>(v) = v.reshape(vt_shape);
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a, u, ut_axis);
      BatchSVDImpl(ncol, nrow,
                   v.FlatToKD<xpu, 3, DType>(s_xpu),
                   s.FlatToKD<xpu, 2, DType>(s_xpu),
                   u.FlatToKD<xpu, 3, DType>(s_xpu),
                   work.FlatToKD<xpu, 1, DType>(s_xpu), s_xpu);
    } else {
      if (a.dptr<DType>() != v.dptr<DType>()) {
        Copy(v.FlatToKD<xpu, 3, DType>(s_xpu), a.FlatToKD<xpu, 3, DType>(s_xpu), s_xpu);
      }
      BatchSVDImpl(nrow, ncol,
                   u.FlatToKD<xpu, 3, DType>(s_xpu),
                   s.FlatToKD<xpu, 2, DType>(s_xpu),
                   v.FlatToKD<xpu, 3, DType>(s_xpu),
                   work.FlatToKD<xpu, 1, DType>(s_xpu), s_xpu);
    }
  }
};

inline void GetOrCheckBroadcastShape(const nnvm::NodeAttrs& attrs,
                                     const mxnet::TShape& a_shape,
                                     const mxnet::TShape& tol_shape,
                                     mxnet::TShape *broadcast_shape = nullptr,
                                     mxnet::TShape *new_tol_shape = nullptr) {
  CHECK_GE(a_shape.ndim(), 2);
  const int a_ndim = a_shape.ndim();
  const int tol_ndim = tol_shape.ndim();
  const int nrow = a_shape[a_ndim - 2];
  const int ncol = a_shape[a_ndim - 1];
  // Get new tol shape.
  mxnet::TShape temp_new_tol_shape(tol_ndim + 1, 1);
  for (int i = 0; i < tol_ndim; ++i) { temp_new_tol_shape[i] = tol_shape[i]; }
  // Get singular value shape.
  mxnet::TShape temp_s_shape(a_ndim - 1, 0);
  for (int i = 0; i < a_ndim - 2; ++i) {
    temp_s_shape[i] = a_shape[i];
  }
  temp_s_shape[a_ndim - 2] = std::min(nrow, ncol);
  // Check binary broadcast shape.
  mxnet::ShapeVector in_shape_vec({ temp_s_shape, temp_new_tol_shape });
  mxnet::ShapeVector out_shape_vec(1, mxnet::TShape());
  mxnet::op::BinaryBroadcastShape(attrs, &in_shape_vec, &out_shape_vec);
  // Assign shape.
  if (broadcast_shape) {
    *broadcast_shape = out_shape_vec[0];
  }
  if (new_tol_shape) {
    *new_tol_shape = temp_new_tol_shape;
  }
}

template<typename xpu, typename DType>
struct WSQ {
  static size_t SVDWorkspaceSizeQuery(const TBlob& a,
                                      const mxnet::TShape& u_shape,
                                      const mxnet::TShape& s_shape,
                                      const mxnet::TShape& v_shape,
                                      const OpContext& ctx) {
    size_t workspace_size = 0;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int a_ndim = a.shape_.ndim();
    const int u_ndim = u_shape.ndim();
    const int s_ndim = s_shape.ndim();
    const int v_ndim = v_shape.ndim();
    mxnet::TShape u_shape2 = Shape2(u_shape[u_ndim - 2], u_shape[u_ndim - 1]);
    mxnet::TShape s_shape1 = Shape1(s_shape[s_ndim - 1]);
    mxnet::TShape v_shape2 = Shape2(v_shape[v_ndim - 2], v_shape[v_ndim - 1]);
    if (xpu::kDevCPU) {
      std::vector<DType> u_vec(u_shape2.Size(), 0);
      std::vector<DType> s_vec(s_shape1.Size(), 0);
      std::vector<DType> v_vec(v_shape2.Size(), 0);
      // Get workspace size in linalg_gesdd.
      workspace_size += linalg_gesdd_workspace_query(
        a.shape_[a_ndim - 2], a.shape_[a_ndim - 1],
        TBlob(u_vec.data(), u_shape2, a.dev_mask(), a.dev_id()).get<xpu, 2, DType>(s),
        TBlob(s_vec.data(), s_shape1, a.dev_mask(), a.dev_id()).get<xpu, 1, DType>(s),
        TBlob(v_vec.data(), v_shape2, a.dev_mask(), a.dev_id()).get<xpu, 2, DType>(s), s);
    } else {
      Storage::Handle u_handle =
        Storage::Get()->Alloc(sizeof(DType) * u_shape2.Size(), Context::GPU());
      Storage::Handle s_handle =
        Storage::Get()->Alloc(sizeof(DType) * s_shape1.Size(), Context::GPU());
      Storage::Handle v_handle =
        Storage::Get()->Alloc(sizeof(DType) * v_shape2.Size(), Context::GPU());
      TBlob u_data(static_cast<DType*>(u_handle.dptr), u_shape2, a.dev_mask(), a.dev_id());
      TBlob s_data(static_cast<DType*>(s_handle.dptr), s_shape1, a.dev_mask(), a.dev_id());
      TBlob v_data(static_cast<DType*>(v_handle.dptr), v_shape2, a.dev_mask(), a.dev_id());
      // Get workspace size in linalg_gesvd.
      if (a.shape_[a_ndim - 2] >= a.shape_[a_ndim - 1]) {
        workspace_size += linalg_gesvd_workspace_query(v_data.get<xpu, 2, DType>(s),
                                                       s_data.get<xpu, 1, DType>(s),
                                                       u_data.get<xpu, 2, DType>(s), s);
      } else {
        workspace_size += linalg_gesvd_workspace_query(u_data.get<xpu, 2, DType>(s),
                                                       s_data.get<xpu, 1, DType>(s),
                                                       v_data.get<xpu, 2, DType>(s), s);
      }
      Storage::Get()->Free(u_handle);
      Storage::Get()->Free(s_handle);
      Storage::Get()->Free(v_handle);
    }
    return workspace_size;
  }

  static size_t MatrixRankNoneTolForwardWSQ(size_t *svd_workspace_size,
                                            const TBlob& a,
                                            const OpContext& ctx) {
    size_t workspace_size = 0;
    mxnet::TShape u_shape, s_shape, v_shape;
    GetPinvShape(a.shape_, &u_shape, &s_shape, &v_shape);
    *svd_workspace_size = SVDWorkspaceSizeQuery(a, u_shape, s_shape, v_shape, ctx);
    workspace_size += *svd_workspace_size;  // For #gesdd_ or #gesvd work space.
    workspace_size += u_shape.Size();       // For UT.
    workspace_size += s_shape.Size();       // For S.
    workspace_size += v_shape.Size();       // For V.
    return workspace_size * sizeof(DType);
  }

  static size_t MatrixRankForwardWSQ(size_t *svd_workspace_size,
                                     const TBlob& a,
                                     const TBlob& tol,
                                     const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx) {
    const mxnet::TShape a_shape = a.shape_;
    const mxnet::TShape tol_shape = tol.shape_;
    size_t workspace_size = 0;
    mxnet::TShape u_shape, s_shape, v_shape;
    GetPinvShape(a.shape_, &u_shape, &s_shape, &v_shape);
    mxnet::TShape broadcast_shape, new_tol_shape;
    GetOrCheckBroadcastShape(attrs, a_shape, tol_shape, &broadcast_shape, &new_tol_shape);
    *svd_workspace_size = SVDWorkspaceSizeQuery(a, u_shape, s_shape, v_shape, ctx);
    workspace_size += *svd_workspace_size;     // For #gesdd_ or #gesvd work space.
    workspace_size += u_shape.Size();          // For UT.
    workspace_size += s_shape.Size();          // For S.
    workspace_size += v_shape.Size();          // For V.
    workspace_size += new_tol_shape.Size();    // For tol with newaxis.
    workspace_size += broadcast_shape.Size();  // For binary broadcast shape.
    return workspace_size * sizeof(DType);
  }
};

template<typename xpu>
void MatrixRankNoneTolForwardImpl(const TBlob& a,
                                  const TBlob& rank,
                                  const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<OpReqType>& req) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape& a_shape = a.shape_;
  const int a_ndim = a.ndim();
  MSHADOW_SGL_DBL_TYPE_SWITCH(a.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      if (a_ndim < 2) {
        mxnet_op::Kernel<VectorRankKernel<req_type>, xpu>::Launch(
          s, 1, a.dptr<DType>(), rank.dptr<int64_t>(), a.Size());
        return;
      }
      // a_ndim >= 2
      const int nrow = a_shape[a_ndim - 2];
      const int ncol = a_shape[a_ndim - 1];
      const MatrixRankNoneTolParam& param = nnvm::get<MatrixRankNoneTolParam>(attrs.parsed);
      CHECK_EQ(param.hermitian, false)
        << "matrix_rank not support param.hermitian = true at present.";
      double finfoEps = a.type_flag_ == mshadow::kFloat32 ? param.finfoEps32 : param.finfoEps64;
      // Step1: Calculate workspace size.
      size_t svd_workspace_size = 0;
      size_t workspace_size =
        WSQ<xpu, DType>::MatrixRankNoneTolForwardWSQ(&svd_workspace_size, a, ctx);
      Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      // Step2: Allocate memory.
      mxnet::TShape s_shape, u_shape, v_shape, ut_shape, vt_shape;
      GetPinvShape(a_shape, &u_shape, &s_shape, &v_shape, &ut_shape, &vt_shape);
      DType *s_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      DType *u_ptr = s_ptr + s_shape.Size();
      DType *v_ptr = u_ptr + u_shape.Size();
      DType *work_ptr = v_ptr + v_shape.Size();
      TBlob s_data(s_ptr, s_shape, a.dev_mask(), a.dev_id());
      TBlob u_data(u_ptr, u_shape, a.dev_mask(), a.dev_id());
      TBlob v_data(v_ptr, v_shape, a.dev_mask(), a.dev_id());
      TBlob work_data(work_ptr, Shape1(svd_workspace_size), a.dev_mask(), a.dev_id());
      // Step3: SVD.
      SVDWrapper::op<xpu, DType>(a, s_data, u_data, ut_shape, v_data, vt_shape, work_data, ctx);
      // Step4: Calculate rank.
      const int data_size = s_data.size(s_data.ndim() - 1);
      const int batch_size = a_ndim == 2 ? 1 : s_shape.ProdShape(0, s_shape.ndim() - 1);
      mxnet_op::Kernel<MatrixRankNoneTolKernel<req_type>, xpu>::Launch(s, batch_size,
                                                                       s_data.dptr<DType>(),
                                                                       rank.dptr<int64_t>(),
                                                                       nrow, ncol, finfoEps,
                                                                       data_size, batch_size);
    });
  });
}

template<typename xpu>
void MatrixRankNoneTolForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (kNullOp == req[0]) { return; }
  CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);

  const TBlob& a = inputs[0];
  const TBlob& rank = outputs[0];
  MatrixRankNoneTolForwardImpl<xpu>(a, rank, attrs, ctx, req);
}

template<typename xpu>
void MatrixRankForwardImpl(const TBlob& a,
                           const TBlob& tol,
                           const TBlob& rank,
                           const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<OpReqType>& req) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& tol_shape = tol.shape_;
  const int a_ndim = a.ndim();
  MSHADOW_SGL_DBL_TYPE_SWITCH(a.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      if (a_ndim < 2) {
        mxnet_op::Kernel<VectorRankKernel<req_type>, xpu>::Launch(
          s, 1, a.dptr<DType>(), rank.dptr<int64_t>(), a.Size());
        return;
      }
      // a_ndim >= 2
      const MatrixRankParam& param = nnvm::get<MatrixRankParam>(attrs.parsed);
      CHECK_EQ(param.hermitian, false)
        << "matrix_rank not support param.hermitian = true at present.";
      mxnet::TShape s_shape, u_shape, v_shape, ut_shape, vt_shape;
      GetPinvShape(a_shape, &u_shape, &s_shape, &v_shape, &ut_shape, &vt_shape);
      mxnet::TShape broadcast_shape, new_tol_shape;
      GetOrCheckBroadcastShape(attrs, a_shape, tol_shape, &broadcast_shape, &new_tol_shape);
      // Step1: Calculate workspace size.
      size_t svd_workspace_size = 0;
      size_t workspace_size =
        WSQ<xpu, DType>::MatrixRankForwardWSQ(&svd_workspace_size, a, tol, attrs, ctx);
      Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      // Step2: Allocate memory.
      DType *s_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      DType *u_ptr = s_ptr + s_shape.Size();
      DType *v_ptr = u_ptr + u_shape.Size();
      DType *work_ptr = v_ptr + v_shape.Size();
      DType *new_tol_ptr = work_ptr + svd_workspace_size;
      DType *broadcast_ptr = new_tol_ptr + new_tol_shape.Size();
      TBlob s_data(s_ptr, s_shape, a.dev_mask(), a.dev_id());
      TBlob u_data(u_ptr, u_shape, a.dev_mask(), a.dev_id());
      TBlob v_data(v_ptr, v_shape, a.dev_mask(), a.dev_id());
      TBlob work_data(work_ptr, Shape1(svd_workspace_size), a.dev_mask(), a.dev_id());
      TBlob new_tol_data(new_tol_ptr, new_tol_shape, a.dev_mask(), a.dev_id());
      TBlob broadcast_data(broadcast_ptr, broadcast_shape, a.dev_mask(), a.dev_id());
      // Step3: SVD.
      SVDWrapper::op<xpu, DType>(a, s_data, u_data, ut_shape, v_data, vt_shape, work_data, ctx);
      // Step4: Calculate broadcast data.
      if (new_tol_data.dptr<DType>() != tol.dptr<DType>()) {
        Copy(new_tol_data.FlatTo1D<xpu, DType>(s), tol.FlatTo1D<xpu, DType>(s), s);
      }
      mxnet::op::BinaryBroadcastCompute<xpu, op::mshadow_op::gt>(attrs, ctx,
                                                                 {s_data, new_tol_data},
                                                                 {kWriteTo}, {broadcast_data});
      // Step5: Calculate rank.
      const int b_ndim  = broadcast_shape.ndim();
      const int data_size = broadcast_data.size(b_ndim - 1);
      const int batch_size = b_ndim == 1 ? 1 : broadcast_shape.ProdShape(0, b_ndim - 1);
      mxnet_op::Kernel<MatrixRankKernel<req_type>, xpu>::Launch(s, batch_size,
                                                                broadcast_data.dptr<DType>(),
                                                                rank.dptr<int64_t>(),
                                                                data_size, batch_size);
    });
  });
}

template<typename xpu>
void MatrixRankForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (kNullOp == req[0]) { return; }
  CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);

  const TBlob& a = inputs[0];
  const TBlob& tol = inputs[1];
  const TBlob& rank = outputs[0];
  MatrixRankForwardImpl<xpu>(a, tol, rank, attrs, ctx, req);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_MATRIX_RANK_INL_H_
