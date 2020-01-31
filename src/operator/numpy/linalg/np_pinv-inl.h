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
 * \file np_pinv-inl.h
 * \brief Placeholder for pinv
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_PINV_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_PINV_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include "../../operator_common.h"
#include "../../mshadow_op.h"
#include "../../tensor/elemwise_binary_op.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../tensor/la_op.h"
#include "../../tensor/la_op-inl.h"
#include "../../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct PinvParam : public dmlc::Parameter<PinvParam> {
  bool hermitian;
  DMLC_DECLARE_PARAMETER(PinvParam) {
    DMLC_DECLARE_FIELD(hermitian)
    .set_default(false)
    .describe("If True, A is assumed to be Hermitian (symmetric if real-valued).");
  }
};

struct PinvScalarRcondParam : public dmlc::Parameter<PinvScalarRcondParam> {
  double rcond;
  bool hermitian;
  DMLC_DECLARE_PARAMETER(PinvScalarRcondParam) {
    DMLC_DECLARE_FIELD(rcond)
    .set_default(1e-15)
    .describe("Cutoff for small singular values.");
    DMLC_DECLARE_FIELD(hermitian)
    .set_default(false)
    .describe("If True, A is assumed to be Hermitian (symmetric if real-valued).");
  }
};

template<typename xpu, typename DType>
int linalg_gesdd_workspace_query(const int m, const int n,
                                 const Tensor<xpu, 2, DType>& UT,
                                 const Tensor<xpu, 1, DType>& S,
                                 const Tensor<xpu, 2, DType>& V,
                                 Stream<xpu>* s = 0);

template<typename xpu, typename DType>
void linalg_gesdd(const int m, const int n,
                  const Tensor<xpu, 2, DType>& UT,
                  const Tensor<xpu, 1, DType>& S,
                  const Tensor<xpu, 2, DType>& V,
                  const Tensor<xpu, 1, DType>& work,
                  Stream<xpu> *s = 0);

template<typename xpu, typename DType>
void BatchSVDImpl(const int m, const int n,
                  const Tensor<xpu, 3, DType>& UT,
                  const Tensor<xpu, 2, DType>& S,
                  const Tensor<xpu, 3, DType>& V,
                  const Tensor<xpu, 1, DType>& work,
                  Stream<xpu> *s = 0);

#define LINALG_CPU_GESDD_WORKSPACE_QUERY(func, DType) \
template<> inline \
int linalg_gesdd_workspace_query<cpu, DType>(const int m, const int n, \
                                             const Tensor<cpu, 2, DType>& UT, \
                                             const Tensor<cpu, 1, DType>& S, \
                                             const Tensor<cpu, 2, DType>& V, \
                                             Stream<cpu> *s) { \
  DType work(0.0); \
  std::vector<int> iwork(8 * std::min(m, n), 0); \
  if (m > n) { \
    MXNET_LAPACK_##func(MXNET_LAPACK_COL_MAJOR, n, m, \
                        UT.dptr_, UT.stride_, S.dptr_, \
                        V.dptr_, V.stride_, \
                        UT.dptr_, UT.stride_, \
                        &work, -1, iwork.data()); \
  } else { \
    MXNET_LAPACK_##func(MXNET_LAPACK_COL_MAJOR, n, m, \
                        V.dptr_, V.stride_, S.dptr_, \
                        V.dptr_, V.stride_, \
                        UT.dptr_, UT.stride_, \
                        &work, -1, iwork.data()); \
  } \
  return static_cast<int>(work); \
}

#define LINALG_CPU_GESDD(func, DType) \
template<> inline \
void linalg_gesdd<cpu, DType>(const int m, \
                              const int n, \
                              const Tensor<cpu, 2, DType>& UT, \
                              const Tensor<cpu, 1, DType>& S, \
                              const Tensor<cpu, 2, DType>& V, \
                              const Tensor<cpu, 1, DType>& work, \
                              Stream<cpu> *s) { \
  std::vector<int> iwork(8 * std::min(m, n), 0); \
  int res(0); \
  if (m > n) { \
    res = MXNET_LAPACK_##func(MXNET_LAPACK_COL_MAJOR, n, m, \
                              UT.dptr_, UT.stride_, S.dptr_, \
                              V.dptr_, V.stride_, \
                              UT.dptr_, UT.stride_, \
                              work.dptr_, work.shape_.Size(), iwork.data()); \
  } else { \
    res = MXNET_LAPACK_##func(MXNET_LAPACK_COL_MAJOR, n, m, \
                              V.dptr_, V.stride_, S.dptr_, \
                              V.dptr_, V.stride_, \
                              UT.dptr_, UT.stride_, \
                              work.dptr_, work.shape_.Size(), iwork.data()); \
  } \
  CHECK_GE(res, 0) << #func << ": the " << -res \
    << "-th argument had an illegal value"; \
  CHECK_LE(res, 0) << #func << " did not converge, updating process failed."; \
}

LINALG_CPU_GESDD_WORKSPACE_QUERY(sgesdd, float)
LINALG_CPU_GESDD_WORKSPACE_QUERY(dgesdd, double)

LINALG_CPU_GESDD(sgesdd, float)
LINALG_CPU_GESDD(dgesdd, double)

#ifdef __CUDACC__

#define LINALG_GPU_GESDD_WORKSPACE_QUERY(DType) \
template<> inline \
int linalg_gesdd_workspace_query<gpu, DType>(const int m, const int n, \
                                             const Tensor<gpu, 2, DType>& U, \
                                             const Tensor<gpu, 1, DType>& S, \
                                             const Tensor<gpu, 2, DType>& VT, \
                                             Stream<gpu> *s) { \
  LOG(FATAL) << "Lapack gesdd workspace query routines is unsupported in gpu!"; \
  return 0; \
}

#define LINALG_GPU_GESDD(DType) \
template<> inline \
void linalg_gesdd<gpu, DType>(const int m, const int n, \
                              const Tensor<gpu, 2, DType>& U, \
                              const Tensor<gpu, 1, DType>& S, \
                              const Tensor<gpu, 2, DType>& VT, \
                              const Tensor<gpu, 1, DType>& work, \
                              Stream<gpu> *s) { \
  LOG(FATAL) << "Lapack gesdd routines is unsupported in gpu!"; \
}

LINALG_GPU_GESDD_WORKSPACE_QUERY(float)
LINALG_GPU_GESDD_WORKSPACE_QUERY(double)

LINALG_GPU_GESDD(float)
LINALG_GPU_GESDD(double)

#endif  // __CUDACC__

#define BATCH_SVD_IMPL_CPU(DType) \
template<> inline \
void BatchSVDImpl<cpu, DType>(const int m, const int n, \
                              const Tensor<cpu, 3, DType>& UT, \
                              const Tensor<cpu, 2, DType>& S, \
                              const Tensor<cpu, 3, DType>& V, \
                              const Tensor<cpu, 1, DType>& work, \
                              Stream<cpu> *s) { \
  for (index_t i = 0; i < S.size(0); ++i) { \
    linalg_gesdd(m, n, UT[i], S[i], V[i], work, s); \
  } \
}

BATCH_SVD_IMPL_CPU(float)
BATCH_SVD_IMPL_CPU(double)

#ifdef __CUDACC__

#define BATCH_SVD_IMPL_GPU(DType) \
template<> inline \
void BatchSVDImpl<gpu, DType>(const int m, const int n, \
                              const Tensor<gpu, 3, DType>& UT, \
                              const Tensor<gpu, 2, DType>& S, \
                              const Tensor<gpu, 3, DType>& V, \
                              const Tensor<gpu, 1, DType>& work, \
                              Stream<gpu> *s) { \
  for (index_t i = 0; i < S.size(0); ++i) { \
    linalg_gesvd(UT[i], S[i], V[i], work, s); \
  } \
}

BATCH_SVD_IMPL_GPU(float)
BATCH_SVD_IMPL_GPU(double)

#endif  // __CUDACC__

struct SingularValSmax {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *smax_ptr, const DType *s_ptr,
                                  const int length, const int lds) {
    const DType *s_iptr = s_ptr + i * lds;
    DType *smax_iptr = smax_ptr + i;
    *smax_iptr = s_iptr[0];
    for (int j = 1; j < length; ++j) {
      *smax_iptr = s_iptr[j] > *smax_iptr ? s_iptr[j] : *smax_iptr;
    }
  }
};

struct DiscardSmallSingularVal {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *s_ptr, const DType *large_ptr) {
    if (large_ptr[i]) {
      s_ptr[i] = DType(1) / s_ptr[i];
    } else {
      s_ptr[i] = DType(0);
    }
  }
};

struct DiscardSmallSingularValWithScalarRcond {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *s_ptr, const int length,
                                  const int lds, const double rcond) {
    DType *s_iptr = s_ptr + i * lds;
    DType smax_i = s_iptr[0];
    for (int j = 1; j < length; ++j) {
      smax_i = s_iptr[j] > smax_i ? s_iptr[j] : smax_i;
    }
    for (int j = 0; j < length; ++j) {
      s_iptr[j] = (s_iptr[j] > rcond * smax_i) ? (DType(1) / s_iptr[j]) : (DType(0));
    }
  }
};

inline void GetPinvShape(const mxnet::TShape& a_shape,
                         mxnet::TShape *ut_shape,
                         mxnet::TShape *s_shape,
                         mxnet::TShape *v_shape,
                         mxnet::TShape *u_shape = nullptr,
                         mxnet::TShape *vt_shape = nullptr) {
  const int a_ndim = a_shape.ndim();
  const int m = a_shape[a_ndim - 2];
  const int n = a_shape[a_ndim - 1];

  // Calculate S shape.
  std::vector<int> s_shape_vec(a_ndim - 1, -1);
  for (int i = 0; i < a_ndim - 2; ++i) {
    s_shape_vec[i] = a_shape[i];
  }
  s_shape_vec[a_ndim - 2] = std::min(m, n);
  *s_shape = mxnet::TShape(s_shape_vec.begin(), s_shape_vec.end());

  std::vector<int> temp_shape_vec(a_ndim, -1);
  for (int i = 0; i < a_ndim - 2; ++i) {
    temp_shape_vec[i] = a_shape[i];
  }
  temp_shape_vec[a_ndim - 2] = std::min(m, n);
  temp_shape_vec[a_ndim - 1] = std::min(m, n);
  if (m >= n) {
    // UT must have same shape as A.
    *ut_shape = a_shape;
    *v_shape = mxnet::TShape(temp_shape_vec.begin(), temp_shape_vec.end());
    if (u_shape && vt_shape) {
      *vt_shape = mxnet::TShape(temp_shape_vec.begin(), temp_shape_vec.end());
      *u_shape = a_shape;
      (*u_shape)[a_ndim - 2] = a_shape[a_ndim - 1];
      (*u_shape)[a_ndim - 1] = a_shape[a_ndim - 2];
    }
  } else {
    // V must have same shape as A.
    *v_shape = a_shape;
    *ut_shape = mxnet::TShape(temp_shape_vec.begin(), temp_shape_vec.end());
    if (u_shape && vt_shape) {
      *u_shape = mxnet::TShape(temp_shape_vec.begin(), temp_shape_vec.end());
      *vt_shape = a_shape;
      (*vt_shape)[a_ndim - 2] = a_shape[a_ndim - 1];
      (*vt_shape)[a_ndim - 1] = a_shape[a_ndim - 2];
    }
  }
}

inline void GetOrCheckCutoffAndLargeShape(const nnvm::NodeAttrs& attrs,
                                          const mxnet::TShape& a_shape,
                                          const mxnet::TShape& rcond_shape,
                                          mxnet::TShape *cutoff_shape = nullptr,
                                          mxnet::TShape *large_shape = nullptr) {
  if (!shape_is_known(a_shape)) { return ; }
  const int a_ndim = a_shape.ndim();
  const int rcond_ndim = rcond_shape.ndim();
  mxnet::TShape s_shape(a_ndim - 1, 1);
  mxnet::TShape smax_shape(a_ndim - 1, 1);
  mxnet::TShape new_rcond_shape(rcond_ndim + 1, 1);
  // Get new rcond shape.
  for (int i = 0; i < rcond_ndim; ++i) {
    new_rcond_shape[i] = rcond_shape[i];
  }
  // Get Smax shape.
  for (int i = 0; i < a_ndim - 2; ++i) {
    s_shape[i] = a_shape[i];
    smax_shape[i] = a_shape[i];
  }
  s_shape[s_shape.ndim() - 1] = std::min(a_shape[a_ndim - 2], a_shape[a_ndim - 1]);
  smax_shape[smax_shape.ndim() - 1] = 1;
  // Check cutoff = rcond[..., newaxis] * smax.
  mxnet::ShapeVector in_shape_vec1({ new_rcond_shape, smax_shape });
  mxnet::ShapeVector out_shape_vec1(1);
  mxnet::op::BinaryBroadcastShape(attrs, &in_shape_vec1, &out_shape_vec1);
  // Check large = s > cutoff.
  mxnet::ShapeVector in_shape_vec2({ s_shape, out_shape_vec1[0] });
  mxnet::ShapeVector out_shape_vec2(1);
  mxnet::op::BinaryBroadcastShape(attrs, &in_shape_vec2, &out_shape_vec2);
  // Check s = divide(1, s, where=large, out=s).
  if (s_shape != out_shape_vec2[0]) {
    LOG(FATAL) << "Error: non-broadcastable output operand with shape "
      << s_shape << " doesn't match the broadcast shape " << out_shape_vec2[0];
  }
  if (cutoff_shape) {
    *cutoff_shape = out_shape_vec1[0];
  }
  if (large_shape) {
    *large_shape = out_shape_vec2[0];
  }
}

template<typename xpu>
size_t SVDWorkspaceSize(const TBlob& a,
                        const TBlob& pinv_a,
                        const mxnet::TShape& u_shape,
                        const mxnet::TShape& s_shape,
                        const mxnet::TShape& v_shape,
                        const std::vector<OpReqType>& req,
                        const OpContext& ctx) {
  if (kNullOp == req[0]) { return 0U; }

  // Zero-size input, no need to launch kernel
  if (0U == a.Size()) { return 0U; }

  size_t work_space_size = 0;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_SGL_DBL_TYPE_SWITCH(pinv_a.type_flag_, OType, {
    const int a_ndim = a.shape_.ndim();
    const int u_ndim = u_shape.ndim();
    const int s_ndim = s_shape.ndim();
    const int v_ndim = v_shape.ndim();
    mxnet::TShape u_shape2 = Shape2(u_shape[u_ndim - 2], u_shape[u_ndim - 1]);
    mxnet::TShape s_shape1 = Shape1(s_shape[s_ndim - 1]);
    mxnet::TShape v_shape2 = Shape2(v_shape[v_ndim - 2], v_shape[v_ndim - 1]);
    if (xpu::kDevCPU) {
      std::vector<OType> u_vec(u_shape2.Size(), 0);
      std::vector<OType> s_vec(s_shape1.Size(), 0);
      std::vector<OType> v_vec(v_shape2.Size(), 0);
      // For workspace size in linalg_gesdd.
      work_space_size += linalg_gesdd_workspace_query(
          a.shape_[a_ndim - 2], a.shape_[a_ndim - 1],
          TBlob(u_vec.data(), u_shape2, a.dev_mask(), a.dev_id()).get<xpu, 2, OType>(s),
          TBlob(s_vec.data(), s_shape1, a.dev_mask(), a.dev_id()).get<xpu, 1, OType>(s),
          TBlob(v_vec.data(), v_shape2, a.dev_mask(), a.dev_id()).get<xpu, 2, OType>(s), s);
    } else {
      Storage::Handle u_handle =
        Storage::Get()->Alloc(sizeof(OType) * u_shape2.Size(), Context::GPU());
      Storage::Handle s_handle =
        Storage::Get()->Alloc(sizeof(OType) * s_shape1.Size(), Context::GPU());
      Storage::Handle v_handle =
        Storage::Get()->Alloc(sizeof(OType) * v_shape2.Size(), Context::GPU());
      TBlob u_data(static_cast<OType*>(u_handle.dptr), u_shape2, a.dev_mask(), a.dev_id());
      TBlob s_data(static_cast<OType*>(s_handle.dptr), s_shape1, a.dev_mask(), a.dev_id());
      TBlob v_data(static_cast<OType*>(v_handle.dptr), v_shape2, a.dev_mask(), a.dev_id());
      // For workspace size in linalg_gesvd.
      if (a.shape_[a_ndim - 2] >= a.shape_[a_ndim - 1]) {
        work_space_size += linalg_gesvd_workspace_query(v_data.get<xpu, 2, OType>(s),
                                                        s_data.get<xpu, 1, OType>(s),
                                                        u_data.get<xpu, 2, OType>(s), s);
      } else {
        work_space_size += linalg_gesvd_workspace_query(u_data.get<xpu, 2, OType>(s),
                                                        s_data.get<xpu, 1, OType>(s),
                                                        v_data.get<xpu, 2, OType>(s), s);
      }
      Storage::Get()->Free(u_handle);
      Storage::Get()->Free(s_handle);
      Storage::Get()->Free(v_handle);
    }
  });
  return work_space_size;
}

// Calculates workspace size of pinv op forward.
template<typename xpu>
size_t PinvForwardWorkspaceSize(const TBlob& a,
                                const TBlob& rcond,
                                const TBlob& pinv_a,
                                const nnvm::NodeAttrs& attrs,
                                const std::vector<OpReqType>& req,
                                const OpContext& ctx) {
  if (kNullOp == req[0]) { return 0U; }
  // Zero-size input, no need to launch kernel
  if (0U == a.Size()) { return 0U; }

  size_t work_space_size = 0;
  mxnet::TShape u_shape, s_shape, v_shape;
  GetPinvShape(a.shape_, &u_shape, &s_shape, &v_shape);

  MSHADOW_SGL_DBL_TYPE_SWITCH(pinv_a.type_flag_, OType, {
    mxnet::TShape smax_shape = s_shape;
    smax_shape[s_shape.ndim() - 1] = 1;
    mxnet::TShape cutoff_shape;
    mxnet::TShape large_shape;
    GetOrCheckCutoffAndLargeShape(attrs, a.shape_, rcond.shape_, &cutoff_shape, &large_shape);
    work_space_size +=  // For #gesdd_ or #gesvd work space size.
      SVDWorkspaceSize<xpu>(a, pinv_a, u_shape, s_shape, v_shape, req, ctx);
    work_space_size += rcond.shape_.Size();  // For rcond.
    work_space_size += 2 * u_shape.Size();   // For UT.
    work_space_size += s_shape.Size();       // For S.
    work_space_size += 2 * v_shape.Size();   // For V.
    work_space_size += smax_shape.Size();    // For Smax.
    work_space_size += cutoff_shape.Size();  // For Cutoff.
    work_space_size += large_shape.Size();   // For Large.
    return work_space_size * sizeof(OType);
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

inline mxnet::TShape GetTransAxis(const mxnet::TShape& in_shape) {
  const int in_ndim = in_shape.ndim();
  std::vector<int> trans_axis(in_ndim, -1);
  for (int i = 0; i < in_ndim - 2; ++i) { trans_axis[i] = i; }
  trans_axis[in_ndim - 2] = in_ndim - 1;
  trans_axis[in_ndim - 1] = in_ndim - 2;
  return mxnet::TShape(trans_axis.begin(), trans_axis.end());
}

template<typename xpu>
void PinvOpForwardImpl(const TBlob& a,
                       const TBlob& rcond,
                       const TBlob& pinv_a,
                       const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<OpReqType>& req,
                       const Tensor<xpu, 1, char>& workspace) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape rcond_shape = rcond.shape_;
  const int a_ndim = a_shape.ndim();
  const int rcond_ndim = rcond_shape.ndim();
  mxnet::TShape rcond_shape_newaxis(rcond_ndim + 1, 1);
  for (int i = 0; i < rcond_ndim; ++i) {
    rcond_shape_newaxis[i] = rcond_shape[i];
  }
  mxnet::TShape s_shape;
  mxnet::TShape u_shape;
  mxnet::TShape ut_shape;
  mxnet::TShape v_shape;
  mxnet::TShape vt_shape;
  GetPinvShape(a_shape, &u_shape, &s_shape, &v_shape, &ut_shape, &vt_shape);
  mxnet::TShape smax_shape = s_shape;
  smax_shape[s_shape.ndim() - 1] = 1;
  mxnet::TShape s_shape_newaxis(s_shape.ndim() + 1, 1);
  for (int i = 0; i < s_shape.ndim(); ++i) {
    s_shape_newaxis[i] = s_shape[i];
  }
  mxnet::TShape cutoff_shape;
  mxnet::TShape large_shape;
  GetOrCheckCutoffAndLargeShape(attrs, a_shape, rcond_shape, &cutoff_shape, &large_shape);

  MSHADOW_SGL_DBL_TYPE_SWITCH(a.type_flag_, AType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(pinv_a.type_flag_, DType, {
      const size_t workspace_size = (workspace.shape_.Size() + sizeof(DType) - 1) / sizeof(DType);
      const size_t lwork = workspace_size - rcond_shape_newaxis.Size()
        - 2 * u_shape.Size() - s_shape.Size() - 2 * v_shape.Size() - smax_shape.Size()
        - cutoff_shape.Size() - large_shape.Size();
      DType *work_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      DType *rcond_ptr = work_ptr + lwork;
      DType *ut_ptr = rcond_ptr + rcond_shape_newaxis.Size();
      DType *u_ptr = ut_ptr + ut_shape.Size();
      DType *s_ptr = u_ptr + u_shape.Size();
      DType *v_ptr = s_ptr + s_shape.Size();
      DType *vt_ptr = v_ptr + v_shape.Size();
      DType *smax_ptr = vt_ptr + vt_shape.Size();
      DType *cutoff_ptr = smax_ptr + smax_shape.Size();
      DType *large_ptr = cutoff_ptr + cutoff_shape.Size();
      // Step1: Calculate SVD.
      TBlob work_data(work_ptr, Shape1(lwork), a.dev_mask(), a.dev_id());
      TBlob u_data(u_ptr, u_shape, a.dev_mask(), a.dev_id());
      TBlob ut_data(ut_ptr, ut_shape, a.dev_mask(), a.dev_id());
      TBlob v_data(v_ptr, v_shape, a.dev_mask(), a.dev_id());
      TBlob vt_data(vt_ptr, vt_shape, a.dev_mask(), a.dev_id());
      TBlob s_data(s_ptr, s_shape, a.dev_mask(), a.dev_id());
      // Noet: Only a_shape[a_ndim - 2] > a_shape[a_ndim - 1], need transpose operation.
      if (a_shape[a_ndim - 2] > a_shape[a_ndim - 1]) {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, a.Size(), u_ptr, a.dptr<AType>());
        mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, u_data, ut_data,  // u_data: src, ut_data: res
                                      GetTransAxis(u_data.shape_));
        BatchSVDImpl(a_shape[a_ndim - 1], a_shape[a_ndim - 2],
                     vt_data.FlatToKD<xpu, 3, DType>(s),
                     s_data.FlatToKD<xpu, 2, DType>(s),
                     ut_data.FlatToKD<xpu, 3, DType>(s),
                     work_data.FlatToKD<xpu, 1, DType>(s), s);
      } else {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, a.Size(), v_ptr, a.dptr<AType>());
        BatchSVDImpl(a_shape[a_ndim - 2], a_shape[a_ndim - 1],
                     u_data.FlatToKD<xpu, 3, DType>(s),
                     s_data.FlatToKD<xpu, 2, DType>(s),
                     v_data.FlatToKD<xpu, 3, DType>(s),
                     work_data.FlatToKD<xpu, 1, DType>(s), s);
      }
      TBlob smax_data(smax_ptr, smax_shape, a.dev_mask(), a.dev_id());
      TBlob cutoff_data(cutoff_ptr, cutoff_shape, a.dev_mask(), a.dev_id());
      TBlob large_data(large_ptr, large_shape, a.dev_mask(), a.dev_id());
      TBlob rcond_data(rcond_ptr, rcond_shape_newaxis, a.dev_mask(), a.dev_id());
      Tensor<xpu, 2, DType> S = s_data.FlatToKD<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> Smax = smax_data.FlatToKD<xpu, 2, DType>(s);
      mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
        s, rcond_shape_newaxis.Size(), rcond_ptr, rcond.dptr<AType>());
      // Step2: Calculate Smax.
      mxnet_op::Kernel<SingularValSmax, xpu>::Launch(
        s, S.size(0), Smax.dptr_, S.dptr_, S.size(1), S.stride_);
      // Step3: Calculate Cutoff.
      std::vector<OpReqType> temp_req({kWriteTo});
      mxnet::op::BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx,
                                                                  {rcond_data, smax_data},
                                                                  temp_req, {cutoff_data});
      // Step4: Calculte Large.
      mxnet::op::BinaryBroadcastCompute<xpu, op::mshadow_op::gt>(attrs, ctx,
                                                                 {s_data, cutoff_data},
                                                                 temp_req, {large_data});
      // Step5: Discard small singular values.
      mxnet_op::Kernel<DiscardSmallSingularVal, xpu>::Launch(
        s, s_data.Size(), s_data.dptr<DType>(), large_data.dptr<DType>());
      // Step6: Calculte matmul(transpose(v), multiply(s[..., newaxis], transpose(u))).
      // Note: No need transpose when a_shape[a_ndim - 2] >= a_shape[a_ndim - 1]
      if (a_shape[a_ndim - 2] <= a_shape[a_ndim - 1]) {
        mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, u_data, ut_data,  // u_data: src, ut_data: res
                                      GetTransAxis(u_data.shape_));
        mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, v_data, vt_data,  // v_data: src, vt_data: res
                                      GetTransAxis(v_data.shape_));
      }
      s_data = s_data.reshape(s_shape_newaxis);
      u_data = ut_data.reshape(ut_shape);
      mxnet::op::BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, {s_data, ut_data},
                                                                  temp_req, {u_data});
      gemm2::op(vt_data.FlatToKD<xpu, 3, DType>(s),
                u_data.FlatToKD<xpu, 3, DType>(s),
                pinv_a.FlatToKD<xpu, 3, DType>(s),
                DType(1), false, false, s);
    });
  });
}

template<typename xpu>
void PinvOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& a = inputs[0];
  const TBlob& rcond = inputs[1];
  const TBlob& pinv_a = outputs[0];
  const mxnet::TShape a_shape = a.shape_;

  if (kNullOp == req[0]) { return; }

  // Zero-size output, no need to launch kernel
  if (0U == a.Size()) { return; }

  size_t workspace_size = PinvForwardWorkspaceSize<xpu>(a, rcond, pinv_a, attrs, req, ctx);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  PinvOpForwardImpl<xpu>(a, rcond, pinv_a, attrs, ctx, req, workspace);
}

// Calculates workspace size of pinv scalar rcond op forward.
template<typename xpu>
size_t PinvScalarRcondForwardWorkspaceSize(const TBlob& a,
                                           const TBlob& pinv_a,
                                           const nnvm::NodeAttrs& attrs,
                                           const std::vector<OpReqType>& req,
                                           const OpContext& ctx) {
  if (kNullOp == req[0]) { return 0U; }
  // Zero-size input, no need to launch kernel
  if (0U == a.Size()) { return 0U; }

  size_t work_space_size = 0;
  mxnet::TShape u_shape, s_shape, v_shape;
  GetPinvShape(a.shape_, &u_shape, &s_shape, &v_shape);

  MSHADOW_SGL_DBL_TYPE_SWITCH(pinv_a.type_flag_, OType, {
    mxnet::TShape smax_shape = s_shape;
    smax_shape[s_shape.ndim() - 1] = 1;
    work_space_size +=  // For #gesdd_ or #gesvd work space size.
      SVDWorkspaceSize<xpu>(a, pinv_a, u_shape, s_shape, v_shape, req, ctx);
    work_space_size += 2 * u_shape.Size();  // For UT.
    work_space_size += s_shape.Size();      // For S.
    work_space_size += 2 * v_shape.Size();  // For V.
    return work_space_size * sizeof(OType);
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

template<typename xpu>
void PinvScalarRcondOpForwardImpl(const TBlob& a,
                                  const TBlob& pinv_a,
                                  const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<OpReqType>& req,
                                  const Tensor<xpu, 1, char>& workspace) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape a_shape = a.shape_;
  const int a_ndim = a_shape.ndim();

  mxnet::TShape s_shape;
  mxnet::TShape u_shape;
  mxnet::TShape ut_shape;
  mxnet::TShape v_shape;
  mxnet::TShape vt_shape;
  GetPinvShape(a_shape, &u_shape, &s_shape, &v_shape, &ut_shape, &vt_shape);
  mxnet::TShape s_shape_newaxis(s_shape.ndim() + 1, 1);
  for (int i = 0; i < s_shape.ndim(); ++i) {
    s_shape_newaxis[i] = s_shape[i];
  }
  MSHADOW_SGL_DBL_TYPE_SWITCH(a.type_flag_, AType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(pinv_a.type_flag_, DType, {
      const double rcond = nnvm::get<PinvScalarRcondParam>(attrs.parsed).rcond;
      const size_t workspace_size = (workspace.shape_.Size() + sizeof(DType) - 1) / sizeof(DType);
      const size_t lwork = workspace_size - 2 * u_shape.Size() - s_shape.Size()
        - 2 * v_shape.Size();
      DType *work_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      DType *u_ptr = work_ptr + lwork;
      DType *ut_ptr = u_ptr + u_shape.Size();
      DType *s_ptr = ut_ptr + ut_shape.Size();
      DType *v_ptr = s_ptr + s_shape.Size();
      DType *vt_ptr = v_ptr + v_shape.Size();
      // Step1: Calculate SVD.
      TBlob work_data(work_ptr, Shape1(lwork), a.dev_mask(), a.dev_id());
      TBlob u_data(u_ptr, u_shape, a.dev_mask(), a.dev_id());
      TBlob ut_data(ut_ptr, ut_shape, a.dev_mask(), a.dev_id());
      TBlob v_data(v_ptr, v_shape, a.dev_mask(), a.dev_id());
      TBlob vt_data(vt_ptr, vt_shape, a.dev_mask(), a.dev_id());
      TBlob s_data(s_ptr, s_shape, a.dev_mask(), a.dev_id());
      Tensor<xpu, 2, DType> S = s_data.FlatToKD<xpu, 2, DType>(s);
      // Noet: Only a_shape[a_ndim - 2] > a_shape[a_ndim - 1], need transpose operation.
      if (a_shape[a_ndim - 2] > a_shape[a_ndim - 1]) {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, a.Size(), u_ptr, a.dptr<AType>());
        mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, u_data, ut_data,  // u_data: src, ut_data: res
                                      GetTransAxis(u_data.shape_));
        BatchSVDImpl(a_shape[a_ndim - 1], a_shape[a_ndim - 2],
                     vt_data.FlatToKD<xpu, 3, DType>(s),
                     s_data.FlatToKD<xpu, 2, DType>(s),
                     ut_data.FlatToKD<xpu, 3, DType>(s),
                     work_data.FlatToKD<xpu, 1, DType>(s), s);
      } else {
        mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
          s, a.Size(), v_ptr, a.dptr<AType>());
        BatchSVDImpl(a_shape[a_ndim - 2], a_shape[a_ndim - 1],
                     u_data.FlatToKD<xpu, 3, DType>(s),
                     s_data.FlatToKD<xpu, 2, DType>(s),
                     v_data.FlatToKD<xpu, 3, DType>(s),
                     work_data.FlatToKD<xpu, 1, DType>(s), s);
      }
      // Step2: Discard small singular values.
      mxnet_op::Kernel<DiscardSmallSingularValWithScalarRcond, xpu>::Launch(
        s, S.size(0), S.dptr_, S.size(1), S.stride_, rcond);
      // Step3: Calculte matmul(transpose(v), multiply(s[..., newaxis], transpose(u))).
      // Note: No need transpose when a_shape[a_ndim - 2] >= a_shape[a_ndim - 1]
      if (a_shape[a_ndim - 2] <= a_shape[a_ndim - 1]) {
        mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, u_data, ut_data,  // u_data: src, ut_data: res
                                      GetTransAxis(u_data.shape_));
        mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, v_data, vt_data,  // v_data: src, vt_data: res
                                      GetTransAxis(v_data.shape_));
      }
      s_data = s_data.reshape(s_shape_newaxis);
      u_data = ut_data.reshape(ut_shape);
      mxnet::op::BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, {s_data, ut_data},
                                                                  {kWriteTo}, {u_data});
      gemm2::op(vt_data.FlatToKD<xpu, 3, DType>(s),
                u_data.FlatToKD<xpu, 3, DType>(s),
                pinv_a.FlatToKD<xpu, 3, DType>(s),
                DType(1), false, false, s);
    });
  });
}

template<typename xpu>
void PinvScalarRcondOpForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& a = inputs[0];
  const TBlob& pinv_a = outputs[0];

  if (kNullOp == req[0]) { return; }
  // Zero-size output, no need to launch kernel
  if (0U == a.Size()) { return; }

  // Calculate workspace size.
  size_t workspace_size = PinvScalarRcondForwardWorkspaceSize<xpu>(a, pinv_a, attrs, req, ctx);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  PinvScalarRcondOpForwardImpl<xpu>(a, pinv_a, attrs, ctx, req, workspace);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_PINV_INL_H_
