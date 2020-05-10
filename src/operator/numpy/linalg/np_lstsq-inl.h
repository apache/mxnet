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
 * \file np_lstsq-inl.h
 * \brief Placeholder for lstlq
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_LSTSQ_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_LSTSQ_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../../operator_common.h"
#include "../../mshadow_op.h"
#include "../../c_lapack_api.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct LstsqParam : public dmlc::Parameter<LstsqParam> {
  double rcond;
  float finfoEps32;
  double finfoEps64;
  bool new_default;
  DMLC_DECLARE_PARAMETER(LstsqParam) {
    DMLC_DECLARE_FIELD(rcond)
    .set_default(-1)
    .describe("Cut-off ratio for small singular values");
    DMLC_DECLARE_FIELD(finfoEps32)
    .set_default(0)
    .describe("Machine limits for float32 type");
    DMLC_DECLARE_FIELD(finfoEps64)
    .set_default(0)
    .describe("Machine limits for float64 type");
    DMLC_DECLARE_FIELD(new_default)
    .set_default(false)
    .describe("Specifies whether rcond is default which is machine precision");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream rcond_s, finfoEps32_s, finfoEps64_s, new_default_s;
    rcond_s << rcond;
    finfoEps32_s << finfoEps32;
    finfoEps64_s << finfoEps64;
    new_default_s << new_default;
    (*dict)["rcond"] = rcond_s.str();
    (*dict)["finfoEps32"] = finfoEps32_s.str();
    (*dict)["finfoEps64"] = finfoEps64_s.str();
    (*dict)["new_default"] = new_default_s.str();
  }
};

template<typename xpu, typename DType>
inline void linalg_gelsd_workspace_query(const int nrow,
                                         const int ncol,
                                         const int nrhs,
                                         int *lwork,
                                         int *liwork,
                                         const Tensor<xpu, 2, DType>& A,
                                         const Tensor<xpu, 2, DType>& B,
                                         const Tensor<xpu, 1, DType>& S);

template<typename xpu, typename DType>
inline void linalg_gelsd(const int nrow,
                         const int ncol,
                         const int nrhs,
                         const DType rcond,
                         int *rank,
                         const Tensor<xpu, 2, DType>& A,
                         const Tensor<xpu, 2, DType>& B,
                         const Tensor<xpu, 1, DType>& SingularValues,
                         const Tensor<xpu, 1, DType>& Work,
                         const Tensor<xpu, 1, int>& Iwork);

struct LstsqTypeTransposeHelper {
  template<typename InDType, typename OutDType>
  MSHADOW_XINLINE static void Map(int i, const InDType *in_ptr, OutDType *out_ptr,
                                  const int nrow, const int ncol, const int ld) {
    if (ld >= nrow && i < nrow * ncol) {
      out_ptr[i / ncol + (i % ncol) * ld] = static_cast<OutDType>(in_ptr[i]);
    }
  }
};

template<int req>
struct ValuesAssignHelper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, DType *out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType& in_data, DType *out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data);
  }
};

template<int req>
struct SolutionAssignHelper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *trans_b_ptr, DType *x_ptr,
                                  const int nrow, const int ncol, const int nrhs, const int ldb) {
    if (i < ncol * nrhs && ldb >= nrow && ldb >= ncol) {
      KERNEL_ASSIGN(x_ptr[i], req, trans_b_ptr[i / nrhs  + (i % nrhs) * ldb]);
    }
  }
};

template<int req>
struct ResidualsAssignHelper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *trans_b_ptr, DType *residuals_ptr,
                                  const int nrow, const int ncol, const int nrhs, const int ldb) {
    if (i < nrhs) {
      DType residuals_values = 0;
      for (int j = ncol; j < nrow; ++j) {
        residuals_values += trans_b_ptr[j + i * ldb] * trans_b_ptr[j + i * ldb];
      }
      KERNEL_ASSIGN(residuals_ptr[i], req, residuals_values);
    }
  }
};

#define LINALG_CPU_GELSD_WORKSPACE_QUERY(func, DType) \
template<> inline void \
linalg_gelsd_workspace_query<cpu, DType>(const int nrow, \
                                         const int ncol, \
                                         const int nrhs, \
                                         int *lwork, \
                                         int *liwork, \
                                         const Tensor<cpu, 2, DType>& A, \
                                         const Tensor<cpu, 2, DType>& B, \
                                         const Tensor<cpu, 1, DType>& S) { \
  CHECK(A.size(0) == ncol && A.size(1) == A.stride_) \
    << "ncol or lda dismatch A shape and lda should >= max(1, nrow)."; \
  CHECK(B.size(0) == nrhs && B.size(1) == B.stride_) \
    << "nrhs or ldb dismatch B shape and ldb should >= max(1, max(nrow, ncol))"; \
  DType temp_work = -1, rcond = -1; \
  int temp_iwork = -1, rank = -1; \
  int info = MXNET_LAPACK_##func(MXNET_LAPACK_COL_MAJOR, nrow, ncol, nrhs, \
                                 A.dptr_, static_cast<int>(A.stride_), \
                                 B.dptr_, static_cast<int>(B.stride_), \
                                 B.dptr_, rcond, &rank, \
                                 &temp_work, -1, &temp_iwork); \
  CHECK_GE(info, 0) << "MXNET_LAPACK_" << #func << ": " \
    << "the " << -info << "-th argument had an illegal value"; \
  *lwork = static_cast<int>(temp_work); \
  *liwork = temp_iwork; \
  return; \
}

#define LINALG_CPU_GELSD(func, DType) \
template<> inline void \
linalg_gelsd<cpu, DType>(const int nrow, \
                         const int ncol, \
                         const int nrhs, \
                         const DType rcond, \
                         int *rank, \
                         const Tensor<cpu, 2, DType>& A, \
                         const Tensor<cpu, 2, DType>& B, \
                         const Tensor<cpu, 1, DType>& SingularValues, \
                         const Tensor<cpu, 1, DType>& Work, \
                         const Tensor<cpu, 1, int>& Iwork) { \
  CHECK(A.size(0) == ncol && A.size(1) == A.stride_) \
    << "ncol or lda dismatch A shape and lda should >= max(1, nrow)."; \
  CHECK(B.size(0) == nrhs && B.size(1) == B.stride_) \
    << "nrhs or ldb dismatch B shape and ldb should >= max(1, max(nrow, ncol))"; \
  CHECK(SingularValues.MSize() >= std::min(nrow, ncol)) \
    << "SingularValues is too small"; \
  const int lwork = Work.MSize(); \
  int info = MXNET_LAPACK_##func(MXNET_LAPACK_COL_MAJOR, nrow, ncol, nrhs, \
                                 A.dptr_, A.stride_, B.dptr_, B.stride_, \
                                 SingularValues.dptr_, rcond, rank, \
                                 Work.dptr_, lwork, Iwork.dptr_); \
  CHECK_GE(info, 0) << "MXNET_LAPACK_" << #func << ": " \
    << "the " << -info << "-th argument had an illegal value"; \
  CHECK_LE(info, 0) << "MXNET_LAPACK_" << #func << ": " \
    << "the algorithm for computing the SVD failed to converge; " \
    << info << " off-diagonal elements of an intermediate bidiagonal " \
    << "form did not converge to zero."; \
  return; \
}

LINALG_CPU_GELSD_WORKSPACE_QUERY(sgelsd, float)
LINALG_CPU_GELSD_WORKSPACE_QUERY(dgelsd, double)

LINALG_CPU_GELSD(sgelsd, float)
LINALG_CPU_GELSD(dgelsd, double)

#ifdef __CUDACC__

#define LINALG_GPU_GELSD_WORKSPACE_QUERY(DType) \
template<> inline void \
linalg_gelsd_workspace_query<gpu, DType>(const int nrow, \
                                         const int ncol, \
                                         const int nrhs, \
                                         int *lwork, \
                                         int *liwork, \
                                         const Tensor<gpu, 2, DType>& A, \
                                         const Tensor<gpu, 2, DType>& B, \
                                         const Tensor<gpu, 1, DType>& S) { \
  LOG(FATAL) << "linalg gesld workspace query is unsupported in gpu!"; \
  return; \
}

#define LINALG_GPU_GELSD(DType) \
template<> inline void \
linalg_gelsd<gpu, DType>(const int nrow, \
                         const int ncol, \
                         const int nrhs, \
                         const DType rcond, \
                         int *rank, \
                         const Tensor<gpu, 2, DType>& A, \
                         const Tensor<gpu, 2, DType>& B, \
                         const Tensor<gpu, 1, DType>& SingularValues, \
                         const Tensor<gpu, 1, DType>& Work, \
                         const Tensor<gpu, 1, int>& Iwork) { \
  LOG(FATAL) << "linalg gesld is unsupported in gpu!"; \
  return; \
}

LINALG_GPU_GELSD_WORKSPACE_QUERY(float)
LINALG_GPU_GELSD_WORKSPACE_QUERY(double)

LINALG_GPU_GELSD(float)
LINALG_GPU_GELSD(double)

#endif  // __CUDACC__

inline bool GetOutputShapes(const mxnet::TShape& a_shape,
                            const mxnet::TShape& b_shape,
                            mxnet::ShapeVector *out_attrs) {
  if (!ndim_is_known(a_shape) || !ndim_is_known(b_shape)) {
    return false;
  }
  const int a_ndim = a_shape.ndim();
  const int b_ndim = b_shape.ndim();
  const int a_nrow = a_shape[0];
  const int a_ncol = a_shape[1];
  const int b_nrow = b_shape[0];
  const int b_nrhs = b_ndim == 2 ? b_shape[1] : 1;
  CHECK_EQ(a_ndim, 2) << a_ndim
    << "-dimensional array given. Array must be two-dimensional";
  CHECK(b_ndim == 1 || b_ndim == 2) << b_ndim
    << "-dimensional array given. Array must be one-dimensional or two-dimensional";
  CHECK_EQ(a_nrow, b_nrow)
    << "Incompatible dimensions of inputs";
  // x_shape
  if (b_ndim == 2) {
    std::vector<int> x_shape_vec({a_ncol, b_nrhs});
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(x_shape_vec.begin(), x_shape_vec.end()));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(1, a_ncol));
  }
  // temp_residuals_shape
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, mxnet::TShape(1, static_cast<dim_t>(std::max(1, b_nrhs))));
  // rank_shape
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, mxnet::TShape(0, 0));
  // s_shape
  if (a_nrow == 0 || a_ncol == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 3, mxnet::TShape(1, 0));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 3, mxnet::TShape(1, std::max(1, std::min(a_nrow, a_ncol))));
  }
  return shape_is_known(*out_attrs);
}

template<typename xpu, typename DType>
size_t LstsqWorkspaceSize(const TBlob& a,
                          const TBlob& b,
                          const mxnet::TShape& x_shape,
                          int *lwork,
                          int *liwork,
                          const OpContext& ctx) {
  const int a_ndim = a.ndim();
  const int b_ndim = b.ndim();
  CHECK(a_ndim == 2 && (b_ndim == 2 || b_ndim == 1)) << "Wrong array ndim";
  CHECK_EQ(a.shape_[0], b.shape_[0]) << "Inputs shape dismatch";

  size_t workspace_size = 0;
  Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
  MSHADOW_SGL_DBL_TYPE_SWITCH(a.type_flag_, AType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(b.type_flag_, BType, {
      const mxnet::TShape& a_shape = a.shape_;
      const mxnet::TShape& b_shape = b.shape_;
      if (xpu::kDevCPU) {
        int nrow = a_shape[0];
        int ncol = a_shape[1];
        int nrhs = b_ndim == 2 ? b_shape[1] : 1;
        int lda = std::max(1, nrow);
        int ldb = std::max(1, std::max(nrow, ncol));
        // Lapack routine can't handle lda = 0 and ldb = 0.
        // If nrow == 0, leading dimension of A_trans and B_trans will be 0.
        if (nrow == 0) { return 0U; }
        if (ncol == 0 && nrhs == 0) { return 0U; }
        // If ncol != 0, need to invoke lapack routine.
        // Lapack routine can't handle n_rhs = 0, so allocate the array one larger in that axis.
        int temp_nrhs = nrhs == 0 ? 1 : nrhs;
        std::vector<DType> temp_a_vec(ncol * lda, 0);
        std::vector<DType> temp_b_vec(temp_nrhs * ldb, 0);
        std::vector<DType> temp_s_vec(std::max(1, std::min(nrow, ncol)), 0);
        mshadow::Tensor<cpu, 2, DType> A(temp_a_vec.data(), Shape2(ncol, lda), lda, s_cpu);
        mshadow::Tensor<cpu, 2, DType> B(temp_b_vec.data(), Shape2(temp_nrhs, ldb), ldb, s_cpu);
        mshadow::Tensor<cpu, 1, DType> S(temp_s_vec.data(), Shape1(temp_s_vec.size()), s_cpu);
        // Invoke lapack workspace query.
        linalg_gelsd_workspace_query<cpu, DType>(nrow, ncol, temp_nrhs,
                                                 lwork, liwork, A, B, S);
        // For A size because on lapack routine exit, A will be overwritten.
        workspace_size += ncol * lda * sizeof(DType);
        // For B size because on lapack routine exit, B will be overwritten by solution result.
        workspace_size += temp_nrhs * ldb * sizeof(DType);
        // For singular values size.
        workspace_size += std::max(1, std::min(nrow, ncol)) * sizeof(DType);
        // For workspace size in linalg_gesld.
        workspace_size += (*lwork) * sizeof(DType) + (*liwork) * sizeof(int);
      }
    });
  });
  return workspace_size;
}

template<typename xpu>
void LstsqOpForwardImpl(const TBlob& a,
                        const TBlob& b,
                        const TBlob& x,
                        const TBlob& temp_residuals,
                        const TBlob& rank,
                        const TBlob& singularValues,
                        bool *empty_residuals,
                        const int& lwork,
                        const int& liwork,
                        std::vector<char> *workspace,
                        const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<OpReqType>& req) {
  // Get param.
  const LstsqParam& param = nnvm::get<LstsqParam>(attrs.parsed);
  double rcond = param.rcond;
  bool new_default = param.new_default;
  double finfoEps = a.type_flag_ == mshadow::kFloat32 ? param.finfoEps32 : param.finfoEps64;
  if (new_default) {
    rcond = finfoEps * std::max(a.shape_[0], a.shape_[1]);
  }
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;
  MSHADOW_SGL_DBL_TYPE_SWITCH(x.type_flag_, DType, {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    const int nrow = a_shape[0];
    const int ncol = a_shape[1];
    const int nrhs = b.ndim() == 2 ? b_shape[1] : 1;
    const int lda = std::max(1, nrow);
    const int ldb = std::max(1, std::max(nrow, ncol));
    const int snum = std::max(1, std::min(nrow, ncol));
    if (nrow == 0) {
      // Assign 0 for all values in x.
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<ValuesAssignHelper<req_type>, xpu>::Launch(
          s, x.Size(), static_cast<DType>(0), x.dptr<DType>());
      });
      // Assign values for rank.
      ASSIGN_DISPATCH(*rank.dptr<int>(), kWriteTo, 0);
      // Assign values for empty_residuals.
      *empty_residuals = true;
      return;
    }
    if (ncol == 0 && nrhs == 0) {
      // Assign values for rank.
      ASSIGN_DISPATCH(*rank.dptr<int>(), kWriteTo, 0);
      // Assign values for empty_residuals.
      *empty_residuals = true;
      return;
    }
    int temp_nrhs = nrhs == 0 ? 1 : nrhs;
    mxnet::TShape trans_a_shape(mxnet::Tuple<dim_t>({ ncol, lda }));
    mxnet::TShape trans_b_shape(mxnet::Tuple<dim_t>({ temp_nrhs, ldb }));
    // Allocate data memory.
    DType *a_ptr = reinterpret_cast<DType*>(workspace->data());
    DType *b_ptr = a_ptr + trans_a_shape.Size();
    DType *s_ptr = b_ptr + trans_b_shape.Size();
    DType *work_ptr = s_ptr + snum;
    int *iwork_ptr = reinterpret_cast<int*>(work_ptr + lwork);
    TBlob trans_a(a_ptr, trans_a_shape, a.dev_mask(), a.dev_id());
    TBlob trans_b(b_ptr, trans_b_shape, b.dev_mask(), b.dev_id());
    TBlob singular_values(s_ptr, Shape1(snum), singularValues.dev_mask(), singularValues.dev_id());
    TBlob work(work_ptr, Shape1(lwork), x.dev_mask(), x.dev_id());
    TBlob iwork(iwork_ptr, Shape1(liwork), x.dev_mask(), x.dev_id());
    // Transpose a to trans_a.
    MSHADOW_SGL_DBL_TYPE_SWITCH(a.type_flag_, AType, {
      mxnet_op::Kernel<LstsqTypeTransposeHelper, xpu>::Launch(s, a.Size(), a.dptr<AType>(),
                                                              trans_a.dptr<DType>(),
                                                              nrow, ncol, lda);
    });
    // If nrhs == 0, assign 0 to trans_b directly.
    // If nrhs != 0, assign 0 to initialize trans_b because trans_b.Size > b.Size when nrow < ncol.
    mxnet_op::Kernel<ValuesAssignHelper<kWriteTo>, xpu>::Launch(
      s, trans_b.Size(), static_cast<DType>(0), trans_b.dptr<DType>());
    if (nrhs != 0) {
      // Transpose b to trans_b.
      MSHADOW_SGL_DBL_TYPE_SWITCH(b.type_flag_, BType, {
        mxnet_op::Kernel<LstsqTypeTransposeHelper, xpu>::Launch(s, b.Size(), b.dptr<BType>(),
                                                                trans_b.dptr<DType>(),
                                                                nrow, nrhs, ldb);
      });
    }
    // Invoke lapack routines.
    linalg_gelsd<xpu, DType>(nrow, ncol, temp_nrhs,
                             rcond, rank.dptr<int>(),
                             trans_a.get<xpu, 2, DType>(s),
                             trans_b.get<xpu, 2, DType>(s),
                             singular_values.get<xpu, 1, DType>(s),
                             work.get<xpu, 1, DType>(s),
                             iwork.get<xpu, 1, int>(s));
    if (ncol != 0 && nrhs == 0) {
      // Assign values for singularValues.
      MXNET_ASSIGN_REQ_SWITCH(req[3], req_type, {
        mxnet_op::Kernel<ValuesAssignHelper<req_type>, xpu>::Launch(
          s, singularValues.Size(), singular_values.dptr<DType>(), singularValues.dptr<DType>());
      });
      // Assign values for empty_residuals.
      *empty_residuals = true;
      return;
    } else {
      // Assign values for x.
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<SolutionAssignHelper<req_type>, xpu>::Launch(
          s, x.Size(), trans_b.dptr<DType>(), x.dptr<DType>(), nrow, ncol, nrhs, ldb);
      });
      // Assign values for residuals and residualsEmpty.
      if (*(rank.dptr<int>()) < ncol || nrow <= ncol) {
        *empty_residuals = true;
      } else {
        *empty_residuals = false;
        MXNET_ASSIGN_REQ_SWITCH(req[1], req_type, {
          mxnet_op::Kernel<ResidualsAssignHelper<req_type>, xpu>::Launch(
            s, temp_residuals.Size(), trans_b.dptr<DType>(),
            temp_residuals.dptr<DType>(), nrow, ncol, nrhs, ldb);
        });
      }
      // Assign values for singularValues.
      MXNET_ASSIGN_REQ_SWITCH(req[3], req_type, {
        mxnet_op::Kernel<ValuesAssignHelper<req_type>, xpu>::Launch(
          s, singularValues.Size(), singular_values.dptr<DType>(), singularValues.dptr<DType>());
      });
    }
  });
}

template<typename DType>
inline void GpuCallbackCpuImpl(const TBlob& a,
                               const TBlob& b,
                               const TBlob& x,
                               const TBlob& rank,
                               const TBlob& singularValues,
                               const NDArray& residuals_ndarray,
                               const mxnet::TShape& temp_residuals_shape,
                               const int& lwork,
                               const int& liwork,
                               std::vector<char> *workspace,
                               const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<OpReqType>& req) {
#if MXNET_USE_CUDA
  MSHADOW_SGL_DBL_TYPE_SWITCH(a.type_flag_, AType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(b.type_flag_, BType, {
      std::vector<AType> a_vec(a.Size(), 0);
      std::vector<BType> b_vec(b.Size(), 0);
      std::vector<DType> x_vec(x.Size(), 0);
      std::vector<DType> temp_residuals_vec(temp_residuals_shape.Size(), 0);
      std::vector<int> rank_vec(rank.Size(), 0);
      std::vector<DType> singularValues_vec(singularValues.Size(), 0);
      mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
      cudaStream_t stream = Stream<gpu>::GetStream(s);
      // Copy inputs from gpu to cpu.
      CUDA_CALL(cudaMemcpyAsync(a_vec.data(), a.dptr<AType>(), sizeof(AType) * a.Size(),
                                cudaMemcpyDeviceToHost, stream));
      CUDA_CALL(cudaMemcpyAsync(b_vec.data(), b.dptr<BType>(), sizeof(BType) * b.Size(),
                                cudaMemcpyDeviceToHost, stream));
      CUDA_CALL(cudaStreamSynchronize(stream));
      mxnet::TBlob a_data(a_vec.data(), a.shape_, cpu::kDevMask, -1);
      mxnet::TBlob b_data(b_vec.data(), b.shape_, cpu::kDevMask, -1);
      mxnet::TBlob x_data(x_vec.data(), x.shape_, cpu::kDevMask, -1);
      mxnet::TBlob rank_data(rank_vec.data(), rank.shape_, cpu::kDevMask, -1);
      mxnet::TBlob temp_residuals_data(temp_residuals_vec.data(),
                                       temp_residuals_shape, cpu::kDevMask, -1);
      mxnet::TBlob singularValues_data(singularValues_vec.data(),
                                       singularValues.shape_, cpu::kDevMask, -1);
      // Op forward implement on cpu.
      bool empty_residuals = false;
      LstsqOpForwardImpl<cpu>(a_data, b_data, x_data,
                              temp_residuals_data, rank_data, singularValues_data,
                              &empty_residuals, lwork, liwork, workspace, attrs, ctx, req);
      if (empty_residuals) {
        // Set residuals to empty deriectly.
        const_cast<NDArray&>(residuals_ndarray).Init(mxnet::TShape(1, 0));
      } else {
        // No need set residuals to empty.
        const_cast<NDArray&>(residuals_ndarray).Init(temp_residuals_shape);
        // Copy back to gpu.
        CUDA_CALL(cudaMemcpyAsync(residuals_ndarray.data().dptr<DType>(), temp_residuals_vec.data(),
                                  sizeof(DType) * temp_residuals_data.Size(),
                                  cudaMemcpyHostToDevice, stream));
      }
      CUDA_CALL(cudaStreamSynchronize(stream));
      // Copy back to gpu.
      CUDA_CALL(cudaMemcpyAsync(x.dptr<DType>(), x_vec.data(), sizeof(DType) * x.Size(),
                                cudaMemcpyHostToDevice, stream));
      CUDA_CALL(cudaMemcpyAsync(rank.dptr<int>(), rank_vec.data(), sizeof(int) * rank.Size(),
                                cudaMemcpyHostToDevice, stream));
      CUDA_CALL(cudaMemcpyAsync(singularValues.dptr<DType>(), singularValues_vec.data(),
                                sizeof(DType) * singularValues.Size(),
                                cudaMemcpyHostToDevice, stream));
      CUDA_CALL(cudaStreamSynchronize(stream));
    });
  });
#else
  LOG(FATAL) << "Please build with USE_CUDA=1 to enable GPU";
#endif  // MXNET_USE_CUDA
}

template<typename xpu>
void LstsqOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<NDArray>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 4U);
  CHECK_EQ(req.size(), 4U);
  CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);
  CHECK(req[1] == kWriteTo || req[1] == kWriteInplace);
  CHECK(req[2] == kWriteTo || req[2] == kWriteInplace);
  CHECK(req[3] == kWriteTo || req[3] == kWriteInplace);
  using namespace mshadow;
  const NDArray& a_ndarray = inputs[0];
  const NDArray& b_ndarray = inputs[1];
  const NDArray& x_ndarray = outputs[0];
  const NDArray& residuals_ndarray = outputs[1];
  const NDArray& rank_ndarray = outputs[2];
  const NDArray& singularValues_ndarray = outputs[3];
  const mxnet::TShape& a_shape = a_ndarray.shape();
  const mxnet::TShape& b_shape = b_ndarray.shape();
  // Force set output shapes.
  mxnet::ShapeVector out_shapes(4);
  GetOutputShapes(a_shape, b_shape, &out_shapes);
  const mxnet::TShape& x_shape = out_shapes[0];
  const mxnet::TShape& temp_residuals_shape = out_shapes[1];
  const mxnet::TShape& rank_shape = out_shapes[2];
  const mxnet::TShape& singularValues_shape = out_shapes[3];

  MSHADOW_SGL_DBL_TYPE_SWITCH(x_ndarray.dtype(), DType, {
    // Allocate workspace.
    int lwork = 0, liwork = 0;
    size_t workspace_size = LstsqWorkspaceSize<cpu, DType>(a_ndarray.data(), b_ndarray.data(),
                                                           x_shape, &lwork, &liwork, ctx);
    std::vector<char> workspace(workspace_size);
    // Force init.
    const_cast<NDArray&>(x_ndarray).Init(x_shape);
    const_cast<NDArray&>(rank_ndarray).Init(rank_shape);
    const_cast<NDArray&>(singularValues_ndarray).Init(singularValues_shape);

    // Allocate temp space for residuals
    std::vector<DType> temp_residuals_vec(temp_residuals_shape.Size(), 0);
    if (xpu::kDevCPU) {
      bool empty_residuals = false;
      TBlob temp_residuals(temp_residuals_vec.data(),
                           Shape1(temp_residuals_vec.size()), cpu::kDevMask, -1);
      LstsqOpForwardImpl<cpu>(a_ndarray.data(), b_ndarray.data(), x_ndarray.data(),
                              temp_residuals, rank_ndarray.data(), singularValues_ndarray.data(),
                              &empty_residuals, lwork, liwork, &workspace, attrs, ctx, req);
      if (empty_residuals) {
        // Set residuals to empty deriectly.
        const_cast<NDArray&>(residuals_ndarray).Init(mxnet::TShape(1, 0));
      } else {
        // No need set residuals to empty.
        const_cast<NDArray&>(residuals_ndarray).Init(temp_residuals.shape_);
        MXNET_ASSIGN_REQ_SWITCH(req[1], req_type, {
          mxnet_op::Kernel<ValuesAssignHelper<req_type>, cpu>::Launch(
            ctx.get_stream<cpu>(), temp_residuals.Size(), temp_residuals.dptr<DType>(),
            residuals_ndarray.data().dptr<DType>());
        });
      }
    } else {
      GpuCallbackCpuImpl<DType>(a_ndarray.data(), b_ndarray.data(), x_ndarray.data(),
                                rank_ndarray.data(), singularValues_ndarray.data(),
                                residuals_ndarray, temp_residuals_shape,
                                lwork, liwork, &workspace, attrs, ctx, req);
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_LSTSQ_INL_H_
