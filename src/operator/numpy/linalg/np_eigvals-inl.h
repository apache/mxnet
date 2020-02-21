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
 * \file np_eigvals-inl.h
 * \brief Placeholder for eigvals
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_EIGVALS_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_EIGVALS_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../../operator_common.h"
#include "../../mshadow_op.h"
#include "../../tensor/la_op.h"
#include "../../tensor/la_op-inl.h"
#include "./np_solve-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

template<int req>
struct eigvals_assign_helper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, DType *out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

struct eigh_eigvalsh_helper {
  template<typename InDType, typename OutDType>
  MSHADOW_XINLINE static void Map(int i, const InDType *in_data, OutDType *out_data,
                                  const int nrow, const int ncol, const int ld,
                                  const int step, bool USE_UP) {
    int idx = i / step, row = (i % step) / ld, col = (i % step) % ld;
    if (row < nrow && col < ncol) {
      if ((USE_UP && row > col) || (!USE_UP && row < col)) {
        out_data[idx * step + col + row * ld] =
          static_cast<OutDType>(in_data[idx * step + row + col * ld]);
      } else {
        out_data[idx * step + col + row * ld] =
          static_cast<OutDType>(in_data[idx * step + col + row * ld]);
      }
    }
  }
};

template<typename xpu, typename DType>
void linalg_geev(char jobvl,
                 char jobvr,
                 const Tensor<xpu, 2, DType>& a,
                 const Tensor<xpu, 1, DType>& wr,
                 const Tensor<xpu, 1, DType>& wi,
                 const Tensor<xpu, 2, DType>& vl,
                 const Tensor<xpu, 2, DType>& vr,
                 const Tensor<xpu, 1, DType>& work_array);

#define LINALG_CPU_EIG(fname, DType) \
template<> inline \
void linalg_geev<cpu, DType>(char jobvl, \
                             char jobvr, \
                             const Tensor<cpu, 2, DType>& a, \
                             const Tensor<cpu, 1, DType>& wr, \
                             const Tensor<cpu, 1, DType>& wi, \
                             const Tensor<cpu, 2, DType>& vl, \
                             const Tensor<cpu, 2, DType>& vr, \
                             const Tensor<cpu, 1, DType>& work_array) { \
  const int n = a.size(1), lda = a.size(0); \
  const int lwork = work_array.shape_.Size(); \
  const int ldvl = vl.size(0), ldvr = vr.size(0); \
  int res(MXNET_LAPACK_##fname(MXNET_LAPACK_COL_MAJOR, jobvl, jobvr, \
                               n, a.dptr_, lda, \
                               wr.dptr_, wi.dptr_, \
                               vl.dptr_, ldvl, \
                               vr.dptr_, ldvr, \
                               work_array.dptr_, lwork)); \
  CHECK_LE(res, 0) << #fname << "the QR algorithm failed to compute all the" \
    << "eigenvalues, and no eigenvectors have been computed; elements " \
    << res + 1 << ":N" << " of WR and WI contain eigenvalues which have converged"; \
  CHECK_GE(res, 0) << #fname << ": the " << -res \
    << "-th argument had an illegal value"; \
}

LINALG_CPU_EIG(sgeev, float)
LINALG_CPU_EIG(dgeev, double)

#ifdef __CUDACC__

#define LINALG_GPU_EIG(fname, DType) \
template<> inline \
void linalg_geev<gpu, DType>(char jobvl, \
                             char jobvr, \
                             const Tensor<gpu, 2, DType>& a, \
                             const Tensor<gpu, 1, DType>& wr, \
                             const Tensor<gpu, 1, DType>& wi, \
                             const Tensor<gpu, 2, DType>& vl, \
                             const Tensor<gpu, 2, DType>& vr, \
                             const Tensor<gpu, 1, DType>& work_array) { \
  LOG(FATAL) << "Lapack _geev routines in gpu is unsupported"; \
}

LINALG_GPU_EIG(sgeev, float)
LINALG_GPU_EIG(dgeev, double)

#endif  // __CUDACC__

struct eig_eigvals {
  template<typename xpu, typename DType>
  static void op(char jobvl,
                 char jobvr,
                 const Tensor<xpu, 3, DType>& a,
                 const Tensor<xpu, 2, DType>& wr,
                 const Tensor<xpu, 2, DType>& wi,
                 const Tensor<xpu, 3, DType>& vl,
                 const Tensor<xpu, 3, DType>& vr,
                 const Tensor<xpu, 1, DType>& work_array) {
    const mxnet::TShape& a_shape = a.shape_;
    const int a_ndim = a_shape.ndim();
    if (jobvl == 'N' && jobvr == 'N') {
      CHECK_GE(work_array.shape_.Size(), 3 * a.shape_[a_ndim - 1])
        << "The dimension of the array WORK in LAPACKE_#GEEV should >= max(1,3*N).";
    } else {
      CHECK_GE(work_array.shape_.Size(), 4 * a.shape_[a_ndim - 1])
        << "If JOBVL = 'V' or JOBVR = 'V', "
        << "the dimension of the array WORK in LAPACKE_#GEEV should >= 4*N.";
    }
    for (int i = 0; i < a_shape[0]; ++i) {
      if (jobvl == 'N' && jobvr == 'N') {
        linalg_geev(jobvl, jobvr, a[i], wr[i], wi[i], vl[0], vr[0], work_array);
      } else if (jobvl == 'N' && jobvr == 'V') {
        linalg_geev(jobvl, jobvr, a[i], wr[i], wi[i], vl[0], vr[i], work_array);
      } else if (jobvl == 'V' && jobvr == 'N') {
        linalg_geev(jobvl, jobvr, a[i], wr[i], wi[i], vl[i], vr[0], work_array);
      } else {
        linalg_geev(jobvl, jobvr, a[i], wr[i], wi[i], vl[i], vr[i], work_array);
      }
    }
  }
};

// Calculates workspace size of eigvals forward op.
// The dimension of the array WORK in LAPACKE_#GEEV should >= max(1,3*N), and
// if JOBVL = 'V' or JOBVR = 'V', LWORK >= 4*N.
// For good performance, LWORK must generally be larger.
template<typename xpu>
size_t EigvalsForwardWorkspaceSize(const TBlob& a,
                                   const TBlob& w,
                                   const std::vector<OpReqType>& req) {
  if (kNullOp == req[0]) { return 0U; }

  // Zero-size input, no need to launch kernel
  if (0U == a.Size()) { return 0U; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, DType, {
    size_t work_space_size = 0;
    size_t n = a.size(a.ndim() - 1);
    work_space_size += a.Size();      // For matrix.
    work_space_size += 2 * w.Size();  // For eigenvalues' real and image component.
    work_space_size += 2 * n * n;     // For left and right eigenvectors temp memory
    work_space_size += 3 * n;         // For workspace size in LAPACKE_#GEEV.
    work_space_size *= sizeof(DType);
    return work_space_size;
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

template<typename xpu>
void EigvalsOpForwardImpl(const TBlob& a,
                          const TBlob& w,
                          const std::vector<OpReqType>& req,
                          std::vector<char> *workspace,
                          mshadow::Stream<xpu> *s) {
  if (kNullOp == req[0]) { return; }
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& w_shape = w.shape_;
  const int a_ndim = a_shape.ndim();

  // Zero-size output, no need to launch kernel
  if (0U == a.Size()) { return; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, DType, {
    const int N = a.size(a_ndim - 1);
    DType *a_ptr =
      reinterpret_cast<DType*>(workspace->data());
    DType *wr_ptr =
      reinterpret_cast<DType*>(workspace->data() + a.Size() * sizeof(DType));
    DType *wi_ptr =
      reinterpret_cast<DType*>(workspace->data() + (w.Size() + a.Size()) * sizeof(DType));
    DType *vl_ptr =
      reinterpret_cast<DType*>(workspace->data() + (2 * w.Size() + a.Size()) * sizeof(DType));
    DType *vr_ptr =
      reinterpret_cast<DType*>(
        workspace->data() + (N * N + 2 * w.Size() + a.Size()) * sizeof(DType));
    DType *work_ptr =
      reinterpret_cast<DType*>(
        workspace->data() + (2 * (N * N + w.Size()) + a.Size()) * sizeof(DType));
    MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
      // Cast type and transpose.
      mxnet_op::Kernel<SolveTypeTransposeHelper, xpu>::Launch(
        s, a.Size(), a.dptr<AType>(), a_ptr, N, N, N * N);
    });
    char jobvl = 'N', jobvr = 'N';
    mxnet::TBlob a_trans_data(a_ptr, a_shape, a.dev_mask(), a.dev_id());
    mxnet::TBlob wr_data(wr_ptr, w_shape, w.dev_mask(), w.dev_id());
    mxnet::TBlob wi_data(wi_ptr, w_shape, w.dev_mask(), w.dev_id());
    mxnet::TBlob vl_data(vl_ptr, Shape3(1, N, N), w.dev_mask(), w.dev_id());
    mxnet::TBlob vr_data(vr_ptr, Shape3(1, N, N), w.dev_mask(), w.dev_id());
    mxnet::TBlob work_data(work_ptr, Shape1(3 * N), a.dev_mask(), a.dev_id());
    eig_eigvals::op(jobvl, jobvr,
                    a_trans_data.FlatToKD<xpu, 3, DType>(s),
                    wr_data.FlatToKD<xpu, 2, DType>(s),
                    wi_data.FlatToKD<xpu, 2, DType>(s),
                    vl_data.get<xpu, 3, DType>(s),
                    vr_data.get<xpu, 3, DType>(s),
                    work_data.get<xpu, 1, DType>(s));
    for (size_t i = 0; i < wi_data.Size(); ++i) {
      CHECK_LE(fabs(wi_ptr[i]), 1e-15)
        << "Complex eigvals is unsupported in linalg temporary.";
    }
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      mxnet_op::Kernel<eigvals_assign_helper<req_type>, xpu>::Launch(
        s, w.Size(), wr_ptr, w.dptr<DType>());
    });
  });
}

template<typename AType, typename WType>
void GpuCallbackCpuImpl(const TBlob& a,
                        const TBlob& w,
                        AType* a_cp_ptr,
                        WType* w_cp_ptr,
                        std::vector<char>* workspace,
                        const OpContext& ctx,
                        const std::vector<OpReqType>& req) {
#if MXNET_USE_CUDA
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  CUDA_CALL(cudaMemcpyAsync(a_cp_ptr, a.dptr<AType>(), sizeof(AType) * a.Size(),
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaMemcpyAsync(w_cp_ptr, w.dptr<WType>(), sizeof(WType) * w.Size(),
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  mxnet::TBlob a_data(a_cp_ptr, a.shape_, cpu::kDevMask);
  mxnet::TBlob w_data(w_cp_ptr, w.shape_, cpu::kDevMask);
  // Op forward implement on cpu.
  EigvalsOpForwardImpl<cpu>(a_data, w_data, req, workspace, ctx.get_stream<cpu>());
  // Copy back to gpu.
  CUDA_CALL(cudaMemcpyAsync(w.dptr<WType>(), w_cp_ptr, sizeof(WType) * w.Size(),
                            cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
#else
  LOG(FATAL) << "Please build with USE_CUDA=1 to enable GPU";
#endif  // MXNET_USE_CUDA
}

template<typename xpu>
void EigvalsOpForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& a = inputs[0];
  const TBlob& w = outputs[0];

  // Calculate workspace size.
  size_t workspace_size = EigvalsForwardWorkspaceSize<cpu>(a, w, req);
  std::vector<char> workspace(workspace_size, 0);

  MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, WType, {
    MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
      if (xpu::kDevCPU) {
        // Op forward implement.
        EigvalsOpForwardImpl<cpu>(a, w, req, &workspace, ctx.get_stream<cpu>());
      } else {
        std::vector<AType> a_vec(a.Size(), 0);
        std::vector<WType> w_vec(w.Size(), 0);
        AType* a_cp_ptr = a_vec.data();
        WType* w_cp_ptr = w_vec.data();
        GpuCallbackCpuImpl(a, w, a_cp_ptr, w_cp_ptr, &workspace, ctx, req);
      }
    });
  });
}

struct EigvalshParam : public dmlc::Parameter<EigvalshParam> {
  char UPLO;
  DMLC_DECLARE_PARAMETER(EigvalshParam) {
    DMLC_DECLARE_FIELD(UPLO)
    .set_default('L')
    .describe("Specifies whether the calculation is done with the lower or upper triangular part.");
  }
};

template<typename xpu>
size_t EighEigvalshForwardWorkspaceSize(const TBlob& a,
                                        const TBlob& w,
                                        const std::vector<OpReqType>& req,
                                        const OpContext& ctx) {
  if (kNullOp == req[0]) { return 0U; }

  // Zero-size input, no need to launch kernel
  if (0U == a.Size()) { return 0U; }

  Stream<xpu> *s = ctx.get_stream<xpu>();
  Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
  MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, WType, {
      Tensor<xpu, 2, AType> a_temp_tensor = a.FlatToKD<xpu, 3, AType>(s)[0];
      Tensor<xpu, 1, WType> w_temp_tensor = w.FlatToKD<xpu, 2, WType>(s)[0];
      size_t work_space_size = 0;
      std::vector<WType> A_data(a_temp_tensor.MSize(), 0);
      std::vector<WType> W_data(w_temp_tensor.MSize(), 0);
      TBlob a_data(A_data.data(), a_temp_tensor.shape_, cpu::kDevMask, -1);
      TBlob w_data(W_data.data(), w_temp_tensor.shape_, cpu::kDevMask, -1);
      work_space_size += a.Size();  // For matrix.
      work_space_size += w.Size();  // For eigenvalues.
      work_space_size +=            // For workspace size in LAPACKE_#SYEVD.
        linalg_syevd_workspace_query(a_data.get<cpu, 2, WType>(s_cpu),
                                     w_data.get<cpu, 1, WType>(s_cpu),
                                     s_cpu);
      return work_space_size * sizeof(WType);
    });
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

template<typename xpu>
void EigvalshOpForwardImpl(const TBlob& a,
                           const TBlob& w,
                           const char& UPLO,
                           const OpContext& ctx,
                           const std::vector<OpReqType>& req,
                           std::vector<char> *workspace) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, DType, {
      const size_t workspace_size = (workspace->size() + sizeof(DType) - 1) / sizeof(DType);
      DType *a_ptr =
        reinterpret_cast<DType*>(workspace->data());
      DType *w_ptr =
        reinterpret_cast<DType*>(workspace->data() + a.Size() * sizeof(DType));
      DType *work_ptr =
        reinterpret_cast<DType*>(workspace->data() + (a.Size() + w.Size()) * sizeof(DType));
      TBlob a_data(a_ptr, a.shape_, a.dev_mask(), a.dev_id());
      TBlob w_data(w_ptr, w.shape_, w.dev_mask(), w.dev_id());
      TBlob work_data(work_ptr, Shape1(workspace_size - a.Size() - w.Size()),
                      w.dev_mask(), w.dev_id());
      Tensor<xpu, 3, DType> A = a_data.FlatToKD<xpu, 3, DType>(s);
      Tensor<xpu, 2, DType> W = w_data.FlatToKD<xpu, 2, DType>(s);
      Tensor<xpu, 1, DType> Work = work_data.get<xpu, 1, DType>(s);
      // Copy used upper triangle part of 'a'.
      mxnet_op::Kernel<eigh_eigvalsh_helper, xpu>::Launch(s, a.Size(),
                                                          a.dptr<AType>(), A.dptr_,
                                                          A.size(1), A.size(2), A.stride_,
                                                          A.size(1) * A.stride_, UPLO == 'U');
      for (index_t i = 0; i < A.size(0); ++i) {
        linalg_syevd(A[i], W[i], Work, s);
      }
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<eigvals_assign_helper<req_type>, xpu>::Launch(
          s, w.Size(), w_ptr, w.dptr<DType>());
      });
    });
  });
}

template<typename AType, typename WType>
void GpuCallbackCpuImpl(const TBlob& a,
                        const TBlob& w,
                        const char& UPLO,
                        AType* a_cp_ptr,
                        WType* w_cp_ptr,
                        std::vector<char> *workspace,
                        const OpContext& ctx,
                        const std::vector<OpReqType>& req) {
#if MXNET_USE_CUDA
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  CUDA_CALL(cudaMemcpyAsync(a_cp_ptr, a.dptr<AType>(), sizeof(AType) * a.Size(),
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaMemcpyAsync(w_cp_ptr, w.dptr<WType>(), sizeof(WType) * w.Size(),
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  mxnet::TBlob a_data(a_cp_ptr, a.shape_, cpu::kDevMask);
  mxnet::TBlob w_data(w_cp_ptr, w.shape_, cpu::kDevMask);
  // Op forward implement on cpu.
  EigvalshOpForwardImpl<cpu>(a_data, w_data, UPLO, ctx, req, workspace);
  // Copy back to gpu.
  CUDA_CALL(cudaMemcpyAsync(w.dptr<WType>(), w_cp_ptr, sizeof(WType) * w.Size(),
                            cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
#else
  LOG(FATAL) << "Please build with USE_CUDA=1 to enable GPU";
#endif  // MXNET_USE_CUDA
}

template<typename xpu>
void EigvalshOpForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& a = inputs[0];
  const TBlob& w = outputs[0];
  char UPLO = nnvm::get<EigvalshParam>(attrs.parsed).UPLO;

  if (kNullOp == req[0]) { return; }
  // Zero-size output, no need to launch kernel
  if (0U == a.Size()) { return; }

  // Calculate workspace size.
  size_t workspace_size = EighEigvalshForwardWorkspaceSize<xpu>(a, w, req, ctx);
  std::vector<char> workspace(workspace_size, 0);

  MSHADOW_SGL_DBL_TYPE_SWITCH(w.type_flag_, WType, {
    MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
      if (xpu::kDevCPU) {
        // Op forward implement.
        EigvalshOpForwardImpl<cpu>(a, w, UPLO, ctx, req, &workspace);
      } else {
        std::vector<AType> a_vec(a.Size(), 0);
        std::vector<WType> w_vec(w.Size(), 0);
        AType* a_cp_ptr = a_vec.data();
        WType* w_cp_ptr = w_vec.data();
        GpuCallbackCpuImpl(a, w, UPLO, a_cp_ptr, w_cp_ptr, &workspace, ctx, req);
      }
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_EIGVALS_INL_H_
