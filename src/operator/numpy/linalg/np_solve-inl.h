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
 * \file np_solve-inl.h
 * \brief Placeholder for solve linear equation
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_SOLVE_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_SOLVE_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../../tensor/la_op.h"
#include "../../tensor/la_op-inl.h"
#include "../../linalg.h"
#include "../../operator_common.h"
#include "../../mshadow_op.h"

namespace mxnet {
namespace op {

using namespace mshadow;

template<typename xpu, typename DType>
void linalg_solve(const Tensor<xpu, 2, DType>& A,
                  const Tensor<xpu, 2, DType>& X,
                  const Tensor<xpu, 1, int>& ipiv,
                  Stream<xpu> *s);

template<typename xpu, typename DType>
void linalg_batch_solve(const Tensor<xpu, 3, DType>& A,
                        const Tensor<xpu, 3, DType>& X,
                        const Tensor<xpu, 2, int>& ipiv,
                        const mxnet::OpContext& ctx);

template<typename xpu, typename DType> inline
int linalg_dn_getrf_workspace_query(const Tensor<xpu, 2, DType>& A,
                                    Stream<xpu> *s);

template<typename xpu, typename DType> inline
void linalg_dn_getrf(const Tensor<xpu, 2, DType>& A,
                     const Tensor<xpu, 1, int>& ipiv,
                     Stream<xpu> *s);

template<typename xpu, typename DType> inline
void linalg_dn_getrs(const Tensor<xpu, 2, DType>& A,
                     const Tensor<xpu, 2, DType>& X,
                     const Tensor<xpu, 1, int>& ipiv,
                     Stream<xpu> *s);

// kernel for transpose
struct SolveTypeTransposeHelper {
  template<typename InDType, typename OutDType>
  MSHADOW_XINLINE static void Map(int i, const InDType *in_data, OutDType *out_data,
                                  const int ncol1, const int ncol2, const int step) {
    int idx = i / step, row = (i % step) / ncol1, col = (i % step) % ncol1;
    out_data[idx * step + row + col * ncol2] = static_cast<OutDType>(in_data[i]);
  }
};

template<typename xpu, typename DType>
inline void check_solve(const Tensor<xpu, 2, DType>& A,
                        const Tensor<xpu, 2, DType>& B) {
  CHECK_EQ(A.size(0), A.size(1)) << "A must bu square matrix";
  CHECK_EQ(A.size(1), B.size(1)) << "A, B have incompatible sizes";
}

#define LINALG_CPU_SOLVE(fname, DType) \
template<> inline \
void linalg_solve<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                              const Tensor<cpu, 2, DType>& X, \
                              const Tensor<cpu, 1, int>& ipiv, \
                              Stream<cpu> *s) { \
  check_solve(A, X); \
  const int N = X.size(1), nrhs = X.size(0); \
  const int lda = (N == 0 ? 1 : N), ldx = (N == 0 ? 1 : N); \
  int res(MXNET_LAPACK_##fname(MXNET_LAPACK_COL_MAJOR, N, nrhs, \
                               A.dptr_, lda, ipiv.dptr_, X.dptr_, ldx)); \
  CHECK_LE(res, 0) << #fname << ": U(" << res << ", " << res \
    << ") is exactly zero. The factorization has been completed," \
    << "but the factor U is exactly singular, so the solution could not be computed."; \
  CHECK_GE(res, 0) << #fname << ": the " << -res \
    << "-th argument had an illegal value"; \
}
LINALG_CPU_SOLVE(sgesv, float)
LINALG_CPU_SOLVE(dgesv, double)

#ifdef __CUDACC__

#if CUDA_VERSION >= 8000

#define LINALG_GPU_DN_GETRF_WORKSPACE_QUERY(fname, DType) \
template<> inline \
int linalg_dn_getrf_workspace_query<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                                Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  int lwork(0); \
  CUSOLVER_CALL(cusolver##fname##_bufferSize(Stream<gpu>::GetSolverHandle(s), \
                                             A.size(1), A.size(1), A.dptr_, \
                                             (A.size(1) == 0 ? 1 : A.size(1)), &lwork)); \
  return lwork; \
}

#define LINALG_GPU_DN_GETRF(fname, DType) \
template<> inline \
void linalg_dn_getrf<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                 const Tensor<gpu, 1, int>& ipiv, \
                                 Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  Storage::Handle info = Storage::Get()->Alloc(sizeof(int), Context::GPU()); \
  const int lwork = linalg_dn_getrf_workspace_query(A, s); \
  Storage::Handle workspace = Storage::Get()->Alloc(sizeof(DType) * lwork, Context::GPU()); \
  CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s), \
                                A.size(1), A.size(1), A.dptr_, (A.size(1) == 0 ? 1 : A.size(1)), \
                                static_cast<DType*>(workspace.dptr), ipiv.dptr_, \
                                static_cast<int*>(info.dptr))); \
  Storage::Get()->Free(info); \
  Storage::Get()->Free(workspace); \
}

#define LINALG_GPU_DN_GETRS(fname, DType) \
template<> inline \
void linalg_dn_getrs<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                 const Tensor<gpu, 2, DType>& X, \
                                 const Tensor<gpu, 1, int>& ipiv, \
                                 Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  const int N = A.size(0), nrhs = X.size(0); \
  const int lda = (A.size(1) == 0 ? 1 : A.size(1)), ldx = (X.size(1) == 0 ? 1 : X.size(1)); \
  Storage::Handle info = Storage::Get()->Alloc(sizeof(int), Context::GPU()); \
  CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s), \
                                CUBLAS_OP_N, N, nrhs, \
                                A.dptr_, lda, ipiv.dptr_, X.dptr_, ldx, \
                                static_cast<int*>(info.dptr))); \
  Storage::Get()->Free(info); \
}

#define LINALG_GPU_SOLVE(DType) \
template<> inline \
void linalg_solve<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                              const Tensor<gpu, 2, DType>& X, \
                              const Tensor<gpu, 1, int>& ipiv, \
                              Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  check_solve(A, X); \
  linalg_dn_getrf(A, ipiv, s); \
  linalg_dn_getrs(A, X, ipiv, s); \
}

#else  // CUDA_VERSION >= 8000

#define LINALG_GPU_DN_GETRF_WORKSPACE_QUERY(fname, DType) \
template<> inline \
int linalg_dn_getrf_workspace_query<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                                Stream<gpu> *s) { \
  LOG(FATAL) << "Dn_getrf_workspace_query requires CUDA version >= 8.0!"; \
}

#define LINALG_GPU_DN_GETRF(fname, DType) \
template<> inline \
void linalg_dn_getrf<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                 const Tensor<gpu, 1, int>& ipiv, \
                                 Stream<gpu> *s) { \
  LOG(FATAL) << "Dn_getrf requires CUDA version >= 8.0!"; \
}

#define LINALG_GPU_DN_GETRS(fname, DType) \
template<> inline \
void linalg_dn_getrs<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                 const Tensor<gpu, 2, DType>& X, \
                                 const Tensor<gpu, 1, int>& ipiv, \
                                 Stream<gpu> *s) { \
  LOG(FATAL) << "Dn_getrs requires CUDA version >= 8.0!"; \
}

#define LINALG_GPU_SOLVE(DType) \
template<> inline \
void linalg_solve<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                              const Tensor<gpu, 2, DType>& X, \
                              const Tensor<gpu, 1, int>& ipiv, \
                              Stream<gpu> *s) { \
  LOG(FATAL) << "gpu solve requires CUDA version >= 8.0!"; \
}

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_DN_GETRF_WORKSPACE_QUERY(DnSgetrf, float)
LINALG_GPU_DN_GETRF_WORKSPACE_QUERY(DnDgetrf, double)

LINALG_GPU_DN_GETRF(DnSgetrf, float)
LINALG_GPU_DN_GETRF(DnDgetrf, double)

LINALG_GPU_DN_GETRS(DnSgetrs, float)
LINALG_GPU_DN_GETRS(DnDgetrs, double)

LINALG_GPU_SOLVE(float)
LINALG_GPU_SOLVE(double)

#endif  // __CUDACC__

#define LINALG_XPU_BATCH_SOLVE(xpu, DType) \
template<> inline \
void linalg_batch_solve<xpu, DType>(const Tensor<xpu, 3, DType>& A, \
                                    const Tensor<xpu, 3, DType>& X, \
                                    const Tensor<xpu, 2, int>& ipiv, \
                                    const mxnet::OpContext& ctx) { \
  Stream<xpu> *s = ctx.get_stream<xpu>(); \
  for (index_t i = 0; i < A.size(0); ++i) { \
    linalg_solve(A[i], X[i], ipiv[i], s); \
  } \
}
LINALG_XPU_BATCH_SOLVE(cpu, float)
LINALG_XPU_BATCH_SOLVE(cpu, double)

#ifdef __CUDACC__

LINALG_XPU_BATCH_SOLVE(gpu, float)
LINALG_XPU_BATCH_SOLVE(gpu, double)

#endif  // __CUDACC__

struct solve {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& X,
                 const Tensor<xpu, 2, int>& ipiv,
                 const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    linalg_batch_solve(A, X, ipiv, ctx);  // ipiv for work_space in Lapacke_#gesv
  }
};

template<typename xpu, int idim, int odim, int inum, int onum, typename laop>
void LaOpForwardSolve(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), inum);
  CHECK_EQ(outputs.size(), onum);
  CHECK_EQ(req.size(), onum);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    const mxnet::TBlob& a_tblob = inputs[0];
    const mxnet::TBlob& b_tblob = inputs[1];
    const mxnet::TBlob& x_tblob = outputs[0];
    const mxnet::TShape& a_shape = a_tblob.shape_;
    mxnet::TShape b_shape(a_shape.ndim(), 1);
    for (int i = 0; i < a_shape.ndim() - 1; ++i) { b_shape[i] = b_tblob.shape_[i]; }
    if (b_tblob.shape_.ndim() == a_shape.ndim()) {
      b_shape[a_shape.ndim() - 1] = b_tblob.shape_[a_shape.ndim() - 1];
    }
    const int ndim = a_shape.ndim();
    mxnet::TShape ipiv_shape(a_shape);
    ipiv_shape[ndim - 1] = 1;
    if (0 == a_shape[ndim - 1] || 0 == a_shape[ndim - 2] ||
        0 == b_shape[ndim - 1] || 0 == b_shape[ndim - 2]) { return; }

    const int work_space_size =
      sizeof(OType) * (a_shape.Size() + b_shape.Size()) + sizeof(int) * ipiv_shape.Size();
    Tensor<xpu, 1, char> work_buffer =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(work_space_size), s);
    MSHADOW_TYPE_SWITCH(a_tblob.type_flag_, AType, {
      // cast type and transpose
      mxnet_op::Kernel<SolveTypeTransposeHelper, xpu>::Launch(
        s, a_shape.Size(),
        a_tblob.dptr<AType>(),
        reinterpret_cast<OType*>(work_buffer.dptr_),
        a_shape[ndim - 1], a_shape[ndim - 2], a_shape[ndim - 1] * a_shape[ndim - 2]);
    });
    MSHADOW_TYPE_SWITCH(b_tblob.type_flag_, BType, {
      // cast type and transpose
      mxnet_op::Kernel<SolveTypeTransposeHelper, xpu>::Launch(
        s, b_shape.Size(),
        b_tblob.dptr<BType>(),
        reinterpret_cast<OType*>(work_buffer.dptr_) + a_shape.Size(),
        b_shape[ndim - 1], b_shape[ndim - 2], b_shape[ndim - 1] * b_shape[ndim - 2]);
    });
    // transpose shape
    int temp = b_shape[ndim - 1];
    b_shape[ndim - 1] = b_shape[ndim - 2];
    b_shape[ndim - 2] = temp;
    mxnet::TBlob a_transpose_tblob(reinterpret_cast<OType*>(work_buffer.dptr_),
      a_shape, a_tblob.dev_mask(), a_tblob.dev_id());
    mxnet::TBlob b_transpose_tblob(reinterpret_cast<OType*>(work_buffer.dptr_) + a_shape.Size(),
      b_shape, b_tblob.dev_mask(), b_tblob.dev_id());
    mxnet::TBlob ipiv_tblob(reinterpret_cast<int*>(
      reinterpret_cast<OType*>(work_buffer.dptr_) + a_shape.Size() + b_shape.Size()),
      ipiv_shape, b_tblob.dev_mask(), b_tblob.dev_id());

    laop::op(a_transpose_tblob.FlatToKD<xpu, idim + 1, OType>(s),
             b_transpose_tblob.FlatToKD<xpu, idim + 1, OType>(s),
             ipiv_tblob.FlatToKD<xpu, idim, int>(s),
             ctx,
             attrs);
    // X = transpose(B)
    mxnet_op::Kernel<SolveTypeTransposeHelper, xpu>::Launch(
      s, b_shape.Size(),
      b_transpose_tblob.dptr<OType>(),
      x_tblob.dptr<OType>(),
      b_shape[ndim - 1], b_shape[ndim - 2], b_shape[ndim - 1] * b_shape[ndim - 2]);
  });
}

// X = (inv_A) * B
struct solve_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dX,
                 const Tensor<xpu, 3, DType>& inv_A,
                 const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& X,
                 const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& dB,
                 const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    // (1) calcualte dB = trans(inv(A)) * dX
    // (2) calcualte dA = dB * trans(X)
    Stream<xpu> *s = ctx.get_stream<xpu>();
    gemm2::op(inv_A, dX, dB, DType(1), true, false, s);
    gemm2::op(dB, X, dA, DType(-1), false, true, s);
  }
};

template<typename xpu, typename DType>
inline void batch_inverse(const Tensor<xpu, 3, DType>& inv_A,
                          const Tensor<xpu, 3, DType>& LU,
                          const Tensor<xpu, 2, int>& pivot,
                          const mxnet::OpContext& ctx);

#define CPU_BATCH_INVERSE(xpu, DType) \
template<> inline \
void batch_inverse<xpu, DType>(const Tensor<xpu, 3, DType>& inv_A, \
                               const Tensor<xpu, 3, DType>& LU, \
                               const Tensor<xpu, 2, int>& pivot, \
                               const mxnet::OpContext& ctx) { \
  Stream<xpu> *s = ctx.get_stream<xpu>(); \
  for (index_t i = 0; i < inv_A.size(0); ++i) { \
    linalg_getrf(inv_A[i], pivot[i], true, s); \
    const Tensor<xpu, 1, DType> work( \
      LU[i].dptr_, Shape1(LU.size(1) * LU.size(2))); \
    linalg_getri(inv_A[i], pivot[i], work, s); \
  } \
}
CPU_BATCH_INVERSE(cpu, float)
CPU_BATCH_INVERSE(cpu, double)

#ifdef __CUDACC__

// GETRF and GETRI only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

#define GPU_BATCH_INVERSE(xpu, DType) \
template<> inline \
void batch_inverse<xpu, DType>(const Tensor<xpu, 3, DType>& inv_A, \
                               const Tensor<xpu, 3, DType>& LU, \
                               const Tensor<xpu, 2, int>& pivot, \
                               const mxnet::OpContext& ctx) { \
  Stream<xpu> *s = ctx.get_stream<xpu>(); \
  if (LU.dptr_ != inv_A.dptr_) Copy(LU, inv_A, s); \
  linalg_batch_getrf(LU, pivot, true, s); \
  linalg_batch_getri(inv_A, LU, pivot, s); \
}

#else  // CUDA_VERSION >= 8000

#define GPU_BATCH_INVERSE(xpu, DType) \
template<> inline \
void batch_inverse<xpu, DType>(const Tensor<xpu, 3, DType>& inv_A, \
                               const Tensor<xpu, 3, DType>& LU, \
                               const Tensor<xpu, 2, int>& pivot, \
                               const mxnet::OpContext& ctx) { \
  LOG(FATAL) << "gpu matrix inverse requires CUDA version >= 8.0!"; \
}

#endif  // CUDA_VERSION >= 8000

GPU_BATCH_INVERSE(gpu, float)
GPU_BATCH_INVERSE(gpu, double)

#endif  // __CUDACC__

template<typename xpu, int idim, int odim, int inum, int onum, typename laop>
void LaOpBackwardSolve(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), inum);
  CHECK_EQ(outputs.size(), onum);
  CHECK_EQ(req.size(), onum);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    const mxnet::TBlob& a_tblob = inputs[1];
    const mxnet::TBlob& b_tblob = inputs[2];
    const mxnet::TBlob& x_tblob = inputs[3];

    const mxnet::TShape& a_shape = a_tblob.shape_;
    mxnet::TShape b_shape(a_shape.ndim(), 1);
    for (int i = 0; i < a_shape.ndim() - 1; ++i) { b_shape[i] = b_tblob.shape_[i]; }
    if (b_tblob.shape_.ndim() == a_shape.ndim()) {
      b_shape[a_shape.ndim() - 1] = b_tblob.shape_[a_shape.ndim() - 1];
    }
    const int ndim = a_shape.ndim();
    const int N = a_shape[ndim - 1];
    if (0 == a_shape[ndim - 1] || 0 == a_shape[ndim - 2] ||
        0 == b_shape[ndim - 1] || 0 == b_shape[ndim - 2]) { return; }

    const Tensor<xpu, idim + 1, OType> A = a_tblob.FlatToKD<xpu, idim + 1, OType>(s);
    int work_space_size = sizeof(OType) * a_shape.Size();  // for inverse(A)
    work_space_size += sizeof(OType) * a_shape.Size();  // for getri work space
    work_space_size += 2 * sizeof(OType) * b_shape.Size();  // for B and X
    work_space_size += sizeof(int) * A.size(0) * N;  // for pivot work space
    Tensor<xpu, 1, char> work_buffer =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(work_space_size), s);

    MSHADOW_TYPE_SWITCH(a_tblob.type_flag_, AType, {
      mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
        s, a_shape.Size(),
        reinterpret_cast<OType*>(work_buffer.dptr_),
        a_tblob.dptr<AType>());
    });
    mxnet::TBlob a_inverse_tblob(reinterpret_cast<OType*>(work_buffer.dptr_),
      a_shape, a_tblob.dev_mask(), a_tblob.dev_id());
    const Tensor<xpu, idim + 1, OType> inv_A = a_inverse_tblob.FlatToKD<xpu, idim + 1, OType>(s);

    mxnet::TBlob lu_tblob(reinterpret_cast<OType*>(work_buffer.dptr_) + a_shape.Size(),
      inv_A.shape_, a_tblob.dev_mask(), a_tblob.dev_id());
    const Tensor<xpu, idim + 1, OType> LU = lu_tblob.FlatToKD<xpu, idim + 1, OType>(s);

    MSHADOW_TYPE_SWITCH(b_tblob.type_flag_, BType, {
      mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
        s, b_shape.Size(),
        reinterpret_cast<OType*>(work_buffer.dptr_) + 2 * a_shape.Size(),
        b_tblob.dptr<BType>());
    });
    mxnet::TBlob b_cp_tblob(reinterpret_cast<OType*>(work_buffer.dptr_) + 2 * a_shape.Size(),
      b_shape, b_tblob.dev_mask(), b_tblob.dev_id());
    const Tensor<xpu, idim + 1, OType> B = b_cp_tblob.FlatToKD<xpu, idim + 1, OType>(s);

    MSHADOW_TYPE_SWITCH(x_tblob.type_flag_, XType, {
      mxnet_op::Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
        s, b_shape.Size(),
        reinterpret_cast<OType*>(work_buffer.dptr_) + 2 * a_shape.Size() + b_shape.Size(),
        x_tblob.dptr<XType>());
    });
    mxnet::TBlob x_cp_tblob(
      reinterpret_cast<OType*>(work_buffer.dptr_) + 2 * a_shape.Size() + b_shape.Size(),
      b_shape, b_tblob.dev_mask(), b_tblob.dev_id());
    const Tensor<xpu, idim + 1, OType> X = x_cp_tblob.FlatToKD<xpu, idim + 1, OType>(s);

    mxnet::TBlob pivot_tblob(reinterpret_cast<int*>(
      reinterpret_cast<OType*>(work_buffer.dptr_) + 2 * a_shape.Size() + 2 * b_shape.Size()),
      Shape2(A.size(0), N), a_tblob.dev_mask(), a_tblob.dev_id());
    const Tensor<xpu, idim, int> pivot = pivot_tblob.FlatToKD<xpu, idim, int>(s);

    // calculate inverse(A) on CPU or GPU
    batch_inverse(inv_A, LU, pivot, ctx);
    laop::op(inputs[0].FlatToKD<xpu, idim + 1, OType>(s),
             inv_A,
             B,
             X,
             outputs[0].FlatToKD<xpu, odim + 1, OType>(s),
             outputs[1].FlatToKD<xpu, odim + 1, OType>(s),
             ctx,
             attrs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_SOLVE_INL_H_
