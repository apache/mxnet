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
 * \file np_qr-inl.h
 * \brief Function definition of the QR Operator.
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_QR_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_QR_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include "../../tensor/la_op.h"
#include "../../tensor/la_op-inl.h"
#include "../../linalg.h"
#include "../../operator_common.h"
#include "../../mshadow_op.h"

namespace mxnet {
namespace op {

using namespace mshadow;

//////////////////////////////// QR ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "geqrf", "orgqr". Please refer to the
// LAPACK documentation for further details.
// Note:
// - Both functions have A as input and output parameter
// - Both functions require extra workspace, passed as 1D tensor
// - We call orgqr after geqrf. Apart from A, they also communicate via the
//   first part of the workspace.

template<typename xpu, typename DType>
void linalg_geqrf(const Tensor<xpu, 2, DType>& A,
                  const Tensor<xpu, 1, DType>& work,
                  Stream<xpu> *s = 0);

template<typename xpu, typename DType>
void linalg_orgqr(const Tensor<xpu, 2, DType>& A,
                  const Tensor<xpu, 1, DType>& work,
                  Stream<xpu> *s = 0);

// This function determines the amount of workspace needed for linalg_geqrf,
// linalg_orgqr. The workspace can be used for both. The first mn entries are
// used to communicate information from geqrf to orgqr (tau).

template<typename xpu, typename DType>
int linalg_qr_workspace_query(const Tensor<xpu, 2, DType>& A,
                              Stream<xpu> *s = 0);

//////////////////////////////// QR ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "geqrf", "orqrq." to be used in QR op.

template<typename xpu, typename DType> inline
void check_geqrf(const Tensor<xpu, 2, DType>& A,
                 const Tensor<xpu, 1, DType>& work) {
  CHECK_LE(A.size(0), work.size(0))
    << "Size of work is too small";
}

// transpose helper
struct QrTypeTransposeHelper {
  template<typename InDType, typename OutDType>
  MSHADOW_XINLINE static void Map(int i, const InDType *in_data, OutDType *out_data,
                                  const int ncol1, const int ncol2, const int step) {
    int idx = i / step, row = (i % step) / ncol1, col = (i % step) % ncol1;
    out_data[idx * step + row + col * ncol2] = static_cast<OutDType>(in_data[i]);
  }
};

#define LINALG_CPU_GEQRF(fname, DType) \
template<> inline \
void linalg_geqrf<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                              const Tensor<cpu, 1, DType>& work, \
                              Stream<cpu> *s) { \
  check_geqrf(A, work); \
  const int m = A.size(1); \
  const int n = A.size(0); \
  const int mn = (m > n ? n : m); \
  int lwork = work.shape_.Size() - mn; \
  int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_COL_MAJOR, m, n, \
                               A.dptr_, A.stride_, work.dptr_, \
                               work.dptr_ + mn, lwork)); \
  CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu."; \
}
LINALG_CPU_GEQRF(sgeqrf, float)
LINALG_CPU_GEQRF(dgeqrf, double)

#define LINALG_CPU_ORGQR(fname, DType) \
template<> inline \
void linalg_orgqr<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                              const Tensor<cpu, 1, DType>& work, \
                              Stream<cpu> *s) { \
  check_geqrf(A, work); \
  const int m = A.size(1); \
  const int n = A.size(0); \
  const int mn = (m > n ? n : m); \
  int lwork = work.shape_.Size() - mn; \
  int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_COL_MAJOR, m, mn, mn, \
                               A.dptr_, A.stride_, work.dptr_, \
                               work.dptr_ + mn, lwork)); \
  CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu."; \
}
LINALG_CPU_ORGQR(sorgqr, float)
LINALG_CPU_ORGQR(dorgqr, double)

#define LINALG_CPU_QR_WORKSPACE_QUERY(prefix, DType) \
template<> inline \
int linalg_qr_workspace_query<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                          Stream<cpu> *s) { \
  const int m(A.size(1)); \
  const int n(A.size(0)); \
  const int mn = (m > n ? n : m); \
  DType work = 0; \
  int ret(MXNET_LAPACK_##prefix##geqrf(MXNET_LAPACK_COL_MAJOR, m, \
                                       n, A.dptr_, A.stride_, &work, \
                                       &work, -1)); \
  CHECK_EQ(ret, 0) << #prefix << "geqrf: Workspace query failed on CPU."; \
  int ws_size(static_cast<int>(work)); \
  ret = MXNET_LAPACK_##prefix##orgqr(MXNET_LAPACK_COL_MAJOR, m, mn, \
                                     mn, A.dptr_, \
                                     A.stride_, &work, &work, -1); \
  CHECK_EQ(ret, 0) << #prefix << "orgqr: Workspace query failed on CPU."; \
  int wsz2(static_cast<int>(work)); \
  if (wsz2 > ws_size) ws_size = wsz2; \
  return ws_size + mn; \
}
LINALG_CPU_QR_WORKSPACE_QUERY(s, float)
LINALG_CPU_QR_WORKSPACE_QUERY(d, double)

#ifdef __CUDACC__

#define LINALG_GPU_GEQRF(fname, DType) \
template<> inline \
void linalg_geqrf<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                              const Tensor<gpu, 1, DType>& work, \
                              Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  check_geqrf(A, work); \
  const int m = A.size(1); \
  const int n = A.size(0); \
  const int mn = (m > n ? n : m); \
  int lwork(work.size(0) - mn); \
  Storage::Handle info = Storage::Get()->Alloc(sizeof(int), Context::GPU()); \
  CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s), \
                m, n, A.dptr_, A.stride_, work.dptr_, \
                work.dptr_ + mn, lwork, static_cast<int *>(info.dptr))); \
  Storage::Get()->Free(info); \
}
LINALG_GPU_GEQRF(DnSgeqrf, float)
LINALG_GPU_GEQRF(DnDgeqrf, double)

// ORGQR only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

#define LINALG_GPU_ORGQR(fname, DType) \
template<> inline \
void linalg_orgqr<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                              const Tensor<gpu, 1, DType>& work, \
                              Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  check_geqrf(A, work); \
  const int m = A.size(1); \
  const int n = A.size(0); \
  const int mn = (m > n ? n : m); \
  int lwork(work.size(0) - mn); \
  Storage::Handle info = Storage::Get()->Alloc(sizeof(int), Context::GPU()); \
  CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s), \
                m, mn, mn, A.dptr_, A.stride_, work.dptr_, \
                work.dptr_ + mn, lwork, static_cast<int *>(info.dptr))); \
  Storage::Get()->Free(info); \
}

#else

#define LINALG_GPU_ORGQR(fname, DType) \
template<> inline \
void linalg_orgqr<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                              const Tensor<gpu, 1, DType>& work, \
                              Stream<gpu> *s) { \
  LOG(FATAL) << "orgqr requires CUDA version >= 8.0!"; \
}

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_ORGQR(DnSorgqr, float)
LINALG_GPU_ORGQR(DnDorgqr, double)

// ORGQR only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

#define LINALG_GPU_QR_WORKSPACE_QUERY(prefix, DType) \
template<> inline \
int linalg_qr_workspace_query<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                          Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  const int m(A.size(1)); \
  const int n(A.size(0)); \
  const int mn = (m > n ? n : m); \
  int work1(0); \
  CUSOLVER_CALL(cusolverDn##prefix##geqrf_bufferSize(Stream<gpu>::GetSolverHandle(s), \
                m, n, A.dptr_, A.stride_, &work1)); \
  int work2(0);  \
  Storage::Handle tau = Storage::Get()->Alloc(sizeof(DType), Context::GPU()); \
  CUSOLVER_CALL(cusolverDn##prefix##orgqr_bufferSize(Stream<gpu>::GetSolverHandle(s), \
                m, mn, mn, A.dptr_, A.stride_, static_cast<DType *>(tau.dptr), &work2)); \
  Storage::Get()->Free(tau); \
  return std::max(work1, work2) + mn; \
}

#else

#define LINALG_GPU_QR_WORKSPACE_QUERY(prefix, DType) \
template<> inline \
int linalg_qr_workspace_query<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                          Stream<gpu> *s) { \
  LOG(FATAL) << "orgqr requires CUDA version >= 8.0!"; \
  return 0; \
}

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_QR_WORKSPACE_QUERY(S, float)
LINALG_GPU_QR_WORKSPACE_QUERY(D, double)

#endif  // __CUDACC__

// (Q, R) = qr(A)
// - Works on transposed A, At of shape (n, m) = Qt of shape (m, k) @ Rt of shape (k, n)
// - k = min(m, n)
// - Needs workspace (DType), size of which is determined by a workspace query

struct qr {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& a,
                 const Tensor<xpu, 3, DType>& q,
                 const Tensor<xpu, 1, DType>& work,
                 const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const mxnet::TShape& a_shape = a.shape_;
    for (index_t i = 0; i < a_shape[0]; ++i) {
      const Tensor<xpu, 2, DType>& ai = a[i];
      const Tensor<xpu, 2, DType>& qi = q[i];
      linalg_geqrf(ai, work, s);  // get R
      Copy(qi, ai, s);
      linalg_orgqr(qi, work, s);  // get Q
    }
  }
};

// Calculate the necessary workspace size
template<typename xpu>
size_t QrForwardWorkspaceSize(const TBlob& a,
                              const TBlob& q,
                              const TBlob& r,
                              const std::vector<OpReqType>& req,
                              const OpContext& ctx) {
  if (kNullOp == req[0]) { return 0U; }
  // Zero-size input, no need to launch kernel
  if (0U == a.Size()) { return 0U; }

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape& a_shape = a.shape_;
  const int a_ndim = a_shape.ndim();
  const int n = a.size(a_ndim - 1);
  const int m = a.size(a_ndim - 2);

  MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(q.type_flag_, DType, {
      size_t work_space_size = 0;
      if (m == n) {
        // For transposed input matrix a and q
        work_space_size += 2 * a.Size();
      } else if (m > n) {
        // For transposed input matrix a and q of same shape and r transpose
        work_space_size += 2 * a.Size();
        work_space_size += r.Size();
      } else {
        // For transposed input matrix a and q of same shape and q transpose
        work_space_size += 2 * a.Size();
        work_space_size += q.Size();
      }
      // For workspace size in query; done for all (m, n) shapes
      Tensor<xpu, 2, AType> a_temp_tensor = a.FlatToKD<xpu, 3, AType>(s)[0];
      if (xpu::kDevCPU) {
        std::vector<DType> A_data(a_temp_tensor.MSize(), 0);
        TBlob a_data(A_data.data(), a_temp_tensor.shape_, a.dev_mask(), a.dev_id());
        work_space_size +=
          linalg_qr_workspace_query(a_data.get<xpu, 2, DType>(s), s);
      } else {
        Storage::Handle a_handle =
          Storage::Get()->Alloc(sizeof(DType) * a_temp_tensor.shape_.Size(), Context::GPU());
        TBlob a_data(static_cast<DType*>(a_handle.dptr), a_temp_tensor.shape_,
                                         a.dev_mask(), a.dev_id());
        work_space_size +=
          linalg_qr_workspace_query(a_data.get<xpu, 2, DType>(s), s);
        Storage::Get()->Free(a_handle);
      }
      return work_space_size * sizeof(DType);
    });
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

template<typename xpu>
void QrOpForwardImpl(const TBlob& a,
                     const TBlob& q,
                     const TBlob& r,
                     const std::vector<OpReqType>& req,
                     const Tensor<xpu, 1, char>& workspace,
                     const OpContext& ctx,
                     const nnvm::NodeAttrs& attrs) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape& a_shape = a.shape_;
  const int a_ndim = a_shape.ndim();

  MSHADOW_SGL_DBL_TYPE_SWITCH(q.type_flag_, DType, {
    const int n = a.size(a_ndim - 1);
    const int m = a.size(a_ndim - 2);
    // a shape transposed
    mxnet::TShape transp_shape(a_shape);
    transp_shape[a_ndim - 1] = m;
    transp_shape[a_ndim - 2] = n;
    // Common for all (m, n) shapes
    const size_t workspace_size = (workspace.shape_.Size() + sizeof(DType) - 1) / sizeof(DType);
    DType *a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    DType *qt_ptr = a_ptr + a_shape.Size();
    TBlob a_trans_data(a_ptr, transp_shape, a.dev_mask(), a.dev_id());
    TBlob q_trans_data(qt_ptr, transp_shape, a.dev_mask(), a.dev_id());
    Tensor<xpu, 3, DType> At = a_trans_data.FlatToKD<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> Qt = q_trans_data.FlatToKD<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> R = r.FlatToKD<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> Q = q.FlatToKD<xpu, 3, DType>(s);

    MSHADOW_TYPE_SWITCH(a.type_flag_, AType, {
      // Cast type and transpose a
        mxnet_op::Kernel<QrTypeTransposeHelper, xpu>::Launch(
          s, a.Size(), a.dptr<AType>(), a_ptr, n, m, m * n);
      });

    if (m == n) {
      DType *work_ptr = qt_ptr + a_shape.Size();
      TBlob work_data(work_ptr, Shape1(workspace_size - a_shape.Size()),
                      a.dev_mask(), a.dev_id());
      Tensor<xpu, 1, DType> Work = work_data.get<xpu, 1, DType>(s);
      qr::op(At,
             Qt,
             Work, ctx, attrs);
      // Transpose into R
      mxnet_op::Kernel<QrTypeTransposeHelper, xpu>::Launch(
        s, r.Size(), At.dptr_, R.dptr_, m, m, m * m);
      // Transpose into Q
      mxnet_op::Kernel<QrTypeTransposeHelper, xpu>::Launch(
        s, q.Size(), Qt.dptr_, Q.dptr_, m, m, m * m);
      // R is triu
      mxnet_op::Kernel<ZeroTriangular, xpu>::Launch(
        s, R.MSize(), m * R.stride_, R.stride_, R.dptr_, true);
    } else if (m > n) {
      // r shape transposed
      mxnet::TShape r_shape(transp_shape);
      r_shape[a_ndim - 1] = n;
      DType *rtemp_ptr = qt_ptr + a_shape.Size();
      DType *work_ptr = rtemp_ptr + r_shape.Size();
      TBlob rtemp_data(rtemp_ptr, r_shape, a.dev_mask(), a.dev_id());
      TBlob work_data(work_ptr, Shape1(workspace_size - 2 * a_shape.Size()),
                      a.dev_mask(), a.dev_id());
      Tensor<xpu, 3, DType> Rtemp = rtemp_data.FlatToKD<xpu, 3, DType>(s);
      Tensor<xpu, 1, DType> Work = work_data.get<xpu, 1, DType>(s);
      qr::op(At,
             Qt,
             Work, ctx, attrs);
      // Final Rt of shape (N, N)
      for (index_t i = 0; i < At.size(0); ++i) {
        const Tensor<xpu, 2, DType>& Ati = At[i];
        const Tensor<xpu, 2, DType>& Rtempi = Rtemp[i];
        Tensor<xpu, 2, DType> Rk(Ati.dptr_, Shape2(n, n), Ati.stride_, s);
        Copy(Rtempi, Rk, s);
      }
      // Transpose into R of shape (N, N)
      mxnet_op::Kernel<QrTypeTransposeHelper, xpu>::Launch(
        s, r.Size(), Rtemp.dptr_, R.dptr_, n, n, n * n);
      // Transpose resulting qt(N, M) into q of shape (M, N)
      mxnet_op::Kernel<QrTypeTransposeHelper, xpu>::Launch(
        s, q.Size(), Qt.dptr_, Q.dptr_, m, n, m * n);
      // R is triu
      mxnet_op::Kernel<ZeroTriangular, xpu>::Launch(
        s, R.MSize(), n * R.stride_, R.stride_, R.dptr_, true);
    // case m < n
    } else {
      // q shape transposed
      mxnet::TShape q_shape(transp_shape);
      q_shape[a_ndim - 2] = m;
      DType *qtemp_ptr = qt_ptr + a_shape.Size();
      DType *work_ptr = qtemp_ptr + q_shape.Size();
      TBlob qtemp_data(qtemp_ptr, q_shape, a.dev_mask(), a.dev_id());
      TBlob work_data(work_ptr, Shape1(workspace_size - 2 * a_shape.Size()),
                      a.dev_mask(), a.dev_id());
      Tensor<xpu, 3, DType> Qtemp = qtemp_data.FlatToKD<xpu, 3, DType>(s);
      Tensor<xpu, 1, DType> Work = work_data.get<xpu, 1, DType>(s);
      qr::op(At,
             Qt,
             Work, ctx, attrs);
      // Transpose into R of shape (M, N)
      mxnet_op::Kernel<QrTypeTransposeHelper, xpu>::Launch(
        s, r.Size(), At.dptr_, R.dptr_, m, n, n * m);
      // Get Qt(M, M) from Qt(N, M)
      for (index_t i = 0; i < Qt.size(0); ++i) {
        const Tensor<xpu, 2, DType>& Qti = Qt[i];
        const Tensor<xpu, 2, DType>& Qtempi = Qtemp[i];
        Tensor<xpu, 2, DType> Qk(Qti.dptr_, Shape2(m, m), Qti.stride_, s);
        Copy(Qtempi, Qk, s);
      }
      // Transpose resulting qt into q of shape (M, M)
      mxnet_op::Kernel<QrTypeTransposeHelper, xpu>::Launch(
        s, q.Size(), Qtemp.dptr_, Q.dptr_, m, m, m * m);
      // R is triu
      mxnet_op::Kernel<ZeroTriangular, xpu>::Launch(
        s, R.MSize(), m * R.stride_, R.stride_, R.dptr_, true);
    }
  });
}

// (A) => (Q, R)
template<typename xpu>
void NumpyLaQrForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& a = inputs[0];
  const TBlob& q = outputs[0];
  const TBlob& r = outputs[1];

  if (kNullOp == req[0]) { return; }
  // Zero-size input, no need to launch kernel
  if (0U == a.Size()) { return; }

  // Calculate workspace size
  size_t workspace_size = QrForwardWorkspaceSize<xpu>(a, q, r, req, ctx);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  // Op
  QrOpForwardImpl<xpu>(a, q, r, req, workspace, ctx, attrs);
}

template<int req>
struct assign_helper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, DType *out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

struct qr_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& dQ,
                 const Tensor<xpu, 3, DType>& dR,
                 const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& Q,
                 const Tensor<xpu, 3, DType>& R,
                 const Tensor<xpu, 3, DType>& M,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    // Implements case m >= n; da = [dq + q@copyltu(M))]@r**(-T)
    // Where M = r@(dr**T) - (dq**T)@q
    // Reference: https://arxiv.org/abs/1710.08717
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (dQ.dptr_ != dA.dptr_) Copy(dA, dQ, s);
    // M = R@dR_T
    trmm::op(R, M, DType(1.0), false, false, false, s);
    // M = R@dR_T - dQ_T@Q
    gemm::op(dA, Q, M, DType(-1.0), DType(1.0), true, false, s);
    // M = copyltu(M)
    mxnet_op::Kernel<CopyTriangularToOppositeSide, xpu>::Launch
      (s, M.MSize(), M.size(1) * M.stride_, M.stride_, M.dptr_, false);
    // dA = dQ + Q@M
    gemm::op(Q, M, dA, DType(1.0), DType(1.0), false, false, s);
    // dA = dA@R_inv_T
    trsm::op(R, dA, DType(1.0), true, false, true, s);
  }
};

template<typename xpu>
size_t QrBackwardWorkspaceSize(const TBlob& a,
                               const TBlob& r,
                               const TBlob& grad_a) {
  if (0U == a.Size()) { return 0U; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(grad_a.type_flag_, DType, {
    size_t work_space_size = 0;
    // for grad a and M
    work_space_size += a.Size();
    work_space_size += r.Size();
    return work_space_size * sizeof(DType);
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

template<typename xpu>
void QrBackwardImpl(const TBlob& grad_a,
                    const TBlob& grad_q,
                    const TBlob& grad_r,
                    const TBlob& a,
                    const TBlob& q,
                    const TBlob& r,
                    const std::vector<OpReqType>& req,
                    const Tensor<xpu, 1, char>& workspace,
                    const OpContext& ctx,
                    const nnvm::NodeAttrs& attrs) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& r_shape = r.shape_;
  const int a_ndim = a_shape.ndim();
  const int n = a.size(a_ndim - 1);

  if (kNullOp == req[0]) { return; }

  if (0U == a_shape.Size()) { return; }

  MSHADOW_SGL_DBL_TYPE_SWITCH(grad_a.type_flag_, DType, {
    // case m >= n; Q of same shape with A and R is (n, n)
    DType *m_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    DType *grad_a_ptr = m_ptr + r_shape.Size();
    TBlob temp_m(m_ptr, r_shape, xpu::kDevMask);
    TBlob grad_a_data(grad_a_ptr, a_shape, xpu::kDevMask);
    // dR_T
    mxnet_op::Kernel<QrTypeTransposeHelper, xpu>::Launch(
      s, r_shape.Size(), grad_r.dptr<DType>(), m_ptr, n, n, n * n);

    qr_backward::op(grad_a_data.FlatToKD<xpu, 3, DType>(s),
                    grad_q.FlatToKD<xpu, 3, DType>(s),
                    grad_r.FlatToKD<xpu, 3, DType>(s),
                    a.FlatToKD<xpu, 3, DType>(s),
                    q.FlatToKD<xpu, 3, DType>(s),
                    r.FlatToKD<xpu, 3, DType>(s),
                    temp_m.FlatToKD<xpu, 3, DType>(s),
                    ctx, attrs);

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
    mxnet_op::Kernel<assign_helper<req_type>, xpu>::Launch(
      s, a_shape.Size(), grad_a_data.dptr<DType>(), grad_a.dptr<DType>());
    });
  });
}

// (dQ, dR, A, Q, R) => (dA)
template<typename xpu>
void NumpyLaQrBackward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 5U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const TBlob& grad_q = inputs[0];
  const TBlob& grad_r = inputs[1];
  const TBlob& a = inputs[2];
  const TBlob& q = inputs[3];
  const TBlob& r = inputs[4];
  const TBlob& grad_a = outputs[0];
  const int a_ndim = a.shape_.ndim();
  const int n = a.size(a_ndim - 1);
  const int m = a.size(a_ndim - 2);

  CHECK_LE(n, m)
    << "QrBackward not implemented when ncols > nrows";

  size_t workspace_size = QrBackwardWorkspaceSize<xpu>(a, r, grad_a);
  Tensor<xpu, 1, char> workspace = ctx.requested[0]
    .get_space_typed<xpu, 1, char>(Shape1(workspace_size), ctx.get_stream<xpu>());
  QrBackwardImpl<xpu>(grad_a, grad_q, grad_r, a, q, r, req, workspace, ctx, attrs);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_QR_INL_H_
