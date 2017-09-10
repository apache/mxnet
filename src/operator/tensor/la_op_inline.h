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
 * \file la_op_inline.h
 * \brief Operators for advanced linear algebra.
 */
#ifndef MXNET_OPERATOR_TENSOR_LA_OP_INLINE_H_
#define MXNET_OPERATOR_TENSOR_LA_OP_INLINE_H_

#include "../linalg.h"

namespace mxnet {
namespace op {

using namespace mshadow;

// Helper functions.
struct CopyLowerToUpper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int matrix_size, int stride, DType* data) {
    // Below computation works even when we are dealing with a batch of matrices.
    const int row((i % matrix_size) / stride), col(i % stride);
    if ( row > col ) data[i + (col - row) * (stride - 1)] = data[i];
  }
};
struct ZeroUpper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int matrix_size, int stride, DType* data) {
    const int row((i % matrix_size) / stride), col(i % stride);
    if ( row < col ) data[i] = 0;
  }
};
struct Scale {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType scale, DType* data) {
    data[i] *= scale;
  }
};

// Forward computations (always using batched processing)
// CHANGE: Added xyz::op(..., ctx, attrs), which calls xyz::op(..., s, attrs)

// D = gemm(A,B,C)
struct gemm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, DType alpha, DType beta,
                 bool tA, bool tB, Stream<xpu> *s) {
    linalg_batch_gemm(A, B, C, alpha, beta, tA, tB, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, const Tensor<xpu, 3, DType>& D,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    if ( C.dptr_ != D.dptr_ ) Copy(D, C, s);
    const LaMatrixMacParam& param = nnvm::get<LaMatrixMacParam>(attrs.parsed);
    op(A, B, D, DType(param.alpha), DType(param.beta), param.transpose_a,
       param.transpose_b, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, const Tensor<xpu, 3, DType>& D,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, C, D, s, attrs);
  }
};

// C = gemm2(A,B)
struct gemm2 {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, Stream<xpu> *s,
                 const nnvm::NodeAttrs& attrs) {
    const LaMatrixMultParam& param = nnvm::get<LaMatrixMultParam>(attrs.parsed);
    gemm::op(A, B, C, DType(param.alpha), DType(0), param.transpose_a,
             param.transpose_b, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, C, s, attrs);
  }
};

// L = potrf(A).
struct potrf {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& L,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    if ( A.dptr_ != L.dptr_ ) Copy(L, A, s);
    linalg_batch_potrf(L, true, s);
    using namespace mxnet_op;
    Kernel<ZeroUpper, xpu>::Launch(s, L.MSize(), L.size(1)*L.stride_, L.stride_, L.dptr_);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& L,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, L, s, attrs);
  }
};

// A = potri(L).
struct potri {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& L, const Tensor<xpu, 3, DType>& A,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    if ( A.dptr_ != L.dptr_ ) Copy(A, L, s);
    linalg_batch_potri(A, true, s);
    using namespace mxnet_op;
    Kernel<CopyLowerToUpper, xpu>::Launch(s, A.MSize(), A.size(1)*A.stride_, A.stride_, A.dptr_);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& L, const Tensor<xpu, 3, DType>& A,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(L, A, s, attrs);
  }
};

// B = trsm(L,A)
struct trsm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& L, const Tensor<xpu, 3, DType>& B,
                 DType alpha, bool rightside, bool transpose, Stream<xpu> *s) {
    linalg_batch_trsm(L, B, alpha, rightside, true, transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& L, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    if ( A.dptr_ != B.dptr_ ) Copy(B, A, s);
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    op(L, B, DType(param.alpha), param.rightside, param.transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& L, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(L, A, B, s, attrs);
  }
};

// B = trmm(L,A)
struct trmm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& L, const Tensor<xpu, 3, DType>& B,
                 DType alpha, bool rightside, bool transpose, Stream<xpu> *s) {
    linalg_batch_trmm(L, B, alpha, rightside, true, transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& L, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, Stream<xpu> *s,
                 const nnvm::NodeAttrs& attrs) {
    if ( A.dptr_ != B.dptr_ ) Copy(B, A, s);
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    op(L, B, DType(param.alpha), param.rightside, param.transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& L, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(L, A, B, s, attrs);
  }
};

// Useful operator that is not part of BLAS/LAPACK.
struct ForwardSumLogDiag {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int N, int stride, DType* A, DType* B) {
    DType sum(0);
    const int offset(i * N * stride);
    for ( int j = 0; j < N; ++j ) {
      sum += log(A[offset+j*(stride+1)]);
    }
    B[i] = sum;
  }
};
struct sumlogdiag {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 1, DType>& B,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    CHECK_EQ(A.size(1), A.size(2)) << "sumlogdiag operator requires square matrices as input.";
    using namespace mxnet_op;
    Kernel<ForwardSumLogDiag, xpu>::Launch(s, A.size(0), A.size(1), A.stride_, A.dptr_, B.dptr_);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 1, DType>& B,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, s, attrs);
  }
};

// B = syrk(A)
struct syrk {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 DType alpha, DType beta, bool tA, Stream<xpu> *s) {
    linalg_batch_syrk(A, B, alpha, beta, tA, s);
    // Symmetric B is in lower triangle: Copy to upper
    using namespace mxnet_op;
    Kernel<CopyLowerToUpper, xpu>::Launch(s, B.MSize(), B.size(1)*B.stride_,
                                          B.stride_, B.dptr_);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    const LaSyrkParam& param = nnvm::get<LaSyrkParam>(attrs.parsed);
    op(A, B, DType(param.alpha), DType(0), param.transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, s, attrs);
  }
};

// (Q, L) = gelqf(A) [LQ factorization]
// More complex than the other cases:
// - Has to reserve workspace, whose size can only be determined by workspace
//   queries. This is done once, and then the workspace is used for all items
//   of the batch
// - Two different LAPACK functions are called (the first, gelqf, returns an
//   internal representation, which has to be converted into Q, L)
struct gelqf {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& Q,
                 const Tensor<xpu, 3, DType>& L, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (A.dptr_ != Q.dptr_) Copy(Q, A, s);
    // From here on, we work on Q only
    // Reserve workspace
    // The size is determined by workspace queries, done on the first items
    // of the batch
    int ws_size(linalg_gelqf_workspace_query(Q[0], s));
    Tensor<xpu, 1, DType> work = ctx.requested[0]
      .get_space_typed<xpu, 1, DType>(Shape1(ws_size), s);
    // Loop over items in batch
    linalg_check_batch_size(A.size(0), Q.size(0), L.size(0));
    int m = Q.size(1);  // Q[i] has shape (m, n)
    for (index_t i = 0; i < A.size(0); ++i) {
      const Tensor<xpu, 2, DType>& Qi = Q[i];
      const Tensor<xpu, 2, DType>& Li = L[i];
      // Call gelqf: Overwrites Qi and part of work. Afterwards, L matrix is
      // in lower triangle of Qi
      linalg_gelqf(Qi, work, s);
      // Copy lower triangle & diagonal of Qi ==> Li.
      // Also, zero the upper triangle.
      // QLeft: First m columns of Qi
      Tensor<xpu, 2, DType> QLeft(Qi.dptr_, Shape2(m, m), Qi.stride_, s);
      Copy(Li, QLeft, s);
      using namespace mxnet_op;
      Kernel<ZeroUpper, xpu>::Launch(s, Li.MSize(), m*Li.stride_, Li.stride_,
                                     Li.dptr_);
      // Call orglq: Input is Qi and part of work. Overwrites Qi by final Q
      // matrix (conversion from internal representation)
      linalg_orglq(Qi, work, s);
    }
  }
};

// Backward operators (always using batch processing)

struct gemm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dD, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& C,
                 const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& dB,
                 const Tensor<xpu, 3, DType>& dC,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    const LaMatrixMacParam& param = nnvm::get<LaMatrixMacParam>(attrs.parsed);
    bool tA(param.transpose_a), tB(param.transpose_b);
    (tA ? gemm::op(B, dD, dA, DType(param.alpha), DType(0), tB, true, s)
        : gemm::op(dD, B, dA, DType(param.alpha), DType(0), false, !tB, s));
    (tB ? gemm::op(dD, A, dB, DType(param.alpha), DType(0), true, tA, s)
        : gemm::op(A, dD, dB, DType(param.alpha), DType(0), !tA, false, s));
    Copy(dC, dD, s);
    using namespace mxnet_op;
    Kernel<Scale, xpu>::Launch(s, dC.MSize(), DType(param.beta), dC.dptr_);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dD, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& C,
                 const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& dB,
                 const Tensor<xpu, 3, DType>& dC,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dD, A, B, C, dA, dB, dC, s, attrs);
  }
};

struct gemm2_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dC, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& dB,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    const LaMatrixMultParam& param = nnvm::get<LaMatrixMultParam>(attrs.parsed);
    bool tA(param.transpose_a), tB(param.transpose_b);
    (tA ? gemm::op(B, dC, dA, DType(param.alpha), DType(0), tB, true, s)
        : gemm::op(dC, B, dA, DType(param.alpha), DType(0), false, !tB, s));
    (tB ? gemm::op(dC, A, dB, DType(param.alpha), DType(0), true, tA, s)
        : gemm::op(A, dC, dB, DType(param.alpha), DType(0), !tA, false, s));
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dC, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& dB,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dC, A, B, dA, dB, s, attrs);
  }
};

struct potrf_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dL, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& dA,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of L = potrf(A).
    //   dA = 0.5 * L**T * copyLTU(L**T * dL) * L**(-1)
    // Here, copyLTU(M) creates a symmetric matrix from the square matrix M
    // by setting the upper triangle to be equal to the lower triangle, leaving
    // lower triangle and diagonal unchanged.
    if ( dL.dptr_ != dA.dptr_ ) {
      Copy(dA, dL, s);
    }
    trmm::op(L, dA, DType(1.0), false, true, s);
    using namespace mxnet_op;
    Kernel<CopyLowerToUpper, xpu>::Launch
           (s, dA.MSize(), dA.size(1)*dA.stride_, dA.stride_, dA.dptr_);
    trsm::op(L, dA, DType(1.0), false, true, s);
    trsm::op(L, dA, DType(0.5), true, false, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dL, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& dA,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dL, L, dA, s, attrs);
  }
};

struct potri_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& dL,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of A = potri(L).
    // dL = -tril( A * (dA + dA**T) * L**(-T)), where tril() extracts lower triangle
    // and diagonal. We must not assume that dA is symmetric.
    // Note: Calling gemm twice here is a bit wasteful, but otherwise the symmetrization
    // of dA would require temporary memory.
    gemm::op(A, dA, dL, DType(1.), DType(0.), false, false, s);
    gemm::op(A, dA, dL, DType(1.), DType(1.), false, true, s);
    trsm::op(L, dL, DType(-1.), true, true, s);
    using namespace mxnet_op;
    Kernel<ZeroUpper, xpu>::Launch(s, dL.MSize(), dL.size(1)*dL.stride_, dL.stride_,
                                   dL.dptr_);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& dL,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dA, L, A, dL, s, attrs);
  }
};

struct trsm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& dL, const Tensor<xpu, 3, DType>& dA,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of B = trsm(L,A).
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    // Compute dA
    if ( dA.dptr_ != dB.dptr_ ) Copy(dA, dB, s);
    trsm::op(L, dA, DType(param.alpha), param.rightside, !param.transpose, s);
    // Compute dL
    const bool da_left(param.rightside == param.transpose);
    DType scale(-1.0/param.alpha);
    (da_left ? gemm::op(dA, B, dL, scale, DType(0), param.transpose, !param.transpose, s)
             : gemm::op(B, dA, dL, scale, DType(0), !param.transpose, param.transpose, s));
    using namespace mxnet_op;
    Kernel<ZeroUpper, xpu>::Launch(s, dL.MSize(), dL.size(1)*dL.stride_, dL.stride_, dL.dptr_);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& dL, const Tensor<xpu, 3, DType>& dA,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dB, L, A, B, dL, dA, s, attrs);
  }
};

struct trmm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& dL,
                 const Tensor<xpu, 3, DType>& dA, Stream<xpu>* s,
                 const nnvm::NodeAttrs& attrs) {
    // Backward of B = trmm(L,A).
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    // Compute dL
    DType scale(param.alpha);
    if (param.rightside == param.transpose) {
      gemm::op(dB, A, dL, scale, DType(0.), param.transpose, !param.transpose, s);
    } else {
      gemm::op(A, dB, dL, scale, DType(0.), !param.transpose, param.transpose, s);
    }
    using namespace mxnet_op;
    Kernel<ZeroUpper, xpu>::Launch(s, dL.MSize(), dL.size(1)*dL.stride_, dL.stride_,
                                   dL.dptr_);
    // Compute dA
    if (dA.dptr_ != dB.dptr_) Copy(dA, dB, s);
    trmm::op(L, dA, scale, param.rightside, !param.transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& dL,
                 const Tensor<xpu, 3, DType>& dA, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dB, L, A, dL, dA, s, attrs);
  }
};

struct BackwardSumLogDiag {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int N, int stride, DType* dB, DType* A, DType* dA) {
    const int offset(i * N * stride);
    for ( int j = 0; j < N; ++j ) {
      dA[offset+j*(stride+1)] = dB[i]/A[offset+j*(stride+1)];
    }
  }
};
struct sumlogdiag_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& dA,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of B = sumlogdiag(A).
    // dB is actually a 1-d tensor but we convert it to a 3-D one before calling
    // this function as the LaOpCaller-adapters can only deal with a uniform
    // dimension for all tensor inputs. This doesn't matter as we will interpret
    // it correctly internally in this function.
    using namespace mxnet_op;
    Kernel<Scale, xpu>::Launch(s, dA.MSize(), DType(0), dA.dptr_);
    Kernel<BackwardSumLogDiag, xpu>::Launch
         (s, A.size(0), A.size(1), A.stride_, dB.dptr_, A.dptr_, dA.dptr_);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& dA,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dB, A, dA, s, attrs);
  }
};

struct syrk_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& dA, Stream<xpu>* s,
                 const nnvm::NodeAttrs& attrs) {
    const LaSyrkParam& param = nnvm::get<LaSyrkParam>(attrs.parsed);
    // Note: Calling gemm twice is a bit wasteful, but the symmetrization of dB
    // would otherwise need temporary memory
    if (param.transpose) {
      gemm::op(A, dB, dA, DType(param.alpha), DType(0.), false, false, s);
      gemm::op(A, dB, dA, DType(param.alpha), DType(1.), false, true, s);
    } else {
      gemm::op(dB, A, dA, DType(param.alpha), DType(0.), false, false, s);
      gemm::op(dB, A, dA, DType(param.alpha), DType(1.), true, false, s);
    }
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& dA, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dB, A, dA, s, attrs);
  }
};

// Have to reserve temporary storage tempM, same shape as dL
struct gelqf_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dQ,
                 const Tensor<xpu, 3, DType>& dL,
                 const Tensor<xpu, 3, DType>& Q,
                 const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& dA,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    // Backward of (Q, L) = gelqf(A):
    //   dA = L**(-T) * (dQ + copyLTU(M) * Q), M = L**T * dL - dQ * Q**T
    // Here, copyLTU(M) creates a symmetric matrix from the square matrix M
    // by setting the upper triangle to be equal to the lower triangle, leaving
    // lower triangle and diagonal unchanged.
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (dQ.dptr_ != dA.dptr_) Copy(dA, dQ, s);
    // Need temporal space, same shape as dL
    Tensor<xpu, 3, DType> tempM = ctx.requested[0]
      .get_space_typed<xpu, 3, DType>(dL.shape_, s);
    Copy(tempM, dL, s);
    trmm::op(L, tempM, DType(1.0), false, true, s);
    gemm::op(dA, Q, tempM, DType(-1.0), DType(1.0), false, true, s);
    Kernel<CopyLowerToUpper, xpu>::Launch
           (s, tempM.MSize(), tempM.size(1)*tempM.stride_, tempM.stride_,
            tempM.dptr_);
    gemm::op(tempM, Q, dA, DType(1.0), DType(1.0), false, false, s);
    trsm::op(L, dA, DType(1.0), false, true, s);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_LA_OP_INLINE_H_
