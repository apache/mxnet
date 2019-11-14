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
 * Copyright (c) 2017 by Contributors
 * \file la_op-inl.h
 * \brief Operators for advanced linear algebra.
 * \note  See https://arxiv.org/pdf/1710.08717.pdf for details of gradient computations.
 */
#ifndef MXNET_OPERATOR_TENSOR_LA_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_LA_OP_INL_H_

#include "../linalg.h"

namespace mxnet {
namespace op {

using namespace mshadow;

// Copies lower/upper triangular part to upper/lower, i.e. to the opposite side.
struct CopyTriangularToOppositeSide {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int matrix_size, int stride, DType* data, bool to_lower) {
    // Below computation works even when we are dealing with a batch of matrices.
    const int row((i % matrix_size) / stride), col(i % stride);
    if (row > col) {
       if (to_lower) {
         data[i] = data[i + (col - row) * (stride - 1)];
       } else {
         data[i + (col - row) * (stride - 1)] = data[i];
       }
    }
  }
};

// Zero's lower/upper triangular part of a matrix.
struct ZeroTriangular {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int matrix_size, int stride, DType* data,
                                  bool zero_lower) {
    const int row((i % matrix_size) / stride), col(i % stride);
    if ((!zero_lower && (row < col)) || (zero_lower && (row > col))) data[i] = 0;
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
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& A, const Tensor<xpu, dim, DType>& B,
                 const Tensor<xpu, dim, DType>& C, DType alpha, DType beta,
                 bool tA, bool tB, Stream<xpu> *s) {
    linalg_batch_gemm(A, B, C, alpha, beta, tA, tB, s);
  }
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& A, const Tensor<xpu, dim, DType>& B,
                 const Tensor<xpu, dim, DType>& C, const Tensor<xpu, dim, DType>& D,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    if ( C.dptr_ != D.dptr_ ) Copy(D, C, s);
    const LaMatrixMacParam& param = nnvm::get<LaMatrixMacParam>(attrs.parsed);
    op(A, B, D, DType(param.alpha), DType(param.beta), param.transpose_a,
       param.transpose_b, s);
  }
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& A, const Tensor<xpu, dim, DType>& B,
                 const Tensor<xpu, dim, DType>& C, const Tensor<xpu, dim, DType>& D,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, C, D, s, attrs);
  }
};

// C = gemm2(A,B)
struct gemm2 {
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& A, const Tensor<xpu, dim, DType>& B,
                 const Tensor<xpu, dim, DType>& C, DType alpha, bool tA, bool tB,
                 Stream<xpu> *s) {
    gemm::op(A, B, C, DType(alpha), DType(0), tA, tB, s);
  }
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& A, const Tensor<xpu, dim, DType>& B,
                 const Tensor<xpu, dim, DType>& C, Stream<xpu> *s,
                 const nnvm::NodeAttrs& attrs) {
    const LaMatrixMultParam& param = nnvm::get<LaMatrixMultParam>(attrs.parsed);
    op(A, B, C, DType(param.alpha), param.transpose_a, param.transpose_b, s);
  }
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& A, const Tensor<xpu, dim, DType>& B,
                 const Tensor<xpu, dim, DType>& C, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, C, s, attrs);
  }
};

// B = potrf(A).
struct potrf {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    const LaCholeskyParam& param = nnvm::get<LaCholeskyParam>(attrs.parsed);
    if (A.shape_.Size() == 0U) {
      return;
    }
    if ( A.dptr_ != B.dptr_ ) Copy(B, A, s);
    linalg_batch_potrf(B, param.lower, s);
    using namespace mxnet_op;
    Kernel<ZeroTriangular, xpu>::Launch(s, B.MSize(), B.size(1)*B.stride_, B.stride_,
                                   B.dptr_, !param.lower);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, s, attrs);
  }
};

// A = potri(B).
struct potri {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& A,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    const LaCholeskyParam& param = nnvm::get<LaCholeskyParam>(attrs.parsed);
    if ( A.dptr_ != B.dptr_ ) Copy(A, B, s);
    linalg_batch_potri(A, param.lower, s);
    using namespace mxnet_op;
    Kernel<CopyTriangularToOppositeSide, xpu>::Launch(s, A.MSize(), A.size(1)*A.stride_, A.stride_,
                                          A.dptr_, !param.lower);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& A,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(B, A, s, attrs);
  }
};

// C = trsm(A,B)
struct trsm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& C,
                 DType alpha, bool rightside, bool lower, bool transpose, Stream<xpu> *s) {
    linalg_batch_trsm(A, C, alpha, rightside, lower, transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    if ( B.dptr_ != C.dptr_ ) Copy(C, B, s);
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    op(A, C, DType(param.alpha), param.rightside, param.lower, param.transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, C, s, attrs);
  }
};

// C = trmm(A,B)
struct trmm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& C,
                 DType alpha, bool rightside, bool lower, bool transpose, Stream<xpu> *s) {
    linalg_batch_trmm(A, C, alpha, rightside, lower, transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, Stream<xpu> *s,
                 const nnvm::NodeAttrs& attrs) {
    if ( B.dptr_ != C.dptr_ ) Copy(C, B, s);
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    op(A, C, DType(param.alpha), param.rightside, param.lower, param.transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(A, B, C, s, attrs);
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

template<bool forward>
struct CopyDiag {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int k, int n, DType* A, DType* B) {
    // Index of the matrix from which the diagonal should be extracted.
    const int matrix(i / (n-abs(k)));
    // Index of the diagonal element that should be extracted.
    const int index(i % (n-abs(k)));
    // row/col that must be looked up.
    const int row(index-(k < 0 ? k : 0)), col(index+(k > 0 ? k :0));
    if (forward) {
      B[i] = A[(matrix*n+row)*n+col];
    } else {
      B[(matrix*n+row)*n+col] = A[i];
    }
  }
};

struct copydiag {
  // Extracts diagonal from matrix.
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 2, DType>& B,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const LaDiagParam& param = nnvm::get<LaDiagParam>(attrs.parsed);
    Kernel<CopyDiag<true>, xpu>::Launch(s, B.MSize(), param.offset, A.size(1), A.dptr_, B.dptr_);
  }
  // Sets diagonal in matrix.
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const LaDiagParam& param = nnvm::get<LaDiagParam>(attrs.parsed);
    Kernel<set_zero, xpu>::Launch(s, B.MSize(), B.dptr_);
    Kernel<CopyDiag<false>, xpu>::Launch(s, A.MSize(), param.offset, B.size(1), A.dptr_, B.dptr_);
  }
};

template<bool forward>
struct CopyTrian {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, bool lower, int k, int n, DType* A, DType* B) {
    // Matrix that this index belongs to.
    const int matrix(i/(n*n));
    // Row/Col that this index represents.
    int row((i/n)%n), col(i%n);
    if ((k > 0) || ((k == 0) && !lower)) {
       // When working on upper triangle we switch to transposed coordinates for indexing.
       int tmp(row);
       row = col;
       col = tmp;
    }
    // Actual row inside the lower triangular matrix after offset adjustment.
    row -= abs(k);
    if (row >= col) {
      // Index in the 1-dimensional array that holds the values of the triangle.
      const int index((row*(row+1))/2+col);
      // Total number of entries in the triangle.
      const int m(((n-abs(k))*(n-abs(k)+1))/2);
      if (forward) {
        B[m*matrix+index] = A[i];
      } else {
        B[i] = A[m*matrix+index];
      }
    }
  }
};

struct copytrian {
  // Extracts triangle from matrix.
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 2, DType>& B,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const LaTrianParam& param = nnvm::get<LaTrianParam>(attrs.parsed);
    Kernel<CopyTrian<true>, xpu>::Launch(s, A.MSize(), param.lower, param.offset,
                                         A.size(1), A.dptr_, B.dptr_);
  }
  // Sets triangle in matrix.
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const LaTrianParam& param = nnvm::get<LaTrianParam>(attrs.parsed);
    Kernel<set_zero, xpu>::Launch(s, B.MSize(), B.dptr_);
    Kernel<CopyTrian<false>, xpu>::Launch(s, B.MSize(), param.lower, param.offset,
                                          B.size(1), A.dptr_, B.dptr_);
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
    Kernel<CopyTriangularToOppositeSide, xpu>::Launch(s, B.MSize(), B.size(1)*B.stride_,
                                          B.stride_, B.dptr_, false);
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
      Kernel<ZeroTriangular, xpu>::Launch(s, Li.MSize(), m*Li.stride_, Li.stride_,
                                     Li.dptr_, false);
      // Call orglq: Input is Qi and part of work. Overwrites Qi by final Q
      // matrix (conversion from internal representation)
      linalg_orglq(Qi, work, s);
    }
  }
};

// If (U, L) = syevd(A) [symmetric eigendecomposition], this helper acts on each row
// of U, deciding whether its sign is flipped or not.
// If u denotes a row, we choose the sign s.t. u_k > 0, where k = argmax|u_j|. In case
// of a tie, the smaller index k decides.
struct SyevdEigenVecSigns {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int n, DType* U, int ldu) {
    DType* urow(U + (i*ldu));
    DType maxval(fabs(urow[0])), uval(0.0);
    int maxind(0);
    for (int i = 1; i < n; ++i) {
      uval = fabs(urow[i]);
      if (uval > maxval) {
        maxval = uval;
        maxind = i;
      }
    }
    if (urow[maxind] < 0.0) {
      // Flip all signs
      for (int i = 0; i < n; ++i) {
        urow[i] = -urow[i];
      }
    }
  }
};

// (U, L) = syevd(A) [symmetric eigendecomposition]
// - Input A must be symmetric, only lower triangle is used
// - U can overwrite A
// - Needs workspace (both DType and int), size of which is determined by a
//   workspace query
struct syevd {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& U,
                 const Tensor<xpu, 2, DType>& L, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    linalg_check_batch_size(A.size(0), U.size(0), L.size(0));
    if (A.dptr_ != U.dptr_) Copy(U, A, s);
    // From here on, we work on U only
    // Reserve workspace (size determined by query)
    int lwork(linalg_syevd_workspace_query(U[0], L[0], s));
    Tensor<xpu, 1, DType> work = ctx.requested[0]
      .get_space_typed<xpu, 1, DType>(Shape1(lwork), s);
    // Loop over items in batch
    for (index_t i = 0; i < U.size(0); ++i) {
      linalg_syevd(U[i], L[i], work, s);
    }
    // Set signs of eigenvectors in a deterministic way
    using namespace mxnet_op;
    Kernel<SyevdEigenVecSigns, xpu>::Launch
      (s, U.size(0)*U.size(1), U.size(1), U.dptr_, U.stride_);
  }
};

// A = inverse(B).
struct inverse {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& A,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    // Since inverse(A) = trans(inverse(trans(A))), so we don't need to transpose
    // A even if we are using the col-major version of getrf and getri routines.
    if (B.shape_.Size() == 0U) {
      return;
    }
    linalg_batch_inverse(A, B, ctx);
  }
};

// this kernel computes sign(det(A)), log(abs(det(A))) from LU decomposition
struct SignedLogDet {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int N, int* pivot,
                                  DType *LU, DType* sign, DType *logdet) {
    int changes(0);
    DType diag_sign(1);
    DType diag_logsum(0);
    int *pivot_mat = pivot + i * N;
    DType *LU_mat = LU + i * N * N;
    for (int j = 0; j < N; ++j) {
      changes += (pivot_mat[j] != (j + 1));
      DType diag = LU_mat[j * (N + 1)];
      diag_sign *= ((DType(0) < diag) - (diag < DType(0)));
      diag_logsum += std::log(std::abs(diag));
    }
    sign[i] = (changes % 2 == 1 ? DType(-1) : DType(1)) * diag_sign;
    logdet[i] = diag_logsum;
  }
};

// det = det(A), the computation method is based on partial pivoting LU decomposition:
//     A = PLU, so det(A) = det(P) * det(L) * det(U),
//     det(P) depends on number of row changes in P
//     det(L) = 1 since L has unit diagnal elemements
//     det(U) = prod(diag(U))
// LU and pivot store the LU decomposition output which will be used in computing gradient
struct det {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 1, DType>& det,
                 const Tensor<xpu, 3, DType>& LU, const Tensor<xpu, 2, int>& pivot,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    if (A.shape_.Size() == 0U) {
      return;
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1, DType> sign = ctx.requested[0]
      .get_space_typed<xpu, 1, DType>(det.shape_, s);
    Copy(LU, A, s);
    // since det(A) = det(trans(A)), so we'll use col-major blas routines here
    linalg_batch_getrf(LU, pivot, false, s);
    using namespace mxnet_op;
    using namespace mshadow::expr;
    Kernel<SignedLogDet, xpu>::Launch(s, pivot.size(0), pivot.size(1), pivot.dptr_,
                                      LU.dptr_, sign.dptr_, det.dptr_);
    const_cast<Tensor<xpu, 1, DType>&>(det) = sign * F<mshadow_op::exp>(det);
  }
};

// sign = sign(det(A))
// logabsdet = log(abs(det(A)))
struct slogdet {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 1, DType>& sign,
                 const Tensor<xpu, 1, DType>& logabsdet, const Tensor<xpu, 3, DType>& LU,
                 const Tensor<xpu, 2, int>& pivot, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    if (A.shape_.Size() == 0U) {
      return;
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Copy(LU, A, s);
    linalg_batch_getrf(LU, pivot, false, s);
    using namespace mxnet_op;
    using namespace mshadow::expr;
    Kernel<SignedLogDet, xpu>::Launch(s, pivot.size(0), pivot.size(1), pivot.dptr_,
                                      LU.dptr_, sign.dptr_, logabsdet.dptr_);
  }
};

// Backward operators (always using batch processing)

struct gemm_backward {
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& dD, const Tensor<xpu, dim, DType>& A,
                 const Tensor<xpu, dim, DType>& B, const Tensor<xpu, dim, DType>& C,
                 const Tensor<xpu, dim, DType>& dA, const Tensor<xpu, dim, DType>& dB,
                 const Tensor<xpu, dim, DType>& dC,
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
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& dD, const Tensor<xpu, dim, DType>& A,
                 const Tensor<xpu, dim, DType>& B, const Tensor<xpu, dim, DType>& C,
                 const Tensor<xpu, dim, DType>& dA, const Tensor<xpu, dim, DType>& dB,
                 const Tensor<xpu, dim, DType>& dC,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dD, A, B, C, dA, dB, dC, s, attrs);
  }
};

struct gemm2_backward {
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& dC, const Tensor<xpu, dim, DType>& A,
                 const Tensor<xpu, dim, DType>& B, const Tensor<xpu, dim, DType>& dA,
                 const Tensor<xpu, dim, DType>& dB,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    const LaMatrixMultParam& param = nnvm::get<LaMatrixMultParam>(attrs.parsed);
    bool tA(param.transpose_a), tB(param.transpose_b);
    (tA ? gemm::op(B, dC, dA, DType(param.alpha), DType(0), tB, true, s)
        : gemm::op(dC, B, dA, DType(param.alpha), DType(0), false, !tB, s));
    (tB ? gemm::op(dC, A, dB, DType(param.alpha), DType(0), true, tA, s)
        : gemm::op(A, dC, dB, DType(param.alpha), DType(0), !tA, false, s));
  }
  template<typename xpu, int dim, typename DType>
  static void op(const Tensor<xpu, dim, DType>& dC, const Tensor<xpu, dim, DType>& A,
                 const Tensor<xpu, dim, DType>& B, const Tensor<xpu, dim, DType>& dA,
                 const Tensor<xpu, dim, DType>& dB,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dC, A, B, dA, dB, s, attrs);
  }
};

struct potrf_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& dA,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of B = potrf(A).
    //   dA = 0.5 * B**(-T) * copyLTU(B**T * dB) * B**(-1)
    // Here, copyLTU(M) creates a symmetric matrix from the square matrix M
    // by setting the upper triangle to be equal to the lower triangle, leaving
    // lower triangle and diagonal unchanged.
    // The function also handles the case when B is upper triangular by appropriate
    // transpositions.
    const LaCholeskyParam& param = nnvm::get<LaCholeskyParam>(attrs.parsed);
    if (dA.shape_.Size() == 0U) {
      return;
    }
    if ( dB.dptr_ != dA.dptr_ ) {
      Copy(dA, dB, s);
    }
    trmm::op(B, dA, DType(1.0), !param.lower, param.lower, true, s);
    using namespace mxnet_op;
    Kernel<CopyTriangularToOppositeSide, xpu>::Launch
           (s, dA.MSize(), dA.size(1)*dA.stride_, dA.stride_, dA.dptr_, !param.lower);
    trsm::op(B, dA, DType(1.0), false, param.lower, param.lower, s);
    trsm::op(B, dA, DType(0.5), true, param.lower, !param.lower, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& dA,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dB, B, dA, s, attrs);
  }
};

struct potri_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& dB,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of A = potri(B).
    // dB = -tril( A * (dA + dA**T) * B**(-T)), where tril() extracts lower triangle
    // and diagonal. We must not assume that dA is symmetric.
    // The function also handles the case when B is upper triangular by appropriate
    // transpositions.
    // Note: Calling gemm twice here is a bit wasteful, but otherwise the symmetrization
    // of dA would require temporary memory.
    const LaCholeskyParam& param = nnvm::get<LaCholeskyParam>(attrs.parsed);
    if (param.lower) {
      gemm::op(A, dA, dB, DType(1.), DType(0.), false, false, s);
      gemm::op(A, dA, dB, DType(1.), DType(1.), false, true, s);
    } else {
      gemm::op(dA, A, dB, DType(1.), DType(0.), false, false, s);
      gemm::op(dA, A, dB, DType(1.), DType(1.), true, false, s);
    }
    trsm::op(B, dB, DType(-1.), param.lower, param.lower, true, s);
    using namespace mxnet_op;
    Kernel<ZeroTriangular, xpu>::Launch(s, dB.MSize(), dB.size(1)*dB.stride_, dB.stride_,
                                   dB.dptr_, !param.lower);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& dB,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dA, B, A, dB, s, attrs);
  }
};

struct trsm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dC, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& C,
                 const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& dB,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of C = trsm(A,B).
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    // Compute dB
    if ( dB.dptr_ != dC.dptr_ ) Copy(dB, dC, s);
    trsm::op(A, dB, DType(param.alpha), param.rightside, param.lower, !param.transpose, s);
    // Compute dA
    const bool da_left(param.rightside == param.transpose);
    DType scale(-1.0/param.alpha);
    (da_left ? gemm::op(dB, C, dA, scale, DType(0), param.transpose, !param.transpose, s)
             : gemm::op(C, dB, dA, scale, DType(0), !param.transpose, param.transpose, s));
    using namespace mxnet_op;
    Kernel<ZeroTriangular, xpu>::Launch(s, dA.MSize(), dA.size(1)*dA.stride_, dA.stride_,
                                   dA.dptr_, !param.lower);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dC, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& C,
                 const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& dB,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dC, A, B, C, dA, dB, s, attrs);
  }
};

struct trmm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dC, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& dB, Stream<xpu>* s,
                 const nnvm::NodeAttrs& attrs) {
    // Backward of C = trmm(A,B).
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    // Compute dA
    DType scale(param.alpha);
    if (param.rightside == param.transpose) {
      gemm::op(dC, B, dA, scale, DType(0.), param.transpose, !param.transpose, s);
    } else {
      gemm::op(B, dC, dA, scale, DType(0.), !param.transpose, param.transpose, s);
    }
    using namespace mxnet_op;
    Kernel<ZeroTriangular, xpu>::Launch(s, dA.MSize(), dA.size(1)*dA.stride_, dA.stride_,
                                   dA.dptr_, !param.lower);
    // Compute dB
    if (dB.dptr_ != dC.dptr_) Copy(dB, dC, s);
    trmm::op(A, dB, scale, param.rightside, param.lower, !param.transpose, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dC, const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& B, const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& dB, const OpContext& ctx,
                 const nnvm::NodeAttrs& attrs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    op(dC, A, B, dA, dB, s, attrs);
  }
};

struct BackwardSumLogDiag {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int M, int stride, DType* dB, DType* A, DType* dA) {
    const int matrix(i / M), row((i % M) / stride), col(i % stride);
    dA[i] = (row == col ? dB[matrix]/A[i] : DType(0));
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
    // Note that A and dA may point to the same memory.
    using namespace mxnet_op;
    Kernel<BackwardSumLogDiag, xpu>::Launch
         (s, dA.MSize(), dA.size(1)*dA.stride_, dA.stride_, dB.dptr_, A.dptr_, dA.dptr_);
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
    trmm::op(L, tempM, DType(1.0), false, true, true, s);
    gemm::op(dA, Q, tempM, DType(-1.0), DType(1.0), false, true, s);
    Kernel<CopyTriangularToOppositeSide, xpu>::Launch
           (s, tempM.MSize(), tempM.size(1)*tempM.stride_, tempM.stride_,
            tempM.dptr_, false);
    gemm::op(tempM, Q, dA, DType(1.0), DType(1.0), false, false, s);
    trsm::op(L, dA, DType(1.0), false, true, true, s);
  }
};

// Helper for syevd_backward. See technical report for details
// Note: Could be parallelized more, but this is subdominant anyway
template<typename DType>
DType syevd_back_helper_eps(DType* X);

template<>
MSHADOW_XINLINE float syevd_back_helper_eps(float* X) {
  return 1e-30;
}

template<>
MSHADOW_XINLINE double syevd_back_helper_eps(double* X) {
  return 1e-100;
}

struct SyevdBackHelper {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int k, int n, DType* X, int ldx, DType* L,
                                  int ldl, DType* dL, int lddl, DType* Y,
                                  int ldy) {
    const int offx(k*n*ldx);
    const int offy(k*n*ldy);
    const int offl(k*ldl);
    const int offdl(k*lddl);
    DType denom(0.0), elem(0.0);
    const DType eps(syevd_back_helper_eps(X));
    // Lower and upper triangle: Loop i > j
    for (int i = 1; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        denom = L[offl+i] - L[offl+j];  // Must be >=0
        if (denom < eps) denom = eps;
        denom *= 2.0;
        elem = (X[offx+i*ldx+j] - X[offx+j*ldx+i])/denom;
        Y[offy+i*ldy+j] = Y[offy+j*ldy+i] = elem;
      }
    }
    // Diagonal
    for (int i = 0; i < n; ++i) {
      Y[offy+i*(ldy+1)] = dL[offdl+i];
    }
  }
};

// Have to reserve temporary storage tempM, same shape as dA.
// dA may overwrite dU
struct syevd_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dU,
                 const Tensor<xpu, 2, DType>& dL,
                 const Tensor<xpu, 3, DType>& U,
                 const Tensor<xpu, 2, DType>& L,
                 const Tensor<xpu, 3, DType>& dA,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    // Backward of (U, L) = syevd(A):
    //   dA = U**T * SyevdBackHelper(dU * U**T, L, dL) * U
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // Need temporal space, same shape as dA
    Tensor<xpu, 3, DType> tempM = ctx.requested[0]
      .get_space_typed<xpu, 3, DType>(dA.shape_, s);
    // This copy is just to make sure there are no invalid values (NaN, infinity) in
    // tempM. gemm multiplies tempM with 0, instead of setting entries to 0.
    Copy(tempM, dU, s);
    gemm::op(dU, U, tempM, DType(1.0), DType(0.0), false, true, s);
    // SyevdBackHelper: tempM => dA
    Kernel<SyevdBackHelper, xpu>::Launch
      (s, dA.size(0), dA.size(1), tempM.dptr_, tempM.stride_, L.dptr_,
       L.stride_, dL.dptr_, dL.stride_, dA.dptr_, dA.stride_);
    gemm::op(U, dA, tempM, DType(1.0), DType(0.0), true, false, s);
    gemm::op(tempM, U, dA, DType(1.0), DType(0.0), false, false, s);
  }
};

struct inverse_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dA,
                 const Tensor<xpu, 3, DType>& A,
                 const Tensor<xpu, 3, DType>& dB,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    // Backward of A = inverse(B)
    if (dB.shape_.Size() == 0U) {
      return;
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> temp = ctx.requested[0]
      .get_space_typed<xpu, 3, DType>(A.shape_, s);
    gemm2::op(dA, A, temp, DType(1), false, true, s);
    gemm2::op(A, temp, dB, DType(-1), true, false, s);
  }
};

// Here we set grad to zero if det = 0
struct StopZeroDetGrad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int grad_step, DType *grad, DType *det, DType zero_det) {
    int batch_ind = i / grad_step;
    if (det[batch_ind] == zero_det) {
      grad[i] = DType(0);
    }
  }
};

// Backward of det(A) is derived from Jacobi's formula.
// The closed form solution is pretty easy when A is invertible.
// For non-invertible A, grad is not backwarded.
struct det_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 1, DType>& ddet,
                 const Tensor<xpu, 1, DType>& det,
                 const Tensor<xpu, 3, DType>& LU,
                 const Tensor<xpu, 2, int>& pivot,
                 const Tensor<xpu, 3, DType>& dA,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mxnet_op;
    if (dA.shape_.Size() == 0U) {
      return;
    }
    // compute inverse(A) and stores it to LU
    linalg_batch_det_backward_helper(LU, pivot, det, dA, DType(0), ctx);
    const_cast<Tensor<xpu, 3, DType>&>(dA) = broadcast_to(reshape(det * ddet, \
      Shape3(det.size(0), 1, 1)), mxnet::TShape(LU.shape_)) * \
      transpose(LU, Shape3(0, 2, 1));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // stop grad for zero det temporarily
    Kernel<StopZeroDetGrad, xpu>::Launch(s, dA.shape_.Size(), dA.size(1) * dA.size(2), \
                                         dA.dptr_, det.dptr_, DType(0));
  }
};

// Backward of slogdet(A) is derived from Jacobi's formula.
// The closed form solution is pretty easy when A is invertible.
// For non-invertible A, grad is not backwarded.
// Grad is not properly defined on sign, so it's not backwarded either.
struct slogdet_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 1, DType>& dlogabsdet,
                 const Tensor<xpu, 1, DType>& sign,
                 const Tensor<xpu, 1, DType>& logabsdet,
                 const Tensor<xpu, 3, DType>& LU,
                 const Tensor<xpu, 2, int>& pivot,
                 const Tensor<xpu, 3, DType>& dA,
                 const OpContext& ctx, const nnvm::NodeAttrs& attrs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mxnet_op;
    if (dA.shape_.Size() == 0U) {
      return;
    }
    // compute inverse(A) and stores it to LU
    linalg_batch_det_backward_helper(LU, pivot, logabsdet, dA, DType(-INFINITY), ctx);
    const_cast<Tensor<xpu, 3, DType>&>(dA) = broadcast_to(reshape(dlogabsdet, \
      Shape3(logabsdet.size(0), 1, 1)), mxnet::TShape(LU.shape_)) * \
      transpose(LU, Shape3(0, 2, 1));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // stop grad for zero det
    Kernel<StopZeroDetGrad, xpu>::Launch(s, dA.shape_.Size(), dA.size(1) * dA.size(2), \
                                         dA.dptr_, logabsdet.dptr_, DType(-INFINITY));
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_LA_OP_INL_H_
