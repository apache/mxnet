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

// D = gemm(A,B,C)
struct gemm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
    const Tensor<xpu, 3, DType>& C, DType alpha, DType beta, bool tA, bool tB, Stream<xpu> *s) {
    linalg_batch_gemm(A, B, C, alpha, beta, tA, tB, s);
  }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, const Tensor<xpu, 3, DType>& D,
                 Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    if ( C.dptr_ != D.dptr_ ) Copy(D, C, s);
    const LaMatrixMacParam& param = nnvm::get<LaMatrixMacParam>(attrs.parsed);
    gemm::op(A, B, D, DType(param.alpha), DType(param.beta),
             param.transpose_a, param.transpose_b, s);
  }
};

// C = gemm2(A,B)
struct gemm2 {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& C, Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    const LaMatrixMultParam& param = nnvm::get<LaMatrixMultParam>(attrs.parsed);
    gemm::op(A, B, C, DType(param.alpha), DType(0), param.transpose_a, param.transpose_b, s);
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
                 const Tensor<xpu, 3, DType>& B, Stream<xpu> *s, const nnvm::NodeAttrs& attrs) {
    if ( A.dptr_ != B.dptr_ ) Copy(B, A, s);
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    op(L, B, DType(param.alpha), param.rightside, param.transpose, s);
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
};

struct potrf_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dL, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& dA,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of L = potrf(A).
    // dA = 0.5 * L**T * symm(L**T * dL # E) * L**(-1) where
    //     '#' denotes Hadamard product
    //      E is the matrix having 1 on diagonal, 0 on upper and 2 on lower triagle
    //      symm(X) = 0.5 * (X + X**T)
    // Hadamard product and symm can be realized by a single copy from lower to upper triangle.
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
};

struct potri_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dA, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& dL,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of A = potri(L).
    // dL = -2 * tril(A * dA * L**(-T)), where tril() extracts lower triangle and diagonal.
    gemm::op(A, dA, dL, DType(1.0), DType(0), false, false, s);
    trsm::op(L, dL, DType(-2.0), true, true, s);
    using namespace mxnet_op;
    Kernel<ZeroUpper, xpu>::Launch(s, dL.MSize(), dL.size(1)*dL.stride_, dL.stride_, dL.dptr_);
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
};

struct trmm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 3, DType>& dB, const Tensor<xpu, 3, DType>& L,
                 const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                 const Tensor<xpu, 3, DType>& dL, const Tensor<xpu, 3, DType>& dA,
                 Stream<xpu>* s, const nnvm::NodeAttrs& attrs) {
    // Backward of B = trmm(L,A).
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    // Compute dL
    const bool db_left(param.rightside == param.transpose);
    DType scale(param.alpha);
    (db_left ? gemm::op(dB, A, dL, scale, DType(0), param.transpose, !param.transpose, s)
             : gemm::op(A, dB, dL, scale, DType(0), !param.transpose, param.transpose, s));
    using namespace mxnet_op;
    Kernel<ZeroUpper, xpu>::Launch(s, dL.MSize(), dL.size(1)*dL.stride_, dL.stride_, dL.dptr_);
    // Compute dA
    if ( dA.dptr_ != dB.dptr_ ) Copy(dA, dB, s);
    trmm::op(L, dA, scale, param.rightside, !param.transpose, s);
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
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_LA_OP_INLINE_H_
