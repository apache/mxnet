/*!
 * Copyright (c) 2017 by Contributors
 * \file la_op_inline.h
 * \brief Operators for advanced linear algebra.
 */
#ifndef MXNET_OPERATOR_TENSOR_LA_OP_INLINE_H_
#define MXNET_OPERATOR_TENSOR_LA_OP_INLINE_H_

#include <mxnet/c_lapack_api.h>

namespace mxnet {
namespace op {

using namespace mshadow;

#define LA_OP_NOT_AVAIL " operator can only be called with float/double data type."

// Signature for single matrix operations (decomposition/inversion).
#define FUNC_SIGNATURE_1(fname, arg1) {CHECK_EQ(MXNET_LAPACK_##fname(MXNET_LAPACK_ROW_MAJOR, 'L', \
  arg1.size(0), arg1.dptr_, arg1.size(0)), 0) << "fname failed in lapack";}

// Signature for matrix-matrix multiplications involving one diagonal matrix.
#define FUNC_SIGNATURE_2(fname, arg1, arg2) \
  { cblas_##fname(CblasRowMajor, (rightside ? CblasRight : CblasLeft), \
                  CblasLower, (transpose ? CblasTrans : CblasNoTrans), \
                  CblasNonUnit, arg2.size(0), arg2.size(1), alpha, arg1.dptr_, \
                  (rightside ? arg2.size(1) : arg2.size(0)), arg2.dptr_, arg2.size(1)); }


// Helper functions.
template<typename DType>
void CopyLowerToUpper(DType *dptr, int N)
  { for (int i = 1; i < N; ++i ) for ( int j = 0; j < i; ++j ) dptr[j*N+i] = dptr[i*N+j]; }
template<typename DType>
void ZeroUpper(DType *dptr, int N)
  { for (int i = 0; i < N; ++i ) for ( int j = i+1; j < N; ++j ) dptr[i*N+j] = 0; }

// Forward operators

// D = gemm(A,B,C)
struct gemm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 const Tensor<xpu, 2, DType>& C, DType alpha, DType beta, bool tA, bool tB)
    { CHECK(false) << "gemm" << LA_OP_NOT_AVAIL; }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 const Tensor<xpu, 2, DType>& C, const Tensor<xpu, 2, DType>& D,
                 const nnvm::NodeAttrs& attrs) {
    if ( C.dptr_ != D.dptr_ ) Copy(D, C);
    const LaMatrixMacParam& param = nnvm::get<LaMatrixMacParam>(attrs.parsed);
    gemm::op(A, B, D, DType(param.alpha), DType(param.beta), param.transpose_a, param.transpose_b);
  }
};
template<>
void gemm::op<cpu, float>(const Tensor<cpu, 2, float>& A, const Tensor<cpu, 2, float>& B,
                          const Tensor<cpu, 2, float>& C,
                          float alpha, float beta, bool tA, bool tB ) {
  CHECK_EQ((tA ? A.size(1) : A.size(0)), C.size(0))
    << "Non compatible matrix dimensions between inputs A and C for gemm operator";
  CHECK_EQ((tB ? B.size(0) : B.size(1)), C.size(1))
    << "Non compatible matrix dimensions between inputs B and C for gemm operator";
  CHECK_EQ((tA ? A.size(0) : A.size(1)), (tB ? B.size(1) : B.size(0)))
    << "Non compatible matrix dimensions between inputs A and B for gemm operator";
  cblas_sgemm(CblasRowMajor, (tA ? CblasTrans : CblasNoTrans), (tB ? CblasTrans : CblasNoTrans),
              (tA ? A.size(1):A.size(0)), (tB ? B.size(0): B.size(1)),
              (tA ? A.size(0):A.size(1)), alpha, A.dptr_, A.size(1), B.dptr_, B.size(1),
              beta, C.dptr_, (tB ? B.size(0): B.size(1)));
}
template<>
void gemm::op<cpu, double>(const Tensor<cpu, 2, double>& A, const Tensor<cpu, 2, double>& B,
                           const Tensor<cpu, 2, double>& C,
                           double alpha, double beta, bool tA, bool tB) {
  CHECK_EQ((tA ? A.size(1) : A.size(0)), C.size(0))
    << "Non compatible matrix dimensions between inputs A and C for gemm operator";
  CHECK_EQ((tB ? B.size(0) : B.size(1)), C.size(1))
    << "Non compatible matrix dimensions between inputs B and C for gemm operator";
  CHECK_EQ((tA ? A.size(0) : A.size(1)), (tB ? B.size(1) : B.size(0)))
    << "Non compatible matrix dimensions between inputs A and B for gemm operator";
  cblas_dgemm(CblasRowMajor, (tA ? CblasTrans : CblasNoTrans), (tB ? CblasTrans : CblasNoTrans),
              (tA ? A.size(1):A.size(0)), (tB ? B.size(0): B.size(1)),
              (tA ? A.size(0):A.size(1)), alpha, A.dptr_, A.size(1), B.dptr_, B.size(1),
              beta, C.dptr_, (tB ? B.size(0): B.size(1)));
}

// C = gemm2(A,B)
struct gemm2 {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 const Tensor<xpu, 2, DType>& C, const nnvm::NodeAttrs& attrs) {
    const LaMatrixMultParam& param = nnvm::get<LaMatrixMultParam>(attrs.parsed);
    gemm::op(A, B, C, DType(param.alpha), DType(0), param.transpose_a, param.transpose_b);
  }
};

// L = potrf(A).
struct potrf {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& L,
                 const nnvm::NodeAttrs& attrs)
    { CHECK(false) << "potrf" << LA_OP_NOT_AVAIL; }
};
template<>
void potrf::op<cpu, float>(const Tensor<cpu, 2, float>& A, const Tensor<cpu, 2, float>& L,
                           const nnvm::NodeAttrs& attrs) {
  if ( A.dptr_ != L.dptr_ ) Copy(L, A);
  FUNC_SIGNATURE_1(spotrf, L);
  ZeroUpper(L.dptr_, L.size(0));
}
template<>
void potrf::op<cpu, double>(const Tensor<cpu, 2, double>& A, const Tensor<cpu, 2, double>& L,
                            const nnvm::NodeAttrs& attrs) {
  if ( A.dptr_ != L.dptr_ ) Copy(L, A);
  FUNC_SIGNATURE_1(dpotrf, L);
  ZeroUpper(L.dptr_, L.size(0));
}

// A = potri(L).
struct potri {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& L, const Tensor<xpu, 2, DType>& A,
                 const nnvm::NodeAttrs& attrs)
    { CHECK(false) << "potri" << LA_OP_NOT_AVAIL; }
};
template<>
void potri::op<cpu, float>(const Tensor<cpu, 2, float>& L, const Tensor<cpu, 2, float>& A,
                           const nnvm::NodeAttrs& attrs) {
  if ( A.dptr_ != L.dptr_ ) Copy(A, L);
  FUNC_SIGNATURE_1(spotri, A);
  CopyLowerToUpper(A.dptr_, A.size(0));
}
template<>
void potri::op<cpu, double>(const Tensor<cpu, 2, double>& A, const Tensor<cpu, 2, double>& L,
                            const nnvm::NodeAttrs& attrs) {
  if ( A.dptr_ != L.dptr_ ) Copy(L, A);
  FUNC_SIGNATURE_1(dpotri, A);
  CopyLowerToUpper(A.dptr_, A.size(0));
}

// B = trsm(L,A)
struct trsm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& L, const Tensor<xpu, 2, DType>& B,
                 DType alpha, bool rightside, bool transpose)
    { CHECK(false) << "trsm" << LA_OP_NOT_AVAIL; }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& L, const Tensor<xpu, 2, DType>& A,
                 const Tensor<xpu, 2, DType>& B, const nnvm::NodeAttrs& attrs) {
    if ( A.dptr_ != B.dptr_ ) Copy(B, A);
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    op(L, B, DType(param.alpha), param.rightside, param.transpose);
  }
};
template<>
void trsm::op<cpu, float>(const Tensor<cpu, 2, float>& L, const Tensor<cpu, 2, float>& B,
                          float alpha, bool rightside, bool transpose) {
  FUNC_SIGNATURE_2(strsm, L, B);
}
template<>
void trsm::op<cpu, double>(const Tensor<cpu, 2, double>& L, const Tensor<cpu, 2, double>& B,
                           double alpha, bool rightside, bool transpose) {
  FUNC_SIGNATURE_2(dtrsm, L, B);
}

// B = trmm(L,A)
struct trmm {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& L, const Tensor<xpu, 2, DType>& B,
                 DType alpha, bool rightside, bool transpose)
    { CHECK(false) << "trmm" << LA_OP_NOT_AVAIL; }
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& L, const Tensor<xpu, 2, DType>& A,
                 const Tensor<xpu, 2, DType>& B, const nnvm::NodeAttrs& attrs) {
    if ( A.dptr_ != B.dptr_ ) Copy(B, A);
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    op(L, B, DType(param.alpha), param.rightside, param.transpose);
  }
};
template<>
void trmm::op<cpu, float>(const Tensor<cpu, 2, float>& L, const Tensor<cpu, 2, float>& B,
                          float alpha, bool rightside, bool transpose) {
  FUNC_SIGNATURE_2(strmm, L, B);
}
template<>
void trmm::op<cpu, double>(const Tensor<cpu, 2, double>& L, const Tensor<cpu, 2, double>& B,
                           double alpha, bool rightside, bool transpose) {
  FUNC_SIGNATURE_2(dtrmm, L, B);
}

// Useful operator that is not part of BLAS/LAPACK.
struct sumlogdiag {
  template<typename xpu, typename DType,
           typename std::enable_if<!std::is_floating_point<DType>::value, int>::type = 0>
  static void op(const Tensor<xpu, 2, DType>& A, DType& L, const nnvm::NodeAttrs& attrs)
    { CHECK(false) << "sumlogdiag operator can only be called with float/double data type."; }
  template<typename xpu, typename DType,
           typename std::enable_if<std::is_floating_point<DType>::value, int>::type = 0>
  static void op(const Tensor<xpu, 2, DType>& A, DType& B, const nnvm::NodeAttrs& attrs) {
    CHECK_EQ(A.size(0), A.size(1)) << "sumlogdiag operator requires a NxN matrix as input.";
    const int N(A.size(0));
    DType sum(0);
    DType *p(A.dptr_);
    for ( int i = 0; i < N; ++i, p += N+1 ) {
      sum += log(*p);
    }
    B = sum;
  }
};

// Backward operators

struct gemm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& dD, const Tensor<xpu, 2, DType>& A,
                 const Tensor<xpu, 2, DType>& B, const Tensor<xpu, 2, DType>& C,
                 const Tensor<xpu, 2, DType>& dA, const Tensor<xpu, 2, DType>& dB,
                 const Tensor<xpu, 2, DType>& dC, const nnvm::NodeAttrs& attrs) {
    const LaMatrixMacParam& param = nnvm::get<LaMatrixMacParam>(attrs.parsed);
    (param.transpose_a ? gemm::op(B, dD, dA, DType(param.alpha), DType(0), param.transpose_b, true)
                  : gemm::op(dD, B, dA, DType(param.alpha), DType(0), false, !param.transpose_b));
    (param.transpose_b ? gemm::op(dD, A, dB, DType(param.alpha), DType(0), true, param.transpose_a)
                  : gemm::op(A, dD, dB, DType(param.alpha), DType(0), !param.transpose_a, false));
    const int N(dC.size(0)*dC.size(1));
    for ( int i = 0; i < N; ++i ) {
      dC.dptr_[i] = param.beta * dD.dptr_[i];
    }
  }
};

struct gemm2_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& dC, const Tensor<xpu, 2, DType>& A,
                 const Tensor<xpu, 2, DType>& B, const Tensor<xpu, 2, DType>& dA,
                 const Tensor<xpu, 2, DType>& dB, const nnvm::NodeAttrs& attrs) {
    const LaMatrixMultParam& param = nnvm::get<LaMatrixMultParam>(attrs.parsed);
    (param.transpose_a ? gemm::op(B, dC, dA, DType(param.alpha), DType(0), param.transpose_b, true)
                   : gemm::op(dC, B, dA, DType(param.alpha), DType(0), false, !param.transpose_b));
    (param.transpose_b ? gemm::op(dC, A, dB, DType(param.alpha), DType(0), true, param.transpose_a)
                   : gemm::op(A, dC, dB, DType(param.alpha), DType(0), !param.transpose_a, false));
  }
};

struct potrf_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& dL, const Tensor<xpu, 2, DType>& L,
                 const Tensor<xpu, 2, DType>& dA, const nnvm::NodeAttrs& attrs) {
    // Backward of L = potrf(A).
    // dA = 0.5 * L**T * symm(L**T * dL # E) * L**(-1) where
    //     '#' denotes Hadamard product
    //      E is the matrix having 1 on diagonal, 0 on upper and 2 on lower triagle
    //      symm(X) = 0.5 * (X + X**T)
    // Hadamard product and symm can be realized by a single copy from lower to upper triangle.
    if ( dL.dptr_ != dA.dptr_ ) {
      Copy(dA, dL);
    }
    trmm::op(L, dA, DType(1.0), false, true);
    CopyLowerToUpper(dA.dptr_, dA.size(0));
    trsm::op(L, dA, DType(1.0), false, true);
    trsm::op(L, dA, DType(0.5), true, false);
  }
};

struct potri_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& dA, const Tensor<xpu, 2, DType>& L,
                 const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& dL,
                 const nnvm::NodeAttrs& attrs) {
    // Backward of A = potri(L).
    // dL = -2 * tril(A * dA * L**(-T)), where tril() extracts lower triangle and diagonal.
    gemm::op(A, dA, dL, DType(1.0), DType(0), false, false);
    trsm::op(L, dL, DType(-2.0), true, true);
    ZeroUpper(dL.dptr_, dL.size(0));
  }
};

struct trsm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& dB, const Tensor<xpu, 2, DType>& L,
                 const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 const Tensor<xpu, 2, DType>& dL, const Tensor<xpu, 2, DType>& dA,
                 const nnvm::NodeAttrs& attrs) {
    // Backward of B = trsm(L,A).
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    // Compute dA
    if ( dA.dptr_ != dB.dptr_ ) Copy(dA, dB);
    trsm::op(L, dA, DType(param.alpha), param.rightside, !param.transpose);
    // Compute dL
    const bool da_left(param.rightside == param.transpose);
    (da_left ?
        gemm::op(dA, B, dL, DType(-1.0/param.alpha), DType(0), param.transpose, !param.transpose)
      : gemm::op(B, dA, dL, DType(-1.0/param.alpha), DType(0), !param.transpose, param.transpose));
    ZeroUpper(dL.dptr_, dL.size(0));
  }
};

struct trmm_backward {
  template<typename xpu, typename DType>
  static void op(const Tensor<xpu, 2, DType>& dB, const Tensor<xpu, 2, DType>& L,
                 const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 const Tensor<xpu, 2, DType>& dL, const Tensor<xpu, 2, DType>& dA,
                 const nnvm::NodeAttrs& attrs) {
    // Backward of B = trmm(L,A).
    const LaTriangMatrixMultParam& param = nnvm::get<LaTriangMatrixMultParam>(attrs.parsed);
    // Compute dL
    const bool db_left(param.rightside == param.transpose);
    (db_left ? gemm::op(dB, A, dL, DType(param.alpha), DType(0), param.transpose, !param.transpose)
           : gemm::op(A, dB, dL, DType(param.alpha), DType(0), !param.transpose, param.transpose));
    ZeroUpper(dL.dptr_, dL.size(0));
    // Compute dA
    if ( dA.dptr_ != dB.dptr_ ) Copy(dA, dB);
    trmm::op(L, dA, DType(param.alpha), param.rightside, !param.transpose);
  }
};

struct sumlogdiag_backward {
  template<typename xpu, typename DType>
  static void op(const DType& dB, const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& dA,
                 const nnvm::NodeAttrs& attrs, bool add) {
    // Backward of B = sumlogdiag(A).
    const int N(A.size(0));
    if ( !add ) {
      for ( int i = 0; i < N*N; ++i ) {
        dA.dptr_[i] = 0;
      }
    }
    for ( int i = 0; i < N; ++i ) {
      dA.dptr_[i*(N+1)] += dB / A.dptr_[i*N+i];
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_LA_OP_INLINE_H_
