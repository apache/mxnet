/*!
 *  Copyright (c) 2014 by Contributors
 */
#ifndef MXNET_OPERATOR_TENSOR_CP_DECOMP_LINALG_H_
#define MXNET_OPERATOR_TENSOR_CP_DECOMP_LINALG_H_

#include <algorithm>

#if MSHADOW_USE_MKL
extern "C" {
  #include <mkl.h>
}
#elif __APPLE__
extern "C" {
  #include <cblas.h>
  #include <clapack.h>
}
#else
#error "The current implementation is only for MKL or Apple vecLib"
#endif


namespace mxnet {
namespace op {
namespace cp_decomp {

using namespace mshadow;
using namespace mshadow::expr;

// Transpose and store
template <typename xpu, typename DType>
inline void transpose(int m, int n, DType *b, int ldb, DType *a, int lda);

template <>
inline void transpose<cpu, float>(int m, int n, 
  float *b, int ldb, float *a, int lda) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      b[j * ldb + i] = a[i * lda + j];
}

template <>
inline void transpose<cpu, double>(int m, int n, 
  double *b, int ldb, double *a, int lda) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      b[j * ldb + i] = a[i * lda + j];
}

// Euclidean norm
template <typename xpu, typename DType>
inline DType nrm2(int n, DType *a, int lda);

// Solve A X = B for row-major matrices A, B, X
template <typename xpu, typename DType>
inline int posv(int n, int nrhs, DType *a, int lda, DType *b, int ldb);

// Compute Q matrix from QR Decomposition for column-major matrix A
template <typename xpu, typename DType>
inline int orgqr(int m, int n, int k, DType *a, int lda);

#if MSHADOW_USE_MKL
template <>
inline int posv<cpu, float>(int n, int nrhs,
    float *a, int lda, float *b, int ldb) {
  return LAPACKE_sposv(LAPACK_ROW_MAJOR, 'U', n, nrhs, a, lda, b, ldb);
}

template <>
inline int posv<cpu, double>(int n, int nrhs,
    double *a, int lda, double *b, int ldb) {
  return LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', n, nrhs, a, lda, b, ldb);
}

template <>
inline int orgqr<cpu, float>(int m, int n, int k, float *a, int lda) {
  float *tau;
  int info;
  int status = 0;

  tau = new float[std::min(m, n)];
  status = LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
  if (status != 0)
    return status;

  status = LAPACKE_sorgqr(LAPACK_COL_MAJOR, m, n, n, a, lda, tau);
  if (status != 0)
    return status;

  return status;
}

template <>
inline int orgqr<cpu, double>(int m, int n, int k, double *a, int lda) {
  double *tau;
  int info;
  int status = 0;

  tau = new double[std::min(m, n)];
  status = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
  if (status != 0)
    return status;

  status = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, a, lda, tau);
  if (status != 0)
    return status;

  return status;
}


#endif

#if __APPLE__
template <>
inline int posv<cpu, float>(int n, int nrhs,
    float *a, int lda, float *b, int ldb) {
  float *b_T = new float[n * nrhs];
  transpose<cpu, float>(n, nrhs, b_T, n, b, ldb);

  char uplo = 'U';
  int status, info;
  status = sposv_(&uplo, &n, &nrhs, a, &lda, b_T, &n, &info);
  transpose<cpu, float>(nrhs, n, b, ldb, b_T, n);

  return status;
}

template <>
inline int posv<cpu, double>(int n, int nrhs,
    double *a, int lda, double *b, int ldb) {
  double *b_T = new double[n * nrhs];
  transpose<cpu, double>(n, nrhs, b_T, n, b, ldb);

  char uplo = 'U';
  int status, info;
  status = dposv_(&uplo, &n, &nrhs, a, &lda, b_T, &n, &info);
  transpose<cpu, double>(nrhs, n, b, ldb, b_T, n);

  return status;
}

template <>
inline int orgqr<cpu, float>(int m, int n, int k, float *a, int lda) {
  float *tau, *work;
  int lwork, info;
  int status = 0;

  tau = new float[std::min(m, n)];
  work = new float[10];
  lwork = -1;
  status = sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0)
    return status;

  lwork = work[0];
  delete work;
  work = new float[lwork];
  status = sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0)
    return status;

  lwork = -1;
  status = sorgqr_(&m, &n, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0)
    return status;

  lwork = work[0];
  delete work;
  work = new float[lwork];
  status = sorgqr_(&m, &n, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0)
    return status;

  return status;
}

template <>
inline int orgqr<cpu, double>(int m, int n, int k, double *a, int lda) {
  double *tau, *work;
  int lwork, info;
  int status = 0;

  tau = new double[std::min(m, n)];
  work = new double[10];
  lwork = -1;
  status = dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0)
    return status;

  lwork = work[0];
  delete work;
  work = new double[lwork];
  status = dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0)
    return status;

  lwork = -1;
  status = dorgqr_(&m, &n, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0)
    return status;

  lwork = work[0];
  delete work;
  work = new double[lwork];
  status = dorgqr_(&m, &n, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0)
    return status;

  return status;
}
#endif

template <>
inline float nrm2<cpu, float>(int n, float *a, int lda) {
  return cblas_snrm2(n, a, lda);
}

template <>
inline double nrm2<cpu, double>(int n, double *a, int lda) {
  return cblas_dnrm2(n, a, lda);
}

}  // cp_decomp
}  // op
}  // mxnet

#endif
