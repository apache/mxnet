/*!
 *  Copyright (c) 2014 by Contributors
 *  \file cp_decomp_linalg.h
 *  \brief Linear algebra routines used by CPDecomp
 *  \author Jencir Lee
 */
#ifndef MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_LINALG_H_
#define MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_LINALG_H_

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

#include <mshadow/tensor.h>
#include <algorithm>


namespace mxnet {
namespace op {
namespace cp_decomp {

using namespace mshadow;
using namespace mshadow::expr;

/*!
 * \brief Re-arrange m-by-n matrix from row-major layout to column-major layout
 *
 * \param m number of rows of input matrix a
 * \param n number of columns of input matrix a
 * \param b output matrix
 * \param ldb leading dimension of b
 * \param a input matrix
 * \param lda leading dimension of a
 */
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

/*!
 * \brief Compute Euclidean norm of a vector
 *
 * This function follows the same signature as the BLAS-1 function nrm2().
 *
 * \param n length of input vector a
 * \param a input vector
 * \param lda every element of a will be accessed at a[i*lda]
 * \return the Euclidean norm of a
 */
template <typename xpu, typename DType>
inline DType nrm2(int n, DType *a, int lda);

/*!
 * \brief Solve A X = B for positive semi-definite real matrix A
 *
 * A, B, X are stored in the row-major layout. A is n-by-n matrix, B n-by-nrhs.
 * This function follows the same signature as the LAPACK driver function <t>posv().
 *
 * \param n number of rows of A
 * \param nrhs number of columns of B
 * \param a matrix A
 * \param lda leading dimension of a
 * \param b matrix B at input. Contains X at output
 * \param ldb leading dimension of b
 * \return return code from the LAPACK driver function <t>posv
 */
template <typename xpu, typename DType>
inline int posv(int n, int nrhs, DType *a, int lda, DType *b, int ldb);

/*!
 * \brief Compute Q matrix from QR Decomposition
 *
 * A is stored in the column-major layout.
 * This function wraps calls to LAPACK functions <t>geqrf(), <t>orgqr(), and follows the same signature as <t>orgqr().
 *
 * \param m number of rows of Q, equal to that of A
 * \param n number of columns of Q
 * \param k use the first k columns of A for QR Decomposition
 * \param a matrix A at input. Contains Q at output
 * \param lda leading dimension of A
 * \return return code from <t>geqrf() or <t>orgqr() in case of error, othewise 0
 */
template <typename xpu, typename DType>
inline int orgqr(int m, int n, int k, DType *a, int lda);

/*!
 * \brief Compute SVD
 *
 * The input matrix is in the row-major layout. This function follows the same signature as the LAPACK counterpart.
 */
template <typename xpu, typename DType>
inline int gesdd(char jobz, int m, int n, DType *a, int lda,
    DType *s, DType *u, int ldu, DType *vt, int ldvt);


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
inline int gesdd<cpu, float>(char jobz, int m, int n, float *a, int lda,
    float *s, float *u, int ldu, float *vt, int ldvt) {
  return LAPACKE_sgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda,
      s, u, ldu, vt, ldvt);
}

template <>
inline int gesdd<cpu, double>(char jobz, int m, int n, double *a, int lda,
    double *s, double *u, int ldu, double *vt, int ldvt) {
  return LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda,
      s, u, ldu, vt, ldvt);
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

  delete [] b_T;
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

  delete [] b_T;
  return status;
}

template <>
inline int gesdd<cpu, float>(char jobz, int m, int n, float *a, int lda,
    float *s, float *u, int ldu, float *vt, int ldvt) {
  int status, info;

  float *work = new float[10];
  int lwork = -1;
  int *iwork = new int[8 * std::min(m, n)];

  status = sgesdd_(&jobz, &n, &m, a, &lda,
      s, vt, &ldvt, u, &ldu,
      work, &lwork, iwork, &info);
  if (status != 0) {
    delete [] work;
    delete [] iwork;
    return status;
  }

  lwork = static_cast<int>(work[0]);
  delete [] work;
  work = new float[lwork];

  status = sgesdd_(&jobz, &n, &m, a, &lda,
      s, vt, &ldvt, u, &ldu,
      work, &lwork, iwork, &info);

  delete [] work;
  delete [] iwork;
  return status;
}

template <>
inline int gesdd<cpu, double>(char jobz, int m, int n, double *a, int lda,
    double *s, double *u, int ldu, double *vt, int ldvt) {
  int status, info;

  double *work = new double[10];
  int lwork = -1;
  int *iwork = new int[8 * std::min(m, n)];

  status = dgesdd_(&jobz, &n, &m, a, &lda,
      s, vt, &ldvt, u, &ldu,
      work, &lwork, iwork, &info);
  if (status != 0) {
    delete [] work;
    delete [] iwork;
    return status;
  }

  lwork = static_cast<int>(work[0]);
  delete [] work;
  work = new double[lwork];

  status = dgesdd_(&jobz, &n, &m, a, &lda,
      s, vt, &ldvt, u, &ldu,
      work, &lwork, iwork, &info);

  delete [] work;
  delete [] iwork;
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
  if (status != 0) {
    delete [] tau;
    delete [] work;
    return status;
  }

  lwork = work[0];
  delete [] work;
  work = new float[lwork];
  status = sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0) {
    delete [] tau;
    delete [] work;
    return status;
  }

  lwork = -1;
  status = sorgqr_(&m, &n, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0) {
    delete [] tau;
    delete [] work;
    return status;
  }

  lwork = work[0];
  delete [] work;
  work = new float[lwork];
  status = sorgqr_(&m, &n, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0) {
    delete [] tau;
    delete [] work;
    return status;
  }

  delete [] tau;
  delete [] work;
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
  if (status != 0) {
    delete [] tau;
    delete [] work;
    return status;
  }

  lwork = work[0];
  delete [] work;
  work = new double[lwork];
  status = dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0) {
    delete [] tau;
    delete [] work;
    return status;
  }

  lwork = -1;
  status = dorgqr_(&m, &n, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0) {
    delete [] tau;
    delete [] work;
    return status;
  }

  lwork = work[0];
  delete [] work;
  work = new double[lwork];
  status = dorgqr_(&m, &n, &n, a, &lda, tau, work, &lwork, &info);
  if (status != 0) {
    delete [] tau;
    delete [] work;
    return status;
  }

  delete [] tau;
  delete [] work;
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

}  // namespace cp_decomp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_LINALG_H_
