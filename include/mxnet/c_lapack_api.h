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
 * \file c_lapack_api.h
 * \brief Unified interface for LAPACK calls from within mxnet.
 *  Purpose is to hide the platform specific differences.
 */
#ifndef MXNET_C_LAPACK_API_H_
#define MXNET_C_LAPACK_API_H_

// Manually maintained list of LAPACK interfaces that can be used
// within MXNET. Conventions:
//    - Interfaces must be compliant with lapacke.h in terms of signature and
//      naming conventions so wrapping a function "foo" which has the
//      signature
//         lapack_int LAPACKE_foo(int, char, lapack_int, float* , lapack_int)
//      within lapacke.h should result in a wrapper with the following signature
//         int MXNET_LAPACK_foo(int, char, int, float* , int)
//      Note that function signatures in lapacke.h will always have as first
//      argument the storage order (row/col-major). All wrappers have to support
//      that argument. The underlying fortran functions will always assume a
//      column-major layout. It is the responsibility of the wrapper function
//      to handle the (usual) case that it is called with data in row-major
//      format, either by doing appropriate transpositions explicitly or using
//      transposition options of the underlying fortran function.
//    - It is ok to assume that matrices are stored in contiguous memory
//      (which removes the need to do special handling for lda/ldb parameters
//      and enables us to save additional matrix transpositions around
//      the fortran calls).
//    - It is desired to add some basic checking in the C++-wrappers in order
//      to catch simple mistakes when calling these wrappers.
//    - Must support compilation without lapack-package but issue runtime error in this case.

#include <dmlc/logging.h>
#include "mshadow/tensor.h"

using namespace mshadow;

extern "C" {
  // Fortran signatures
  #define MXNET_LAPACK_FSIGNATURE1(func, dtype) \
    void func##_(char* uplo, int* n, dtype* a, int* lda, int *info);

  MXNET_LAPACK_FSIGNATURE1(spotrf, float)
  MXNET_LAPACK_FSIGNATURE1(dpotrf, double)
  MXNET_LAPACK_FSIGNATURE1(spotri, float)
  MXNET_LAPACK_FSIGNATURE1(dpotri, double)

  void dposv_(char *uplo, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb, int *info);

  void sposv_(char *uplo, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb, int *info);
}

#define MXNET_LAPACK_ROW_MAJOR 101
#define MXNET_LAPACK_COL_MAJOR 102

#define CHECK_LAPACK_CONTIGUOUS(a, b) \
  CHECK_EQ(a, b) << "non contiguous memory for array in lapack call";

#define CHECK_LAPACK_UPLO(a) \
  CHECK(a == 'U' || a == 'L') << "neither L nor U specified as triangle in lapack call";

inline char loup(char uplo, bool invert) { return invert ? (uplo == 'U' ? 'L' : 'U') : uplo; }


/*!
 * \brief Transpose matrix data in memory
 *
 * Equivalently we can see it as flipping the layout of the matrix
 * between row-major and column-major.
 *
 * \param m number of rows of input matrix a
 * \param n number of columns of input matrix a
 * \param b output matrix
 * \param ldb leading dimension of b
 * \param a input matrix
 * \param lda leading dimension of a
 */
template <typename xpu, typename DType>
inline void flip(int m, int n, DType *b, int ldb, DType *a, int lda);

template <>
inline void flip<cpu, float>(int m, int n,
  float *b, int ldb, float *a, int lda) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      b[j * ldb + i] = a[i * lda + j];
}

template <>
inline void flip<cpu, double>(int m, int n,
  double *b, int ldb, double *a, int lda) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      b[j * ldb + i] = a[i * lda + j];
}


#if MXNET_USE_LAPACK

  #define MXNET_LAPACK_CWRAPPER1(func, dtype) \
  inline int MXNET_LAPACK_##func(int matrix_layout, char uplo, int n, dtype* a, int lda ) { \
    CHECK_LAPACK_CONTIGUOUS(n, lda); \
    CHECK_LAPACK_UPLO(uplo); \
    char o(loup(uplo, (matrix_layout == MXNET_LAPACK_ROW_MAJOR))); \
    int ret(0); \
    func##_(&o, &n, a, &lda, &ret); \
    return ret; \
  }
  MXNET_LAPACK_CWRAPPER1(spotrf, float)
  MXNET_LAPACK_CWRAPPER1(dpotrf, double)
  MXNET_LAPACK_CWRAPPER1(spotri, float)
  MXNET_LAPACK_CWRAPPER1(dpotri, double)

  inline int mxnet_lapack_sposv(int matrix_layout, char uplo, int n, int nrhs,
    float *a, int lda, float *b, int ldb) {
    int info;
    if (matrix_layout == MXNET_LAPACK_ROW_MAJOR) {
      // Transpose b to b_t of shape (nrhs, n)
      float *b_t = new float[nrhs * n];
      flip<cpu, float>(n, nrhs, b_t, n, b, ldb);
      sposv_(&uplo, &n, &nrhs, a, &lda, b_t, &n, &info);
      flip<cpu, float>(nrhs, n, b, ldb, b_t, n);
      delete [] b_t;
      return info;
    }
    sposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
  }

  inline int mxnet_lapack_dposv(int matrix_layout, char uplo, int n, int nrhs,
    double *a, int lda, double *b, int ldb) {
    int info;
    if (matrix_layout == MXNET_LAPACK_ROW_MAJOR) {
      // Transpose b to b_t of shape (nrhs, n)
      double *b_t = new double[nrhs * n];
      flip<cpu, double>(n, nrhs, b_t, n, b, ldb);
      dposv_(&uplo, &n, &nrhs, a, &lda, b_t, &n, &info);
      flip<cpu, double>(nrhs, n, b, ldb, b_t, n);
      delete [] b_t;
      return info;
    }
    dposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
  }

#else

  // use pragma message instead of warning
  #pragma message("Warning: lapack usage not enabled, linalg-operators will not be available." \
     " Ensure that lapack library is installed and build with USE_LAPACK=1 to get lapack" \
     " functionalities.")

  // Define compilable stubs.
  #define MXNET_LAPACK_CWRAPPER1(func, dtype) \
  inline int MXNET_LAPACK_##func(int matrix_layout, char uplo, int n, dtype* a, int lda ) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  #define MXNET_LAPACK_UNAVAILABLE(func) \
  inline int mxnet_lapack_##func(...) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  MXNET_LAPACK_CWRAPPER1(spotrf, float)
  MXNET_LAPACK_CWRAPPER1(dpotrf, double)
  MXNET_LAPACK_CWRAPPER1(spotri, float)
  MXNET_LAPACK_CWRAPPER1(dpotri, double)

  MXNET_LAPACK_UNAVAILABLE(sposv)
  MXNET_LAPACK_UNAVAILABLE(dposv)

#endif

template <typename DType>
inline int MXNET_LAPACK_posv(int matrix_layout, char uplo, int n, int nrhs,
  DType *a, int lda, DType *b, int ldb);

template <>
inline int MXNET_LAPACK_posv<float>(int matrix_layout, char uplo, int n,
  int nrhs, float *a, int lda, float *b, int ldb) {
  return mxnet_lapack_sposv(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
}

template <>
inline int MXNET_LAPACK_posv<double>(int matrix_layout, char uplo, int n,
  int nrhs, double *a, int lda, double *b, int ldb) {
  return mxnet_lapack_dposv(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
}

#endif  // MXNET_C_LAPACK_API_H_
