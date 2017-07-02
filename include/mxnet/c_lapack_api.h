/*!
 *  Copyright (c) 2017 by Contributors
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

extern "C" {
  // Fortran signatures
  #define MXNET_LAPACK_FSIGNATURE1(func, dtype) \
    void func##_(char* uplo, int* n, dtype* a, int* lda, int *info);

  MXNET_LAPACK_FSIGNATURE1(spotrf, float)
  MXNET_LAPACK_FSIGNATURE1(dpotrf, double)
  MXNET_LAPACK_FSIGNATURE1(spotri, float)
  MXNET_LAPACK_FSIGNATURE1(dpotri, double)
}

#define MXNET_LAPACK_ROW_MAJOR 101
#define MXNET_LAPACK_COL_MAJOR 102

#define CHECK_LAPACK_CONTIGUOUS(a, b) \
  CHECK_EQ(a, b) << "non contiguous memory for array in lapack call";

#define CHECK_LAPACK_UPLO(a) \
  CHECK(a == 'U' || a == 'L') << "neither L nor U specified as triangle in lapack call";

inline char loup(char uplo, bool invert) { return invert ? (uplo == 'U' ? 'L' : 'U') : uplo; }

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
  MXNET_LAPACK_CWRAPPER1(spotrf, float)
  MXNET_LAPACK_CWRAPPER1(dpotrf, double)
  MXNET_LAPACK_CWRAPPER1(spotri, float)
  MXNET_LAPACK_CWRAPPER1(dpotri, double)

#endif

#endif  // MXNET_C_LAPACK_API_H_
