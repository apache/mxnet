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

#include "c_lapack_api.h"

#if (MSHADOW_USE_MKL && MXNET_USE_LAPACK)
#elif MXNET_USE_LAPACK
#else
  // use pragma message instead of warning
  #pragma message("Warning: lapack usage not enabled, linalg-operators will not be available." \
     " Ensure that lapack library is installed and build with USE_LAPACK=1 to get lapack" \
     " functionalities.")

  // Define compilable stubs.
  #define MXNET_LAPACK_CWRAPPER1(func, dtype) \
  int MXNET_LAPACK_##func(int matrix_layout, char uplo, int n, dtype* a, int lda) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  #define MXNET_LAPACK_CWRAPPER2(func, dtype) \
  int MXNET_LAPACK_##func(int matrix_layout, int m, int n, dtype* a, \
                          int lda, dtype* tau, dtype* work, int lwork) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  #define MXNET_LAPACK_CWRAPPER3(func, dtype) \
  int MXNET_LAPACK_##func(int matrix_layout, char uplo, int n, dtype *a, \
                          int lda, dtype *w, dtype *work, int lwork, \
                          int *iwork, int liwork) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  #define MXNET_LAPACK_CWRAPPER4(func, dtype) \
  int MXNET_LAPACK_##func(int matrix_layout, int m, int n, \
                          dtype *a, int lda, int *ipiv) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  #define MXNET_LAPACK_CWRAPPER5(func, dtype) \
  int MXNET_LAPACK_##func(int matrix_layout, int n, dtype *a, int lda, \
                          int *ipiv, dtype *work, int lwork) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  #define MXNET_LAPACK_CWRAPPER6(func, dtype) \
  int MXNET_LAPACK_##func(int matrix_layout, int m, int n, dtype* ut, \
                          int ldut, dtype* s, dtype* v, int ldv, \
                          dtype* work, int lwork) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  #define MXNET_LAPACK_UNAVAILABLE(func) \
  int mxnet_lapack_##func(...) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }
  MXNET_LAPACK_CWRAPPER1(spotrf, float)
  MXNET_LAPACK_CWRAPPER1(dpotrf, double)
  MXNET_LAPACK_CWRAPPER1(spotri, float)
  MXNET_LAPACK_CWRAPPER1(dpotri, double)

  MXNET_LAPACK_UNAVAILABLE(sposv)
  MXNET_LAPACK_UNAVAILABLE(dposv)

  MXNET_LAPACK_CWRAPPER2(sgelqf, float)
  MXNET_LAPACK_CWRAPPER2(dgelqf, double)
  MXNET_LAPACK_CWRAPPER2(sorglq, float)
  MXNET_LAPACK_CWRAPPER2(dorglq, double)

  MXNET_LAPACK_CWRAPPER3(ssyevd, float)
  MXNET_LAPACK_CWRAPPER3(dsyevd, double)

  MXNET_LAPACK_CWRAPPER4(sgetrf, float)
  MXNET_LAPACK_CWRAPPER4(dgetrf, double)

  MXNET_LAPACK_CWRAPPER5(sgetri, float)
  MXNET_LAPACK_CWRAPPER5(dgetri, double)

  MXNET_LAPACK_CWRAPPER6(sgesvd, float)
  MXNET_LAPACK_CWRAPPER6(dgesvd, double)

#endif  // MSHADOW_USE_MKL == 0
