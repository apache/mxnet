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
 * \file dot_engine-inl.h
 * \brief definitions of how Matrix Multiplications can be evaluated
 * \author Tianqi Chen
 */
#ifndef MSHADOW_DOT_ENGINE_INL_H_
#define MSHADOW_DOT_ENGINE_INL_H_

#include <vector>
#include "./base.h"
#include "./extension/implicit_gemm.h"

#ifdef __CUDACC__
#include "./cuda/tensor_gpu-inl.cuh"
#endif  // #ifdef __CUDACC__

namespace mshadow {
 /*!
* \brief CPU/GPU: Get a batched view of the src array. dst[i] = src + i * stride
* \param dst 2D pointer
* \param src 1D pointer
* \param num number of batches
* \param stride size of each batch
* \param stream
*/
template<typename Device, typename DType>
inline void GetBatchedView(DType **dst, DType *src, int num, int stride,
                           Stream<Device> *stream);
template<typename DType>
inline void GetBatchedView(DType **dst, DType *src, int num, int stride,
                           Stream<cpu> *stream) {
  for (int i = 0; i < num; i++) {
    dst[i] = src + i * stride;
  }
}
#ifdef __CUDACC__
namespace cuda {};
template<typename DType>
inline void GetBatchedView(DType **dst, DType *src, int num, int stride,
                           Stream<gpu> *stream) {
  cuda::GetBatchedView(dst, src, num, stride, stream);
}
#endif  // #ifdef __CUDACC__

namespace expr {
//---------------------------------------------------------------------
// Matrix Multiplications, depends on BLAS Engine
//---------------------------------------------------------------------
template<typename SV, typename Device, int ddim, int ldim,
         int rdim, bool ltrans, bool rtrans, typename DType>
struct DotEngine {
  inline static void Eval(Tensor<Device, ddim, DType> *p_dst,
                          const Tensor<Device, ldim, DType> &lhs,
                          const Tensor<Device, rdim, DType> &rhs,
                          DType scale);
};
// handles the dot, use CblasColMajor
template<typename Device, typename DType = default_real_t>
struct BLASEngine {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<Device> *stream) {
  }
  inline static void gemm(Stream<Device> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, DType alpha,
                          const DType *A, int lda, const DType *B, int ldb,
                          DType beta, DType *C, int ldc) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemm(Stream<Device> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, DType alpha,
                                  const DType *A, int lda, const DType *B, int ldb,
                                  DType beta, DType *C, int ldc, int batch_count,
                                  DType **workspace) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void gemv(Stream<Device> *stream,
                          bool trans, int m, int n,
                          DType alpha, const DType *A, int lda,
                          const DType *X, int incX,
                          DType beta, DType *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<Device> *stream,
                                  bool trans, int m, int n,
                                  DType alpha, const DType *A, int lda,
                                  const DType *X, int incX,
                                  DType beta, DType *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<Device> *stream,
                         int m, int n, DType alpha,
                         const DType *X, int incX,
                         const DType *Y, int incY, DType *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<Device> *stream,
                         int m, int n, DType alpha,
                         const DType *X, int incX,
                         const DType *Y, int incY, DType *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<Device> *stream,
                         int n,
                         const DType* X, int incX,
                         const DType* Y, int incY,
                         DType* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

#if MSHADOW_STAND_ALONE
template<>
struct BLASEngine<cpu, float> {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
    if (alpha == 1.0f && beta == 0.0f) {
      bool transpose_left = transb;
      bool transpose_right = transa;
      Tensor<cpu, 2, float> lhs((float*)B, Shape2(transpose_left ? k : n, transpose_left ? n : k));  // NOLINT(*)
      Tensor<cpu, 2, float> rhs((float*)A, Shape2(transpose_right ? m : k, transpose_right ? k : m));  // NOLINT(*)
      Tensor<cpu, 2, float> dst(C, Shape2(m, n));
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      } else {
        LOG(FATAL) << "Not implmented!";
      }
    } else {
      LOG(FATAL) << "Not implmented!";
    }
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, float alpha,
                                  const float *A, int lda, const float *B, int ldb,
                                  float beta, float *C, int ldc, int batch_count,
                                  float **workspace) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          float alpha, const float *A, int lda,
                          const float *X, int incX,
                          float beta, float *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  float alpha, const float *A, int lda,
                                  const float *X, int incX,
                                  float beta, float *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

template<>
struct BLASEngine<cpu, double> {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda, const double *B, int ldb,
                          double beta, double *C, int ldc) {
    if (alpha == 1.0f && beta == 0.0f) {
      bool transpose_left = transb;
      bool transpose_right = transa;
      Tensor<cpu, 2, double> lhs((double*)B, Shape2(transpose_left ? k : n, transpose_left ? n : k));  // NOLINT(*)
      Tensor<cpu, 2, double> rhs((double*)A, Shape2(transpose_right ? m : k, transpose_right ? k : m));  // NOLINT(*)
      Tensor<cpu, 2, double> dst(C, Shape2(m, n));
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      } else {
        LOG(FATAL) << "Not implmented!";
      }
    } else {
      LOG(FATAL) << "Not implmented!";
    }
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, double alpha,
                                  const double *A, int lda, const double *B, int ldb,
                                  double beta, double *C, int ldc, int batch_count,
                                  double **workspace) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          double alpha, const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  double alpha, const double *A, int lda,
                                  const double *X, int incX,
                                  double beta, double *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

#elif (MSHADOW_USE_MKL || MSHADOW_USE_CBLAS)  // NOLINT(*)
template<>
struct BLASEngine<cpu, float> {
  inline static CBLAS_TRANSPOSE GetT(bool t) {
    return t ? CblasTrans : CblasNoTrans;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
    cblas_sgemm(CblasColMajor, GetT(transa), GetT(transb),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, float alpha,
                                  const float *A, int lda, const float *B, int ldb,
                                  float beta, float *C, int ldc, int batch_count,
                                  float **workspace) {
#if (MSHADOW_USE_MKL && INTEL_MKL_VERSION >= 20160000)
  // since same m/n/k is used for all single gemms, so we put all gemms into one group
  const int GROUP_SIZE = 1;
  MKL_INT p_m[GROUP_SIZE] = {m};
  MKL_INT p_n[GROUP_SIZE] = {n};
  MKL_INT p_k[GROUP_SIZE] = {k};
  MKL_INT p_lda[GROUP_SIZE] = {lda};
  MKL_INT p_ldb[GROUP_SIZE] = {ldb};
  MKL_INT p_ldc[GROUP_SIZE] = {ldc};

  float p_alpha[GROUP_SIZE] = {alpha};
  float p_beta[GROUP_SIZE] = {beta};

  CBLAS_TRANSPOSE cblas_a_trans = GetT(transa);
  CBLAS_TRANSPOSE cblas_b_trans = GetT(transb);

  MKL_INT p_group_sizeb[GROUP_SIZE] = {batch_count};
  CBLAS_TRANSPOSE p_transa[GROUP_SIZE] = {cblas_a_trans};
  CBLAS_TRANSPOSE p_transb[GROUP_SIZE] = {cblas_b_trans};

  std::vector<const float*> pp_A(batch_count, nullptr);
  std::vector<const float*> pp_B(batch_count, nullptr);
  std::vector<float*> pp_C(batch_count, nullptr);

  auto m_k = m * k;
  auto k_n = k * n;
  auto m_n = m * n;

  for (int i = 0; i < batch_count; i++) {
    pp_A[i] = A + i * m_k;
    pp_B[i] = B + i * k_n;
    pp_C[i] = C + i * m_n;
  }

  cblas_sgemm_batch(CblasColMajor, p_transa, p_transb,
                    p_m, p_n, p_k, p_alpha, pp_A.data(), p_lda, pp_B.data(),
                    p_ldb, p_beta, pp_C.data(), p_ldc, GROUP_SIZE, p_group_sizeb);
#else
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
#endif
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          float alpha, const float *A, int lda,
                          const float *X, int incX,
                          float beta, float *Y, int incY) {
    cblas_sgemv(CblasColMajor, GetT(trans), m, n, alpha,
                A, lda, X, incX, beta, Y, incY);
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  float alpha, const float *A, int lda,
                                  const float *X, int incX,
                                  float beta, float *Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda,
           X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    cblas_sger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY,
          A + i * lda * n, lda);
    }
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float* ret) {
    *ret = cblas_sdot(n, X, incX, Y, incY);
  }
};

template<>
struct BLASEngine<cpu, double> {
  inline static CBLAS_TRANSPOSE GetT(bool t) {
    return t ? CblasTrans : CblasNoTrans;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda, const double *B, int ldb,
                          double beta, double *C, int ldc) {
    cblas_dgemm(CblasColMajor, GetT(transa), GetT(transb),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, double alpha,
                                  const double *A, int lda, const double *B, int ldb,
                                  double beta, double *C, int ldc, int batch_count,
                                  double **workspace) {
#if (MSHADOW_USE_MKL && INTEL_MKL_VERSION >= 20160000)
  // since same m/n/k is used for all single gemms, so we put all gemms into one group
  const int GROUP_SIZE = 1;
  MKL_INT p_m[GROUP_SIZE] = {m};
  MKL_INT p_n[GROUP_SIZE] = {n};
  MKL_INT p_k[GROUP_SIZE] = {k};
  MKL_INT p_lda[GROUP_SIZE] = {lda};
  MKL_INT p_ldb[GROUP_SIZE] = {ldb};
  MKL_INT p_ldc[GROUP_SIZE] = {ldc};

  double p_alpha[GROUP_SIZE] = {alpha};
  double p_beta[GROUP_SIZE] = {beta};

  CBLAS_TRANSPOSE cblas_a_trans = GetT(transa);
  CBLAS_TRANSPOSE cblas_b_trans = GetT(transb);

  MKL_INT p_group_sizeb[GROUP_SIZE] = {batch_count};
  CBLAS_TRANSPOSE p_transa[GROUP_SIZE] = {cblas_a_trans};
  CBLAS_TRANSPOSE p_transb[GROUP_SIZE] = {cblas_b_trans};

  std::vector<const double*> pp_A(batch_count, nullptr);
  std::vector<const double*> pp_B(batch_count, nullptr);
  std::vector<double*> pp_C(batch_count, nullptr);

  auto m_k = m * k;
  auto k_n = k * n;
  auto m_n = m * n;

  for (int i = 0; i < batch_count; i++) {
    pp_A[i] = A + i * m_k;
    pp_B[i] = B + i * k_n;
    pp_C[i] = C + i * m_n;
  }

  cblas_dgemm_batch(CblasColMajor, p_transa, p_transb,
                    p_m, p_n, p_k, p_alpha, pp_A.data(), p_lda, pp_B.data(),
                    p_ldb, p_beta, pp_C.data(), p_ldc, GROUP_SIZE, p_group_sizeb);
#else
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
#endif
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n, double alpha,
                          const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    cblas_dgemv(CblasColMajor, GetT(trans), m, n, alpha,
                A, lda, X, incX, beta, Y, incY);
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  double alpha, const double *A, int lda,
                                  const double *X, int incX,
                                  double beta, double *Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda,
           X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    cblas_dger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY,
          A + i * lda * n, lda);
    }
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double* ret) {
    *ret = cblas_ddot(n, X, incX, Y, incY);
  }
};
#endif  // MSHADOW_USE_CBLAS || MSHADOW_USE_MKL || MSHADOW_STAND_ALONE
// CuBLAS redirect code
#if MSHADOW_USE_CUDA
// All CuBLAS goes to here, use legacy API: not threadsafe
template<>
struct BLASEngine<gpu, half::half_t> {
  inline static cublasOperation_t GetT(bool t) {
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
  inline static void SetStream(Stream<gpu> *stream) {
    cublasStatus_t err = cublasSetStream(Stream<gpu>::GetBlasHandle(stream),
                    Stream<gpu>::GetStream(stream));
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas set stream fail";
  }
  inline static void gemm(Stream<gpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, half::half_t alpha,
                          const half::half_t *A, int lda,
                          const half::half_t *B, int ldb, half::half_t beta,
                          half::half_t *C, int ldc) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 7050
  // Always use pseudo-fp16: fp32 compute with fp16 I/O.
  float alpha_f = float(alpha);  // NOLINT(*)
  float beta_f = float(beta);  // NOLINT(*)
  #if CUDA_VERSION >= 8000
    cublasStatus_t err = cublasSgemmEx(Stream<gpu>::GetBlasHandle(stream),
                                       GetT(transa), GetT(transb), m, n, k, &alpha_f,
                                       A, CUDA_R_16F, lda, B, CUDA_R_16F,
                                       ldb, &beta_f, C, CUDA_R_16F, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas SgemmEx fail";
  #else
    cublasStatus_t err = cublasSgemmEx(Stream<gpu>::GetBlasHandle(stream),
                                       GetT(transa), GetT(transb), m, n, k, &alpha_f,
                                       A, CUBLAS_DATA_HALF, lda, B, CUBLAS_DATA_HALF,
                                       ldb, &beta_f, C, CUBLAS_DATA_HALF, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas SgemmEx fail";
  #endif  // CUDA_VERSION >= 8000
#else
    LOG(FATAL) << "Require CUDA version >= 7.5!";
#endif  // defined(CUDA_VERSION) && CUDA_VERSION >= 7050
  }
  inline static void batched_gemm(Stream<gpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, half::half_t alpha,
                                  const half::half_t *A, int lda, const half::half_t *B, int ldb,
                                  half::half_t beta, half::half_t *C, int ldc, int batch_count,
                                  half::half_t **workspace) {
#if defined(__CUDACC__) && CUDA_VERSION >= 9000
    int major = stream->prop.major;
    int minor = stream->prop.minor;
    // fp16 is not supported before ARCH 53
    if ((major > 5) || (major == 5 && minor >= 3)) {
      const __half* A_h = reinterpret_cast<const __half*>(A);
      const __half* B_h = reinterpret_cast<const __half*>(B);
      __half* alpha_h = reinterpret_cast<__half*>(&alpha);
      __half* beta_h = reinterpret_cast<__half*>(&beta);
      __half* C_h = reinterpret_cast<__half*>(C);
      cublasStatus_t err = cublasHgemmStridedBatched(Stream<gpu>::GetBlasHandle(stream),
        GetT(transa), GetT(transb), m, n, k, alpha_h,
        A_h, lda, m * k,
        B_h, ldb, k * n,
        beta_h, C_h, ldc, m * n,
        batch_count);
      CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: HgemmStridedBatched fail";
      return;
    }
#endif
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<gpu> *stream,
                          bool trans, int m, int n, half::half_t alpha,
                          const half::half_t *A, int lda,
                          const half::half_t *X, int incX, half::half_t beta,
                          half::half_t *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<gpu> *stream,
                                  bool trans, int m, int n,
                                  half::half_t alpha, const half::half_t *A, int lda,
                                  const half::half_t *X, int incX,
                                  half::half_t beta, half::half_t *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<gpu> *stream,
                         int m, int n, half::half_t alpha,
                         const half::half_t *X, int incX,
                         const half::half_t *Y, int incY, half::half_t *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<gpu> *stream,
                         int m, int n, half::half_t alpha,
                         const half::half_t *X, int incX, const half::half_t *Y, int incY,
                         half::half_t *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<gpu> *stream,
                         int n,
                         const half::half_t* X, int incX,
                         const half::half_t* Y, int incY,
                         half::half_t *ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

template<>
struct BLASEngine<gpu, float> {
  inline static cublasOperation_t GetT(bool t) {
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
  inline static void SetStream(Stream<gpu> *stream) {
    cublasStatus_t err = cublasSetStream(Stream<gpu>::GetBlasHandle(stream),
                    Stream<gpu>::GetStream(stream));
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: set stream fail";
  }
  inline static void gemm(Stream<gpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda,
                          const float *B, int ldb, float beta,
                          float *C, int ldc) {
    cublasStatus_t err = cublasSgemm(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha,
                A, lda, B, ldb, &beta, C, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Sgemm fail";
  }
  inline static void batched_gemm(Stream<gpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, float alpha,
                                  const float *A, int lda, const float *B, int ldb,
                                  float beta, float *C, int ldc, int batch_count,
                                  float **workspace) {
#if defined(__CUDACC__) && CUDA_VERSION >= 4010 && CUDA_VERSION < 8000
    // Cast DType* to DType** using workspace as a buffer
    bool alloc_workspace = false;
    if (workspace == NULL) {
      // Allocate the workspace if it's NULL.
      // TODO(sxjscience) Try to move the allocation inside Tensor, which is thread-safe.
      cudaMalloc(reinterpret_cast<void**>(&workspace), 3 * batch_count * sizeof(float*));
      alloc_workspace = true;
    }
    GetBatchedView(workspace, const_cast<float*>(A), batch_count, m * k, stream);
    GetBatchedView(workspace + batch_count,
                   const_cast<float*>(B), batch_count, k * n, stream);
    GetBatchedView(workspace + 2 * batch_count, C, batch_count, m * n, stream);
    cublasStatus_t err = cublasSgemmBatched(Stream<gpu>::GetBlasHandle(stream),
                                            GetT(transa), GetT(transb), m, n, k, &alpha,
                                            (const float**)workspace, lda,
                                            (const float**)(workspace + batch_count), ldb,
                                            &beta, workspace + 2 * batch_count, ldc, batch_count);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: SgemmBatched fail";
    if (alloc_workspace) {
      cudaFree(workspace);
    }
#elif defined(__CUDACC__) && CUDA_VERSION >= 8000
    cublasStatus_t err = cublasSgemmStridedBatched(Stream<gpu>::GetBlasHandle(stream),
      GetT(transa), GetT(transb), m, n, k, &alpha,
      A, lda, m * k,
      B, ldb, k * n,
      &beta, C, ldc, m * n,
      batch_count);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: SgemmStridedBatched fail";
#else
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
#endif  // defined(__CUDACC__) && CUDA_VERSION >= 4010
  }
  inline static void gemv(Stream<gpu> *stream,
                          bool trans, int m, int n, float alpha,
                          const float *A, int lda,
                          const float *X, int incX, float beta,
                          float *Y, int incY) {
    cublasStatus_t err = cublasSgemv(Stream<gpu>::GetBlasHandle(stream),
                GetT(trans), m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Sgemv fail";
  }
  inline static void batched_gemv(Stream<gpu> *stream,
                                  bool trans, int m, int n,
                                  float alpha, const float *A, int lda,
                                  const float *X, int incX,
                                  float beta, float *Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda,
           X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<gpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    cublasStatus_t err = cublasSger(Stream<gpu>::GetBlasHandle(stream),
                                    m, n, &alpha, X, incX, Y, incY, A, lda);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Sger fail";
  }
  inline static void batched_ger(Stream<gpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY,
          A + i * lda * n, lda);
    }
  }
  inline static void dot(Stream<gpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float *ret) {
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t err = cublasSdot(Stream<gpu>::GetBlasHandle(stream),
                                    n, X, incX, Y, incY, ret);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_HOST);
  }
};

template<>
struct BLASEngine<gpu, double> {
  inline static cublasOperation_t GetT(bool t) {
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
  inline static void SetStream(Stream<gpu> *stream) {
    cublasStatus_t err = cublasSetStream(Stream<gpu>::GetBlasHandle(stream),
                    Stream<gpu>::GetStream(stream));
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: set stream fail";
  }
  inline static void gemm(Stream<gpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda,
                          const double *B, int ldb,
                          double beta, double *C, int ldc) {
    cublasStatus_t err = cublasDgemm(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha,
                A, lda, B, ldb, &beta, C, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dgemm fail";
  }
  inline static void batched_gemm(Stream<gpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, double alpha,
                                  const double *A, int lda, const double *B, int ldb,
                                  double beta, double *C, int ldc, int batch_count,
                                  double **workspace) {
#if defined(__CUDACC__) && CUDA_VERSION >= 4010 && CUDA_VERSION < 8000
    // Cast DType* to DType** using workspace as a buffer
    bool alloc_workspace = false;
    if (workspace == NULL) {
      // Allocate the workspace if it's NULL.
      // TODO(sxjscience) Try to move the allocation inside Tensor, which is thread-safe.
      cudaMalloc(reinterpret_cast<void**>(&workspace), 3 * batch_count * sizeof(double*));
      alloc_workspace = true;
    }
    GetBatchedView(workspace, const_cast<double*>(A), batch_count, m * k, stream);
    GetBatchedView(workspace + batch_count,
                   const_cast<double*>(B), batch_count, k * n, stream);
    GetBatchedView(workspace + 2 * batch_count, C, batch_count, m * n, stream);
    cublasStatus_t err = cublasDgemmBatched(Stream<gpu>::GetBlasHandle(stream),
                                            GetT(transa), GetT(transb), m, n, k, &alpha,
                                            (const double**)workspace, lda,
                                            (const double**)(workspace + batch_count), ldb,
                                            &beta, workspace + 2 * batch_count, ldc, batch_count);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: DgemmBatched fail";
    if (alloc_workspace) {
      cudaFree(workspace);
    }
#elif defined(__CUDACC__) && CUDA_VERSION >= 8000
    cublasStatus_t err = cublasDgemmStridedBatched(Stream<gpu>::GetBlasHandle(stream),
      GetT(transa), GetT(transb), m, n, k, &alpha,
      A, lda, m * k,
      B, ldb, k * n,
      &beta, C, ldc, m * n,
      batch_count);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: DgemmStridedBatched fail";
#else
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
#endif  // defined(__CUDACC__) && CUDA_VERSION >= 4010
  }
  inline static void gemv(Stream<gpu> *stream,
                          bool trans, int m, int n, double alpha,
                          const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    cublasStatus_t err = cublasDgemv(Stream<gpu>::GetBlasHandle(stream),
                GetT(trans), m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dgemv fail";
  }
  inline static void batched_gemv(Stream<gpu> *stream,
                                  bool trans, int m, int n,
                                  double alpha, const double *A, int lda,
                                  const double *X, int incX,
                                  double beta, double *Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda,
           X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<gpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    cublasStatus_t err = cublasDger(Stream<gpu>::GetBlasHandle(stream),
                                    m, n, &alpha, X, incX, Y, incY, A, lda);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dger fail";
  }
  inline static void batched_ger(Stream<gpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY,
          A + i * lda * n, lda);
    }
  }
  inline static void dot(Stream<gpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double *ret) {
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t err = cublasDdot(Stream<gpu>::GetBlasHandle(stream),
                                    n, X, incX, Y, incY, ret);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_HOST);
  }
};
#endif  // MSHADOW_USE_CUDA
// helper function to decide which shape we are in
inline Shape<2> GetShape(const Shape<2> &shape, bool transpose) {
  return transpose ? Shape2(shape[1], shape[0]) : shape;
}
// dst = dot(lhs[.T], rhs[.T])
template<typename SV, typename xpu,
         bool transpose_left, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 2, 2, 2, transpose_left, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 2, DType> *p_dst,
                          const Tensor<xpu, 2, DType> &lhs,
                          const Tensor<xpu, 2, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 2, DType> &dst = *p_dst;
#if MSHADOW_STAND_ALONE
    if (xpu::kDevMask == cpu::kDevMask && scale == 1.0f) {
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      }
    }
#endif
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    Shape<2> sleft = GetShape(lhs.shape_, transpose_left);
    Shape<2> sright = GetShape(rhs.shape_, transpose_right);
    CHECK(dst.size(0) == sleft[0] && dst.size(1) == sright[1] && sleft[1] == sright[0])
      << "dot-gemm: matrix shape mismatch";
    // use column major argument to compatible with most BLAS
    BLASEngine<xpu, DType>::gemm
        (dst.stream_,
         transpose_right , transpose_left,
         transpose_right ? rhs.size(0) : rhs.size(1),
         transpose_left  ? lhs.size(1) : lhs.size(0),
         transpose_right ? rhs.size(1) : rhs.size(0),
         DType(scale * SV::AlphaBLAS()),
         rhs.dptr_, rhs.stride_,
         lhs.dptr_, lhs.stride_,
         DType(SV::BetaBLAS()),
         dst.dptr_, dst.stride_);
  }
};
template<typename SV, typename xpu, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 1, 1, 2, false, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 1, DType> *p_dst,
                          const Tensor<xpu, 1, DType> &lhs,
                          const Tensor<xpu, 2, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 1, DType> &dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    Shape<2> sright = GetShape(rhs.shape_, transpose_right);
    CHECK(dst.size(0) == sright[1] && lhs.size(0) == sright[0])
      << "dot-gemv: matrix shape mismatch"
      << "dst: " << dst.shape_ << "\n"
      << "lhs: " << lhs.shape_ << "\n"
      << "rhs: " << sright << "\n";
    BLASEngine<xpu, DType>::gemv
        (dst.stream_,
         transpose_right,
         rhs.size(1), rhs.size(0), scale * SV::AlphaBLAS(),
         rhs.dptr_, rhs.stride_,
         lhs.dptr_, 1, SV::BetaBLAS(),
         dst.dptr_, 1);
  }
};
template<typename SV, typename xpu, typename DType>
struct DotEngine<SV, xpu, 2, 1, 1, true, false, DType> {
  inline static void Eval(Tensor<xpu, 2, DType> *p_dst,
                          const Tensor<xpu, 1, DType> &lhs,
                          const Tensor<xpu, 1, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 2, DType> &dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    CHECK(dst.size(0) == lhs.size(0) && dst.size(1) == rhs.size(0))
      << "dot-ger: matrix shape mismatch"
      << "dst: " << dst.shape_ << "\n"
      << "lhs: " << lhs.shape_ << "\n"
      << "rhs: " << rhs.shape_;
    if (SV::BetaBLAS() == 0.0f) {
      BLASEngine<xpu, DType>::ger
          (dst.stream_, rhs.size(0), lhs.size(0), scale * SV::AlphaBLAS(),
           rhs.dptr_, 1, lhs.dptr_, 1, dst.dptr_, dst.stride_);
    } else {
      DotEngine<SV, xpu, 2, 2, 2, true, false,
                DType>::Eval(p_dst, lhs.FlatTo2D(), rhs.FlatTo2D(), scale);
    }
  }
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_DOT_ENGINE_INL_H_
