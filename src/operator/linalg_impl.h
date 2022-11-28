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
 * \file linalg_impl.h
 * \brief Implementation of unified tensor interface for advanced linear algebra functions
 * (specifically BLAS3/LAPACK) from within mxnet.
 */
#ifndef MXNET_OPERATOR_LINALG_IMPL_H_
#define MXNET_OPERATOR_LINALG_IMPL_H_

#include <mxnet/op_attr_types.h>

#include <algorithm>

#include "../common/cuda/utils.h"
#include "mxnet_op.h"

// Convenience functions.
inline void linalg_check_batch_size(int A, int B, int C) {
  CHECK_EQ(A, B) << "Inconsistent batch size between arguments to linear algebra operator";
  CHECK_EQ(A, C) << "Inconsistent batch size between arguments to linear algebra operator";
  CHECK_GT(A, 0) << "Zero batch size for arguments to linear algebra operator";
}

#ifdef __CUDACC__
#define EPHEMERAL_GPU_STORAGE_ALLOC(func, var, dtype, size)                          \
  Storage::Handle var = Storage::Get()->Alloc(sizeof(dtype) * size, Context::GPU()); \
  var.profiler_scope  = "<ephemeral>:";                                              \
  var.name            = #func "_" #var;
#endif

//////////////////////////////// GEMM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "gemm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is C = gemm(A,B,C), so C is input and output parameter.

template <typename xpu, typename DType>
inline void check_gemm(const Tensor<xpu, 2, DType>& A,
                       const Tensor<xpu, 2, DType>& B,
                       const Tensor<xpu, 2, DType>& C,
                       DType alpha,
                       DType beta,
                       bool tA,
                       bool tB) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ((tA ? A.size(1) : A.size(0)), C.size(0))
      << "Non compatible matrix dimensions between inputs A and C for gemm";
  CHECK_EQ((tB ? B.size(0) : B.size(1)), C.size(1))
      << "Non compatible matrix dimensions between inputs B and C for gemm";
  CHECK_EQ((tA ? A.size(0) : A.size(1)), (tB ? B.size(1) : B.size(0)))
      << "Non compatible matrix dimensions between inputs A and B for gemm";
}

template <typename xpu, typename DType>
void linalg_gemm_axis(const Tensor<xpu, 3, DType>& A,
                      const Tensor<xpu, 3, DType>& B,
                      const Tensor<xpu, 3, DType>& C,
                      DType alpha,
                      DType beta,
                      bool tA,
                      bool tB,
                      Stream<xpu>* s = 0);

#if (MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1)

#define LINALG_CPU_GEMM(fname, DType)                                 \
  template <>                                                         \
  inline void linalg_gemm<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                      const Tensor<cpu, 2, DType>& B, \
                                      const Tensor<cpu, 2, DType>& C, \
                                      DType alpha,                    \
                                      DType beta,                     \
                                      bool tA,                        \
                                      bool tB,                        \
                                      Stream<cpu>* s) {               \
    check_gemm(A, B, C, alpha, beta, tA, tB);                         \
    cblas_##fname(CblasRowMajor,                                      \
                  (tA ? CblasTrans : CblasNoTrans),                   \
                  (tB ? CblasTrans : CblasNoTrans),                   \
                  C.size(0),                                          \
                  C.size(1),                                          \
                  (tA ? A.size(0) : A.size(1)),                       \
                  alpha,                                              \
                  A.dptr_,                                            \
                  A.stride_,                                          \
                  B.dptr_,                                            \
                  B.stride_,                                          \
                  beta,                                               \
                  C.dptr_,                                            \
                  C.stride_);                                         \
  }

#define LINALG_XPU_BATCH_GEMM(xpu, DType)                                   \
  template <>                                                               \
  inline void linalg_batch_gemm<xpu, DType>(const Tensor<xpu, 3, DType>& A, \
                                            const Tensor<xpu, 3, DType>& B, \
                                            const Tensor<xpu, 3, DType>& C, \
                                            DType alpha,                    \
                                            DType beta,                     \
                                            bool tA,                        \
                                            bool tB,                        \
                                            Stream<xpu>* s) {               \
    linalg_check_batch_size(A.size(0), B.size(0), C.size(0));               \
    for (index_t i = 0; i < A.size(0); ++i) {                               \
      linalg_gemm(A[i], B[i], C[i], alpha, beta, tA, tB, s);                \
    }                                                                       \
  }

// Batched gemm where the batch coordinate is given by the second axis.
#define LINALG_CPU_GEMM_AXIS(fname, DType)                                 \
  template <>                                                              \
  inline void linalg_gemm_axis<cpu, DType>(const Tensor<cpu, 3, DType>& A, \
                                           const Tensor<cpu, 3, DType>& B, \
                                           const Tensor<cpu, 3, DType>& C, \
                                           DType alpha,                    \
                                           DType beta,                     \
                                           bool tA,                        \
                                           bool tB,                        \
                                           Stream<cpu>* s) {               \
    linalg_check_batch_size(A.size(1), B.size(1), C.size(1));              \
    for (index_t i = 0; i < A.size(1); ++i) {                              \
      cblas_##fname(CblasRowMajor,                                         \
                    (tA ? CblasTrans : CblasNoTrans),                      \
                    (tB ? CblasTrans : CblasNoTrans),                      \
                    C.size(0),                                             \
                    C.size(2),                                             \
                    (tA ? A.size(0) : A.size(2)),                          \
                    alpha,                                                 \
                    A.dptr_ + i * A.stride_,                               \
                    A.size(1) * A.stride_,                                 \
                    B.dptr_ + i * B.stride_,                               \
                    B.size(1) * B.stride_,                                 \
                    beta,                                                  \
                    C.dptr_ + i * C.stride_,                               \
                    C.size(1) * C.stride_);                                \
    }                                                                      \
  }

LINALG_CPU_GEMM_AXIS(sgemm, float)
LINALG_CPU_GEMM_AXIS(dgemm, double)

// Version where matrix rows are given by the second axis.
#define LINALG_XPU_BATCH_GEMM_AXIS(xpu, DType)                              \
  template <>                                                               \
  inline void linalg_batch_gemm<xpu, DType>(const Tensor<xpu, 4, DType>& A, \
                                            const Tensor<xpu, 4, DType>& B, \
                                            const Tensor<xpu, 4, DType>& C, \
                                            DType alpha,                    \
                                            DType beta,                     \
                                            bool tA,                        \
                                            bool tB,                        \
                                            Stream<xpu>* s) {               \
    linalg_check_batch_size(A.size(0), B.size(0), C.size(0));               \
    for (index_t i = 0; i < A.size(0); ++i) {                               \
      linalg_gemm_axis(A[i], B[i], C[i], alpha, beta, tA, tB, s);           \
    }                                                                       \
  }

#else

#define LINALG_CPU_GEMM(fname, DType)                                                             \
  template <>                                                                                     \
  inline void linalg_gemm<cpu, DType>(const Tensor<cpu, 2, DType>& A,                             \
                                      const Tensor<cpu, 2, DType>& B,                             \
                                      const Tensor<cpu, 2, DType>& C,                             \
                                      DType alpha,                                                \
                                      DType beta,                                                 \
                                      bool tA,                                                    \
                                      bool tB,                                                    \
                                      Stream<cpu>* s) {                                           \
    LOG(FATAL) << "linalg_gemm (without req arg) not implemented by mxnet for cpu, needs cblas!"; \
  }

#define LINALG_XPU_BATCH_GEMM(xpu, DType)                                             \
  template <>                                                                         \
  inline void linalg_batch_gemm<xpu, DType>(const Tensor<xpu, 3, DType>& A,           \
                                            const Tensor<xpu, 3, DType>& B,           \
                                            const Tensor<xpu, 3, DType>& C,           \
                                            DType alpha,                              \
                                            DType beta,                               \
                                            bool tA,                                  \
                                            bool tB,                                  \
                                            Stream<xpu>* s) {                         \
    LOG(FATAL) << "linalg_batch_gemm not implemented by mxnet for cpu, needs cblas!"; \
  }

#define LINALG_XPU_BATCH_GEMM_AXIS(xpu, DType)                                        \
  template <>                                                                         \
  inline void linalg_batch_gemm<xpu, DType>(const Tensor<xpu, 4, DType>& A,           \
                                            const Tensor<xpu, 4, DType>& B,           \
                                            const Tensor<xpu, 4, DType>& C,           \
                                            DType alpha,                              \
                                            DType beta,                               \
                                            bool tA,                                  \
                                            bool tB,                                  \
                                            Stream<xpu>* s) {                         \
    LOG(FATAL) << "linalg_batch_gemm not implemented by mxnet for cpu, needs cblas!"; \
  }

#endif  // MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1

LINALG_CPU_GEMM(sgemm, float)
LINALG_CPU_GEMM(dgemm, double)

LINALG_XPU_BATCH_GEMM(cpu, float)
LINALG_XPU_BATCH_GEMM(cpu, double)

LINALG_XPU_BATCH_GEMM_AXIS(cpu, float)
LINALG_XPU_BATCH_GEMM_AXIS(cpu, double)

// Specialization of linalg_gemm<cpu, DType> for DType=mshadow::half::half_t.
template <>
inline void linalg_gemm<cpu, mshadow::half::half_t>(const Tensor<cpu, 2, mshadow::half::half_t>& A,
                                                    const Tensor<cpu, 2, mshadow::half::half_t>& B,
                                                    const Tensor<cpu, 2, mshadow::half::half_t>& C,
                                                    mshadow::half::half_t alpha,
                                                    mshadow::half::half_t beta,
                                                    bool tA,
                                                    bool tB,
                                                    Stream<cpu>* s) {
  LOG(FATAL) << "FP16 gemm on cpu not implemented!";
}

#ifdef __CUDACC__

// cublas col-major processing accounted for by switching first two operands

#define LINALG_GPU_GEMM(fname, DType)                                 \
  template <>                                                         \
  inline void linalg_gemm<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                      const Tensor<gpu, 2, DType>& B, \
                                      const Tensor<gpu, 2, DType>& C, \
                                      DType alpha,                    \
                                      DType beta,                     \
                                      bool tA,                        \
                                      bool tB,                        \
                                      Stream<gpu>* s) {               \
    using namespace mxnet;                                            \
    using mshadow::gpu;                                               \
    CHECK_NOTNULL(s);                                                 \
    check_gemm(A, B, C, alpha, beta, tA, tB);                         \
    CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s),          \
                              (tB ? CUBLAS_OP_T : CUBLAS_OP_N),       \
                              (tA ? CUBLAS_OP_T : CUBLAS_OP_N),       \
                              C.size(1),                              \
                              C.size(0),                              \
                              (tB ? B.size(1) : B.size(0)),           \
                              &alpha,                                 \
                              B.dptr_,                                \
                              B.stride_,                              \
                              A.dptr_,                                \
                              A.stride_,                              \
                              &beta,                                  \
                              C.dptr_,                                \
                              C.stride_))                             \
  }

// Use cublasSgemmEx when it is available (CUDA >= 7.5). Resolves precision issues with
// cublasSgemm. Please see https://github.com/apache/mxnet/pull/11630
#if CUDA_VERSION >= 7050
template <>
inline void linalg_gemm<gpu, float>(const Tensor<gpu, 2, float>& A,
                                    const Tensor<gpu, 2, float>& B,
                                    const Tensor<gpu, 2, float>& C,
                                    float alpha,
                                    float beta,
                                    bool tA,
                                    bool tB,
                                    Stream<gpu>* s) {
  using namespace mxnet;
  using mshadow::gpu;
  CHECK_NOTNULL(s);
  check_gemm(A, B, C, alpha, beta, tA, tB);
#if CUDA_VERSION >= 8000
  cudaDataType_t full_datatype = CUDA_R_32F;
#else
  cublasDataType_t full_datatype = CUBLAS_DATA_FULL;
#endif
  auto handle                  = Stream<gpu>::GetBlasHandle(s);
  cublasMath_t saved_math_mode = SetCublasMathMode(handle, VERSION_ADJUSTED_TF32_MATH);
  CUBLAS_CALL(cublasSgemmEx(handle,
                            (tB ? CUBLAS_OP_T : CUBLAS_OP_N),
                            (tA ? CUBLAS_OP_T : CUBLAS_OP_N),
                            C.size(1),
                            C.size(0),
                            (tB ? B.size(1) : B.size(0)),
                            &alpha,
                            B.dptr_,
                            full_datatype,
                            B.stride_,
                            A.dptr_,
                            full_datatype,
                            A.stride_,
                            &beta,
                            C.dptr_,
                            full_datatype,
                            C.stride_));
  CUBLAS_CALL(cublasSetMathMode(handle, saved_math_mode));
}

#else
LINALG_GPU_GEMM(Sgemm, float)
#endif
LINALG_GPU_GEMM(Dgemm, double)

// Version where matrix rows are given by first axis.
#define LINALG_GPU_GEMM_AXIS(fname, DType)                                                \
  template <>                                                                             \
  inline void linalg_gemm_axis<gpu, DType>(const Tensor<gpu, 3, DType>& A,                \
                                           const Tensor<gpu, 3, DType>& B,                \
                                           const Tensor<gpu, 3, DType>& C,                \
                                           DType alpha,                                   \
                                           DType beta,                                    \
                                           bool tA,                                       \
                                           bool tB,                                       \
                                           Stream<gpu>* s) {                              \
    using namespace mxnet;                                                                \
    using mshadow::gpu;                                                                   \
    CHECK_NOTNULL(s);                                                                     \
    linalg_check_batch_size(A.size(1), B.size(1), C.size(1));                             \
    auto handle                  = Stream<gpu>::GetBlasHandle(s);                         \
    cublasMath_t saved_math_mode = SetCublasMathMode(handle, VERSION_ADJUSTED_TF32_MATH); \
    CUBLAS_CALL(cublas##fname(handle,                                                     \
                              (tB ? CUBLAS_OP_T : CUBLAS_OP_N),                           \
                              (tA ? CUBLAS_OP_T : CUBLAS_OP_N),                           \
                              C.size(2),                                                  \
                              C.size(0),                                                  \
                              (tB ? B.size(2) : B.size(0)),                               \
                              &alpha,                                                     \
                              B.dptr_,                                                    \
                              B.size(1) * B.stride_,                                      \
                              B.stride_,                                                  \
                              A.dptr_,                                                    \
                              A.size(1) * A.stride_,                                      \
                              A.stride_,                                                  \
                              &beta,                                                      \
                              C.dptr_,                                                    \
                              C.size(1) * C.stride_,                                      \
                              C.stride_,                                                  \
                              A.size(1)))                                                 \
    CUBLAS_CALL(cublasSetMathMode(handle, saved_math_mode));                              \
  }
LINALG_GPU_GEMM_AXIS(SgemmStridedBatched, float)
LINALG_GPU_GEMM_AXIS(DgemmStridedBatched, double)

// Specialization of linalg_gemm<gpu, DType> for DType=mshadow::half::half_t.
// Specialization of linalg_gemm<gpu, DType> for DType=mshadow::half::half_t.
template <>
inline void linalg_gemm<gpu, mshadow::half::half_t>(const Tensor<gpu, 2, mshadow::half::half_t>& A,
                                                    const Tensor<gpu, 2, mshadow::half::half_t>& B,
                                                    const Tensor<gpu, 2, mshadow::half::half_t>& C,
                                                    mshadow::half::half_t alpha,
                                                    mshadow::half::half_t beta,
                                                    bool tA,
                                                    bool tB,
                                                    Stream<gpu>* s) {
  using namespace mxnet;
  using namespace mxnet::common::cuda;
  using mshadow::gpu;
  CHECK_NOTNULL(s);
  check_gemm(A, B, C, alpha, beta, tA, tB);

#if CUDA_VERSION >= 7050
  auto blas_handle = Stream<gpu>::GetBlasHandle(s);
#if CUDA_VERSION >= 9000
  auto cublas_math_mode   = GetEnvAllowTensorCore() ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  auto previous_math_mode = SetCublasMathMode(blas_handle, cublas_math_mode);
#endif

// As of cuda8, cublas adopted the cuda datatype, rather than maintaining its own datatype.
#if CUDA_VERSION >= 8000
  cudaDataType_t half_datatype = CUDA_R_16F;
#else
  cublasDataType_t half_datatype = CUBLAS_DATA_HALF;
#endif
  auto algo                       = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  using TrueFP16Type              = mshadow::half::half_t;
  using PseudoFP16Type            = typename CublasType<mshadow::half::half_t>::ScaleType;
  TrueFP16Type trueFP16_alpha     = static_cast<TrueFP16Type>(alpha);
  TrueFP16Type trueFP16_beta      = static_cast<TrueFP16Type>(beta);
  PseudoFP16Type pseudoFP16_alpha = static_cast<PseudoFP16Type>(alpha);
  PseudoFP16Type pseudoFP16_beta  = static_cast<PseudoFP16Type>(beta);
  const void* alpha_ptr;
  const void* beta_ptr;
  cudaDataType_t computeType;
  bool use_true_fp16 = dmlc::GetEnv("MXNET_FC_TRUE_FP16", false);
  if (use_true_fp16) {
    alpha_ptr   = &trueFP16_alpha;
    beta_ptr    = &trueFP16_beta;
    computeType = CublasType<TrueFP16Type>::kCudaFlag;
  } else {
    alpha_ptr   = &pseudoFP16_alpha;
    beta_ptr    = &pseudoFP16_beta;
    computeType = CublasType<PseudoFP16Type>::kCudaFlag;
  }
  if (SupportsFloat16Compute(s->dev_id)) {
    CUBLAS_CALL(cublasGemmEx(blas_handle,
                             (tB ? CUBLAS_OP_T : CUBLAS_OP_N),
                             (tA ? CUBLAS_OP_T : CUBLAS_OP_N),
                             C.size(1),
                             C.size(0),
                             (tB ? B.size(1) : B.size(0)),
                             alpha_ptr,
                             B.dptr_,
                             half_datatype,
                             B.stride_,
                             A.dptr_,
                             half_datatype,
                             A.stride_,
                             beta_ptr,
                             C.dptr_,
                             half_datatype,
                             C.stride_,
                             computeType,
                             algo));
  } else {
    // pseudo-fp16 (fp32 math with fp16 I/O)
    if (use_true_fp16)
      common::LogOnce("MXNET_FC_TRUE_FP16 was set but this architecture does not support it.");
    float alpha_f = static_cast<float>(alpha);
    float beta_f  = static_cast<float>(beta);
    CUBLAS_CALL(cublasSgemmEx(blas_handle,
                              (tB ? CUBLAS_OP_T : CUBLAS_OP_N),
                              (tA ? CUBLAS_OP_T : CUBLAS_OP_N),
                              C.size(1),
                              C.size(0),
                              (tB ? B.size(1) : B.size(0)),
                              &alpha_f,
                              B.dptr_,
                              half_datatype,
                              B.stride_,
                              A.dptr_,
                              half_datatype,
                              A.stride_,
                              &beta_f,
                              C.dptr_,
                              half_datatype,
                              C.stride_));
  }
#if CUDA_VERSION >= 9000
  SetCublasMathMode(blas_handle, previous_math_mode);
#endif
#else
  LOG(FATAL) << "FP16 gemm requires CUDA version >= 7.5!";
#endif  // CUDA_VERSION >= 7050
}

// As of cuda8, cublas has implemented a strided version of batch gemm.
#if CUDA_VERSION < 8000
LINALG_XPU_BATCH_GEMM(gpu, float)
LINALG_XPU_BATCH_GEMM(gpu, double)
LINALG_XPU_BATCH_GEMM_AXIS(gpu, float)
LINALG_XPU_BATCH_GEMM_AXIS(gpu, double)
#else
#define LINALG_GPU_BATCH_GEMM(fname, DType)                                               \
  template <>                                                                             \
  inline void linalg_batch_gemm<gpu, DType>(const Tensor<gpu, 3, DType>& A,               \
                                            const Tensor<gpu, 3, DType>& B,               \
                                            const Tensor<gpu, 3, DType>& C,               \
                                            DType alpha,                                  \
                                            DType beta,                                   \
                                            bool tA,                                      \
                                            bool tB,                                      \
                                            Stream<gpu>* s) {                             \
    using namespace mxnet;                                                                \
    using mshadow::gpu;                                                                   \
    CHECK_NOTNULL(s);                                                                     \
    linalg_check_batch_size(A.size(0), B.size(0), C.size(0));                             \
    check_gemm(A[0], B[0], C[0], alpha, beta, tA, tB);                                    \
    using namespace mshadow::cuda;                                                        \
    auto handle                  = Stream<gpu>::GetBlasHandle(s);                         \
    cublasMath_t saved_math_mode = SetCublasMathMode(handle, VERSION_ADJUSTED_TF32_MATH); \
    CUBLAS_CALL(cublas##fname(handle,                                                     \
                              (tB ? CUBLAS_OP_T : CUBLAS_OP_N),                           \
                              (tA ? CUBLAS_OP_T : CUBLAS_OP_N),                           \
                              C.size(2),                                                  \
                              C.size(1),                                                  \
                              (tB ? B.size(2) : B.size(1)),                               \
                              &alpha,                                                     \
                              B.dptr_,                                                    \
                              B.stride_,                                                  \
                              static_cast<int64_t>(B.size(1) * B.stride_),                \
                              A.dptr_,                                                    \
                              A.stride_,                                                  \
                              static_cast<int64_t>(A.size(1) * A.stride_),                \
                              &beta,                                                      \
                              C.dptr_,                                                    \
                              C.stride_,                                                  \
                              static_cast<int64_t>(C.size(1) * C.stride_),                \
                              A.size(0)))                                                 \
    CUBLAS_CALL(cublasSetMathMode(handle, saved_math_mode));                              \
  }

LINALG_GPU_BATCH_GEMM(DgemmStridedBatched, double)

#if CUDA_VERSION < 9010
LINALG_GPU_BATCH_GEMM(SgemmStridedBatched, float)
#else
template <>
inline void linalg_batch_gemm<gpu, float>(const Tensor<gpu, 3, float>& A,
                                          const Tensor<gpu, 3, float>& B,
                                          const Tensor<gpu, 3, float>& C,
                                          float alpha,
                                          float beta,
                                          bool tA,
                                          bool tB,
                                          Stream<gpu>* s) {
  using namespace mxnet;
  using mshadow::gpu;
  CHECK_NOTNULL(s);
  linalg_check_batch_size(A.size(0), B.size(0), C.size(0));
  check_gemm(A[0], B[0], C[0], alpha, beta, tA, tB);
  auto blas_handle    = Stream<gpu>::GetBlasHandle(s);
  bool use_tensor_ops = GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion();

  using namespace mshadow::cuda;
  auto cublas_math_mode   = use_tensor_ops ? CUBLAS_TENSOR_OP_MATH : VERSION_ADJUSTED_TF32_MATH;
  auto previous_math_mode = SetCublasMathMode(blas_handle, cublas_math_mode);

  // cublasGemmStridedBatchedEx is only supported for GPU with architecture
  // capabilities equal or greater than 5.0. Fall back to
  // cublasSgemmStridedBatched, which doesn't support implicit conversion
  // to half-precision to use TensorCores
  auto cc_major = (s->prop).major;
  if ((cc_major >= 5) && use_tensor_ops) {
    CUBLAS_CALL(cublasGemmStridedBatchedEx(blas_handle,
                                           (tB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                           (tA ? CUBLAS_OP_T : CUBLAS_OP_N),
                                           C.size(2),
                                           C.size(1),
                                           (tB ? B.size(2) : B.size(1)),
                                           &alpha,
                                           B.dptr_,
                                           CUDA_R_32F,
                                           B.stride_,
                                           B.size(1) * B.stride_,
                                           A.dptr_,
                                           CUDA_R_32F,
                                           A.stride_,
                                           A.size(1) * A.stride_,
                                           &beta,
                                           C.dptr_,
                                           CUDA_R_32F,
                                           C.stride_,
                                           C.size(1) * C.stride_,
                                           A.size(0),
                                           CUDA_R_32F,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    CUBLAS_CALL(cublasSgemmStridedBatched(blas_handle,
                                          (tB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                          (tA ? CUBLAS_OP_T : CUBLAS_OP_N),
                                          C.size(2),
                                          C.size(1),
                                          (tB ? B.size(2) : B.size(1)),
                                          &alpha,
                                          B.dptr_,
                                          B.stride_,
                                          B.size(1) * B.stride_,
                                          A.dptr_,
                                          A.stride_,
                                          A.size(1) * A.stride_,
                                          &beta,
                                          C.dptr_,
                                          C.stride_,
                                          C.size(1) * C.stride_,
                                          A.size(0)));
  }
  SetCublasMathMode(blas_handle, previous_math_mode);
}
#endif  // CUDA_VERSION < 9010

// Version where matrix rows are given by second axis.
#define LINALG_GPU_BATCH_GEMM_AXIS(fname, DType)                                          \
  template <>                                                                             \
  inline void linalg_batch_gemm<gpu, DType>(const Tensor<gpu, 4, DType>& A,               \
                                            const Tensor<gpu, 4, DType>& B,               \
                                            const Tensor<gpu, 4, DType>& C,               \
                                            DType alpha,                                  \
                                            DType beta,                                   \
                                            bool tA,                                      \
                                            bool tB,                                      \
                                            Stream<gpu>* s) {                             \
    using namespace mxnet;                                                                \
    using mshadow::gpu;                                                                   \
    CHECK_NOTNULL(s);                                                                     \
    linalg_check_batch_size(A.size(0), B.size(0), C.size(0));                             \
    linalg_check_batch_size(A.size(2), B.size(2), C.size(2));                             \
    auto handle                  = Stream<gpu>::GetBlasHandle(s);                         \
    cublasMath_t saved_math_mode = SetCublasMathMode(handle, VERSION_ADJUSTED_TF32_MATH); \
    for (index_t i = 0; i < A.size(2); ++i) {                                             \
      CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s),                            \
                                (tB ? CUBLAS_OP_T : CUBLAS_OP_N),                         \
                                (tA ? CUBLAS_OP_T : CUBLAS_OP_N),                         \
                                C.size(3),                                                \
                                C.size(1),                                                \
                                (tB ? B.size(3) : B.size(1)),                             \
                                &alpha,                                                   \
                                B.dptr_ + i * B.stride_,                                  \
                                B.size(2) * B.stride_,                                    \
                                B.size(1) * B.size(2) * B.stride_,                        \
                                A.dptr_ + i * A.stride_,                                  \
                                A.size(2) * A.stride_,                                    \
                                A.size(1) * A.size(2) * A.stride_,                        \
                                &beta,                                                    \
                                C.dptr_ + i * C.stride_,                                  \
                                C.size(2) * C.stride_,                                    \
                                C.size(1) * C.size(2) * C.stride_,                        \
                                A.size(0)))                                               \
    }                                                                                     \
    SetCublasMathMode(handle, saved_math_mode);                                           \
  }

LINALG_GPU_BATCH_GEMM_AXIS(SgemmStridedBatched, float)
LINALG_GPU_BATCH_GEMM_AXIS(DgemmStridedBatched, double)

#endif  // CUDA < 8000

#endif  // __CUDACC__

/*!
 * \brief Performs gemm, setting alpha and beta as appropriate for `req`.
 *
 * \param A the first operand of the gemm
 * \param B the second operand of the gemm
 * \param C the data to be assigned
 * \param tA whether the `A` operand should be transposed first.
 * \param tB whether the `B` operand should be transposed first.
 * \param s the stream to perform the operation
 * \param req the assignment request
 */
template <typename xpu, typename DType>
inline void linalg_gemm(const Tensor<xpu, 2, DType>& A,
                        const Tensor<xpu, 2, DType>& B,
                        const Tensor<xpu, 2, DType>& C,
                        bool tA,
                        bool tB,
                        Stream<xpu>* s,
                        mxnet::OpReqType req) {
  using namespace mxnet;
  switch (req) {
    case kNullOp:
      break;
    case kWriteTo:
    case kWriteInplace:
      linalg_gemm(A, B, C, DType(1.0), DType(0.0), tA, tB, s);
      break;
    case kAddTo:
      linalg_gemm(A, B, C, DType(1.0), DType(1.0), tA, tB, s);
      break;
    default:
      LOG(FATAL) << "not reached";
  }
}

#if (MSHADOW_USE_CBLAS == 0 && MSHADOW_USE_MKL == 0)

// A template for a cpu linalg_gemm implementation using mshadow::dot()
#define LINALG_CPU_GEMM_NO_CBLAS(DType)                                 \
  template <>                                                           \
  inline void linalg_gemm<cpu, DType>(const Tensor<cpu, 2, DType>& A,   \
                                      const Tensor<cpu, 2, DType>& B,   \
                                      const Tensor<cpu, 2, DType>& C,   \
                                      bool tA,                          \
                                      bool tB,                          \
                                      Stream<cpu>* s,                   \
                                      mxnet::OpReqType req) {           \
    using namespace mxnet;                                              \
    using mshadow::cpu;                                                 \
    switch (req) {                                                      \
      case kNullOp:                                                     \
        break;                                                          \
      case kWriteTo:                                                    \
      case kWriteInplace:                                               \
        if (tA) {                                                       \
          if (tB) {                                                     \
            const_cast<Tensor<cpu, 2, DType>&>(C) = dot(A.T(), B.T());  \
          } else {                                                      \
            const_cast<Tensor<cpu, 2, DType>&>(C) = dot(A.T(), B);      \
          }                                                             \
        } else {                                                        \
          if (tB) {                                                     \
            const_cast<Tensor<cpu, 2, DType>&>(C) = dot(A, B.T());      \
          } else {                                                      \
            const_cast<Tensor<cpu, 2, DType>&>(C) = dot(A, B);          \
          }                                                             \
        }                                                               \
        break;                                                          \
      case kAddTo:                                                      \
        if (tA) {                                                       \
          if (tB) {                                                     \
            const_cast<Tensor<cpu, 2, DType>&>(C) += dot(A.T(), B.T()); \
          } else {                                                      \
            const_cast<Tensor<cpu, 2, DType>&>(C) += dot(A.T(), B);     \
          }                                                             \
        } else {                                                        \
          if (tB) {                                                     \
            const_cast<Tensor<cpu, 2, DType>&>(C) += dot(A, B.T());     \
          } else {                                                      \
            const_cast<Tensor<cpu, 2, DType>&>(C) += dot(A, B);         \
          }                                                             \
        }                                                               \
        break;                                                          \
      default:                                                          \
        LOG(FATAL) << "not reached";                                    \
    }                                                                   \
  }

LINALG_CPU_GEMM_NO_CBLAS(float)
LINALG_CPU_GEMM_NO_CBLAS(double)

#endif  // (MSHADOW_USE_CBLAS == 0 && MSHADOW_USE_MKL == 0)

//////////////////////////////// TRSM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "trsm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is B = trsm(A,B), so B is input and output parameter.

template <typename xpu, typename DType>
inline void check_trsm(const Tensor<xpu, 2, DType>& A,
                       const Tensor<xpu, 2, DType>& B,
                       DType alpha,
                       bool rightside,
                       bool lower,
                       bool transpose) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1)) << "First input of trsm is not a square matrix.";
  CHECK(!rightside || (B.size(1) == A.size(0)))
      << "Non compatible matrix dimensions between inputs A and B for trsm";
  CHECK(rightside || (B.size(0) == A.size(1)))
      << "Non compatible matrix dimensions between inputs A and B for trsm";
}

#if (MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1)

#define LINALG_CPU_TRSM(fname, DType)                                 \
  template <>                                                         \
  inline void linalg_trsm<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                      const Tensor<cpu, 2, DType>& B, \
                                      DType alpha,                    \
                                      bool rightside,                 \
                                      bool lower,                     \
                                      bool transpose,                 \
                                      Stream<cpu>* s) {               \
    check_trsm(A, B, alpha, rightside, lower, transpose);             \
    cblas_##fname(CblasRowMajor,                                      \
                  (rightside ? CblasRight : CblasLeft),               \
                  (lower ? CblasLower : CblasUpper),                  \
                  (transpose ? CblasTrans : CblasNoTrans),            \
                  CblasNonUnit,                                       \
                  B.size(0),                                          \
                  B.size(1),                                          \
                  alpha,                                              \
                  A.dptr_,                                            \
                  A.stride_,                                          \
                  B.dptr_,                                            \
                  B.stride_);                                         \
  }

#define LINALG_XPU_BATCH_TRSM(xpu, DType)                                   \
  template <>                                                               \
  inline void linalg_batch_trsm<xpu, DType>(const Tensor<xpu, 3, DType>& A, \
                                            const Tensor<xpu, 3, DType>& B, \
                                            DType alpha,                    \
                                            bool rightside,                 \
                                            bool lower,                     \
                                            bool transpose,                 \
                                            Stream<xpu>* s) {               \
    linalg_check_batch_size(A.size(0), B.size(0), B.size(0));               \
    for (index_t i = 0; i < A.size(0); ++i) {                               \
      linalg_trsm(A[i], B[i], alpha, rightside, lower, transpose, s);       \
    }                                                                       \
  }

#else

#define LINALG_CPU_TRSM(fname, DType)                                 \
  template <>                                                         \
  inline void linalg_trsm<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                      const Tensor<cpu, 2, DType>& B, \
                                      DType alpha,                    \
                                      bool rightside,                 \
                                      bool lower,                     \
                                      bool transpose,                 \
                                      Stream<cpu>* s) {               \
    LOG(FATAL) << "linalg_trsm not implemented, needs cblas!";        \
  }

#define LINALG_XPU_BATCH_TRSM(xpu, DType)                                   \
  template <>                                                               \
  inline void linalg_batch_trsm<xpu, DType>(const Tensor<xpu, 3, DType>& A, \
                                            const Tensor<xpu, 3, DType>& B, \
                                            DType alpha,                    \
                                            bool rightside,                 \
                                            bool lower,                     \
                                            bool transpose,                 \
                                            Stream<xpu>* s) {               \
    LOG(FATAL) << "linalg_batch_trsm not implemented, needs cblas!";        \
  }

#endif  // MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1

LINALG_CPU_TRSM(strsm, float)
LINALG_CPU_TRSM(dtrsm, double)

LINALG_XPU_BATCH_TRSM(cpu, float)
LINALG_XPU_BATCH_TRSM(cpu, double)

#ifdef __CUDACC__

// cublas col-major processing accounted for by switching sides and fill mode

#define LINALG_GPU_TRSM(fname, DType)                                                    \
  template <>                                                                            \
  inline void linalg_trsm<gpu, DType>(const Tensor<gpu, 2, DType>& A,                    \
                                      const Tensor<gpu, 2, DType>& B,                    \
                                      DType alpha,                                       \
                                      bool rightside,                                    \
                                      bool lower,                                        \
                                      bool transpose,                                    \
                                      Stream<gpu>* s) {                                  \
    using namespace mxnet;                                                               \
    using mshadow::gpu;                                                                  \
    CHECK_NOTNULL(s);                                                                    \
    check_trsm(A, B, alpha, rightside, lower, transpose);                                \
    CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s),                             \
                              (rightside ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT),        \
                              (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                              (transpose ? CUBLAS_OP_T : CUBLAS_OP_N),                   \
                              CUBLAS_DIAG_NON_UNIT,                                      \
                              B.size(1),                                                 \
                              B.size(0),                                                 \
                              &alpha,                                                    \
                              A.dptr_,                                                   \
                              A.stride_,                                                 \
                              B.dptr_,                                                   \
                              B.stride_));                                               \
  }
LINALG_GPU_TRSM(Strsm, float)
LINALG_GPU_TRSM(Dtrsm, double)

LINALG_XPU_BATCH_TRSM(gpu, float)
LINALG_XPU_BATCH_TRSM(gpu, double)

#endif  // __CUDACC__

//////////////////////////////// TRMM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "trmm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is B = trmm(A,B), so B is input and output parameter.

template <typename xpu, typename DType>
inline void check_trmm(const Tensor<xpu, 2, DType>& A,
                       const Tensor<xpu, 2, DType>& B,
                       DType alpha,
                       bool rightside,
                       bool lower,
                       bool transpose) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1)) << "First input of trmm is not a square matrix.";
  CHECK(!rightside || (B.size(1) == A.size(0)))
      << "Non compatible matrix dimensions between inputs A and B for trmm";
  CHECK(rightside || (B.size(0) == A.size(1)))
      << "Non compatible matrix dimensions between inputs A and B for trmm";
}

#if (MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1)

#define LINALG_CPU_TRMM(fname, DType)                                 \
  template <>                                                         \
  inline void linalg_trmm<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                      const Tensor<cpu, 2, DType>& B, \
                                      DType alpha,                    \
                                      bool rightside,                 \
                                      bool lower,                     \
                                      bool transpose,                 \
                                      Stream<cpu>* s) {               \
    check_trmm(A, B, alpha, rightside, lower, transpose);             \
    cblas_##fname(CblasRowMajor,                                      \
                  (rightside ? CblasRight : CblasLeft),               \
                  (lower ? CblasLower : CblasUpper),                  \
                  (transpose ? CblasTrans : CblasNoTrans),            \
                  CblasNonUnit,                                       \
                  B.size(0),                                          \
                  B.size(1),                                          \
                  alpha,                                              \
                  A.dptr_,                                            \
                  A.stride_,                                          \
                  B.dptr_,                                            \
                  B.stride_);                                         \
  }

#else

#define LINALG_CPU_TRMM(fname, DType)                                 \
  template <>                                                         \
  inline void linalg_trmm<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                      const Tensor<cpu, 2, DType>& B, \
                                      DType alpha,                    \
                                      bool rightside,                 \
                                      bool lower,                     \
                                      bool transpose,                 \
                                      Stream<cpu>* s) {               \
    LOG(FATAL) << "linalg_trmm not implemented, needs cblas!";        \
  }

#endif  // MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1

#define LINALG_XPU_BATCH_TRMM(xpu, DType)                                   \
  template <>                                                               \
  inline void linalg_batch_trmm<xpu, DType>(const Tensor<xpu, 3, DType>& A, \
                                            const Tensor<xpu, 3, DType>& B, \
                                            DType alpha,                    \
                                            bool rightside,                 \
                                            bool lower,                     \
                                            bool transpose,                 \
                                            Stream<xpu>* s) {               \
    linalg_check_batch_size(A.size(0), B.size(0), B.size(0));               \
    for (index_t i = 0; i < A.size(0); ++i) {                               \
      linalg_trmm(A[i], B[i], alpha, rightside, lower, transpose, s);       \
    }                                                                       \
  }

LINALG_CPU_TRMM(strmm, float)
LINALG_CPU_TRMM(dtrmm, double)

LINALG_XPU_BATCH_TRMM(cpu, float)
LINALG_XPU_BATCH_TRMM(cpu, double)

#ifdef __CUDACC__

// cublas col-major processing accounted for by switching sides and fill mode
// doing in-place computation by supplying B as second and third matrix
#define LINALG_GPU_TRMM(fname, DType)                                                    \
  template <>                                                                            \
  inline void linalg_trmm<gpu, DType>(const Tensor<gpu, 2, DType>& A,                    \
                                      const Tensor<gpu, 2, DType>& B,                    \
                                      DType alpha,                                       \
                                      bool rightside,                                    \
                                      bool lower,                                        \
                                      bool transpose,                                    \
                                      Stream<gpu>* s) {                                  \
    using namespace mxnet;                                                               \
    using mshadow::gpu;                                                                  \
    CHECK_NOTNULL(s);                                                                    \
    check_trmm(A, B, alpha, rightside, lower, transpose);                                \
    CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s),                             \
                              (rightside ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT),        \
                              (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                              (transpose ? CUBLAS_OP_T : CUBLAS_OP_N),                   \
                              CUBLAS_DIAG_NON_UNIT,                                      \
                              B.size(1),                                                 \
                              B.size(0),                                                 \
                              &alpha,                                                    \
                              A.dptr_,                                                   \
                              A.stride_,                                                 \
                              B.dptr_,                                                   \
                              B.stride_,                                                 \
                              B.dptr_,                                                   \
                              B.stride_));                                               \
  }
LINALG_GPU_TRMM(Strmm, float)
LINALG_GPU_TRMM(Dtrmm, double)

LINALG_XPU_BATCH_TRMM(gpu, float)
LINALG_XPU_BATCH_TRMM(gpu, double)

#endif  // __CUDACC__

//////////////////////////////// POTRF ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "potrf". Please refer to the LAPACK-documentation
// for further information about the function and its parameters.
// Note that this is A = potrf(A), so A is input and output parameter.

static const char* potrf_errstr =
    "This may happen when the input matrix is either not symmetric or not positive definite.";

template <typename xpu, typename DType>
inline void check_potrf(const Tensor<xpu, 2, DType>& A, bool lower) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1)) << "No square matrix as input to potrf.";
}

#define LINALG_CPU_POTRF(fname, DType)                                                \
  template <>                                                                         \
  inline void linalg_potrf<cpu, DType>(                                               \
      const Tensor<cpu, 2, DType>& A, bool lower, Stream<cpu>* s) {                   \
    check_potrf(A, lower);                                                            \
    int ret(MXNET_LAPACK_##fname(                                                     \
        MXNET_LAPACK_ROW_MAJOR, (lower ? 'L' : 'U'), A.size(0), A.dptr_, A.stride_)); \
    CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu. " << potrf_errstr;       \
  }
LINALG_CPU_POTRF(spotrf, float)
LINALG_CPU_POTRF(dpotrf, double)

#define LINALG_CPU_BATCH_POTRF(DType)                               \
  template <>                                                       \
  inline void linalg_batch_potrf<cpu, DType>(                       \
      const Tensor<cpu, 3, DType>& A, bool lower, Stream<cpu>* s) { \
    for (index_t i = 0; i < A.size(0); ++i) {                       \
      linalg_potrf(A[i], lower);                                    \
    }                                                               \
  }
LINALG_CPU_BATCH_POTRF(float)
LINALG_CPU_BATCH_POTRF(double)

#if defined(__CUDACC__) && MXNET_USE_CUSOLVER == 1

#define LINALG_GPU_BUFFSIZE_POTRF(fname, DType)                                                  \
  inline int linalg_potrf_buffsize(const Tensor<gpu, 2, DType>& A, bool lower, Stream<gpu>* s) { \
    using namespace mxnet;                                                                       \
    using mshadow::gpu;                                                                          \
    CHECK_NOTNULL(s);                                                                            \
    int buffsize(0);                                                                             \
    CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s),                               \
                                  (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER),     \
                                  A.size(0),                                                     \
                                  A.dptr_,                                                       \
                                  A.stride_,                                                     \
                                  &buffsize));                                                   \
    return buffsize;                                                                             \
  }
LINALG_GPU_BUFFSIZE_POTRF(DnSpotrf_bufferSize, float)
LINALG_GPU_BUFFSIZE_POTRF(DnDpotrf_bufferSize, double)

#define LINALG_GPU_POTRF(fname, DType)                                                       \
  template <>                                                                                \
  inline void linalg_potrf<gpu, DType>(                                                      \
      const Tensor<gpu, 2, DType>& A, bool lower, Stream<gpu>* s) {                          \
    using namespace mxnet;                                                                   \
    using mshadow::gpu;                                                                      \
    CHECK_NOTNULL(s);                                                                        \
    check_potrf(A, lower);                                                                   \
    int buffsize(linalg_potrf_buffsize(A, lower, s));                                        \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_potrf, buffer, DType, buffsize);                      \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_potrf, info, int, 1);                                 \
    CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s),                           \
                                  (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                                  A.size(0),                                                 \
                                  A.dptr_,                                                   \
                                  A.stride_,                                                 \
                                  static_cast<DType*>(buffer.dptr),                          \
                                  buffsize,                                                  \
                                  static_cast<int*>(info.dptr)));                            \
    Storage::Get()->Free(buffer);                                                            \
    Storage::Get()->Free(info);                                                              \
  }
LINALG_GPU_POTRF(DnSpotrf, float)
LINALG_GPU_POTRF(DnDpotrf, double)

#define LINALG_GPU_BATCH_POTRF(fname, DType)                                                   \
  template <>                                                                                  \
  inline void linalg_batch_potrf<gpu, DType>(                                                  \
      const Tensor<gpu, 3, DType>& A, bool lower, Stream<gpu>* s) {                            \
    using namespace mxnet;                                                                     \
    using mshadow::gpu;                                                                        \
    CHECK_NOTNULL(s);                                                                          \
    CHECK_GT(A.size(0), 0);                                                                    \
    check_potrf(A[0], lower);                                                                  \
    int buffsize(linalg_potrf_buffsize(A[0], lower, s));                                       \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_batch_potrf, buffer, DType, buffsize);                  \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_batch_potrf, info, int, 1);                             \
    for (mshadow::index_t i = 0; i < A.size(0); ++i) {                                         \
      CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s),                           \
                                    (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                                    A[i].size(0),                                              \
                                    A[i].dptr_,                                                \
                                    A[i].stride_,                                              \
                                    static_cast<DType*>(buffer.dptr),                          \
                                    buffsize,                                                  \
                                    static_cast<int*>(info.dptr)));                            \
    }                                                                                          \
    Storage::Get()->Free(buffer);                                                              \
    Storage::Get()->Free(info);                                                                \
  }
LINALG_GPU_BATCH_POTRF(DnSpotrf, float)
LINALG_GPU_BATCH_POTRF(DnDpotrf, double)

#endif

//////////////////////////////// POTRI ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "potri". Please refer to the LAPACK-documentation
// for further information about the function and its parameters.
// Note that this is A = potri(A), so A is input and output parameter.

static const char* potri_errstr =
    "This may happen when the input matrix is not a Cholesky factorization obtained"
    " by a prior call of the potrf-operator.";

template <typename xpu, typename DType>
inline void check_potri(const Tensor<xpu, 2, DType>& A, bool lower) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1)) << "No square matrix as input to potri.";
}

#define LINALG_CPU_POTRI(fname, DType)                                                \
  template <>                                                                         \
  inline void linalg_potri<cpu, DType>(                                               \
      const Tensor<cpu, 2, DType>& A, bool lower, Stream<cpu>* s) {                   \
    check_potri(A, lower);                                                            \
    int ret(MXNET_LAPACK_##fname(                                                     \
        MXNET_LAPACK_ROW_MAJOR, (lower ? 'L' : 'U'), A.size(0), A.dptr_, A.stride_)); \
    CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu. " << potri_errstr;       \
  }
LINALG_CPU_POTRI(spotri, float)
LINALG_CPU_POTRI(dpotri, double)

#define LINALG_CPU_BATCH_POTRI(DType)                               \
  template <>                                                       \
  inline void linalg_batch_potri<cpu, DType>(                       \
      const Tensor<cpu, 3, DType>& A, bool lower, Stream<cpu>* s) { \
    for (index_t i = 0; i < A.size(0); ++i) {                       \
      linalg_potri(A[i], lower);                                    \
    }                                                               \
  }
LINALG_CPU_BATCH_POTRI(float)
LINALG_CPU_BATCH_POTRI(double)

#ifdef __CUDACC__

// Initializes multiple identity matrices on the same vector.
template <typename DType>
__global__ void linalgInitIdentityGPU(DType* a, int stride, int lda, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    // index relative to the matrix.
    int index(i % stride);
    a[i] = (index / lda == index % lda ? DType(1.0) : DType(0));
  }
}

// There is no direct support for potri in cuda. We emulate the function by two calls to trsm.
#define LINALG_GPU_POTRI(DType)                                                                \
  template <>                                                                                  \
  inline void linalg_potri<gpu, DType>(                                                        \
      const Tensor<gpu, 2, DType>& A, bool lower, Stream<gpu>* s) {                            \
    using namespace mxnet;                                                                     \
    CHECK_NOTNULL(s);                                                                          \
    check_potri(A, lower);                                                                     \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_potri, buffer, DType, A.MSize());                       \
    using namespace mshadow::cuda;                                                             \
    int ngrid = std::min(kMaxGridNum,                                                          \
                         static_cast<int>((A.MSize() + kBaseThreadNum - 1) / kBaseThreadNum)); \
    linalgInitIdentityGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(   \
        static_cast<DType*>(buffer.dptr), A.MSize(), A.stride_, A.MSize());                    \
    MSHADOW_CUDA_POST_KERNEL_CHECK(linalgInitIdentityGPU);                                     \
    Tensor<gpu, 2, DType> B((DType*)buffer.dptr, A.shape_, A.stride_, s);                      \
    linalg_trsm(A, B, DType(1.0), false, lower, !lower, s);                                    \
    linalg_trsm(A, B, DType(1.0), false, lower, lower, s);                                     \
    Copy(A, B, s);                                                                             \
    B.dptr_ = 0;                                                                               \
    Storage::Get()->Free(buffer);                                                              \
  }
LINALG_GPU_POTRI(float)
LINALG_GPU_POTRI(double)

#define LINALG_GPU_BATCH_POTRI(DType)                                                          \
  template <>                                                                                  \
  inline void linalg_batch_potri<gpu, DType>(                                                  \
      const Tensor<gpu, 3, DType>& A, bool lower, Stream<gpu>* s) {                            \
    using namespace mxnet;                                                                     \
    CHECK_NOTNULL(s);                                                                          \
    CHECK_GT(A.size(0), 0);                                                                    \
    check_potri(A[0], lower);                                                                  \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_batch_potri, buffer, DType, A.MSize());                 \
    using namespace mshadow::cuda;                                                             \
    int ngrid = std::min(kMaxGridNum,                                                          \
                         static_cast<int>((A.MSize() + kBaseThreadNum - 1) / kBaseThreadNum)); \
    linalgInitIdentityGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(   \
        static_cast<DType*>(buffer.dptr), A.size(1) * A.stride_, A.stride_, A.MSize());        \
    MSHADOW_CUDA_POST_KERNEL_CHECK(linalgInitIdentityGPU);                                     \
    Tensor<gpu, 3, DType> B((DType*)buffer.dptr, A.shape_, A.stride_, s);                      \
    linalg_batch_trsm(A, B, DType(1.0), false, lower, !lower, s);                              \
    linalg_batch_trsm(A, B, DType(1.0), false, lower, lower, s);                               \
    Copy(A, B, s);                                                                             \
    B.dptr_ = 0;                                                                               \
    Storage::Get()->Free(buffer);                                                              \
  }
LINALG_GPU_BATCH_POTRI(float)
LINALG_GPU_BATCH_POTRI(double)

#endif

//////////////////////////////// SYRK ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "syrk". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is B = syrk(A, B), so B is input and output parameter.

template <typename xpu, typename DType>
inline void check_syrk(const Tensor<xpu, 2, DType>& A,
                       const Tensor<xpu, 2, DType>& B,
                       DType alpha,
                       DType beta,
                       bool tA) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(B.size(0), B.size(1)) << "B must be square symmetric matrix for syrk";
  CHECK_EQ((tA ? A.size(1) : A.size(0)), B.size(0))
      << "Non compatible matrix dimensions between inputs A and B for syrk";
}

#if (MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1)

#define LINALG_CPU_SYRK(fname, DType)                                 \
  template <>                                                         \
  inline void linalg_syrk<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                      const Tensor<cpu, 2, DType>& B, \
                                      DType alpha,                    \
                                      DType beta,                     \
                                      bool tA,                        \
                                      Stream<cpu>* s) {               \
    check_syrk(A, B, alpha, beta, tA);                                \
    cblas_##fname(CblasRowMajor,                                      \
                  CblasLower,                                         \
                  (tA ? CblasTrans : CblasNoTrans),                   \
                  B.size(0),                                          \
                  (tA ? A.size(0) : A.size(1)),                       \
                  alpha,                                              \
                  A.dptr_,                                            \
                  A.stride_,                                          \
                  beta,                                               \
                  B.dptr_,                                            \
                  B.stride_);                                         \
  }

#else

#define LINALG_CPU_SYRK(fname, DType)                                           \
  template <>                                                                   \
  inline void linalg_syrk<cpu, DType>(const Tensor<cpu, 2, DType>& A,           \
                                      const Tensor<cpu, 2, DType>& B,           \
                                      DType alpha,                              \
                                      DType beta,                               \
                                      bool tA,                                  \
                                      Stream<cpu>* s) {                         \
    LOG(FATAL) << "linalg_syrk not implemented by mxnet for cpu, needs cblas!"; \
  }

#endif  // MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1

#define LINALG_XPU_BATCH_SYRK(xpu, DType)                       \
  template <>                                                   \
  inline void linalg_batch_syrk(const Tensor<xpu, 3, DType>& A, \
                                const Tensor<xpu, 3, DType>& B, \
                                DType alpha,                    \
                                DType beta,                     \
                                bool tA,                        \
                                Stream<xpu>* s) {               \
    linalg_check_batch_size(A.size(0), B.size(0), B.size(0));   \
    for (index_t i = 0; i < A.size(0); ++i) {                   \
      linalg_syrk(A[i], B[i], alpha, beta, tA, s);              \
    }                                                           \
  }

LINALG_CPU_SYRK(ssyrk, float)
LINALG_CPU_SYRK(dsyrk, double)
LINALG_XPU_BATCH_SYRK(cpu, float)
LINALG_XPU_BATCH_SYRK(cpu, double)

#ifdef __CUDACC__

// cublas col-major processing accounted for by switching transpose and fill mode
#define LINALG_GPU_SYRK(fname, DType)                                 \
  template <>                                                         \
  inline void linalg_syrk<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                      const Tensor<gpu, 2, DType>& B, \
                                      DType alpha,                    \
                                      DType beta,                     \
                                      bool tA,                        \
                                      Stream<gpu>* s) {               \
    using namespace mxnet;                                            \
    using mshadow::gpu;                                               \
    CHECK_NOTNULL(s);                                                 \
    check_syrk(A, B, alpha, beta, tA);                                \
    CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s),          \
                              CUBLAS_FILL_MODE_UPPER,                 \
                              (tA ? CUBLAS_OP_N : CUBLAS_OP_T),       \
                              B.size(1),                              \
                              (tA ? A.size(0) : A.size(1)),           \
                              &alpha,                                 \
                              A.dptr_,                                \
                              A.stride_,                              \
                              &beta,                                  \
                              B.dptr_,                                \
                              B.stride_));                            \
  }

LINALG_GPU_SYRK(Ssyrk, float)
LINALG_GPU_SYRK(Dsyrk, double)
LINALG_XPU_BATCH_SYRK(gpu, float)
LINALG_XPU_BATCH_SYRK(gpu, double)

#endif  // __CUDACC__

//////////////////////////////// GELQF ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK functions "gelqf", "orglq".

template <typename xpu, typename DType>
inline void check_gelqf(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 1, DType>& work) {
  // Any checking that helps user debug potential problems.
  CHECK_LE(A.size(0), A.size(1)) << "A must have num(rows) <= num(columns)";
  CHECK_LE(A.size(0), work.size(0)) << "Size of work is too small";
}

#define LINALG_CPU_GELQF(fname, DType)                                                     \
  template <>                                                                              \
  inline void linalg_gelqf<cpu, DType>(                                                    \
      const Tensor<cpu, 2, DType>& A, const Tensor<cpu, 1, DType>& work, Stream<cpu>* s) { \
    check_gelqf(A, work);                                                                  \
    int m(A.size(0));                                                                      \
    int lwork(work.size(0) - m);                                                           \
    int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_ROW_MAJOR,                                   \
                                 m,                                                        \
                                 A.size(1),                                                \
                                 A.dptr_,                                                  \
                                 A.stride_,                                                \
                                 work.dptr_,                                               \
                                 work.dptr_ + m,                                           \
                                 lwork));                                                  \
    CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu.";                             \
  }
LINALG_CPU_GELQF(sgelqf, float)
LINALG_CPU_GELQF(dgelqf, double)

#define LINALG_CPU_ORGLQ(fname, DType)                                                     \
  template <>                                                                              \
  inline void linalg_orglq<cpu, DType>(                                                    \
      const Tensor<cpu, 2, DType>& A, const Tensor<cpu, 1, DType>& work, Stream<cpu>* s) { \
    check_gelqf(A, work);                                                                  \
    int m(A.size(0));                                                                      \
    int lwork(work.size(0) - m);                                                           \
    int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_ROW_MAJOR,                                   \
                                 m,                                                        \
                                 A.size(1),                                                \
                                 A.dptr_,                                                  \
                                 A.stride_,                                                \
                                 work.dptr_,                                               \
                                 work.dptr_ + m,                                           \
                                 lwork));                                                  \
    CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu.";                             \
  }
LINALG_CPU_ORGLQ(sorglq, float)
LINALG_CPU_ORGLQ(dorglq, double)

#define LINALG_CPU_GELQF_WORKSPACE_QUERY(prefix, DType)                               \
  template <>                                                                         \
  inline int linalg_gelqf_workspace_query<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                                      Stream<cpu>* s) {               \
    int m(A.size(0));                                                                 \
    DType work = 0;                                                                   \
    int ret(MXNET_LAPACK_##prefix##gelqf(                                             \
        MXNET_LAPACK_ROW_MAJOR, m, A.size(1), A.dptr_, A.stride_, &work, &work, -1)); \
    CHECK_EQ(ret, 0) << #prefix << "gelqf: Workspace query failed on CPU.";           \
    int ws_size(static_cast<int>(work));                                              \
    ret = MXNET_LAPACK_##prefix##orglq(                                               \
        MXNET_LAPACK_ROW_MAJOR, m, A.size(1), A.dptr_, A.stride_, &work, &work, -1);  \
    CHECK_EQ(ret, 0) << #prefix << "orglq: Workspace query failed on CPU.";           \
    int wsz2(static_cast<int>(work));                                                 \
    if (wsz2 > ws_size)                                                               \
      ws_size = wsz2;                                                                 \
    return ws_size + m;                                                               \
  }
LINALG_CPU_GELQF_WORKSPACE_QUERY(s, float)
LINALG_CPU_GELQF_WORKSPACE_QUERY(d, double)

#ifdef __CUDACC__

#define LINALG_GPU_GELQF(fname, DType)                                                     \
  template <>                                                                              \
  inline void linalg_gelqf<gpu, DType>(                                                    \
      const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 1, DType>& work, Stream<gpu>* s) { \
    using namespace mxnet;                                                                 \
    using mshadow::gpu;                                                                    \
    CHECK_NOTNULL(s);                                                                      \
    check_gelqf(A, work);                                                                  \
    int m(A.size(0));                                                                      \
    int lwork(work.size(0) - m);                                                           \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_gelqf, info, int, 1);                               \
    CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s),                         \
                                  A.size(1),                                               \
                                  m,                                                       \
                                  A.dptr_,                                                 \
                                  A.stride_,                                               \
                                  work.dptr_,                                              \
                                  work.dptr_ + m,                                          \
                                  lwork,                                                   \
                                  static_cast<int*>(info.dptr)));                          \
    Storage::Get()->Free(info);                                                            \
  }
// Col-major QR-decomposition results in row-major LQ decomposition.
LINALG_GPU_GELQF(DnSgeqrf, float)
LINALG_GPU_GELQF(DnDgeqrf, double)

// ORGLQ only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

#define LINALG_GPU_ORGLQ(fname, DType)                                                     \
  template <>                                                                              \
  inline void linalg_orglq<gpu, DType>(                                                    \
      const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 1, DType>& work, Stream<gpu>* s) { \
    using namespace mxnet;                                                                 \
    using mshadow::gpu;                                                                    \
    CHECK_NOTNULL(s);                                                                      \
    check_gelqf(A, work);                                                                  \
    int m(A.size(0));                                                                      \
    int lwork(work.size(0) - m);                                                           \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_orglq, info, int, 1);                               \
    CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s),                         \
                                  A.size(1),                                               \
                                  m,                                                       \
                                  m,                                                       \
                                  A.dptr_,                                                 \
                                  A.stride_,                                               \
                                  work.dptr_,                                              \
                                  work.dptr_ + m,                                          \
                                  lwork,                                                   \
                                  static_cast<int*>(info.dptr)));                          \
    Storage::Get()->Free(info);                                                            \
  }

#else

#define LINALG_GPU_ORGLQ(fname, DType)                                                     \
  template <>                                                                              \
  inline void linalg_orglq<gpu, DType>(                                                    \
      const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 1, DType>& work, Stream<gpu>* s) { \
    LOG(FATAL) << "orglq requires CUDA version >= 8.0!";                                   \
  }

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_ORGLQ(DnSorgqr, float)
LINALG_GPU_ORGLQ(DnDorgqr, double)

// ORGLQ only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

#define LINALG_GPU_GELQF_WORKSPACE_QUERY(prefix, DType)                                 \
  template <>                                                                           \
  inline int linalg_gelqf_workspace_query<gpu, DType>(const Tensor<gpu, 2, DType>& A,   \
                                                      Stream<gpu>* s) {                 \
    using namespace mxnet;                                                              \
    using mshadow::gpu;                                                                 \
    int m(A.size(0));                                                                   \
    int work1(0);                                                                       \
    CUSOLVER_CALL(cusolverDn##prefix##geqrf_bufferSize(                                 \
        Stream<gpu>::GetSolverHandle(s), A.size(1), m, A.dptr_, A.stride_, &work1));    \
    int work2(0);                                                                       \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_gelqf_workspace_query, tau, DType, 1);           \
    CUSOLVER_CALL(cusolverDn##prefix##orgqr_bufferSize(Stream<gpu>::GetSolverHandle(s), \
                                                       A.size(1),                       \
                                                       m,                               \
                                                       m,                               \
                                                       A.dptr_,                         \
                                                       A.stride_,                       \
                                                       static_cast<DType*>(tau.dptr),   \
                                                       &work2));                        \
    Storage::Get()->Free(tau);                                                          \
    return std::max(work1, work2) + m;                                                  \
  }

#else

#define LINALG_GPU_GELQF_WORKSPACE_QUERY(prefix, DType)                               \
  template <>                                                                         \
  inline int linalg_gelqf_workspace_query<gpu, DType>(const Tensor<gpu, 2, DType>& A, \
                                                      Stream<gpu>* s) {               \
    LOG(FATAL) << "orglq requires CUDA version >= 8.0!";                              \
    return 0;                                                                         \
  }

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_GELQF_WORKSPACE_QUERY(S, float)
LINALG_GPU_GELQF_WORKSPACE_QUERY(D, double)

#endif  // __CUDACC__

//////////////////////////////// SYEVD ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "syevd"

template <typename xpu, typename DType>
inline void check_syevd(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 1, DType>& L) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1)) << "A must be square symmetric matrix";
  CHECK_EQ(A.size(0), L.size(0)) << "A, L have incompatible sizes";
}

#define LINALG_CPU_SYEVD(fname, DType)                                                            \
  template <>                                                                                     \
  inline void linalg_syevd<cpu, DType>(const Tensor<cpu, 2, DType>& A,                            \
                                       const Tensor<cpu, 1, DType>& L,                            \
                                       const Tensor<cpu, 1, DType>& work,                         \
                                       Stream<cpu>* s) {                                          \
    check_syevd(A, L);                                                                            \
    DType workTmp(0);                                                                             \
    lapack_index_t liwork(0);                                                                     \
    MXNET_LAPACK_##fname(MXNET_LAPACK_ROW_MAJOR,                                                  \
                         'L',                                                                     \
                         A.size(0),                                                               \
                         A.dptr_,                                                                 \
                         A.stride_,                                                               \
                         L.dptr_,                                                                 \
                         &workTmp,                                                                \
                         -1,                                                                      \
                         &liwork,                                                                 \
                         -1);                                                                     \
    lapack_index_t lwork = static_cast<lapack_index_t>(workTmp);                                  \
    if /*constexpr*/ (sizeof(lapack_index_t) > sizeof(DType)) {                                   \
      /* For alligning iwork pointer address */                                                   \
      constexpr lapack_index_t round_mask =                                                       \
          static_cast<lapack_index_t>(sizeof(lapack_index_t) / sizeof(DType)) - 1;                \
      lwork = (lwork + round_mask) & ~round_mask;                                                 \
    }                                                                                             \
    lapack_index_t* iwork = static_cast<lapack_index_t*>(static_cast<void*>(work.dptr_ + lwork)); \
    int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_ROW_MAJOR,                                          \
                                 'L',                                                             \
                                 A.size(0),                                                       \
                                 A.dptr_,                                                         \
                                 A.stride_,                                                       \
                                 L.dptr_,                                                         \
                                 work.dptr_,                                                      \
                                 lwork,                                                           \
                                 iwork,                                                           \
                                 liwork));                                                        \
    CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu.";                                    \
  }
LINALG_CPU_SYEVD(ssyevd, float)
LINALG_CPU_SYEVD(dsyevd, double)

// Mangle temp storage requirements for DType and int into a single
// request as we can only allocate one temp space per operator. We
// partition this temp space into two chunks again when calling sseyvd.
// Returned is the number of elements of type DType that the temp space
// needs to accomodate. This also makes this function signature equivalent
// to the work space query on GPU.
#define LINALG_CPU_SYEVD_WORKSPACE_QUERY(func, DType)                                   \
  template <>                                                                           \
  inline lapack_index_t linalg_syevd_workspace_query<cpu, DType>(                       \
      const Tensor<cpu, 2, DType>& A, const Tensor<cpu, 1, DType>& L, Stream<cpu>* s) { \
    DType work(0);                                                                      \
    lapack_index_t liwork(0);                                                           \
    MXNET_LAPACK_##func(MXNET_LAPACK_ROW_MAJOR,                                         \
                        'L',                                                            \
                        A.size(0),                                                      \
                        A.dptr_,                                                        \
                        A.stride_,                                                      \
                        L.dptr_,                                                        \
                        &work,                                                          \
                        -1,                                                             \
                        &liwork,                                                        \
                        -1);                                                            \
    lapack_index_t lwork = static_cast<lapack_index_t>(work);                           \
    if /*constexpr*/ (sizeof(DType) != sizeof(lapack_index_t)) {                        \
      if /*constexpr*/ (sizeof(DType) > sizeof(lapack_index_t)) {                       \
        /* Convert memory size needed for liwork to lwork units [Dtype] */              \
        liwork = (sizeof(lapack_index_t) * liwork + sizeof(DType) - 1) / sizeof(DType); \
      } else {                                                                          \
        /* Convert memory size needed for liwork to lwork units [Dtype] */              \
        liwork *= sizeof(lapack_index_t) / sizeof(DType);                               \
        /* For alligning iwork pointer address */                                       \
        constexpr lapack_index_t round_mask =                                           \
            static_cast<lapack_index_t>(sizeof(lapack_index_t) / sizeof(DType)) - 1;    \
        lwork = (lwork + round_mask) & ~round_mask;                                     \
      }                                                                                 \
    }                                                                                   \
    return lwork + liwork;                                                              \
  }
LINALG_CPU_SYEVD_WORKSPACE_QUERY(ssyevd, float)
LINALG_CPU_SYEVD_WORKSPACE_QUERY(dsyevd, double)

#ifdef __CUDACC__

// SYEVD only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

// Row-major vs. col-major handled by using upper triangular
// in cusolver-call.
#define LINALG_GPU_SYEVD(fname, DType)                                    \
  template <>                                                             \
  inline void linalg_syevd<gpu, DType>(const Tensor<gpu, 2, DType>& A,    \
                                       const Tensor<gpu, 1, DType>& L,    \
                                       const Tensor<gpu, 1, DType>& work, \
                                       Stream<gpu>* s) {                  \
    using namespace mxnet;                                                \
    using mshadow::gpu;                                                   \
    CHECK_NOTNULL(s);                                                     \
    check_syevd(A, L);                                                    \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_syevd, info, int, 1);              \
    CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s),        \
                                  CUSOLVER_EIG_MODE_VECTOR,               \
                                  CUBLAS_FILL_MODE_UPPER,                 \
                                  A.size(0),                              \
                                  A.dptr_,                                \
                                  A.stride_,                              \
                                  L.dptr_,                                \
                                  work.dptr_,                             \
                                  work.size(0),                           \
                                  static_cast<int*>(info.dptr)));         \
    Storage::Get()->Free(info);                                           \
  }

#define LINALG_GPU_SYEVD_WORKSPACE_QUERY(fname, DType)                                  \
  template <>                                                                           \
  inline int linalg_syevd_workspace_query<gpu, DType>(                                  \
      const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 1, DType>& L, Stream<gpu>* s) { \
    using namespace mxnet;                                                              \
    using mshadow::gpu;                                                                 \
    int lwork(0);                                                                       \
    CUSOLVER_CALL(cusolver##fname##_bufferSize(Stream<gpu>::GetSolverHandle(s),         \
                                               CUSOLVER_EIG_MODE_VECTOR,                \
                                               CUBLAS_FILL_MODE_UPPER,                  \
                                               A.size(0),                               \
                                               A.dptr_,                                 \
                                               A.stride_,                               \
                                               L.dptr_,                                 \
                                               &lwork));                                \
    return lwork;                                                                       \
  }

#else

#define LINALG_GPU_SYEVD(fname, DType)                                    \
  template <>                                                             \
  inline void linalg_syevd<gpu, DType>(const Tensor<gpu, 2, DType>& A,    \
                                       const Tensor<gpu, 1, DType>& L,    \
                                       const Tensor<gpu, 1, DType>& work, \
                                       Stream<gpu>* s) {                  \
    LOG(FATAL) << "syevd requires CUDA version >= 8.0!";                  \
  }

#define LINALG_GPU_SYEVD_WORKSPACE_QUERY(fname, DType)                                  \
  template <>                                                                           \
  inline int linalg_syevd_workspace_query<gpu, DType>(                                  \
      const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 1, DType>& L, Stream<gpu>* s) { \
    LOG(FATAL) << "syevd requires CUDA version >= 8.0!";                                \
    return 0;                                                                           \
  }

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_SYEVD(DnSsyevd, float)
LINALG_GPU_SYEVD(DnDsyevd, double)

LINALG_GPU_SYEVD_WORKSPACE_QUERY(DnSsyevd, float)
LINALG_GPU_SYEVD_WORKSPACE_QUERY(DnDsyevd, double)

#endif  // __CUDACC__

//////////////////////////////// GESVD ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "gesvd"

template <typename xpu, typename DType>
inline void check_gesvd(const Tensor<xpu, 2, DType>& UT,
                        const Tensor<xpu, 1, DType>& L,
                        const Tensor<xpu, 2, DType>& V) {
  // Any checking that helps user debug potential problems.
  CHECK_LE(V.size(0), V.size(1))
      << "The second to last dimension of A must be less or equal to the "
      << "last dimension";
  CHECK_EQ(UT.size(0), UT.size(1)) << "UT must be square matrix";
  CHECK_EQ(V.size(0), L.size(0)) << "V, L have incompatible sizes";
  CHECK_EQ(V.size(0), UT.size(0)) << "V, UT must have compatible sizes";
}

#define LINALG_CPU_GESVD(fname, DType)                                    \
  template <>                                                             \
  inline void linalg_gesvd<cpu, DType>(const Tensor<cpu, 2, DType>& UT,   \
                                       const Tensor<cpu, 1, DType>& L,    \
                                       const Tensor<cpu, 2, DType>& V,    \
                                       const Tensor<cpu, 1, DType>& work, \
                                       Stream<cpu>* s) {                  \
    check_gesvd(UT, L, V);                                                \
    int lwork(work.size(0));                                              \
    int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_ROW_MAJOR,                  \
                                 V.size(0),                               \
                                 V.size(1),                               \
                                 UT.dptr_,                                \
                                 UT.stride_,                              \
                                 L.dptr_,                                 \
                                 V.dptr_,                                 \
                                 V.stride_,                               \
                                 work.dptr_,                              \
                                 lwork));                                 \
    CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu.";            \
  }

LINALG_CPU_GESVD(sgesvd, float)
LINALG_CPU_GESVD(dgesvd, double)

#define LINALG_CPU_GESVD_WORKSPACE_QUERY(func, DType)                                     \
  template <>                                                                             \
  inline size_t linalg_gesvd_workspace_query<cpu, DType>(const Tensor<cpu, 2, DType>& UT, \
                                                         const Tensor<cpu, 1, DType>& L,  \
                                                         const Tensor<cpu, 2, DType>& V,  \
                                                         Stream<cpu>* s) {                \
    DType work(0.0);                                                                      \
    MXNET_LAPACK_##func(MXNET_LAPACK_ROW_MAJOR,                                           \
                        V.size(0),                                                        \
                        V.size(1),                                                        \
                        UT.dptr_,                                                         \
                        UT.stride_,                                                       \
                        L.dptr_,                                                          \
                        V.dptr_,                                                          \
                        V.stride_,                                                        \
                        &work,                                                            \
                        -1);                                                              \
    return static_cast<size_t>(work);                                                     \
  }
LINALG_CPU_GESVD_WORKSPACE_QUERY(sgesvd, float)
LINALG_CPU_GESVD_WORKSPACE_QUERY(dgesvd, double)

#ifdef __CUDACC__

// GESVD only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

#define LINALG_GPU_GESVD(fname, DType)                                    \
  template <>                                                             \
  inline void linalg_gesvd<gpu, DType>(const Tensor<gpu, 2, DType>& UT,   \
                                       const Tensor<gpu, 1, DType>& L,    \
                                       const Tensor<gpu, 2, DType>& V,    \
                                       const Tensor<gpu, 1, DType>& work, \
                                       Stream<gpu>* s) {                  \
    using namespace mxnet;                                                \
    using mshadow::gpu;                                                   \
    check_gesvd(UT, L, V);                                                \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_gesvd, info, int, 1);              \
    CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s),        \
                                  'O',                                    \
                                  'S',                                    \
                                  V.size(1),                              \
                                  V.size(0),                              \
                                  V.dptr_,                                \
                                  V.stride_,                              \
                                  L.dptr_,                                \
                                  V.dptr_,                                \
                                  V.stride_,                              \
                                  UT.dptr_,                               \
                                  UT.stride_,                             \
                                  work.dptr_,                             \
                                  work.size(0),                           \
                                  V.dptr_,                                \
                                  static_cast<int*>(info.dptr)));         \
    Storage::Get()->Free(info);                                           \
  }

#define LINALG_GPU_GESVD_WORKSPACE_QUERY(fname, DType)                                    \
  template <>                                                                             \
  inline size_t linalg_gesvd_workspace_query<gpu, DType>(const Tensor<gpu, 2, DType>& UT, \
                                                         const Tensor<gpu, 1, DType>& L,  \
                                                         const Tensor<gpu, 2, DType>& V,  \
                                                         Stream<gpu>* s) {                \
    using namespace mxnet;                                                                \
    using mshadow::gpu;                                                                   \
    int lwork(0);                                                                         \
    CUSOLVER_CALL(cusolver##fname##_bufferSize(                                           \
        Stream<gpu>::GetSolverHandle(s), V.size(1), V.size(0), &lwork));                  \
    return lwork;                                                                         \
  }

#else

#define LINALG_GPU_GESVD(fname, DType)                                    \
  template <>                                                             \
  inline void linalg_gesvd<gpu, DType>(const Tensor<gpu, 2, DType>& UT,   \
                                       const Tensor<gpu, 1, DType>& L,    \
                                       const Tensor<gpu, 2, DType>& V,    \
                                       const Tensor<gpu, 1, DType>& work, \
                                       Stream<gpu>* s) {                  \
    LOG(FATAL) << "gesvd requires CUDA version >= 8.0!";                  \
  }

#define LINALG_GPU_GESVD_WORKSPACE_QUERY(fname, DType)                                    \
  template <>                                                                             \
  inline size_t linalg_gesvd_workspace_query<gpu, DType>(const Tensor<gpu, 2, DType>& UT, \
                                                         const Tensor<gpu, 1, DType>& L,  \
                                                         const Tensor<gpu, 2, DType>& V,  \
                                                         Stream<gpu>* s) {                \
    LOG(FATAL) << "gesvd requires CUDA version >= 8.0!";                                  \
    return 0;                                                                             \
  }

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_GESVD(DnSgesvd, float)
LINALG_GPU_GESVD(DnDgesvd, double)

LINALG_GPU_GESVD_WORKSPACE_QUERY(DnSgesvd, float)
LINALG_GPU_GESVD_WORKSPACE_QUERY(DnDgesvd, double)

#endif  // __CUDACC__

//////////////////////////////// GETRF ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "getrf"

// The input of this function should be col-major for performance.
// Tensor work holds space for ipiv in getrf
#define LINALG_CPU_GETRF(fname, DType)                                                   \
  template <>                                                                            \
  inline void linalg_getrf<cpu, DType>(const Tensor<cpu, 2, DType>& A,                   \
                                       const Tensor<cpu, 1, lapack_index_t>& pivot,      \
                                       bool check_singular,                              \
                                       Stream<cpu>* s) {                                 \
    int ret(MXNET_LAPACK_##fname(                                                        \
        MXNET_LAPACK_COL_MAJOR, A.size(1), A.size(0), A.dptr_, A.stride_, pivot.dptr_)); \
    CHECK_GE(ret, 0) << #fname << " failed in lapack on cpu.";                           \
    if (check_singular) {                                                                \
      CHECK_EQ(ret, 0) << "the input matrix is non-convertible";                         \
    }                                                                                    \
  }

LINALG_CPU_GETRF(sgetrf, float)
LINALG_CPU_GETRF(dgetrf, double)

#define LINALG_CPU_BATCH_GETRF(fname, DType, IndexT)                              \
  template <>                                                                     \
  inline void linalg_batch_getrf<cpu, DType>(const Tensor<cpu, 3, DType>& A,      \
                                             const Tensor<cpu, 2, IndexT>& pivot, \
                                             bool check_singular,                 \
                                             Stream<cpu>* s) {                    \
    for (IndexT i = 0; i < A.size(0); ++i) {                                      \
      linalg_getrf(A[i], pivot[i], check_singular);                               \
    }                                                                             \
  }

LINALG_CPU_BATCH_GETRF(sgetrf, float, LapackIndex<cpu>::IndexT)
LINALG_CPU_BATCH_GETRF(dgetrf, double, LapackIndex<cpu>::IndexT)

#ifdef __CUDACC__

// "getrfBatched" and "getriBatched" in cuBLAS must have DType *matrices[] as input
// to store the pointers of each batch matrix. This kernel is used to build the
// pointer array.
struct set_matrix {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i, DType** p, DType* m, int step) {
    p[i] = m + i * step;
  }
};

// GETRF only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

// Since there is no "getri" in cuSolver, we are using batched version of
// "getrf" and "getri" in cuBLAS here. These routines are good for large
// batches of small matrices, so performance issue may happen when computing
// large matices. We leave it here until MAGMA which has "getri" is introduced
// into MXNet.
#define LINALG_GPU_BATCH_GETRF(fname, DType)                                              \
  template <>                                                                             \
  inline void linalg_batch_getrf<gpu, DType>(const Tensor<gpu, 3, DType>& A,              \
                                             const Tensor<gpu, 2, int>& pivot,            \
                                             bool check_singular,                         \
                                             Stream<gpu>* s) {                            \
    using namespace mxnet;                                                                \
    using namespace mxnet::op::mxnet_op;                                                  \
    CHECK_NOTNULL(s);                                                                     \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_batch_getrf, info, int, A.size(0));                \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_batch_getrf, A_ptr_buf, DType*, A.size(0));        \
    DType** A_ptr = static_cast<DType**>(A_ptr_buf.dptr);                                 \
    Kernel<set_matrix, gpu>::Launch(s, A.size(0), A_ptr, A.dptr_, A.size(1) * A.size(2)); \
    CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s),                              \
                              A.size(1),                                                  \
                              A_ptr,                                                      \
                              A.size(2),                                                  \
                              pivot.dptr_,                                                \
                              static_cast<int*>(info.dptr),                               \
                              A.size(0)))                                                 \
    Storage::Get()->Free(info);                                                           \
    Storage::Get()->Free(A_ptr_buf);                                                      \
  }

#else

#define LINALG_GPU_BATCH_GETRF(fname, DType)                                   \
  template <>                                                                  \
  inline void linalg_batch_getrf<gpu, DType>(const Tensor<gpu, 3, DType>& A,   \
                                             const Tensor<gpu, 2, int>& pivot, \
                                             bool check_singular,              \
                                             Stream<gpu>* s) {                 \
    LOG(FATAL) << "batched getrf requires CUDA version >= 8.0!";               \
  }

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_BATCH_GETRF(SgetrfBatched, float)
LINALG_GPU_BATCH_GETRF(DgetrfBatched, double)

#endif  // __CUDACC__

//////////////////////////////// GETRI ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "getri"

// The input of this function should be col-major for performance.
#define LINALG_CPU_GETRI(fname, DType)                                              \
  template <>                                                                       \
  inline void linalg_getri<cpu, DType>(const Tensor<cpu, 2, DType>& LU,             \
                                       const Tensor<cpu, 1, lapack_index_t>& pivot, \
                                       const Tensor<cpu, 1, DType>& work,           \
                                       Stream<cpu>* s) {                            \
    int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_COL_MAJOR,                            \
                                 LU.size(0),                                        \
                                 LU.dptr_,                                          \
                                 LU.stride_,                                        \
                                 pivot.dptr_,                                       \
                                 work.dptr_,                                        \
                                 work.size(0)));                                    \
    CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu.";                      \
  }
LINALG_CPU_GETRI(sgetri, float)
LINALG_CPU_GETRI(dgetri, double)

template <typename xpu, typename DType>
lapack_index_t linalg_getri_workspace_query(const Tensor<xpu, 2, DType>& A, Stream<cpu>* s) {
  LOG(FATAL) << "it only takes float or double Tensor";
  return 0;
}

// Query workspace for "getri"
#define LINALG_CPU_GETRI_WORKSPACE_QUERY(func, DType)                                            \
  template <>                                                                                    \
  inline lapack_index_t linalg_getri_workspace_query<cpu, DType>(const Tensor<cpu, 2, DType>& A, \
                                                                 Stream<cpu>* s) {               \
    DType lwork(0);                                                                              \
    MXNET_LAPACK_##func(                                                                         \
        MXNET_LAPACK_COL_MAJOR, A.size(0), A.dptr_, A.stride_, nullptr, &lwork, -1);             \
    return static_cast<lapack_index_t>(lwork);                                                   \
  }

LINALG_CPU_GETRI_WORKSPACE_QUERY(sgetri, float)
LINALG_CPU_GETRI_WORKSPACE_QUERY(dgetri, double)

#ifdef __CUDACC__

// GETRI only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

// Since there is no "getri" in cuSolver, we are using batched version of
// "getrf" and "getri" in cuBLAS here. These routines are good for large
// batches of small matrices, so performance issue may happen when computing
// large matices. We leave it here until MAGMA which has "getri" is introduced
// into MXNet.
#define LINALG_GPU_BATCH_GETRI(fname, DType)                                                   \
  template <>                                                                                  \
  inline void linalg_batch_getri<gpu, DType>(const Tensor<gpu, 3, DType>& A,                   \
                                             const Tensor<gpu, 3, DType>& LU,                  \
                                             const Tensor<gpu, 2, int>& pivot,                 \
                                             Stream<gpu>* s) {                                 \
    using namespace mxnet;                                                                     \
    using namespace mxnet::op::mxnet_op;                                                       \
    CHECK_NOTNULL(s);                                                                          \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_batch_getri, info, int, A.size(0));                     \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_batch_getri, A_ptr_buf, DType*, A.size(0));             \
    DType** A_ptr = static_cast<DType**>(A_ptr_buf.dptr);                                      \
    EPHEMERAL_GPU_STORAGE_ALLOC(linalg_batch_getri, LU_ptr_buf, DType*, A.size(0));            \
    DType** LU_ptr = static_cast<DType**>(LU_ptr_buf.dptr);                                    \
    Kernel<set_matrix, gpu>::Launch(s, A.size(0), A_ptr, A.dptr_, A.size(1) * A.size(2));      \
    Kernel<set_matrix, gpu>::Launch(s, LU.size(0), LU_ptr, LU.dptr_, LU.size(1) * LU.size(2)); \
    CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s),                                   \
                              A.size(1),                                                       \
                              const_cast<const DType**>(LU_ptr),                               \
                              LU.size(2),                                                      \
                              const_cast<const int*>(pivot.dptr_),                             \
                              A_ptr,                                                           \
                              A.size(2),                                                       \
                              static_cast<int*>(info.dptr),                                    \
                              A.size(0)))                                                      \
    Storage::Get()->Free(info);                                                                \
    Storage::Get()->Free(A_ptr_buf);                                                           \
    Storage::Get()->Free(LU_ptr_buf);                                                          \
  }

#else

#define LINALG_GPU_BATCH_GETRI(fname, DType)                                   \
  template <>                                                                  \
  inline void linalg_batch_getri<gpu, DType>(const Tensor<gpu, 3, DType>& A,   \
                                             const Tensor<gpu, 3, DType>& LU,  \
                                             const Tensor<gpu, 2, int>& pivot, \
                                             Stream<gpu>* s) {                 \
    LOG(FATAL) << "batched getri requires CUDA version >= 8.0!";               \
  }

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_BATCH_GETRI(SgetriBatched, float)
LINALG_GPU_BATCH_GETRI(DgetriBatched, double)

#endif  // __CUDACC__

//////////////////////////////// INVERSE ////////////////////////////////////////////

// CPU/GPU-versions of matrix inverse combining LAPACK function "getrf" and "getri"

// Note A = inverse(B)
#define LINALG_CPU_BATCH_INVERSE(xpu, DType)                                                       \
  template <>                                                                                      \
  inline void linalg_batch_inverse<xpu, DType>(const Tensor<xpu, 3, DType>& A,                     \
                                               const Tensor<xpu, 3, DType>& B,                     \
                                               const mxnet::OpContext& ctx) {                      \
    Stream<xpu>* s = ctx.get_stream<xpu>();                                                        \
    lapack_index_t lwork(linalg_getri_workspace_query(A[0], s));                                   \
    lapack_index_t workspace_size =                                                                \
        (sizeof(lapack_index_t) * A.size(1) + sizeof(DType) * lwork + sizeof(DType) - 1) /         \
        sizeof(DType);                                                                             \
    Tensor<xpu, 1, DType> workspace =                                                              \
        ctx.requested[0].get_space_typed<xpu, 1, DType>(Shape1(workspace_size), s);                \
    const Tensor<xpu, 1, lapack_index_t> pivot(reinterpret_cast<lapack_index_t*>(workspace.dptr_), \
                                               Shape1(A.size(1)));                                 \
    const Tensor<xpu, 1, DType> work(reinterpret_cast<DType*>(pivot.dptr_ + pivot.MSize()),        \
                                     Shape1(lwork));                                               \
    if (A.dptr_ != B.dptr_)                                                                        \
      Copy(A, B, s);                                                                               \
    for (lapack_index_t i = 0; i < A.size(0); ++i) {                                               \
      linalg_getrf(A[i], pivot, true, s);                                                          \
      linalg_getri(A[i], pivot, work, s);                                                          \
    }                                                                                              \
  }
LINALG_CPU_BATCH_INVERSE(cpu, float)
LINALG_CPU_BATCH_INVERSE(cpu, double)

#ifdef __CUDACC__

// GETRF and GETRI only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

#define LINALG_GPU_BATCH_INVERSE(xpu, DType)                                                  \
  template <>                                                                                 \
  inline void linalg_batch_inverse<xpu, DType>(const Tensor<xpu, 3, DType>& A,                \
                                               const Tensor<xpu, 3, DType>& B,                \
                                               const mxnet::OpContext& ctx) {                 \
    Stream<xpu>* s     = ctx.get_stream<xpu>();                                               \
    int pivot_size     = sizeof(int) * A.size(0) * A.size(1);                                 \
    int matrix_size    = sizeof(DType) * A.shape_.Size();                                     \
    int workspace_size = (pivot_size + matrix_size + sizeof(DType) - 1) / sizeof(DType);      \
    Tensor<xpu, 1, DType> workspace =                                                         \
        ctx.requested[0].get_space_typed<xpu, 1, DType>(Shape1(workspace_size), s);           \
    const Tensor<xpu, 2, int> pivot(reinterpret_cast<int*>(workspace.dptr_),                  \
                                    Shape2(A.size(0), A.size(1)));                            \
    int offset = pivot.MSize() & 1 ? pivot.MSize() + 1 : pivot.MSize();                       \
    const Tensor<xpu, 3, DType> LU(reinterpret_cast<DType*>(pivot.dptr_ + offset), A.shape_); \
    Copy(LU, B, s);                                                                           \
    linalg_batch_getrf(LU, pivot, true, s);                                                   \
    linalg_batch_getri(A, LU, pivot, s);                                                      \
  }

#else

#define LINALG_GPU_BATCH_INVERSE(xpu, DType)                                   \
  template <>                                                                  \
  inline void linalg_batch_inverse<xpu, DType>(const Tensor<xpu, 3, DType>& A, \
                                               const Tensor<xpu, 3, DType>& B, \
                                               const mxnet::OpContext& ctx) {  \
    LOG(FATAL) << "gpu matrix inverse requires CUDA version >= 8.0!";          \
  }

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_BATCH_INVERSE(gpu, float)
LINALG_GPU_BATCH_INVERSE(gpu, double)

#endif  // __CUDACC__

//////////////////////////////// DET ////////////////////////////////////////////

// CPU/GPU-versions of helper functions used in matrix determinant operators

#define LINALG_CPU_BATCH_DET_HELPER(xpu, DType, IndexT)                                         \
  template <>                                                                                   \
  inline void linalg_batch_det_backward_helper<xpu, DType>(const Tensor<xpu, 3, DType>& LU,     \
                                                           const Tensor<xpu, 2, IndexT>& pivot, \
                                                           const Tensor<xpu, 1, DType>& det,    \
                                                           const Tensor<xpu, 3, DType>& temp,   \
                                                           const DType zero_det,                \
                                                           const mxnet::OpContext& ctx) {       \
    Stream<xpu>* s = ctx.get_stream<xpu>();                                                     \
    lapack_index_t lwork(linalg_getri_workspace_query(LU[0], s));                               \
    Tensor<xpu, 1, DType> work =                                                                \
        ctx.requested[0].get_space_typed<xpu, 1, DType>(Shape1(lwork), s);                      \
    for (index_t i = 0; i < LU.size(0); ++i) {                                                  \
      if (det[i] != zero_det) {                                                                 \
        linalg_getri(LU[i], pivot[i], work, s);                                                 \
      }                                                                                         \
    }                                                                                           \
  }

LINALG_CPU_BATCH_DET_HELPER(cpu, float, LapackIndex<cpu>::IndexT)
LINALG_CPU_BATCH_DET_HELPER(cpu, double, LapackIndex<cpu>::IndexT)

// GETRF and GETRI only available with cuda8 or higher.
#if CUDA_VERSION >= 8000

#define LINALG_GPU_BATCH_DET_HELPER(xpu, DType)                                               \
  template <>                                                                                 \
  inline void linalg_batch_det_backward_helper<xpu, DType>(const Tensor<xpu, 3, DType>& LU,   \
                                                           const Tensor<xpu, 2, int>& pivot,  \
                                                           const Tensor<xpu, 1, DType>& det,  \
                                                           const Tensor<xpu, 3, DType>& temp, \
                                                           const DType zero_det,              \
                                                           const mxnet::OpContext& ctx) {     \
    Stream<xpu>* s = ctx.get_stream<xpu>();                                                   \
    linalg_batch_getri(temp, LU, pivot, s);                                                   \
    Copy(LU, temp, s);                                                                        \
  }

#else

#define LINALG_GPU_BATCH_DET_HELPER(xpu, DType)                                               \
  template <>                                                                                 \
  inline void linalg_batch_det_backward_helper<xpu, DType>(const Tensor<xpu, 3, DType>& LU,   \
                                                           const Tensor<xpu, 2, int>& pivot,  \
                                                           const Tensor<xpu, 1, DType>& det,  \
                                                           const Tensor<xpu, 3, DType>& temp, \
                                                           const DType zero_det,              \
                                                           const mxnet::OpContext& ctx) {     \
    LOG(FATAL) << "gpu matrix inverse requires CUDA version >= 8.0!";                         \
  }

#endif  // CUDA_VERSION >= 8000

LINALG_GPU_BATCH_DET_HELPER(gpu, float)
LINALG_GPU_BATCH_DET_HELPER(gpu, double)

#endif  // MXNET_OPERATOR_LINALG_IMPL_H_
