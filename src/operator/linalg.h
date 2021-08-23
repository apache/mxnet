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
 * \file linalg.h
 * \brief Unified tensor interface for advanced linear algebra functions
 * (specifically BLAS3/LAPACK) from within mxnet.
 */
#ifndef MXNET_OPERATOR_LINALG_H_
#define MXNET_OPERATOR_LINALG_H_

#include <mshadow/tensor.h>
#include <mxnet/op_attr_types.h>

#include "./c_lapack_api.h"
using namespace mshadow;

// The purpose of this header is to expose the interfaces of the advanced
// linear algebra functions without clutter by the implementations. In contrast
// to the implementations in linalg_inline.h, no macros are used to generate
// similar functions that just differ by name/type in order to improve readability.
//
// Guidelines for extensions:
// For any type of computation the following should be provided at minimum:
//   - 1 templated function supporting cpu/gpu float/double in non-batch mode
//   - 1 templated function supporting cpu/gpu float/double in batch mode
// Naming conventions:
//   - linalg_<func>()
//   - linalg_batch_<func>()
// Signatures of CPU/GPU versions should be equivalent whenever possible including
// that a stream is supplied to the cpu-versions as (optional) last argument.
// The batched versions all work on tensors with one more dimension as the
// non-batched ones and the first/highest dimension iterates over the elements
// within the batch.

//////////////////////////////// GEMM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "gemm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is C = gemm(A,B,C), so C is input and output parameter.
// C = alpha * A * B + beta * C
template<typename xpu, typename DType>
void linalg_gemm(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 const Tensor<xpu, 2, DType>& C, DType alpha, DType beta,
                 bool tA, bool tB, Stream<xpu> *s = 0);

template<typename xpu, typename DType>
void linalg_batch_gemm(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                       const Tensor<xpu, 3, DType>& C, DType alpha, DType beta,
                       bool tA, bool tB, Stream<xpu> *s = 0);

// Version of batch gemmm where rows are indexed at axis 1 and columns at axis 3.
template<typename xpu, typename DType>
void linalg_batch_gemm(const Tensor<xpu, 4, DType>& A, const Tensor<xpu, 4, DType>& B,
                       const Tensor<xpu, 4, DType>& C, DType alpha, DType beta,
                       bool tA, bool tB, Stream<xpu> *s = 0);


template<typename xpu, typename DType>
inline void linalg_gemm(const Tensor<xpu, 2, DType>& A,
                        const Tensor<xpu, 2, DType>& B,
                        const Tensor<xpu, 2, DType>& C,
                        bool tA, bool tB,
                        Stream<xpu> *s = 0,
                        mxnet::OpReqType req = mxnet::kWriteTo);

//////////////////////////////// TRSM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "trsm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is B = trsm(A,B), so B is input and output parameter.
template<typename xpu, typename DType>
void linalg_trsm(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 DType alpha, bool rightside, bool lower, bool transpose, Stream<xpu> *s = 0);

template<typename xpu, typename DType>
inline void linalg_batch_trsm(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                   DType alpha, bool rightside, bool lower, bool transpose, Stream<xpu> *s = 0);

//////////////////////////////// TRMM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "trmm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is B = trmm(A,B), so B is input and output parameter.

template<typename xpu, typename DType>
void linalg_trmm(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 DType alpha, bool rightside, bool lower, bool transpose, Stream<xpu> *s = 0);

template<typename xpu, typename DType>
void linalg_batch_trmm(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B,
                    DType alpha, bool rightside, bool lower, bool transpose, Stream<xpu> *s = 0);

//////////////////////////////// POTRF ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "potrf". Please refer to the LAPACK-documentation
// for further information about the function and its parameters.
// Note that this is A = potrf(A), so A is input and output parameter.

template<typename xpu, typename DType>
void linalg_potrf(const Tensor<xpu, 2, DType>& A, bool lower, Stream<xpu> *s = 0);

template<typename xpu, typename DType>
void linalg_batch_potrf(const Tensor<xpu, 3, DType>& A, bool lower, Stream<xpu> *s = 0);

//////////////////////////////// POTRI ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "potri". Please refer to the LAPACK-documentation
// for further information about the function and its parameters.
// Note that this is A = potri(A), so A is input and output parameter.

template<typename xpu, typename DType>
void linalg_potri(const Tensor<xpu, 2, DType>& A, bool lower, Stream<xpu> *s = 0);

template<typename xpu, typename DType>
void linalg_batch_potri(const Tensor<xpu, 3, DType>& A, bool lower, Stream<xpu> *s = 0);

//////////////////////////////// SYRK ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "syrk". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is B = syrk(A, B), so that B is input and output parameter.

template<typename xpu, typename DType>
void linalg_syrk(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                 DType alpha, DType beta, bool tA, Stream<xpu> *s = 0);

template<typename xpu, typename DType>
void linalg_batch_syrk(const Tensor<xpu, 3, DType>& A,
                       const Tensor<xpu, 3, DType>& B, DType alpha, DType beta,
                       bool tA, Stream<xpu> *s = 0);

//////////////////////////////// GELQF ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK functions "gelqf", "orglq". Please refer to the
// LAPACK documentation for further details.
// Note:
// - Both functions have A as input and output parameter
// - Both functions require extra workspace, passed as 1D tensor
// - We call orglq after gelqf. Apart from A, they also communicate via the
//   first part of the workspace.

template<typename xpu, typename DType>
void linalg_gelqf(const Tensor<xpu, 2, DType>& A,
                  const Tensor<xpu, 1, DType>& work, Stream<xpu> *s = 0);

template<typename xpu, typename DType>
void linalg_orglq(const Tensor<xpu, 2, DType>& A,
                  const Tensor<xpu, 1, DType>& work, Stream<xpu> *s = 0);

// This function determines the amount of workspace needed for linalg_gelqf,
// linalg_orglq. The workspace can be used for both. The first m entries are
// used to communicate information from gelqf to orglq.
template<typename xpu, typename DType>
int linalg_gelqf_workspace_query(const Tensor<xpu, 2, DType>& A,
                                 Stream<xpu> *s = 0);

//////////////////////////////// SYEVD ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "syevd". Please refer to the
// LAPACK documentation for further details.
// Note:
// - A is input and output parameter (overwritten by U)
// - Input A is symmetric, we access the lower triangle only

template<typename xpu, typename DType>
void linalg_syevd(const Tensor<xpu, 2, DType>& A,
                  const Tensor<xpu, 1, DType>& L,
                  const Tensor<xpu, 1, DType>& work,
                  Stream<xpu> *s = 0);

// This function determines the amount of workspace needed for linalg_syevd
// which is returned as number of elements of type DType.
template<typename xpu, typename DType, typename IndexT = typename LapackIndex<xpu>::IndexT>
IndexT linalg_syevd_workspace_query(const Tensor<xpu, 2, DType>& A,
                                 const Tensor<xpu, 1, DType>& L,
                                 Stream<xpu> *s = 0);

//////////////////////////////// GESVD ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "gesvd". Please refer to the
// LAPACK documentation for further details.
// Note: V is input and output parameter (it overwrites A)

template<typename xpu, typename DType>
void linalg_gesvd(const Tensor<xpu, 2, DType>& UT,
                  const Tensor<xpu, 1, DType>& L,
                  const Tensor<xpu, 2, DType>& V,
                  const Tensor<xpu, 1, DType>& work,
                  Stream<xpu>* s = 0);

// This function determines the amount of workspace needed for linalg_gesvd
// which is returned as number of elements of type DType.
template<typename xpu, typename DType>
size_t linalg_gesvd_workspace_query(const Tensor<xpu, 2, DType>& UT,
                                 const Tensor<xpu, 1, DType>& L,
                                 const Tensor<xpu, 2, DType>& V,
                                 Stream<xpu>* s = 0);

//////////////////////////////// GETRF ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "getrf". Please refer to the
// LAPACK documentation for further details.

// Note:
// - A is input and output parameter (overwritten by LU)
// - Param check_singular is only useful in cpu version. If check_singular is false,
//   don't throw error when A is non-invertible matrix.
template<typename xpu, typename DType>
void linalg_getrf(const Tensor<xpu, 2, DType>& A,
                  const Tensor<xpu, 1, lapack_index_t>& pivot,
                  bool check_singular,
                  Stream<xpu> *s = 0);

template<typename xpu, typename DType, typename IndexT>
void linalg_batch_getrf(const Tensor<xpu, 3, DType>& A,
                        const Tensor<xpu, 2, IndexT>& pivot,
                        bool check_singular,
                        Stream<xpu> *s = 0);

//////////////////////////////// GETRI ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "getri". Please refer to the
// LAPACK documentation for further details.

// Note:
// - pivot and LU is the output of getrf(A)
// - LU is also the output parameter (overwritten by inverse(A))
template<typename xpu, typename DType>
void linalg_getri(const Tensor<xpu, 2, DType>& LU,
                  const Tensor<xpu, 1, lapack_index_t>& pivot, \
                  const Tensor<xpu, 1, DType>& work,
                  Stream<xpu> *s = 0);

// Note that this function only implements GPU version with "getriBatched" in cuBLAS.
// Unlike lapack routines in cpu, it is computed out-of-place, so the final matrix
// inverse is stored in A.
template<typename xpu, typename DType, typename IndexT>
void linalg_batch_getri(const Tensor<xpu, 3, DType>& A,
                        const Tensor<xpu, 3, DType>& LU,
                        const Tensor<xpu, 2, IndexT>& pivot,
                        Stream<xpu> *s = 0);

//////////////////////////////// INVERSE ////////////////////////////////////////////

// CPU/GPU-versions of matrix inverse combining LAPACK function "getrf" and "getri"
// Note that A = inverse(B)
template<typename xpu, typename DType, typename IndexT = typename LapackIndex<xpu>::IndexT>
void linalg_batch_inverse(const Tensor<xpu, 3, DType>& A,
                          const Tensor<xpu, 3, DType>& B,
                          const mxnet::OpContext& ctx);

//////////////////////////////// DET ////////////////////////////////////////////

// CPU/GPU-versions of helper functions used in matrix determinant operators

// Helper function in determinant backward computation: compute matrix inverse
// from LU and pivot using temp workspace, the result is stored back to LU
template<typename xpu, typename DType, typename IndexT>
void linalg_batch_det_backward_helper(const Tensor<xpu, 3, DType>& LU,
                                      const Tensor<xpu, 2, IndexT>& pivot,
                                      const Tensor<xpu, 1, DType>& det,
                                      const Tensor<xpu, 3, DType>& temp,
                                      const DType zero_det,
                                      const mxnet::OpContext& ctx);

#ifdef __CUDACC__
#if CUDA_VERSION < 11000
#define VERSION_ADJUSTED_TF32_MATH CUBLAS_DEFAULT_MATH
#else
#define VERSION_ADJUSTED_TF32_MATH CUBLAS_TF32_TENSOR_OP_MATH
#endif
#endif  // __CUDACC__

#include "linalg_impl.h"

#endif  // MXNET_OPERATOR_LINALG_H_
