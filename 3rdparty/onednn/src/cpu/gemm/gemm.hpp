/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_GEMM_GEMM_HPP
#define CPU_GEMM_GEMM_HPP

#include "oneapi/dnnl/dnnl_types.h"

#include "common/bfloat16.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/os_blas.hpp"

#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

dnnl_status_t extended_sgemm(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *alpha,
        const float *A, const dim_t *lda, const float *B, const dim_t *ldb,
        const float *beta, float *C, const dim_t *ldc,
        const float *bias = nullptr, bool force_jit_gemm = false);

template <typename b_dt>
dnnl_status_t gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const dim_t *M, const dim_t *N, const dim_t *K,
        const float *alpha, const int8_t *A, const dim_t *lda, const int8_t *ao,
        const b_dt *B, const dim_t *ldb, const b_dt *bo, const float *beta,
        int32_t *c, const dim_t *ldc, const int32_t *co);

dnnl_status_t gemm_bf16bf16f32(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *alpha,
        const bfloat16_t *A, const dim_t *lda, const bfloat16_t *B,
        const dim_t *ldb, const float *beta, float *C, const dim_t *ldc);

#if defined(USE_CBLAS)
#define GEMM_IMPL_STR "gemm:blas"
#elif DNNL_X64
#define GEMM_IMPL_STR "gemm:jit"
#else
#define GEMM_IMPL_STR "gemm:ref"
#endif

#if USE_MKL_IGEMM
#define IGEMM_S8U8S32_IMPL_STR "igemm_s8u8s32:blas"
#define IGEMM_S8S8S32_IMPL_STR "igemm_s8s8s32:blas"
#elif DNNL_X64
#define IGEMM_S8U8S32_IMPL_STR "igemm_s8u8s32:jit"
#define IGEMM_S8S8S32_IMPL_STR "igemm_s8s8s32:jit"
#else
#define IGEMM_S8U8S32_IMPL_STR "igemm_s8u8s32:ref"
#define IGEMM_S8S8S32_IMPL_STR "igemm_s8s8s32:ref"
#endif

#if !defined(USE_MKL_IGEMM) && defined(DNNL_X64)
#define IGEMM_S8U8S32_ISA_STR \
    JIT_IMPL_NAME_HELPER(IGEMM_S8U8S32_IMPL_STR ":", \
            mayiuse(avx512_core_vnni) \
                    ? avx512_core_vnni \
                    : (mayiuse(avx512_core) ? avx512_core : isa_any), \
            "")
#else
#define IGEMM_S8U8S32_ISA_STR IGEMM_S8U8S32_IMPL_STR
#endif

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_GEMM_GEMM_HPP
