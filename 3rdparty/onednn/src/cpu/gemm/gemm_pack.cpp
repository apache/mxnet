/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "cpu/platform.hpp"

#include "cpu/gemm/gemm_pack.hpp"

#if DNNL_X64
#include "cpu/x64/gemm/gemm_pack.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

bool pack_sgemm_supported() {
#if DNNL_X64
    return x64::pack_sgemm_supported();
#endif
    return false;
}
bool pack_gemm_bf16bf16f32_supported() {
#if DNNL_X64
    return x64::pack_gemm_bf16bf16f32_supported();
#endif
    return false;
}

dnnl_status_t sgemm_pack_get_size(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, size_t *size, bool *pack) {
#if DNNL_X64
    return x64::sgemm_pack_get_size(
            identifier, transa, transb, M, N, K, lda, ldb, size, pack);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_bf16bf16f32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack) {
#if DNNL_X64
    return x64::gemm_bf16bf16f32_pack_get_size(
            identifier, transa, transb, M, N, K, lda, ldb, size, pack);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_s8u8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack) {
#if DNNL_X64
    return x64::gemm_s8u8s32_pack_get_size(
            identifier, transa, transb, M, N, K, lda, ldb, size, pack);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_s8s8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack) {
#if DNNL_X64
    return x64::gemm_s8s8s32_pack_get_size(
            identifier, transa, transb, M, N, K, lda, ldb, size, pack);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t sgemm_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const float *src, float *dst) {
#if DNNL_X64
    return x64::sgemm_pack(
            identifier, transa, transb, M, N, K, lda, ldb, src, dst);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_bf16bf16f32_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const bfloat16_t *src,
        bfloat16_t *dst) {
#if DNNL_X64
    return x64::gemm_bf16bf16f32_pack(
            identifier, transa, transb, M, N, K, lda, ldb, src, dst);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_s8u8s32_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const void *src, void *dst) {
#if DNNL_X64
    return x64::gemm_s8u8s32_pack(
            identifier, transa, transb, M, N, K, lda, ldb, src, dst);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_s8s8s32_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const void *src, void *dst) {
#if DNNL_X64
    return x64::gemm_s8s8s32_pack(
            identifier, transa, transb, M, N, K, lda, ldb, src, dst);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t sgemm_compute(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *A,
        const dim_t *lda, const float *B, const dim_t *ldb, const float *beta,
        float *C, const dim_t *ldc) {
#if DNNL_X64
    return x64::sgemm_compute(
            transa, transb, M, N, K, A, lda, B, ldb, beta, C, ldc);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_bf16bf16f32_compute(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const bfloat16_t *A,
        const dim_t *lda, const bfloat16_t *B, const dim_t *ldb,
        const float *beta, float *C, const dim_t *ldc) {
#if DNNL_X64
    return x64::gemm_bf16bf16f32_compute(
            transa, transb, M, N, K, A, lda, B, ldb, beta, C, ldc);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_s8u8s32_compute(const char *transa, const char *transb,
        const char *offsetc, const dim_t *M, const dim_t *N, const dim_t *K,
        const int8_t *A, const dim_t *lda, const uint8_t *B, const dim_t *ldb,
        const float *beta, int32_t *C, const dim_t *ldc, const int32_t *co) {
#if DNNL_X64
    return x64::gemm_s8u8s32_compute(
            transa, transb, offsetc, M, N, K, A, lda, B, ldb, beta, C, ldc, co);
#endif
    return dnnl_unimplemented;
}

dnnl_status_t gemm_s8s8s32_compute(const char *transa, const char *transb,
        const char *offsetc, const dim_t *M, const dim_t *N, const dim_t *K,
        const int8_t *A, const dim_t *lda, const int8_t *B, const dim_t *ldb,
        const float *beta, int32_t *C, const dim_t *ldc, const int32_t *co) {
#if DNNL_X64
    return x64::gemm_s8s8s32_compute(
            transa, transb, offsetc, M, N, K, A, lda, B, ldb, beta, C, ldc, co);
#endif
    return dnnl_unimplemented;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
