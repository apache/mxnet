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

#include <cstdint>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/gemm/f32/ref_gemm_f32.hpp"

#include "cpu/gemm/s8x8s32/ref_gemm_s8x8s32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <typename b_dt>
dnnl_status_t ref_gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const dim_t *M, const dim_t *N, const dim_t *K,
        const float *alpha, const int8_t *A, const dim_t *LDA, const int8_t *ao,
        const b_dt *B, const dim_t *LDB, const b_dt *bo, const float *beta,
        int32_t *C, const dim_t *LDC, const int32_t *co) {

    if (*M == 0 || *N == 0 || *K == 0) return dnnl_success;

    if (!(utils::one_of(*transa, 'n', 'N', 't', 'T')
                && utils::one_of(*transb, 'n', 'N', 't', 'T')))
        return dnnl_unimplemented;

    bool OCisR = (*offsetc == 'R' || *offsetc == 'r');
    bool OCisC = (*offsetc == 'C' || *offsetc == 'c');
    bool AisN = (*transa == 'N' || *transa == 'n');
    bool BisN = (*transb == 'N' || *transb == 'n');

    dim_t m = *M, n = *N, k = *K, lda = *LDA, ldb = *LDB, ldc = *LDC;
    size_t sizeA = AisN ? lda * k : lda * m;
    size_t sizeB = BisN ? ldb * n : ldb * k;
    size_t sizeC = ldc * n;

    double *dA = (double *)malloc(sizeA * sizeof(double), PAGE_4K);
    double *dB = (double *)malloc(sizeB * sizeof(double), PAGE_4K);
    double *dC = (double *)malloc(sizeC * sizeof(double), PAGE_4K);

    if (utils::any_null(dA, dB, dC)) {
        free(dA);
        free(dB);
        free(dC);
        return dnnl_out_of_memory;
    }

    auto da_setter = [=](dim_t i, dim_t j, double v) { dA[j * lda + i] = v; };
    auto db_setter = [=](dim_t i, dim_t j, double v) { dB[j * ldb + i] = v; };

    auto ia_accessor = [=](dim_t i, dim_t j) { return A[j * lda + i]; };
    auto ib_accessor = [=](dim_t i, dim_t j) { return B[j * ldb + i]; };

    const int a_rows = AisN ? m : k;
    const int a_cols = AisN ? k : m;
    dnnl::impl::parallel_nd(a_cols, a_rows, [&](dim_t j, dim_t i) {
        da_setter(i, j,
                static_cast<double>(ia_accessor(i, j))
                        - static_cast<double>(ao[0]));
    });

    const dim_t b_rows = BisN ? k : n;
    const dim_t b_cols = BisN ? n : k;
    dnnl::impl::parallel_nd(b_cols, b_rows, [&](dim_t j, dim_t i) {
        db_setter(i, j,
                static_cast<double>(ib_accessor(i, j))
                        - static_cast<double>(bo[0]));
    });
    double one = 1.0, zero = 0.0;
    ref_gemm<double>(transa, transb, M, N, K, &one, dA, LDA, dB, LDB, &zero, dC,
            LDC, nullptr);

    auto i2d = [=](int32_t v) { return static_cast<double>(v); };
    auto f2d = [=](float v) { return static_cast<double>(v); };

    dnnl::impl::parallel_nd(n, m, [&](dim_t j, dim_t i) {
        double coffset = OCisR ? i2d(co[j]) : OCisC ? i2d(co[i]) : i2d(co[0]);
        double val = ((*beta == 0.0f) ? 0.0 : f2d(*beta) * i2d(C[i + j * ldc]))
                + f2d(*alpha) * dC[i + j * ldc] + coffset;
        C[i + j * ldc] = out_round<int32_t>(saturate<int32_t>(val));
    });

    free(dA);
    free(dB);
    free(dC);
    return dnnl_success;
}

template dnnl_status_t ref_gemm_s8x8s32<uint8_t>(const char *transa,
        const char *transb, const char *offsetc, const dim_t *M, const dim_t *N,
        const dim_t *K, const float *alpha, const int8_t *A, const dim_t *LDA,
        const int8_t *ao, const uint8_t *B, const dim_t *LDB, const uint8_t *bo,
        const float *beta, int32_t *C, const dim_t *LDC, const int32_t *co);

template dnnl_status_t ref_gemm_s8x8s32<int8_t>(const char *transa,
        const char *transb, const char *offsetc, const dim_t *M, const dim_t *N,
        const dim_t *K, const float *alpha, const int8_t *A, const dim_t *LDA,
        const int8_t *ao, const int8_t *B, const dim_t *LDB, const int8_t *bo,
        const float *beta, int32_t *C, const dim_t *LDC, const int32_t *co);

} // namespace cpu
} // namespace impl
} // namespace dnnl
