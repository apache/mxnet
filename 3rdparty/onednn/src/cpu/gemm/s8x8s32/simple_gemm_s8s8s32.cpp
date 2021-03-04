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
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/gemm/s8x8s32/simple_gemm_s8s8s32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

void compensation_init(const char *offsetC, int32_t *compensation, dim_t len,
        const int32_t *oc) {
    bool OCisC = (*offsetC == 'C' || *offsetC == 'c');
    bool OCisF = (*offsetC == 'F' || *offsetC == 'f');

    if (OCisF && (*oc) != 0) {
        for (dim_t i = 0; i < len; i++)
            compensation[i] = *oc;
    } else if (OCisC) {
        for (dim_t i = 0; i < len; i++)
            compensation[i] = oc[i];
    } else {
        for (dim_t i = 0; i < len; i++)
            compensation[i] = 0;
    }
}

void compensation_compute(bool transa, dim_t m, dim_t k, float alpha,
        const int8_t *a, dim_t lda, int32_t *compensation) {
    if (!transa) {
        const int L2_cache_size = platform::get_per_core_cache_size(2);
        const int blocking_factor = nstl::min(k, L2_cache_size / lda + 1);
        const dim_t npanels = k / blocking_factor;
        const bool has_tile = k % blocking_factor > 0;

        parallel_nd(npanels, m, [&](dim_t j, dim_t i) {
            int32_t val = 0;
            for (dim_t jb = 0; jb < blocking_factor; jb++) {
                val += a[(i + j * blocking_factor * lda) + jb * lda];
            }
            if (alpha != 1.0f) {
                val = out_round<int32_t>(
                        saturate<int32_t>((double)val * alpha * -128.0));
            } else {
                val *= -128;
            }
            fetch_and_add(&compensation[i], val);
        });

        if (has_tile) {
            parallel_nd(m, [=](dim_t i) {
                int32_t val = 0;
                for (dim_t j = npanels * blocking_factor; j < k; j++) {
                    val += a[i + j * lda];
                }
                if (alpha != 1.0f) {
                    val = out_round<int32_t>(
                            saturate<int32_t>((double)val * alpha * -128.0));
                } else {
                    val *= -128;
                }
                fetch_and_add(&compensation[i], val);
            });
        }
    } else {
        parallel_nd(m, [=](dim_t i) {
            int32_t val = 0;
            for (dim_t j = 0; j < k; j++) {
                val += a[j + i * lda];
            }
            if (alpha != 1.0f) {
                val = out_round<int32_t>(
                        saturate<int32_t>((double)val * alpha * -128.0));
            } else {
                val *= -128;
            }
            compensation[i] += val;
        });
    }
}

void copy_and_shift_b(bool transb, dim_t k, dim_t n, uint8_t *b_u8,
        dim_t ldb_u8, const int8_t *b_s8, dim_t ldb_s8) {
    const dim_t b_cols = transb ? k : n;

    parallel_nd(b_cols, [=](dim_t j) {
        const dim_t b_rows = transb ? n : k;

        uint8_t *pb_u8 = b_u8 + j * ldb_u8;
        const int8_t *pb_s8 = b_s8 + j * ldb_s8;

        for (dim_t i = 0; i < b_rows; i++) {
            (*pb_u8) = (*pb_s8) + 128;
            pb_u8++;
            pb_s8++;
        }
    });
}

/**
 * gemm_s8s8s32 operation is defined as follows:
 * C = alpha * op(A) * (op(B) + B_shift) + beta * C + C_offset + compensation
 *
 * where
 *  - compensation is a vector of length m that contains computed compensation
 *   that may contain C_offset if applicable. The compensation is applied inside
 *   gemm_s8u8s32 as a C_offset
 *  - B_shift is a k-by-n matrix, every element of B_shift is equal to 128
 *
 *  What is the compensation:
 *  In order to prepare the matrix B for gemm_s8u8s32 call the B_shift is applied:
 *  C = alpha * op(A) * (op(B) + B_shift) + beta * C + C_offset =
 *  alpha * op(A) * op(B) + alpha * op(A) * B_shift + beta * C + C_offset
 *  compensation = -alpha * op(A) * B_shift
 *  Since B_shift is a matrix, every element of which is equal to 128 then
 *  - if op(A) = A: compensation contains sum of the elements in each row
 *   scaled by -128 * alpha
 *  - if op(A) = A**T: compensation contains sum of the elements in each column
 *   scaled by -128 * alpha
 *
 * The rest of parameters is described in dnnl.h
 */
dnnl_status_t simple_gemm_s8s8s32(const char *transA, const char *transB,
        const char *offsetC, const dim_t *m, const dim_t *n, const dim_t *k,
        const float *alpha, const int8_t *a, const dim_t *lda, const int8_t *oa,
        const int8_t *b, const dim_t *ldb, const int8_t *ob, const float *beta,
        int32_t *c, const dim_t *ldc, const int32_t *oc) {
    if (*oa != 0 || *ob != 0) return dnnl_unimplemented;

    dim_t M = *m, N = *n, K = *k;
    bool transa = (*transA == 'T' || *transA == 't');
    bool transb = (*transB == 'T' || *transB == 't');
    dim_t ld = transb ? N : K;

    uint8_t *b_u8 = (uint8_t *)malloc(
            sizeof(uint8_t) * K * N, platform::get_cache_line_size());
    uint8_t ob_u8 = 0;
    int32_t *compensation = (int32_t *)malloc(
            sizeof(int32_t) * M, platform::get_cache_line_size());

    if (utils::any_null(b_u8, compensation)) {
        free(b_u8);
        free(compensation);
        return dnnl_out_of_memory;
    }

    compensation_init(offsetC, compensation, M, oc);
    compensation_compute(transa, M, K, *alpha, a, *lda, compensation);
    copy_and_shift_b(transb, K, N, b_u8, ld, b, *ldb);

    status_t st = gemm_s8x8s32(transA, transB, "C", m, n, k, alpha, a, lda, oa,
            b_u8, &ld, &ob_u8, beta, c, ldc, compensation);
    if (st != dnnl_success) return st;

    if ((*offsetC == 'R' || *offsetC == 'r'))
        parallel_nd(M, N, [=](dim_t i, dim_t j) { c[i + j * *ldc] += oc[j]; });

    free(b_u8);
    free(compensation);

    return st;
}
} // namespace cpu
} // namespace impl
} // namespace dnnl
