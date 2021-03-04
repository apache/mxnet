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

#ifndef CPU_X64_GEMM_AMX_IGEMM_K_HPP
#define CPU_X64_GEMM_AMX_IGEMM_K_HPP

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename T>
struct gemm_amx_traits {
    static constexpr dim_t UNROLL_M = 1;
    static constexpr dim_t UNROLL_N = 1;
    static constexpr dim_t UNROLL_K = 1;
    static constexpr int NTILES_M = 1;
    static constexpr int NTILES_N = 1;
    static constexpr dim_t UNROLL_MM = 1;
    static constexpr dim_t UNROLL_NN = 1;
    static constexpr dim_t UNROLL_KK = 1;
};

template <>
struct gemm_amx_traits<bfloat16_t> {
    static constexpr dim_t UNROLL_M = 16;
    static constexpr dim_t UNROLL_N = 16;
    static constexpr dim_t UNROLL_K = 2;
    static constexpr int NTILES_M = 2;
    static constexpr int NTILES_N = 2;
    static constexpr dim_t UNROLL_MM = UNROLL_M * NTILES_M;
    static constexpr dim_t UNROLL_NN = UNROLL_N * NTILES_N;
    static constexpr dim_t UNROLL_KK = UNROLL_K * 16;
};

template <>
struct gemm_amx_traits<int8_t> {
    static constexpr dim_t UNROLL_M = 16;
    static constexpr dim_t UNROLL_N = 16;
    static constexpr dim_t UNROLL_K = 4;
    static constexpr int NTILES_M = 2;
    static constexpr int NTILES_N = 2;
    static constexpr dim_t UNROLL_MM = UNROLL_M * NTILES_M;
    static constexpr dim_t UNROLL_NN = UNROLL_N * NTILES_N;
    static constexpr dim_t UNROLL_KK = UNROLL_K * 16;
};

template <>
struct gemm_amx_traits<uint8_t> {
    static constexpr dim_t UNROLL_M = 16;
    static constexpr dim_t UNROLL_N = 16;
    static constexpr dim_t UNROLL_K = 4;
    static constexpr int NTILES_M = 2;
    static constexpr int NTILES_N = 2;
    static constexpr dim_t UNROLL_MM = UNROLL_M * NTILES_M;
    static constexpr dim_t UNROLL_NN = UNROLL_N * NTILES_N;
    static constexpr dim_t UNROLL_KK = UNROLL_K * 16;
};

template <typename a_t, typename b_t, typename c_t>
struct amx_gemm {

public:
    static void packAN_amx(const dim_t *p_m, const dim_t *p_n, const a_t *a,
            const dim_t *p_lda, const float *alpha, a_t *b, const dim_t *dummy1,
            const dim_t *dummy2, c_t *a_row_sum) {
        constexpr dim_t um = UNROLL_M_;
        constexpr dim_t uk = UNROLL_K_;
        constexpr dim_t umm = UNROLL_MM_;
        constexpr dim_t ukk = UNROLL_KK_;
        constexpr dim_t ntm = NTILES_M_;

        UNUSED(alpha);
        UNUSED(dummy1);
        UNUSED(dummy2);

        dim_t m = *p_m;
        dim_t n = *p_n;
        dim_t lda = *p_lda;

        for_(dim_t i = 0; i < n; i += umm)
        for_(dim_t tj = 0; tj < m; tj += ukk)
        for_(dim_t ti = 0; ti < ntm; ti++)
        for_(dim_t j = 0; j < ukk; j += uk)
        for_(dim_t ii = 0; ii < nstl::min(um, n - i - ti * um); ii++)
        for (dim_t jj = 0; jj < uk; jj++) {
            if (j + jj + tj < m) {
                *b = *(a + j + jj + tj + (i + ii + ti * um) * lda);

                // Compute a_row_sum.
                if (a_row_sum) {
                    if (j + jj + tj == 0) a_row_sum[i + ii + ti * um] = 0;
                    a_row_sum[i + ii + ti * um] += *b;
                }
            } else {
                *b = 0;
            }
            b++;
        }
    };

    static void packAT_amx(const dim_t *p_m, const dim_t *p_n, const a_t *a,
            const dim_t *p_lda, const float *alpha, a_t *b, const dim_t *dummy1,
            const dim_t *dummy2, c_t *a_row_sum) {
        constexpr dim_t um = UNROLL_M_;
        constexpr dim_t uk = UNROLL_K_;
        constexpr dim_t umm = UNROLL_MM_;
        constexpr dim_t ukk = UNROLL_KK_;
        constexpr dim_t ntm = NTILES_M_;

        UNUSED(alpha);
        UNUSED(dummy1);
        UNUSED(dummy2);

        dim_t m = *p_m;
        dim_t n = *p_n;
        dim_t lda = *p_lda;

        for (dim_t i = 0; i < n; i += umm)
            for (dim_t tj = 0; tj < m; tj += ukk)
                for (dim_t ti = 0; ti < ntm; ti++)
                    for (dim_t j = 0; j < ukk; j += uk)
                        for (dim_t ii = 0; ii < nstl::min(um, n - i - ti * um);
                                ii++)
                            for (dim_t jj = 0; jj < uk; jj++) {
                                if (j + jj + tj < m) {
                                    *b = *(a + i + ii + ti * um
                                            + (j + jj + tj) * lda);

                                    // Compute a_row_sum.
                                    if (a_row_sum) {
                                        if (j + jj + tj == 0)
                                            a_row_sum[i + ii + ti * um] = 0;
                                        a_row_sum[i + ii + ti * um] += *b;
                                    }
                                } else {
                                    *b = 0;
                                }
                                b++;
                            }
    };

    static void packBN_amx(const dim_t *p_m, const dim_t *p_n, const b_t *a,
            const dim_t *p_lda, const float *alpha, b_t *b, const dim_t *dummy1,
            const dim_t *dummy2, c_t *b_col_sum) {
        constexpr dim_t unn = UNROLL_NN_;
        constexpr dim_t ukk = UNROLL_KK_;

        UNUSED(alpha);
        UNUSED(dummy1);
        UNUSED(dummy2);

        dim_t m = *p_m;
        dim_t n = *p_n;
        dim_t lda = *p_lda;

        for (dim_t i = 0; i < n; i += unn)
            for (dim_t j = 0; j < m; j += ukk)
                for (dim_t ii = 0; ii < nstl::min(unn, n - i); ii++)
                    for (dim_t jj = 0; jj < ukk; jj++) {
                        if ((j + jj < m) && (i + ii < n)) {
                            *b = *(a + j + jj + (i + ii) * lda);

                            // Compute b_col_sum.
                            if (b_col_sum) {
                                if (j + jj == 0) b_col_sum[i + ii] = 0;
                                b_col_sum[i + ii] += *b;
                            }
                        } else {
                            *b = 0;
                        }
                        b++;
                    }
    };

    static void packBT_amx(const dim_t *p_m, const dim_t *p_n, const b_t *a,
            const dim_t *p_lda, const float *alpha, b_t *b, const dim_t *dummy1,
            const dim_t *dummy2, c_t *b_col_sum) {
        constexpr dim_t unn = UNROLL_NN_;
        constexpr dim_t ukk = UNROLL_KK_;

        UNUSED(alpha);
        UNUSED(dummy1);
        UNUSED(dummy2);

        dim_t m = *p_m;
        dim_t n = *p_n;
        dim_t lda = *p_lda;

        for (dim_t i = 0; i < n; i += unn)
            for (dim_t j = 0; j < m; j += ukk)
                for (dim_t ii = 0; ii < nstl::min(unn, n - i); ii++)
                    for (dim_t jj = 0; jj < ukk; jj++) {
                        if ((j + jj < m) && (i + ii < n)) {
                            *b = *(a + i + ii + (j + jj) * lda);

                            // Compute b_col_sum.
                            if (b_col_sum) {
                                if (j + jj == 0) b_col_sum[i + ii] = 0;
                                b_col_sum[i + ii] += *b;
                            }
                        } else {
                            *b = 0;
                        }
                        b++;
                    }
    };

private:
    static constexpr dim_t UNROLL_M_ = gemm_amx_traits<a_t>::UNROLL_M;
    static constexpr dim_t UNROLL_N_ = gemm_amx_traits<a_t>::UNROLL_N;
    static constexpr dim_t UNROLL_K_ = gemm_amx_traits<a_t>::UNROLL_K;
    static constexpr int NTILES_M_ = gemm_amx_traits<a_t>::NTILES_M;
    static constexpr int NTILES_N_ = gemm_amx_traits<a_t>::NTILES_N;
    static constexpr dim_t UNROLL_MM_ = gemm_amx_traits<a_t>::UNROLL_MM;
    static constexpr dim_t UNROLL_NN_ = gemm_amx_traits<a_t>::UNROLL_NN;
    static constexpr dim_t UNROLL_KK_ = gemm_amx_traits<a_t>::UNROLL_KK;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_AMX_IGEMM_K_HPP
