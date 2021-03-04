/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "common/bfloat16.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"

#include "cpu/platform.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/gemm_info.hpp"
#include "cpu/x64/gemm/gemm_utils.hpp"
#include "cpu/x64/gemm/gemv_driver.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// gemv kernel when A is non-transposed incy == 1 and any stride on X.
template <typename a_t, typename b_t, typename c_t>
static inline void gemv_n_kernel(const dim_t m, const dim_t n, float alpha,
        const a_t *__restrict a, const dim_t lda, const b_t *__restrict x,
        const dim_t incx, c_t *__restrict y, const dim_t incy,
        const gemm_info_t<a_t, b_t, c_t> *arg) {
    assert(incy == 1);

    auto gemv_n_kern = arg->gemv_kernel[no_trans];
    if (gemv_n_kern) {
        gemv_n_kern(&m, &n, &alpha, a, &lda, x, &incx, y, &incy);
    } else {
        if (incx == 1) {
            for (dim_t i = 0; i < n; i++) {
                for (dim_t j = 0; j < m; j++) {
                    y[j] += alpha * x[i] * a[j + i * lda];
                }
            }
        } else {
            dim_t idx = incx < 0 ? (1 - n) * incx : 0;
            for (dim_t i = 0; i < n; i++) {
                for (dim_t j = 0; j < m; j++) {
                    y[j] += alpha * x[idx] * a[j + i * lda];
                }
                idx += incx;
            }
        }
    }
}

// gemv kernel when A is transposed incx == 1 and any stride on Y.
template <typename a_t, typename b_t, typename c_t>
static inline void gemv_t_kernel(const dim_t m, const dim_t n, float alpha,
        const a_t *__restrict a, const dim_t lda, const b_t *__restrict x,
        const dim_t incx, c_t *__restrict y, const dim_t incy,
        const gemm_info_t<a_t, b_t, c_t> *arg) {
    assert(incx == 1);

    auto gemv_t_kern = arg->gemv_kernel[do_trans];
    if (gemv_t_kern) {
        gemv_t_kern(&m, &n, &alpha, a, &lda, x, &incx, y, &incy);
    } else {
        if (incy == 1) {
            for (dim_t i = 0; i < n; i++) {
                c_t temp = (c_t)0;
                for (dim_t j = 0; j < m; j++) {
                    temp += x[j] * a[j + i * lda];
                }
                y[i] += temp * alpha;
            }
        } else {
            dim_t idy = incy < 0 ? (1 - n) * incy : 0;
            for (dim_t i = 0; i < n; i++) {
                c_t temp = (c_t)0;
                for (dim_t j = 0; j < m; j++) {
                    temp += x[j] * a[j + i * lda];
                }
                y[idy] += temp * alpha;

                idy += incy;
            }
        }
    }
}

#define M_BLK 512
template <typename a_t, typename b_t, typename c_t>
static inline void gemv_kernel_driver(const int trans, const dim_t m,
        const dim_t n, const float alpha, const a_t *a, const dim_t lda,
        const b_t *x, const dim_t incx, const float beta, c_t *y,
        const dim_t incy, const gemm_info_t<a_t, b_t, c_t> *arg) {
    // Set dimensions of X and Y vectors based on transpose type.
    dim_t x_dim = trans == no_trans ? n : m;
    dim_t y_dim = trans == no_trans ? m : n;

    if (y_dim <= 0) return;

    // Set the indices for y and x vectors based on incx/incy
    dim_t idx_x = incx < 0 ? (1 - x_dim) * incx : 0;
    dim_t idx_y = incy < 0 ? (1 - y_dim) * incy : 0;

    // Scale the Y vector
    if (beta != 1.0f) {
        if (incy == 1) {
            if (beta == 0.0f) {
                for (dim_t i = 0; i < y_dim; i++) {
                    y[i] = (c_t)0.0f;
                }
            } else {
                for (dim_t i = 0; i < y_dim; i++) {
                    y[i] *= beta;
                }
            }
        } else {
            if (beta == 0.0f) {
                for (dim_t i = 0, inc = idx_y; i < y_dim; i++) {
                    y[inc] = (c_t)0.0f;
                    inc += incy;
                }
            } else {
                for (dim_t i = 0, inc = idx_y; i < y_dim; i++) {
                    y[inc] *= beta;
                    inc += incy;
                }
            }
        }
    }

    if (x_dim <= 0 || alpha == 0.0f) return;

    if (trans == no_trans) { // A is not transpose.
        if (incy == 1) {
            gemv_n_kernel(m, n, alpha, a, lda, x, incx, y, incy, arg);
        } else {
            // Allocate temporary buffer for y vector.
#if !defined(_MSC_VER)
            c_t ytmp[M_BLK];
#else
            c_t *ytmp = (c_t *)_alloca(sizeof(*ytmp) * M_BLK);
#endif

            dim_t m_blk = 0;
            for (dim_t i = 0; i < m; i += m_blk) {
                m_blk = m - i;
                if (m_blk > M_BLK) m_blk = M_BLK;

                for (dim_t j = 0; j < m_blk; j++)
                    ytmp[j] = (c_t)0.0;

                // Call unit-stride kernel.
                gemv_n_kernel(m_blk, n, alpha, a, lda, x, incx, ytmp, 1, arg);

                // Add matrix-vector result back to y vector.
                for (dim_t j = 0, inc = idx_y; j < m_blk; j++) {
                    y[inc] += ytmp[j];
                    inc += incy;
                }
                a += m_blk;
                y += m_blk * incy;
            }
        }
    } else { // Matrix A is transpose.
        if (incx == 1) {
            gemv_t_kernel(m, n, alpha, a, lda, x, incx, y, incy, arg);
        } else {
            // Allocate temporary buffer for x vector.
#if !defined(_MSC_VER)
            b_t xtmp[M_BLK];
#else
            b_t *xtmp = (b_t *)_alloca(sizeof(*xtmp) * M_BLK);
#endif
            dim_t m_blk = 0;
            for (dim_t i = 0; i < m; i += m_blk) {
                m_blk = m - i;
                if (m_blk > M_BLK) m_blk = M_BLK;

                // Copy a block of x vector to temporary buffer.
                for (dim_t j = 0, inc = idx_x; j < m_blk; j++) {
                    xtmp[j] = x[inc];
                    inc += incx;
                }

                // Call unit-stride kernel.
                gemv_t_kernel(m_blk, n, alpha, a, lda, xtmp, 1, y, incy, arg);

                a += m_blk;
                x += m_blk * incx;
            }
        }
    }
}
#undef M_BLK

#define M_MIN 128
#define N_MIN 128
#define BAND_MIN 32
#define MN_MIN_N 1536
#define MN_MIN_T 2048
#define M_LARGE 20000
#define N_LARGE 20000
#define M_SMALL 200
#define N_SMALL 200
#define CONST1_AVX2 288
#define CONST2_AVX2 41700
#define MIN_WIDTH 32
// Check if threading is beneficial.
template <typename a_t>
static inline int thread_checker(
        int nthr, const dim_t m, const dim_t n, int trans) {
    constexpr bool is_f32
            = utils::one_of(data_traits<a_t>::data_type, data_type::f32);

    if (is_f32) {
        // Threshold based on performance measurement with warm and cold cache
        // to decide when threading is beneficial.
        if (mayiuse(avx2)) {
            if (m * n + CONST1_AVX2 * n < CONST2_AVX2) { return 1; }
        } else {
            if (m < M_MIN && n < N_MIN) {
                // Execute in sequential mode for small n and m.
                return 1;
            }
        }

        if (m >= M_LARGE && n <= N_SMALL) {
            // Execute in parallel mode.
            return nthr;
        }

        dim_t bandt = n / nthr; // size per thread.

        if (nthr <= 12 && bandt < BAND_MIN) {
            if (m * bandt < MN_MIN_T) { return 1; }
        } else if (nthr <= 12 && m * bandt < 2 * MN_MIN_T) {
            return 1;
        } else if (nthr > 12 && bandt * m < 2 * MN_MIN_T) {
            if (bandt == 0) {
                return 1;
            } else {
                return nstl::min(nstl::max(n * m / (2 * MN_MIN_N), dim_t(1)),
                        dim_t(nthr));
            }
        }
    } else {
        if (trans) {
            if (MIN_WIDTH * nthr > m) nthr = utils::div_up(m, MIN_WIDTH);
        } else {
            if (MIN_WIDTH * nthr > n) nthr = utils::div_up(n, MIN_WIDTH);
        }
    }

    return nthr;
}
#undef M_MIN
#undef N_MIN
#undef BAND_MIN
#undef MN_MIN_N
#undef MN_MIN_T
#undef M_LARGE
#undef N_LARGE
#undef M_SMALL
#undef N_SMALL
#undef CONST1_AVX2
#undef CONST2_AVX2
#undef MIN_WIDTH

template <typename T>
static inline void decompose_vector(const dim_t m, const dim_t nthr,
        const dim_t ithr, T *addr, dim_t *offset, dim_t *size) {
    dim_t loffset = 0;
    dim_t lsize = 0;

    if (ithr >= nthr) {
        *offset = loffset;
        *size = lsize;
        return;
    }

    if (addr == nullptr) {
        dim_t xthr = m % nthr;
        dim_t width = m / nthr;

        if (ithr < xthr) {
            lsize = width + 1;
            loffset = ithr * lsize;
        } else {
            lsize = width;
            loffset = m - (nthr - ithr) * lsize;
        }
    }

    *offset = loffset;
    *size = lsize;
}

static inline void part_1d(const dim_t k, const int ithr, const int nthr,
        dim_t &off, dim_t &size) {
    if (ithr >= nthr) {
        size = 0;
        off = 0;
        return;
    }
    size = utils::div_up(k, nthr);
    off = ithr * size;
    if (off > k) off = k;
    if (off + size > k) size = k - off;
}

template <typename a_t, typename b_t, typename c_t>
static inline void gemv_threading_driver(const int trans, const dim_t m,
        const dim_t n, const float alpha, const a_t *a, const dim_t lda,
        const b_t *x, const dim_t incx, const float beta, c_t *y,
        const dim_t incy, const gemm_info_t<a_t, b_t, c_t> *arg) {
    constexpr bool is_f32
            = utils::one_of(data_traits<a_t>::data_type, data_type::f32);

    // Quick return if possible.
    if (m <= 0 || n <= 0) return;

    auto nthr_max = (dnnl_in_parallel()) ? 1 : dnnl_get_max_threads();
    auto nthr_goal = thread_checker<a_t>(nthr_max, m, n, trans);

    if (nthr_goal == 1) {
        gemv_kernel_driver(
                trans, m, n, alpha, a, lda, x, incx, beta, y, incy, arg);
        return;
    }

    c_t *ybuf = nullptr;
    if (trans == no_trans && dnnl_thr_syncable() && !is_f32)
        ybuf = (c_t *)malloc(sizeof(*ybuf) * m * (nthr_goal - 1), PAGE_4K);

    // Always use the maximum number of threads to avoid OMP overhead that can
    // occur due to change thread counts.
    auto nthr_spawn = dnnl_thr_syncable() ? nthr_max : nthr_goal;
    parallel(nthr_spawn, [&](int ithr, int nthr) {
        int nthr_eff = nstl::min(nthr_goal, nthr);
        if (is_f32) {
            dim_t band, disp;
            decompose_vector(n, nthr_eff, ithr, (c_t *)nullptr, &disp, &band);

            dim_t ydisp = disp * incy;
            if (incy < 0) ydisp = ydisp + (-n + band) * incy;

            disp = disp * lda;

            auto a_loc = a + disp;
            auto x_loc = x;
            auto y_loc = y + ydisp;
            gemv_kernel_driver(trans, m, band, alpha, a_loc, lda, x_loc, incx,
                    beta, y_loc, incy, arg);
        } else {
            dim_t thread_m = m, off_m = 0;
            dim_t thread_n = n, off_n = 0;
            dim_t band = 1;

            // Default effective values.
            auto a_eff = a;
            auto x_eff = x;
            auto y_eff = y;
            auto incy_eff = incy;
            auto beta_eff = beta;

            if (trans == do_trans) {
                part_1d(n, ithr, nthr_eff, off_n, thread_n);
                a_eff += off_m + off_n * lda;
                y_eff += off_n * incy;
                band = thread_n;
            } else if (ybuf) {
                part_1d(n, ithr, nthr_eff, off_n, thread_n);
                a_eff += off_m + off_n * lda;
                x_eff += off_n * incx;
                if (ithr != 0) {
                    y_eff = ybuf + m * (ithr - 1);
                    incy_eff = 1;
                    beta_eff = 0.0;
                }
            } else {
                // Fallback for no_trans with no extra buffer.
                part_1d(m, ithr, nthr_eff, off_m, thread_m);
                a_eff += off_m + off_n * lda;
                y_eff += off_m * incy;
                band = thread_m;
            }

            // Buffers for y need to be set to zero for reduction case.
            assert(IMPLICATION(ybuf, band > 0));

            if (band > 0 && ithr < nthr_eff)
                gemv_kernel_driver(trans, thread_m, thread_n, alpha, a_eff, lda,
                        x_eff, incx, beta_eff, y_eff, incy_eff, arg);

            // Do reduction for multiple buffers if needed.
            if (ybuf) {
                dnnl_thr_barrier();
                // Reduction in each thread.
                part_1d(m, ithr, nthr_eff, off_m, thread_m);
                for (int buf_id = 0; buf_id < nthr_eff - 1; buf_id++)
                    for (dim_t i = off_m; i < off_m + thread_m; i++)
                        y[i * incy] += ybuf[i + buf_id * m];
            }
        }
    });

    free(ybuf);
}

template <>
dnnl_status_t jump_to_gemv(const gemm_info_t<int8_t, uint8_t, int32_t> *arg) {
    return dnnl_unimplemented;
}

template <>
dnnl_status_t jump_to_gemv(const gemm_info_t<int8_t, int8_t, int32_t> *arg) {
    return dnnl_unimplemented;
}

template <typename a_t, typename b_t, typename c_t>
dnnl_status_t jump_to_gemv(const gemm_info_t<a_t, b_t, c_t> *arg) {
    constexpr bool is_f32
            = utils::one_of(data_traits<a_t>::data_type, data_type::f32);

    int transa = arg->transa;
    int transb = arg->transb;

    dim_t m = arg->m;
    dim_t n = arg->n;
    dim_t k = arg->k;

    dim_t lda = arg->lda;
    dim_t ldb = arg->ldb;
    dim_t ldc = arg->ldc;

    float alpha = arg->alpha;
    float beta = arg->beta;

    const a_t *a = arg->a;
    const b_t *b = arg->b;
    c_t *c = arg->c;

    if (k == 0) return dnnl_success;

    auto packing = (arg->packing != pack_type::none);
    auto do_a = (arg->packing == pack_type::pack_a);
    gemm_pack_storage_t *pack_dst = arg->pack_dst;

    if (n == 1 && (transa == do_trans || packing)) {
        if (!packing) {
            gemv_threading_driver(do_trans, k, m, alpha, a, lda, b,
                    transb == no_trans ? 1 : ldb, beta, c, 1, arg);
        } else {
            if (do_a) {
                gemm_utils::prep_gemm_pack<a_t, c_t>(
                        do_a, do_trans, m, k, pack_dst);
            } else {
                gemm_utils::prep_gemm_pack<b_t, c_t>(
                        do_a, no_trans, k, n, pack_dst);
            }

            if (arg->measure_only) return dnnl_success;

            if (do_a) {
                gemm_utils::pack_no_copy(a, lda, m, k, transa, alpha, pack_dst);
            } else {
                gemm_utils::pack_no_copy(b, ldb, k, n, transb, alpha, pack_dst);
            }
        }
        return dnnl_success;
    } else if (n == 1 && transa == no_trans && !is_f32 && !packing) {
        gemv_threading_driver(no_trans, m, k, alpha, a, lda, b,
                transb == no_trans ? 1 : ldb, beta, c, 1, arg);
        return dnnl_success;
    }

    if (m == 1 && (transb == no_trans || packing)) {
        if (!packing) {
            gemv_threading_driver(do_trans, k, n, alpha, b, ldb, a,
                    transa == no_trans ? lda : 1, beta, c, ldc, arg);
        } else {
            if (do_a) {
                gemm_utils::prep_gemm_pack<a_t, c_t>(
                        do_a, do_trans, m, k, pack_dst);
            } else {
                gemm_utils::prep_gemm_pack<b_t, c_t>(
                        do_a, no_trans, k, n, pack_dst);
            }

            if (arg->measure_only) return dnnl_success;

            if (do_a) {
                gemm_utils::pack_no_copy(a, lda, m, k, transa, alpha, pack_dst);
            } else {
                gemm_utils::pack_no_copy(b, ldb, k, n, transb, alpha, pack_dst);
            }
        }
        return dnnl_success;
    } else if (m == 1 && transb == do_trans && !is_f32 && !packing) {
        gemv_threading_driver(no_trans, n, k, alpha, b, ldb, a,
                transa == no_trans ? lda : 1, beta, c, ldc, arg);
        return dnnl_success;
    }

    return dnnl_unimplemented;
}

template // Instatiate gemv_f32
        dnnl_status_t
        jump_to_gemv<float, float, float>(
                const gemm_info_t<float, float, float> *arg);
template // Instatiate gemv_bf16bf16f32
        dnnl_status_t
        jump_to_gemv<bfloat16_t, bfloat16_t, float>(
                const gemm_info_t<bfloat16_t, bfloat16_t, float> *arg);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
