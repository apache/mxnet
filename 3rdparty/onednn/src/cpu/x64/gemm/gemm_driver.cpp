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
#if defined(_MSC_VER)
#include <malloc.h>
#endif

#include "oneapi/dnnl/dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/dnnl_traits.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/f32/gemm_utils_f32.hpp"
#include "cpu/gemm/gemm_msan_unpoison.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/gemm_driver.hpp"
#include "cpu/x64/gemm/gemm_info.hpp"
#include "cpu/x64/gemm/gemm_partition.hpp"
#include "cpu/x64/gemm/gemm_threading.hpp"
#include "cpu/x64/gemm/gemm_utils.hpp"
#include "cpu/x64/gemm/gemv_driver.hpp"

#include "cpu/x64/gemm/f32/jit_avx512_common_gemm_f32.hpp"
#include "cpu/x64/gemm/f32/jit_avx512_core_gemm_smalln_tn_f32_kern.hpp"
#include "cpu/x64/gemm/f32/jit_avx_gemm_f32.hpp"

#include "cpu/x64/gemm/s8x8s32/jit_avx512_core_gemv_s8x8s32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename c_type>
struct alignas(64) gemm_per_thread_t {
    volatile int32_t result;
    volatile int32_t compute_done;
    int32_t thr_k_stride;
    int32_t nthr_k;
    dim_t ldc_local;
    dim_t ldc_global;
    c_type *c_local;
    c_type *volatile c_global;
    gemm_slice_t slice;
};

template <typename T>
int get_vector_length() {
    int v_bytes;

    if (mayiuse(avx512_core))
        v_bytes = cpu_isa_traits<avx512_core>::vlen;
    else if (mayiuse(avx))
        v_bytes = cpu_isa_traits<avx>::vlen;
    else
        v_bytes = cpu_isa_traits<sse41>::vlen;

    return v_bytes / sizeof(T);
}

template <typename c_type>
static inline void round_to_nearest(c_type *rounded_val, double fp_val) {
    if (fp_val >= 0.) {
        fp_val += 0.5;
        if (fp_val > INT32_MAX) { fp_val = INT32_MAX; }
    } else {
        fp_val -= 0.5;
        if (fp_val < INT32_MIN) { fp_val = INT32_MIN; }
    }
    *rounded_val = (c_type)fp_val;
}

template <typename c_type>
static inline void add_results(const dim_t m, const dim_t n, const float alpha,
        const float beta, const c_type *c_partial_sum, const dim_t ldcp,
        c_type *c_data, const dim_t ldc, const c_type *co,
        offset_type offsetc) {

    constexpr bool is_int8 = data_traits<c_type>::data_type == data_type::s32;

    for (dim_t j = 0; j < n; ++j) {
        for (dim_t i = 0; i < m; ++i) {
            c_type ctemp = c_partial_sum[i + j * ldcp];

            if (alpha == 1.0f) {
                if (beta == 0.0f) {
                    c_data[i + j * ldc] = ctemp;
                } else {
                    if (is_int8) {
                        double c_float
                                = (double)beta * (double)c_data[i + j * ldc];
                        c_float += (double)ctemp;
                        round_to_nearest(&c_data[i + j * ldc], c_float);
                    } else {
                        c_data[i + j * ldc] *= beta;
                        c_data[i + j * ldc] += ctemp;
                    }
                }
            } else if (alpha == -1.0f) {
                if (beta == 0.0f) {
                    c_data[i + j * ldc] = -ctemp;
                } else {
                    if (is_int8) {
                        double c_float
                                = (double)beta * (double)c_data[i + j * ldc];
                        c_float -= (double)ctemp;
                        round_to_nearest(&c_data[i + j * ldc], c_float);
                    } else {
                        c_data[i + j * ldc] *= beta;
                        c_data[i + j * ldc] -= ctemp;
                    }
                }
            } else {
                if (beta == 0.0f) {
                    if (is_int8) {
                        double c_float = alpha * (double)ctemp;
                        round_to_nearest(&c_data[i + j * ldc], c_float);
                    } else {
                        c_data[i + j * ldc] = alpha * ctemp;
                    }

                } else {
                    if (is_int8) {
                        double c_float = alpha * (double)ctemp
                                + beta * (double)c_data[i + j * ldc];
                        round_to_nearest(&c_data[i + j * ldc], c_float);
                    } else {
                        c_data[i + j * ldc] *= beta;
                        c_data[i + j * ldc] += alpha * ctemp;
                    }
                }
            }

            if (offsetc == offset_type::fixed) {
                c_data[i + j * ldc] += co[0];
            } else if (offsetc == offset_type::row) {
                c_data[i + j * ldc] += co[j];
            } else if (offsetc == offset_type::column) {
                c_data[i + j * ldc] += co[i];
            }
        }
    }
}

template <typename a_type, typename b_type, typename c_type>
static inline dim_t get_k_padd(
        int ithr, dim_t k, const gemm_info_t<a_type, b_type, c_type> *arg) {
    if (arg->a_packed) {
        dim_t block_m, block_k;
        arg->a_packed->get_blocking(ithr, block_m, block_k);
        return block_k;
    } else if (arg->b_packed) {
        dim_t block_n, block_k;
        arg->b_packed->get_blocking(ithr, block_k, block_n);
        return block_k;
    } else {
        dim_t k_padd = 0;

        if (k <= arg->bk_traditional) {
            k_padd = utils::rnd_up(k, arg->uk);
            k_padd = nstl::max(dim_t(128), k_padd);
        } else if (k < 2 * arg->bk)
            k_padd = utils::rnd_up((k + 1) / 2, arg->uk);
        else
            k_padd = arg->bk;

        return k_padd;
    }
}

template <typename a_type, typename b_type, typename c_type>
static inline dim_t get_m_padd(
        int ithr, dim_t m, const gemm_info_t<a_type, b_type, c_type> *arg) {
    if (arg->a_packed) {
        dim_t block_m, block_k;
        arg->a_packed->get_blocking(ithr, block_m, block_k);
        return block_m;
    } else
        return utils::rnd_up(
                nstl::min(nstl::max(m, arg->um), arg->bm), arg->um);
}

template <typename a_type, typename b_type, typename c_type>
static inline dim_t get_m_padd_parallel_a(int ithr, dim_t m,
        const gemm_info_t<a_type, b_type, c_type> *arg, int nthrs) {
    auto m_padd = get_m_padd(ithr, m, arg);

    if (!arg->a_packed) {
        constexpr auto multiplier = 10;

        m_padd *= nstl::max(nthrs, multiplier);
        if (m_padd > m) m_padd = utils::rnd_up(m, arg->um);
    }

    return m_padd;
}

template <typename a_type, typename b_type, typename c_type>
static inline dim_t get_n_padd(int ithr, dim_t n, dim_t k,
        const gemm_info_t<a_type, b_type, c_type> *arg) {
    if (arg->b_packed) {
        dim_t block_n, block_k;
        arg->b_packed->get_blocking(ithr, block_k, block_n);
        return block_n;
    } else {
        auto bn = (k < arg->blocking_small_k) ? arg->bn_small_k : arg->bn;
        return utils::rnd_up(nstl::min(nstl::max(n, arg->un), bn), arg->un);
    }
}

static inline void *align(void *ptr, size_t alignment) {
    return (void *)utils::rnd_up((uintptr_t)ptr, alignment);
}

template <typename scale_t, typename mat_t>
void scale_matrix(
        dim_t m, dim_t n, scale_t alpha, mat_t *__restrict p_mat, dim_t ld) {
    if (data_traits<mat_t>::data_type == data_type::f32) {
        for (dim_t j = 0; j < n; j++) {
            for (dim_t i = 0; i < m; i++) {
                p_mat[i + j * ld] = (mat_t)((scale_t)p_mat[i + j * ld] * alpha);
            }
        }
    }
}

template <typename mat_t>
static void sum_matrices(dim_t m, dim_t n, mat_t *__restrict dst, dim_t ld_dst,
        mat_t *__restrict src, dim_t ld_src) {

    for (dim_t j = 0; j < n; j++) {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < m; i++)
            dst[i + j * ld_dst] += src[i + j * ld_src];
    }
}

template <typename c_type>
static void sum_k_blocks(
        int ithr, gemm_per_thread_t<c_type> *thread_arg, bool wait) {

    auto m = thread_arg[ithr].slice.m;
    auto n = thread_arg[ithr].slice.n;
    auto ithr_k = thread_arg[ithr].slice.ithr_k;
    auto nthr_k = thread_arg[ithr].nthr_k;
    auto stride = thread_arg[ithr].thr_k_stride;
    dim_t n0, nn;

    partition_1d(ithr_k, nthr_k, n, n0, nn);

    auto get_thread_arg = [&](int thr_k) -> gemm_per_thread_t<c_type> & {
        return thread_arg[ithr + (thr_k - ithr_k) * stride];
    };

    auto wait_thread = [&](int thr_k) {
        if (wait) {
            auto &tk_arg = get_thread_arg(thr_k);
            while (!tk_arg.compute_done) {}
        }
    };

    auto add_thread_results = [&](int thr_k) {
        auto &tk_arg = get_thread_arg(thr_k);

        sum_matrices(m, nn, tk_arg.c_global + n0 * tk_arg.ldc_global,
                tk_arg.ldc_global, tk_arg.c_local + n0 * tk_arg.ldc_local,
                tk_arg.ldc_local);
    };

    // First accumulate this thread's results while they are in cache.
    if (ithr_k > 0) {
        wait_thread(0);
        add_thread_results(ithr_k);
    }

    // Then accumulate the others.
    for (int thr_k = 1; thr_k < nthr_k; thr_k++) {
        if (thr_k != ithr_k) {
            wait_thread(thr_k);
            add_thread_results(thr_k);
        }
    }
}

template <typename a_type, typename b_type, typename c_type>
static dnnl_status_t pack_no_copy(gemm_info_t<a_type, b_type, c_type> *arg) {

    if (arg->packing == pack_type::pack_a) {
        return gemm_utils::pack_no_copy(arg->a, arg->lda, arg->m, arg->k,
                arg->transa, arg->alpha, arg->pack_dst);
    } else {
        return gemm_utils::pack_no_copy(arg->b, arg->ldb, arg->k, arg->n,
                arg->transb, arg->alpha, arg->pack_dst);
    }
}

template <typename a_type, typename b_type, typename c_type>
static dnnl_status_t gemm_packing_driver(int ithr, dim_t m, dim_t n, dim_t k,
        const a_type *a, const b_type *b,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    if (m <= 0 || n <= 0) return dnnl_success;

    gemm_pack_storage_t *pack_dst = arg->pack_dst;

    if (!pack_dst->is_first_thread_in_slice(ithr)) return dnnl_success;

    dim_t block_r, block_c;
    pack_dst->get_blocking(ithr, block_r, block_c);

    auto do_a = (arg->packing == pack_type::pack_a);
    auto mn = do_a ? m : n;
    auto mn_padd = do_a ? block_r : block_c;
    auto k_padd = do_a ? block_c : block_r;
    dim_t mn_stride, k_stride;

    if (do_a) {
        mn_stride = (arg->transa == no_trans) ? 1 : arg->lda;
        k_stride = (arg->transa == no_trans) ? arg->lda : 1;
    } else {
        mn_stride = (arg->transb == no_trans) ? arg->ldb : 1;
        k_stride = (arg->transb == no_trans) ? 1 : arg->ldb;
    }

    dim_t blk_k = 0;
    for (dim_t Bk = 0; Bk < k; Bk += k_padd, blk_k++) {
        dim_t nk = nstl::min(k - Bk, k_padd);

        for (dim_t Bmn = 0; Bmn < mn; Bmn += mn_padd) {
            dim_t nmn = nstl::min(mn - Bmn, mn_padd);

            if (do_a) {
                auto a_src = a + mn_stride * Bmn + k_stride * Bk;
                auto a_dst = pack_dst->matrix<a_type>(ithr, Bmn, Bk);
                auto a_row_sum = pack_dst->row_sums<c_type>(ithr, Bmn, blk_k);

                arg->copyA(&nk, &nmn, a_src, &arg->lda, &arg->alpha, a_dst,
                        nullptr, nullptr, a_row_sum);
            } else {
                auto b_src = b + mn_stride * Bmn + k_stride * Bk;
                auto b_dst = pack_dst->matrix<b_type>(ithr, Bk, Bmn);
                auto b_col_sum = pack_dst->col_sums<c_type>(ithr, blk_k, Bmn);

                arg->copyB(&nk, &nmn, b_src, &arg->ldb, &arg->alpha, b_dst,
                        nullptr, nullptr, b_col_sum);
            }
        }
    }

    return dnnl_success;
}

template <typename a_type, typename b_type, typename c_type>
void gemm_kernel(dim_t m, dim_t n, const dim_t k, const float alpha,
        const a_type *a, const b_type *b, float beta, c_type *c,
        const dim_t ldc, const c_type *a_row_sum, const c_type *b_col_sum,
        const c_type *co, offset_type offsetc,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

#if DNNL_WITH_SYCL
    std::vector<c_type> col_offset_vec(m);
    std::vector<c_type> row_offset_vec(n);
    c_type *col_offset = col_offset_vec.data();
    c_type *row_offset = row_offset_vec.data();
#else
    // Since m and n are limited by blocking, stack overflow may not happen;
    // it's up to 32kB
#if !defined(_MSC_VER)
    c_type col_offset[m];
    c_type row_offset[n];
#else
    c_type *col_offset = (c_type *)_alloca(sizeof(*col_offset) * m);
    c_type *row_offset = (c_type *)_alloca(sizeof(*row_offset) * n);
#endif
#endif

    bool col_req = false;
    bool row_req = false;

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    constexpr bool is_bf16 = data_traits<a_type>::data_type == data_type::bf16;
    constexpr bool is_f32 = data_traits<a_type>::data_type == data_type::f32;
    bool is_int8_amx = is_int8 && mayiuse(avx512_core_bf16_amx_int8);
    bool is_bf16_amx = is_bf16 && mayiuse(avx512_core_bf16_amx_bf16);
    bool is_amx = is_int8_amx || is_bf16_amx;

    if (is_int8) {
        c_type ao = arg->ao;
        c_type bo = arg->bo;
        c_type co_0 = offsetc == offset_type::none ? 0 : co[0];

        if (bo != 0 || offsetc == offset_type::column) col_req = true;
        if (ao != 0 || offsetc == offset_type::row) row_req = true;

        // It needs one of column or row offsets, but it doesn't need both
        if ((ao != 0 && bo != 0)
                || (offsetc == offset_type::fixed && co_0 != 0)) {
            if (!col_req && !row_req) {
                if (m <= n) {
                    col_req = true;
                } else {
                    row_req = true;
                }
            }
        }

        if (col_req) {
            for (dim_t i = 0; i < m; i++)
                col_offset[i] = 0;

            if (offsetc == offset_type::column) {
                for (dim_t i = 0; i < m; i++)
                    col_offset[i] += co[i];
            }

            if (bo != 0 && a_row_sum) {
                for (dim_t i = 0; i < m; i++)
                    col_offset[i] -= bo * a_row_sum[i];
            }
        }

        if (row_req) {
            for (dim_t i = 0; i < n; i++)
                row_offset[i] = 0;

            if (offsetc == offset_type::row) {
                for (dim_t i = 0; i < n; i++)
                    row_offset[i] += co[i];
            }

            if (ao != 0 && b_col_sum) {
                for (dim_t i = 0; i < n; i++)
                    row_offset[i] -= ao * b_col_sum[i];
            }
        }

        if (offsetc == offset_type::fixed && co_0 != 0) {
            if (col_req) {
                for (dim_t i = 0; i < m; i++)
                    col_offset[i] += co_0;
            } else {
                for (dim_t i = 0; i < n; i++)
                    row_offset[i] += co_0;
            }
        }

        if (ao != 0 && bo != 0) {
            if (col_req) {
                for (dim_t i = 0; i < m; i++)
                    col_offset[i] += (c_type)k * ao * bo;
            } else {
                for (dim_t i = 0; i < n; i++)
                    row_offset[i] += (c_type)k * ao * bo;
            }
        }
    }

    bool isBeta0 = beta == 0.0f;

    dim_t align_m = 0;
    dim_t align_n = 0;
    dim_t align_k = k;
    if (is_amx) {
        align_m = m - utils::rnd_dn(m, arg->um);
        align_n = n - utils::rnd_dn(n, arg->un);
        align_k = utils::rnd_up(k, arg->uk);
    }
    m -= align_m;
    n -= align_n;

    /* Column and row offsets are ignored by non-integer compute kernels.
     * Scaling is done only for bfloat16 kernels.
     */
    if (m > 0) {
        if (n > 0) {
            arg->kernel[isBeta0][col_req][row_req](&m, &n, &align_k, &alpha, a,
                    b, c, ldc, col_offset, row_offset);
        }
        if (align_n > 0) {
            arg->kernel[isBeta0][col_req][row_req](&m, &align_n, &align_k,
                    &alpha, a, b + n * align_k, c + n * ldc, ldc, col_offset,
                    row_offset + n);
        }
    }
    if (align_m > 0) {
        if (n > 0) {
            arg->kernel[isBeta0][col_req][row_req](&align_m, &n, &align_k,
                    &alpha, a + m * align_k, b, c + m, ldc, col_offset + m,
                    row_offset);
        }
        if (align_n > 0) {
            arg->kernel[isBeta0][col_req][row_req](&align_m, &align_n, &align_k,
                    &alpha, a + m * align_k, b + n * align_k, c + m + n * ldc,
                    ldc, col_offset + m, row_offset + n);
        }
    }

    m += align_m;
    n += align_n;
    msan_unpoison_matrix(c, m, n, ldc, sizeof(*c));

    // sgemm kernels don't support bias yet.
    if (is_f32) {
        if (co && offsetc == offset_type::column) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    c[i + j * ldc] += co[i];
                }
            }
        }
    }

    // AMX igemm kernels don't support row & col sums yet.
    if (is_int8_amx) {
        for (dim_t j = 0; j < n; j++) {
            for (dim_t i = 0; i < m; i++) {
                if (row_req) c[i + j * ldc] += row_offset[j];
                if (col_req) c[i + j * ldc] += col_offset[i];
            }
        }
    }
}

template <typename a_type, typename b_type, typename c_type>
static dnnl_status_t gemm_kernel_driver(int ithr, dim_t m, dim_t n, dim_t k,
        const a_type *a, const b_type *b, float beta, c_type *c, dim_t ldc,
        offset_type offsetc, const c_type *co,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    if (arg->packing != pack_type::none)
        return gemm_packing_driver(ithr, m, n, k, a, b, arg);

    if (m <= 0 || n <= 0) return dnnl_success;

    dim_t lda = arg->lda;
    dim_t ldb = arg->ldb;

    float alpha = arg->alpha;

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    constexpr bool is_bf16 = data_traits<a_type>::data_type == data_type::bf16;
    bool is_int8_amx = is_int8 && mayiuse(avx512_core_bf16_amx_int8);
    bool is_bf16_amx = is_bf16 && mayiuse(avx512_core_bf16_amx_bf16);
    bool is_amx = is_int8_amx || is_bf16_amx;

    const std::shared_ptr<const gemm_pack_storage_t> &a_packed = arg->a_packed;
    const std::shared_ptr<const gemm_pack_storage_t> &b_packed = arg->b_packed;

    // Scaling C matrix.
    if (!is_int8 && beta != 1.0f && beta != 0.0f) {
        scale_matrix(m, n, beta, c, ldc);
        beta = 1.0f;
    }

    // Quick exit for C = beta * C
    if (!is_int8 && alpha == 0.0f) {
        if (beta == 0.0f) scale_matrix(m, n, beta, c, ldc);

        return dnnl_success;
    }

    // Get block sizes.
    dim_t k_padd = get_k_padd(ithr, k, arg);
    dim_t m_padd = get_m_padd(ithr, m, arg);
    dim_t n_padd = get_n_padd(ithr, n, k, arg);

    // Padding for temporary buffer for C
    dim_t ldc_buf = gemm_utils::get_ld_padd<c_type>(m_padd);

    dim_t strideAm = (arg->transa == no_trans) ? 1 : lda;
    dim_t strideAn = (arg->transa != no_trans) ? 1 : lda;
    dim_t strideBm = (arg->transb == no_trans) ? 1 : ldb;
    dim_t strideBn = (arg->transb != no_trans) ? 1 : ldb;

    size_t a_buf_nelems = m_padd * k_padd;
    size_t b_buf_nelems = k_padd * n_padd;
    // A and B buffers need more space due to zero-padding.
    if (is_amx) {
        a_buf_nelems = utils::rnd_up(m_padd, arg->um)
                * utils::rnd_up(k_padd, arg->uk);
        b_buf_nelems = utils::rnd_up(k_padd, arg->uk)
                * utils::rnd_up(n_padd, arg->un);
    }
    size_t a_row_sum_nelems = m_padd;
    size_t b_col_sum_nelems = n_padd;

    if (a_packed) a_buf_nelems = a_row_sum_nelems = 0;
    if (b_packed) b_buf_nelems = b_col_sum_nelems = 0;

    size_t mem_size = a_buf_nelems * sizeof(*a) + PAGE_4K
            + b_buf_nelems * sizeof(*b) + PAGE_4K;

    if (is_int8) {
        mem_size += a_row_sum_nelems * sizeof(*c) + PAGE_4K
                + b_col_sum_nelems * sizeof(*c) + PAGE_4K;
    }

    bool need_c_buffer
            = (is_int8 && (alpha != 1.0f || (beta != 1.0f && beta != 0.0f)))
            // AMX bfloat16 kernels don't support alpha scaling yet,
            // so we need to use accumulation buffer even if beta == 0.
            || (is_bf16_amx && alpha != 1.0f);

    if (need_c_buffer) {
        size_t c_buf_nelems = ldc_buf * n_padd;
        mem_size += c_buf_nelems * sizeof(*c) + PAGE_4K;
    }

    char *mem = nullptr;

    if (mem_size > 0) {
        mem = (char *)malloc(mem_size, 128);
        if (!mem) return dnnl_out_of_memory;
    }

    a_type *bufferA = (a_type *)align(mem, PAGE_4K);
    b_type *bufferB = (b_type *)align(bufferA + a_buf_nelems, PAGE_4K);

    c_type *a_row_sum = nullptr;
    c_type *b_col_sum = nullptr;
    if (is_int8) {
        a_row_sum = (c_type *)align(bufferB + b_buf_nelems, PAGE_4K);
        b_col_sum = (c_type *)align(a_row_sum + a_row_sum_nelems, PAGE_4K);
    }

    c_type *bufferC = nullptr;
    if (need_c_buffer) {
        if (is_int8)
            bufferC = (c_type *)align(b_col_sum + b_col_sum_nelems, PAGE_4K);
        else
            bufferC = (c_type *)align(bufferB + b_buf_nelems, PAGE_4K);
    }

    int a_block_copied = 0;
    dim_t sizeM = 0;
    for (dim_t Bm = 0; Bm < m; Bm += sizeM) {
        sizeM = m - Bm;
        if (sizeM > m_padd) sizeM = m_padd;

        dim_t sizeK = 0;
        dim_t blk_k = 0;
        for (dim_t Bk = 0; Bk < k; Bk += sizeK, blk_k++) {
            sizeK = k - Bk;
            if (sizeK > k_padd) sizeK = k_padd;

            // Scale C blocks by beta only for the first time
            auto beta_eff = (Bk == 0) ? beta : 1.0f;

            // Apply C offset when to the last k-block of the partial sum.
            auto offsetc_eff = offset_type::none;
            if (Bk + sizeK == k) offsetc_eff = offsetc;

            dim_t sizeN = 0;
            for (dim_t Bn = 0; Bn < n; Bn += sizeN) {
                sizeN = n - Bn;
                if (sizeN > n_padd) sizeN = n_padd;

                if (b_packed) {
                    bufferB = b_packed->matrix<b_type>(ithr, Bk, Bn);
                    if (is_int8)
                        b_col_sum = b_packed->col_sums<c_type>(ithr, blk_k, Bn);
                } else {
                    const b_type *b_block = b + Bk * strideBm + Bn * strideBn;
                    const float one = 1.0f;

                    /* Column sum argument is ignored for non-integer kernels
                     * and scaling factor is ignored by 8-bit and 16-bit copy
                     * kernels.
                     */
                    arg->copyB(&sizeK, &sizeN, b_block, &ldb, &one, bufferB,
                            nullptr, nullptr, b_col_sum);
                }

                dim_t sizeUM = 0;
                for (dim_t Um = 0; Um < sizeM; Um += sizeUM) {
                    sizeUM = sizeM - Um;
                    if (sizeUM > arg->um) sizeUM = arg->um;

                    /* Use the whole A buffer only if we have multiple B
                     * blocks for k-dimension, otherwise we are wasting cache
                     * to store B and C blocks.
                     */
                    dim_t Um_forA = 0;
                    if (sizeN < n) Um_forA = Um;

                    a_type *bufferA_eff = nullptr;
                    c_type *a_row_sum_eff = nullptr;

                    if (a_packed) {
                        Um_forA = Um;

                        // TODO Can we simplify this!
                        dim_t buf_shift = 0;
                        if (is_amx)
                            buf_shift = Um_forA * utils::rnd_up(sizeK, arg->uk);
                        else
                            buf_shift = Um_forA * sizeK;

                        bufferA_eff = a_packed->matrix<a_type>(ithr, Bm, Bk)
                                + buf_shift;

                        if (is_int8)
                            a_row_sum_eff = a_packed->row_sums<c_type>(
                                                    ithr, Bm, blk_k)
                                    + Um_forA;
                    } else {
                        // TODO Can we simplify this!
                        dim_t buf_shift = 0;
                        if (is_amx)
                            buf_shift = Um_forA * utils::rnd_up(sizeK, arg->uk);
                        else
                            buf_shift = Um_forA * sizeK;

                        bufferA_eff = bufferA + buf_shift;
                        a_row_sum_eff
                                = a_row_sum ? a_row_sum + Um_forA : nullptr;

                        if (!a_block_copied) {
                            const a_type *a_block
                                    = a + (Bm + Um) * strideAm + Bk * strideAn;

                            /* Row sum argument is ignored for non-integer
                             * kernels and scaling factor is ignored by 8-bit
                             * and 16-bit copy kernels.
                             */
                            arg->copyA(&sizeK, &sizeUM, a_block, &lda, &alpha,
                                    bufferA_eff, nullptr, nullptr,
                                    a_row_sum_eff);
                        }
                    }

                    c_type *c_block = c + (Bm + Um) + Bn * ldc;

                    dim_t co_stride = 0;
                    if (offsetc_eff == offset_type::row)
                        co_stride = Bn;
                    else if (offsetc_eff == offset_type::column)
                        co_stride = Bm + Um;

                    if (need_c_buffer) {
                        gemm_kernel(sizeUM, sizeN, sizeK, 1.0f, bufferA_eff,
                                bufferB, 0.0f, bufferC + Um, ldc_buf,
                                a_row_sum_eff, b_col_sum, (c_type *)nullptr,
                                offset_type::none, arg);

                        /* Finish the block adding the necessary alpha, beta
                         * and offsets.
                         */
                        add_results(sizeUM, sizeN, alpha, beta_eff,
                                bufferC + Um, ldc_buf, c_block, ldc,
                                co + co_stride, offsetc_eff);
                    } else {
                        gemm_kernel(sizeUM, sizeN, sizeK, alpha, bufferA_eff,
                                bufferB, beta_eff, c_block, ldc, a_row_sum_eff,
                                b_col_sum, co + co_stride, offsetc_eff, arg);
                    }
                }
                a_block_copied = 1;
            }
            a_block_copied = 0;
        }
    }

    free(mem);

    return dnnl_success;
}

template <typename a_type, typename b_type, typename c_type>
static dnnl_status_t kernel_driver_parallel_acopiedbcopy(int ithr, dim_t m,
        dim_t n, dim_t k, dim_t blk_k, dim_t Bk, const a_type *bufferA,
        const b_type *b, float beta, c_type *c, offset_type offsetc,
        const c_type *co, const c_type *a_row_sum,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    dim_t ldb = arg->ldb;
    dim_t ldc = arg->ldc;

    float alpha = arg->alpha;

    const std::shared_ptr<const gemm_pack_storage_t> &b_packed = arg->b_packed;

    if (m <= 0 || n <= 0) { return dnnl_success; }

    // Padding along N dimension.
    dim_t n_padd = get_n_padd(ithr, n, k, arg);

    // Padding for temporary buffer for C
    dim_t ldc_buf = gemm_utils::get_ld_padd<c_type>(m);

    dim_t strideBn = (arg->transb != 0) ? 1 : ldb;

    size_t b_buf_nelems = k * n_padd;
    size_t b_col_sum_nelems = n_padd;

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    constexpr bool is_bf16 = data_traits<a_type>::data_type == data_type::bf16;
    bool is_int8_amx = is_int8 && mayiuse(avx512_core_bf16_amx_int8);
    bool is_bf16_amx = is_bf16 && mayiuse(avx512_core_bf16_amx_bf16);
    bool is_amx = is_int8_amx || is_bf16_amx;

    // B buffer needs to large due to zero-padding.
    if (is_amx)
        b_buf_nelems
                = utils::rnd_up(k, arg->uk) * utils::rnd_up(n_padd, arg->un);

    if (b_packed) b_buf_nelems = b_col_sum_nelems = 0;

    size_t mem_size = b_buf_nelems * sizeof(*b) + PAGE_4K;

    if (is_int8) { mem_size += b_col_sum_nelems * sizeof(*c) + PAGE_4K; }

    bool need_c_buffer
            = (is_int8 && (alpha != 1.0f || (beta != 1.0f && beta != 0.0f)))
            // AMX bfloat16 kernels don't support alpha scaling yet,
            // so we need to use accumulation buffer even if beta == 0.
            || (is_bf16_amx && alpha != 1.0f);

    if (need_c_buffer) {
        size_t c_buf_nelems = ldc_buf * n_padd;
        mem_size += c_buf_nelems * sizeof(*c) + PAGE_4K;
    }

    char *mem = nullptr;

    if (mem_size > 0) {
        mem = (char *)malloc(mem_size, 128);
        if (!mem) return dnnl_out_of_memory;
    }

    b_type *bufferB = (b_type *)align(mem, PAGE_4K);

    c_type *b_col_sum = nullptr;
    if (is_int8) {
        b_col_sum = (c_type *)align(bufferB + b_buf_nelems, PAGE_4K);
    }

    c_type *bufferC = nullptr;
    if (need_c_buffer) {
        if (is_int8)
            bufferC = (c_type *)align(b_col_sum + b_col_sum_nelems, PAGE_4K);
        else
            bufferC = (c_type *)align(bufferB + b_buf_nelems, PAGE_4K);
    }

    dim_t sizeN = 0;
    for (dim_t Bn = 0; Bn < n; Bn += sizeN) {
        sizeN = n - Bn;
        if (sizeN > n_padd) sizeN = n_padd;

        if (b_packed) {
            bufferB = b_packed->matrix<b_type>(ithr, Bk, Bn);
            if (is_int8)
                b_col_sum = b_packed->col_sums<c_type>(ithr, blk_k, Bn);
        } else {
            const b_type *b_block = b + Bn * strideBn;
            const float one = 1.0f;

            /* Column sum argument is ignored for non-integer kernels and
             * scaling factor is ignored by 8-bit and 16-bit copy kernels.
             */
            arg->copyB(&k, &sizeN, b_block, &ldb, &one, bufferB, nullptr,
                    nullptr, b_col_sum);
        }

        dim_t co_stride = 0;
        if (offsetc == offset_type::fixed) {
            co_stride = 0;
        } else if (offsetc == offset_type::row) {
            co_stride = Bn;
        } else if (offsetc == offset_type::column) {
            co_stride = 0;
        }

        c_type *c_block = c + Bn * ldc;
        if (need_c_buffer) {
            gemm_kernel(m, sizeN, k, 1.0f, bufferA, bufferB, 0.0f, bufferC,
                    ldc_buf, a_row_sum, b_col_sum, (c_type *)nullptr,
                    offset_type::none, arg);

            // Finish the block adding the necessary alpha, beta and offsets.
            add_results(m, sizeN, alpha, beta, bufferC, ldc_buf, c_block, ldc,
                    co + co_stride, offsetc);
        } else {
            gemm_kernel(m, sizeN, k, alpha, bufferA, bufferB, beta, c_block,
                    ldc, a_row_sum, b_col_sum, co + co_stride, offsetc, arg);
        }
    }

    free(mem);

    return dnnl_success;
}

static inline bool nocopy_checker_avx2(const int nthr, const int transa,
        const int transb, const dim_t m, const dim_t n, const dim_t k,
        const dim_t lda, const dim_t ldb, const dim_t ldc) {
    static const dim_t BM_NOCOPY_AVX2 = 64;
    static const dim_t MN_NOCOPY_AVX2 = 128;
    static const dim_t N_TRANSB_PER_THR = 1;
    static const dim_t K_TRANSB_PER_THR = 1;
    static const dim_t N_NOTRANSB_PER_THR = 16;
    static const dim_t K_NOTRANSB_PER_THR = 2;
    static const double FORCE_NOCOPY_THRESH = 0.0038;

    // Crude threshold to nocopy kernels if copy overhead is significant.
    if (1.0 / m + 1.0 / n >= FORCE_NOCOPY_THRESH) { return true; }

    if (m <= 378 && n <= 378 && k >= nthr * 378) return false;

    if (m >= nthr * 378 && k >= nthr * 378) return false;

    if (transb == no_trans) {
        if (m <= MN_NOCOPY_AVX2 && n <= MN_NOCOPY_AVX2) return true;
        if (n <= nthr * N_NOTRANSB_PER_THR) return true;
        if (k <= nthr * K_NOTRANSB_PER_THR) return true;
        if (m <= BM_NOCOPY_AVX2 && n >= nthr * N_NOTRANSB_PER_THR) return true;
    } else {
        if (m <= MN_NOCOPY_AVX2 && n <= MN_NOCOPY_AVX2) return true;
        if (n <= nthr * N_TRANSB_PER_THR) return true;
        if (k <= nthr * K_TRANSB_PER_THR) return true;
    }

    return false;
}

static inline bool nocopy_checker_avx512(int nthr, const int transa,
        const int transb, const dim_t m, const dim_t n, const dim_t k,
        const dim_t lda, const dim_t ldb, const dim_t ldc) {
    // Constants definition
    static const dim_t BAD_LD_MULT = 256;
    static const dim_t VERYBAD_LD_MULT = 1024;
    static const dim_t M_TRANSB_PER_THR = 28;
    static const dim_t N_TRANSB_PER_THR = 28;
    static const dim_t K_TRANSB_PER_THR = 1;
    static const dim_t MN_NOTRANSB_PER_THR = 28;
    static const dim_t K_NOTRANSB_PER_THR = 1;
    static const double FORCE_NOCOPY_THRESH = 0.00196;

    bool is_NT_case = transa == no_trans && transb == do_trans;
    bool is_TN_case = transa == do_trans && transb == no_trans;

    bool is_lda_bad = lda % BAD_LD_MULT == 0;
    bool is_ldb_bad = ldb % BAD_LD_MULT == 0;
    bool is_ldc_bad = ldc % BAD_LD_MULT == 0;
    bool is_ld_bad = is_lda_bad || is_ldb_bad || is_ldc_bad;

    bool is_lda_verybad = lda % VERYBAD_LD_MULT == 0;

    // Copy-based performs better for TN case with small N in sequential case.
    if (nthr == 1 && is_TN_case && m > 100
            && ((m < 1200 && n < 200 && k < 1200)
                    || (is_lda_bad && is_ldb_bad)))
        return false;

    // Crude threshold for nocopy kernels if copy overhead is significant.
    if (1.0 / m + 1.0 / n >= FORCE_NOCOPY_THRESH
            && !(is_lda_verybad && is_NT_case)) {
        return true;
    }

    // Copy strategy usually performs better than nocopy on "bad" leading
    // dimensions.
    if (is_ld_bad) {
        bool use_copy_based = false;

        if (m >= 32 && n > 16) use_copy_based = true;

        // Nocopy outperforms copy-based in certain conditions.
        if (m >= 32 && n == 16
                && (k >= 6400 || transa == do_trans || m == 4096))
            use_copy_based = true;

        if (use_copy_based) return false;
    }

    if (m <= 378 && n <= 378 && k >= nthr * 378) return false;

    if (m >= nthr * 378 && k >= nthr * 378) return false;

    if (transb == no_trans) {
        if (m <= nthr * MN_NOTRANSB_PER_THR) return true;
        if (n <= nthr * MN_NOTRANSB_PER_THR) return true;
        if (k <= nthr * K_NOTRANSB_PER_THR) return true;
    } else {
        if (m <= nthr * M_TRANSB_PER_THR && m >= n) return true;
        if (n <= nthr * N_TRANSB_PER_THR) return true;
        if (k <= nthr * K_TRANSB_PER_THR) return true;
    }
    return false;
}

template <typename a_type, typename b_type, typename c_type>
static inline bool nocopy_checker(
        int nthr, const gemm_info_t<a_type, b_type, c_type> *arg) {

    if (data_traits<a_type>::data_type != data_type::f32) return false;

    if (!mayiuse(avx)) return false;

    if (arg->force_nocopy) return true;

    auto m = arg->m, n = arg->n, k = arg->k;
    auto lda = arg->lda, ldb = arg->ldb, ldc = arg->ldc;
    auto transa = arg->transa, transb = arg->transb;
    auto packing = arg->packing;

    if (packing != pack_type::none) ldc = 64;

    if (arg->a_packed || arg->b_packed)
        return false;
    else if (mayiuse(avx512_core))
        return nocopy_checker_avx512(
                nthr, transa, transb, m, n, k, lda, ldb, ldc);
    else
        return nocopy_checker_avx2(
                nthr, transa, transb, m, n, k, lda, ldb, ldc);
}

template <typename a_type, typename b_type, typename c_type>
static inline void set_thread_opts_nopack(int nthrs, int nthrs_spawn,
        gemm_threading_t &thread_info,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    static constexpr dim_t N2D_MAX = 384;
    static constexpr dim_t M2D_MIN = 384;

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    bool isSgemm = data_traits<a_type>::data_type == data_type::f32;

    dim_t m = arg->m;
    dim_t n = arg->n;
    dim_t k = arg->k;

    thread_info.nthrs_m = 0;
    thread_info.nthrs_n = 0;
    thread_info.nthrs_k = 0;
    thread_info.copy = copy_type::nonshared;
    thread_info.partition = partition_type::row_1d;

    // TODO Check if we can use dynamic scheduling for sgemm.
    // TODO Check if we should use 3D blocking.
    thread_info.nthrs_k = 1;
    thread_info.thread_k = k;

    bool condition_2D_bsrc = false;
    if (isSgemm) {
        // If m is large and n is small then do 1D partitioning for AVX2.
        if (!mayiuse(avx512_core) && n <= N2D_MAX && (m >= nthrs * M2D_MIN))
            condition_2D_bsrc = false;
        else
            condition_2D_bsrc
                    = ((n > nthrs * N2D_MAX) || (n <= nthrs * N2D_MAX / 2))
                    && (m >= 2 * M2D_MIN);
    } else {
        int scale = mayiuse(avx512_core) ? nthrs : 20;
        condition_2D_bsrc = (256 * m > scale * n) && (scale * m < 256 * n);
    }

    // TODO Check if we should use k-partitioning.

    int condition_1D_copya = false;
    if (mayiuse(avx512_core)) {
        const dim_t thresh = isSgemm ? N2D_MAX / 4 : 68;
        if (m >= 1000 && (n >= nthrs * thresh)) {
            condition_2D_bsrc = false;
            condition_1D_copya = true;
        }
    } else {
        if (m >= 1000 && n >= 4000) {
            condition_2D_bsrc = false;
            condition_1D_copya = true;
        }
    }

    // If A or B offset is non-zero, we need to keep 1D_copya to reduce update
    // overhead.
    // TODO: the reasons seems to be in copy_sum_bx routines. At least,
    //       after simple optimization of copy_sum_ax for avx512, similar
    //       restriction on offset B became unnecessary. Revisit.
    if (is_int8 && arg->ao != 0 && (arg->bo != 0 || mayiuse(avx512_core))) {
        condition_2D_bsrc = false;
        condition_1D_copya = true;
    }

    if (condition_2D_bsrc) {
        int nthrs_m = 1;
        int nthrs_n = nthrs;

        if (isSgemm) {
            while ((nthrs_n % 2 == 0)
                    && (n / nthrs > N2D_MAX || n / nthrs_n <= N2D_MAX / 2)
                    && (m / nthrs_m >= 2 * M2D_MIN) && (nthrs_m < 4)) {
                nthrs_m *= 2;
                nthrs_n /= 2;
            }

            thread_info.nthrs_m = nthrs_m;
            thread_info.nthrs_n = nthrs_n;
            thread_info.partition = partition_type::col_major_2d;
        } else {
            if (m == 800 && n == 300) {
                // TODO: Expand this branch to other problem sizes.

                auto &thread_m = thread_info.thread_m;
                auto &thread_n = thread_info.thread_n;

                const dim_t block_m = arg->um * 4;
                constexpr dim_t block_n = 64;
                constexpr dim_t small_m = 16;
                constexpr dim_t small_n = 2;

                std::tie(nthrs_m, nthrs_n)
                        = gemm_utils::calc_nthr_2d(nthrs, m, n, block_m,
                                block_n, small_m, small_n, thread_m, thread_n);

                thread_info.nthrs_m = nthrs_m;
                thread_info.nthrs_n = nthrs_n;
                thread_info.partition = partition_type::mnk_3d;

            } else if ((n <= 64 || n >= 256)) {
                while (((nthrs_n > 1) && (n / nthrs_n < arg->un)
                               && (m / nthrs_m >= 2 * arg->um)
                               && mayiuse(avx512_core))
                        || ((nthrs_n % 2 == 0)
                                && (n / nthrs > N2D_MAX
                                        || n / nthrs_n <= N2D_MAX / 2)
                                && (m / nthrs_m >= 2 * M2D_MIN)
                                && (nthrs_m < 4))) {
                    nthrs_m *= 2;
                    nthrs_n /= 2;
                }

                thread_info.nthrs_m = nthrs_m;
                thread_info.nthrs_n = nthrs_n;
                thread_info.partition = partition_type::col_major_2d;
            } else {
                // Use 3D decomposition from pack api without k-partitioning.
                set_thread_opts_pack(nthrs, thread_info, arg, false);
            }
        }

    } else if (condition_1D_copya && dnnl_thr_syncable()) {
        // Use parallel copy A algorithm
        thread_info.copy = copy_type::shared_a;
        thread_info.partition = partition_type::col_1d;
        thread_info.nthrs_m = 1;
        thread_info.nthrs_n = nthrs_spawn; // Using all spawned threads.
    } else {
        auto veclen = get_vector_length<c_type>();

        if (m > n && (m >= nthrs * veclen || n < nthrs)) {
            if (n <= 20 && is_int8) {
                // Use 3D decomposition forcing m-blocking only.
                set_thread_opts_pack(
                        nthrs, thread_info, arg, false, true, false);
            } else {
                thread_info.partition = partition_type::row_1d;
                thread_info.nthrs_m = nthrs;
                thread_info.nthrs_n = 1;
            }
        } else {
            thread_info.partition = partition_type::col_1d;
            thread_info.nthrs_m = 1;
            thread_info.nthrs_n = nthrs;
        }
    }
}

template <typename a_type, typename b_type, typename c_type>
static inline void set_thread_opts_pack(int nthrs,
        gemm_threading_t &thread_info,
        const gemm_info_t<a_type, b_type, c_type> *arg,
        bool do_k_blocking = true, bool do_m_blocking = true,
        bool do_n_blocking = true) {

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    constexpr bool is_bf16 = data_traits<a_type>::data_type == data_type::bf16;

    bool do_m_blocking_only = do_m_blocking && !do_n_blocking;

    auto m = arg->m, n = arg->n, k = arg->k;

    auto &nthr_m = thread_info.nthrs_m;
    auto &nthr_n = thread_info.nthrs_n;
    auto &nthr_k = thread_info.nthrs_k;
    auto &thread_m = thread_info.thread_m;
    auto &thread_n = thread_info.thread_n;
    auto &thread_k = thread_info.thread_k;
    auto &block_m = thread_info.block_m;
    auto &block_n = thread_info.block_n;
    auto &block_k = thread_info.block_k;

    constexpr auto MBLK = 64;
    constexpr auto NBLK = 64;
    auto KBLK = is_int8 ? 3072 : 256;
    KBLK = do_m_blocking_only && is_int8 ? 384 : KBLK;

    nthr_m = nthr_n = nthr_k = 1;
    thread_info.copy = copy_type::nonshared;
    thread_info.partition = partition_type::mnk_3d;

    auto choose_blocking
            = [](dim_t size_z, dim_t &thread_z, int &nthr_z, dim_t block_z_init,
                      dim_t &block_z, dim_t block_align) {
                  thread_z = utils::div_up(size_z, nthr_z);
                  auto num_blk = utils::div_up(thread_z, block_z_init);
                  block_z = utils::div_up(thread_z, num_blk);
                  block_z = utils::rnd_up(block_z, block_align);
                  thread_z = num_blk * block_z;
                  if (thread_z * nthr_z > size_z)
                      nthr_z = utils::div_up(size_z, thread_z);
              };

    auto choose_m_blocking = [&]() {
        auto align = get_vector_length<c_type>();
        align = do_m_blocking_only ? arg->um : align;
        choose_blocking(m, thread_m, nthr_m, arg->bm, block_m, align);
    };
    auto choose_n_blocking = [&]() {
        choose_blocking(n, thread_n, nthr_n, arg->bn, block_n, arg->un);
    };
    auto choose_k_blocking = [&]() {
        auto align = nstl::max(arg->uk, dim_t(4));
        choose_blocking(k, thread_k, nthr_k, arg->bk, block_k, align);
    };

    // Choose k blocking.
    if ((m / MBLK + n / NBLK) < nthrs && do_k_blocking) {
        for (int nk = 1; nk <= 4 && k >= ((KBLK + 1) * nk); nk++)
            if (nthrs % nk == 0) nthr_k = nk;

        // Sacrifice one thread and try again if parallelism is too small in
        // n-dimension.
        if (nthr_k == 1 && nthrs > 1 && do_m_blocking_only) {
            nthrs--;
            for (int nk = 1; nk <= 4 && k >= ((KBLK + 1) * nk); nk++)
                if (nthrs % nk == 0) nthr_k = nk;
        }

        // Allow up to 2 threads to be sacrificed for large k >> m, n.
        if (nthr_k < 4 && k >= m * 4 && k >= n * 4 && nthrs > 10 && is_bf16) {
            for (int nk = 1; nk <= 4 && k >= ((KBLK + 1) * nk); nk++)
                if (nthrs % nk <= 2) nthr_k = nk;
        }
    }

    choose_k_blocking();

    // Choose m/n blocking.
    auto min_mblk = mayiuse(avx512_core) ? (MBLK / 2) : arg->um;
    min_mblk = do_m_blocking ? min_mblk : m;
    min_mblk = do_m_blocking_only ? arg->um : min_mblk;
    auto min_nblk = do_n_blocking ? NBLK / 2 : n;

    std::tie(nthr_m, nthr_n) = partition_2d_minblk(m, n, MBLK, NBLK, min_mblk,
            min_nblk, arg->um, arg->un, nthrs / nthr_k,
            do_m_blocking && do_n_blocking && do_k_blocking);

    auto nthr_m_init = nthr_m, nthr_n_init = nthr_n;

    choose_m_blocking();
    choose_n_blocking();

    if (is_int8 && do_m_blocking && do_n_blocking) {
        // If we lost a thread in one dimension because we padded the blocking
        // size, try to rebalance the other dimensions.
        if ((nthr_n != nthr_n_init)
                && ((nthr_m + 1) * nthr_n * nthr_k <= nthrs)) {
            nthr_m++;
            choose_m_blocking();
        }

        if ((nthr_m != nthr_m_init)
                && (nthr_m * (nthr_n + 1) * nthr_k <= nthrs)) {
            nthr_n++;
            choose_n_blocking();
        }
    }
}

template <typename a_type, typename b_type, typename c_type>
static inline int set_thread_opts(int nthrs, int nthrs_spawn,
        gemm_threading_t &thread_info,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    thread_info.block_m = thread_info.block_n = thread_info.block_k = -1;
    thread_info.thread_m = thread_info.thread_n = thread_info.thread_k = -1;

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    constexpr bool is_bf16 = data_traits<a_type>::data_type == data_type::bf16;

    if (nocopy_checker(nthrs, arg)) {
        thread_info.copy = copy_type::no_copy;
        thread_info.partition = partition_type::mnk_3d;
        int nthrs_m = 0;
        int nthrs_n = 0;
        int nthrs_k = 0;
        dim_t BM = 0;
        dim_t BN = 0;
        dim_t BK = 0;
        auto m = arg->m, n = arg->n, k = arg->k;

        if (mayiuse(avx512_core)) {
            cpu::gemm_utils::calc_nthr_nocopy_avx512_common(m, n, k, nthrs,
                    &nthrs_m, &nthrs_n, &nthrs_k, &BM, &BN, &BK);
        } else {
            cpu::gemm_utils::calc_nthr_nocopy_avx(m, n, k, nthrs, &nthrs_m,
                    &nthrs_n, &nthrs_k, &BM, &BN, &BK);
        }

        // Block information is being ignored. We will create partitioning
        // later.
        thread_info.nthrs_m = nthrs_m;
        thread_info.nthrs_n = nthrs_n;
        thread_info.nthrs_k = nthrs_k;
    } else {
        if (arg->packing != pack_type::none && (is_int8 || is_bf16))
            set_thread_opts_pack(nthrs, thread_info, arg);
        else
            set_thread_opts_nopack(nthrs, nthrs_spawn, thread_info, arg);
    }

    return thread_info.nthrs_m * thread_info.nthrs_n * thread_info.nthrs_k;
}

template <typename a_type, typename b_type, typename c_type>
static inline std::tuple<const a_type *, const b_type *, c_type *,
        const c_type *>
decompose_matrices(const gemm_slice_t &slice,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    dim_t stride_am = (arg->transa == no_trans) ? 1 : arg->lda;
    dim_t stride_ak = (arg->transa != no_trans) ? 1 : arg->lda;
    dim_t stride_bn = (arg->transb != no_trans) ? 1 : arg->ldb;
    dim_t stride_bk = (arg->transb == no_trans) ? 1 : arg->ldb;

    auto a = arg->a + slice.off_m * stride_am + slice.off_k * stride_ak;
    auto b = arg->b + slice.off_n * stride_bn + slice.off_k * stride_bk;
    auto c = arg->c + slice.off_m + slice.off_n * arg->ldc;

    dim_t co_stride;
    switch (arg->offsetc) {
        case offset_type::row: co_stride = slice.off_n; break;
        case offset_type::column: co_stride = slice.off_m; break;
        default: co_stride = 0; break;
    }
    auto co = arg->co + co_stride;

    return std::make_tuple(a, b, c, co);
}

template <typename a_type, typename b_type, typename c_type>
static dnnl_status_t parallel_a_copy(const int ithr, const int nthrs,
        const dim_t m, const dim_t n, const dim_t k, const a_type *a,
        const b_type *b, float beta, c_type *c, dim_t ldc, offset_type offsetc,
        const c_type *co, const gemm_info_t<a_type, b_type, c_type> *arg,
        char **p_shared_mem) {

    if (arg->packing != pack_type::none)
        return gemm_packing_driver(ithr, m, n, k, a, b, arg);

    const dim_t lda = arg->lda;
    const dim_t ldb = arg->ldb;
    const dim_t strideAm = (arg->transa == no_trans) ? 1 : lda;
    const dim_t strideAn = (arg->transa != no_trans) ? 1 : lda;
    const dim_t strideBm = (arg->transb == no_trans) ? 1 : ldb;

    float alpha = arg->alpha;

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    constexpr bool is_bf16 = data_traits<a_type>::data_type == data_type::bf16;
    bool is_int8_amx = is_int8 && mayiuse(avx512_core_bf16_amx_int8);
    bool is_bf16_amx = is_bf16 && mayiuse(avx512_core_bf16_amx_bf16);
    bool is_amx = is_int8_amx || is_bf16_amx;

    const std::shared_ptr<const gemm_pack_storage_t> &a_packed = arg->a_packed;

    // Scaling C matrix.
    if (!is_int8 && beta != 1.0f && beta != 0.0f) {
        scale_matrix(m, n, beta, c, ldc);
        beta = 1.0f;
    }

    // Padding along M, K dimensions.
    dim_t m_padd = get_m_padd_parallel_a(ithr, m, arg, nthrs);
    dim_t k_padd = get_k_padd(ithr, k, arg);

    size_t a_buf_nelems = m_padd * k_padd;

    // A buffer needs more space due to zero-padding.
    if (is_amx)
        a_buf_nelems = utils::rnd_up(m_padd, arg->um)
                * utils::rnd_up(k_padd, arg->uk);

    // Allocate shared memory for A and its row sum buffers in master thread.
    char *mem = nullptr;
    a_type *bufferA = nullptr;
    c_type *a_row_sum = nullptr;

    if (!a_packed) {
        if (ithr == 0) { // If thread master
            size_t mem_size = (a_buf_nelems * sizeof(*a) + PAGE_4K);

            if (is_int8) {
                size_t a_row_sum_nelems = m_padd;
                mem_size += a_row_sum_nelems * sizeof(*c) + PAGE_4K;
            }

            *p_shared_mem = (char *)malloc(mem_size, 128);
        }

        dnnl_thr_barrier();

        mem = *p_shared_mem;
        bufferA = (a_type *)align(mem, PAGE_4K);

        if (is_int8)
            a_row_sum = (c_type *)align(bufferA + a_buf_nelems, PAGE_4K);

        if (!mem) return dnnl_out_of_memory;
    }

    dnnl_status_t result = dnnl_success; // Return status

    dim_t sizeK = 0;
    dim_t blk_k = 0;
    for (dim_t Bk = 0; Bk < k; Bk += sizeK, blk_k++) {
        sizeK = k - Bk;
        if (sizeK > k_padd) sizeK = k_padd;

        // Scale C blocks by beta only for the first term of partial sum.
        auto beta_eff = (Bk == 0) ? beta : 1.0f;

        // Apply C offset for the last k-block of the partial sum.
        auto offsetc_eff = offset_type::none;
        if (Bk + sizeK == k) offsetc_eff = offsetc;

        dim_t sizeM = 0;
        for (dim_t Bm = 0; Bm < m; Bm += sizeM) {
            sizeM = m - Bm;
            if (sizeM > m_padd) sizeM = m_padd;

            if ((ithr < nthrs) && !a_packed) {
                dim_t band = (sizeM + nthrs - 1) / nthrs;
                band = utils::rnd_up(band, arg->um);

                dim_t offset = band * ithr;

                // If offset is too large don't use that thread for copying.
                if (offset >= sizeM) {
                    offset = 0;
                    band = 0;
                }

                // Handle the tail of the copy.
                if (offset + band > sizeM) { band = sizeM - offset; }

                if (band > 0) {
                    const a_type *a_block
                            = a + (Bm + offset) * strideAm + Bk * strideAn;

                    dim_t buf_shift = 0;
                    if (is_amx)
                        buf_shift = offset * utils::rnd_up(sizeK, arg->uk);
                    else
                        buf_shift = offset * sizeK;

                    /* Row sum argument is ignored for non-integer kernels and
                     * scaling factor is ignored by 8-bit and 16-bit copy
                     * kernels.
                     */
                    c_type *a_row_sum_eff
                            = a_row_sum ? a_row_sum + offset : nullptr;
                    arg->copyA(&sizeK, &band, a_block, &lda, &alpha,
                            bufferA + buf_shift, nullptr, nullptr,
                            a_row_sum_eff);
                }
            }
            if (!a_packed)
                dnnl_thr_barrier(); // Wait for finishing parallel copy.

            const b_type *b_block = b + Bk * strideBm;
            c_type *c_block = c + Bm;

            dim_t co_stride = 0;
            if (offsetc_eff == offset_type::fixed) {
                co_stride = 0;
            } else if (offsetc_eff == offset_type::row) {
                co_stride = 0;
            } else if (offsetc_eff == offset_type::column) {
                co_stride = Bm;
            }

            auto bufferA_eff
                    = a_packed ? a_packed->matrix<a_type>(0, Bm, Bk) : bufferA;
            auto a_row_sum_eff = a_packed
                    ? a_packed->row_sums<c_type>(0, Bm, blk_k)
                    : a_row_sum;

            auto this_result = kernel_driver_parallel_acopiedbcopy(ithr, sizeM,
                    n, sizeK, blk_k, Bk, bufferA_eff, b_block, beta_eff,
                    c_block, offsetc_eff, co + co_stride, a_row_sum_eff, arg);

            if (this_result != dnnl_success) result = this_result;

            if (!a_packed)
                dnnl_thr_barrier(); // Wait for kernel computations to finish.
        }
    }

    // Free memory allocated in master thread
    if (ithr == 0 && !a_packed) free(mem);

    return result;
}

template <typename T>
static inline void adjust_thread_count(dim_t m, dim_t n, dim_t k, int *nthrs) {

    const double omp_overhead_small_core = 3.0e+3;
    const double omp_intercept_big_core = 4.0e+3;
    const double omp_slope_big_core = 5.0e+2;

    auto veclen = get_vector_length<T>();
    const double fp_per_cycle = 2.0 * 2.0 * veclen;

    if (mayiuse(avx2) && !mayiuse(avx512_core))
        if (m > 10 * n && n < *nthrs)
            if (m / *nthrs < veclen * 3)
                *nthrs = nstl::max(m / veclen / 3, dim_t(1));

    double gemm_cycles = m * n * k / fp_per_cycle;
    if (data_traits<T>::data_type == data_type::f32) {
        gemm_cycles *= 2.0;
    } else {
        gemm_cycles *= 8.0;
    }

    int i = *nthrs;

    // Use a different model for omp overheads if nthrs is <= 4
    if (*nthrs <= 4 && omp_overhead_small_core > 0) {
        double omp_cycles = omp_overhead_small_core;
        if (gemm_cycles < omp_cycles) {
            *nthrs = 1;
            return;
        } else {
            while (i > 1) {
                if (omp_cycles * i < gemm_cycles * (i - 1)) break;
                --i;
            }
        }
    } else {
        if (gemm_cycles < (omp_intercept_big_core + 2 * omp_slope_big_core)) {
            *nthrs = 1;
            return;
        }

        // adaptive decrement to march faster·
        while (i > 1) {
            double omp_cycles = omp_intercept_big_core + i * omp_slope_big_core;
            if (omp_cycles * i < gemm_cycles * (i - 1)) break;

            if (i < 10)
                i -= 2;
            else if (i < 30)
                i -= 4;
            else
                i -= 8;
        }
    }

    if (i < 1) i = 1;

    *nthrs = i;
}

template <typename a_type, typename b_type, typename c_type>
static dnnl_status_t call_no_copy_sgemm(
        gemm_info_t<a_type, b_type, c_type> *arg) {

    if (arg->packing == pack_type::none) {
        auto transa_char = (arg->transa != do_trans) ? "N" : "T";
        auto transb_char = (arg->transb != do_trans) ? "N" : "T";

        if (mayiuse(avx512_core))
            return jit_avx512_common_gemm_f32(transa_char, transb_char, &arg->m,
                    &arg->n, &arg->k, &arg->alpha, (float *)arg->a, &arg->lda,
                    (float *)arg->b, &arg->ldb, &arg->beta, (float *)arg->c,
                    &arg->ldc, (float *)arg->co);
        else
            return jit_avx_gemm_f32(transa_char, transb_char, &arg->m, &arg->n,
                    &arg->k, &arg->alpha, (float *)arg->a, &arg->lda,
                    (float *)arg->b, &arg->ldb, &arg->beta, (float *)arg->c,
                    &arg->ldc, (float *)arg->co);
    } else
        return pack_no_copy(arg);
}

template <typename a_type, typename b_type, typename c_type>
static dnnl_status_t gemm_threading_driver(
        gemm_info_t<a_type, b_type, c_type> *arg) {

    auto packing = (arg->packing != pack_type::none);
    auto is_a_packed = (arg->transa == packed);
    auto is_b_packed = (arg->transb == packed);
    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    constexpr bool is_bf16 = data_traits<a_type>::data_type == data_type::bf16;

    if ((arg->m <= 0) || (arg->n <= 0)) return dnnl_success;

    if (!is_a_packed && !is_b_packed && jump_to_gemv_s8x8s32(arg))
        return dnnl_success;

    if (!is_a_packed && !is_b_packed
            && jump_to_gemm_smalln_tn(arg) == dnnl_success)
        return dnnl_success;

    if (!is_a_packed && !is_b_packed && jump_to_gemv(arg) == dnnl_success)
        return dnnl_success;

    if (is_a_packed && arg->bo != 0)
        if (!arg->a_packed->has_row_sums()) return dnnl_invalid_arguments;

    if (is_b_packed && arg->ao != 0)
        if (!arg->b_packed->has_col_sums()) return dnnl_invalid_arguments;

    auto nthr_max = (dnnl_in_parallel()) ? 1 : dnnl_get_max_threads();
    int nthr_goal = nthr_max;

    adjust_thread_count<c_type>(arg->m, arg->n, arg->k, &nthr_goal);

    const gemm_threading_t *force_threading = nullptr;
    gemm_threading_t force_k_decomp;

    // Initialize per-thread data.
    // Note: to support k blocking with non-packed GEMM, threading must be
    //   chosen now and force_threading set.
    if (!packing) {
        // Override choice of thread count if data is pre-packed for a particular
        //  number of threads.
        if (is_a_packed && is_b_packed)
            if (arg->a_packed->threading() != arg->b_packed->threading())
                return dnnl_invalid_arguments;
        if (is_a_packed)
            force_threading = &arg->a_packed->threading();
        else if (is_b_packed)
            force_threading = &arg->b_packed->threading();
        else if (arg->m <= 768 && arg->n <= 768 && arg->k >= 2048 && is_bf16) {
            // Try k-partitioning.
            set_thread_opts_pack(nthr_goal, force_k_decomp, arg);

            // Decide partition type later if no partitions in k-dimension.
            if (force_k_decomp.nthrs_k > 1) force_threading = &force_k_decomp;
        } else if (arg->n <= 128 && arg->k >= 3072 && is_int8) {
            // Use k-partitioning if necessary.
            // Use 3D decomposition from pack api without n-partitioning.
            set_thread_opts_pack(
                    nthr_goal, force_k_decomp, arg, true, true, false);

            // Decide partition type later if no partitions in k-dimension.
            if (force_k_decomp.nthrs_k > 1 && force_k_decomp.nthrs_m > 1)
                force_threading = &force_k_decomp;
        }

        if (force_threading) {
            nthr_goal = force_threading->nthrs();
            arg->update_blocking(*force_threading);
        }
    } else {
        // Prepare packed data layout.
        gemm_pack_storage_t *pack_dst = arg->pack_dst;
        bool do_a = (arg->packing == pack_type::pack_a);

        pack_dst->which() = do_a ? matrix_id::a : matrix_id::b;
        pack_dst->setup(nthr_goal, do_a && is_int8, !do_a && is_int8);

        auto &thread_info = pack_dst->threading();
        force_threading = &thread_info;

        nthr_goal = set_thread_opts(nthr_goal, nthr_max, thread_info, arg);
        arg->update_blocking(thread_info);

        if (thread_info.copy != copy_type::no_copy) {
            for (int ithr = 0; ithr < nthr_goal; ithr++) {
                if (!pack_dst->is_first_thread_in_slice(ithr)) continue;

                auto slice = thread_info.get_thread_slice(
                        ithr, arg->m, arg->n, arg->k);

                auto m = slice.m, n = slice.n, k = slice.k;

                auto m_padd = (thread_info.copy == copy_type::shared_a)
                        ? get_m_padd_parallel_a(
                                ithr, m, arg, thread_info.nthrs())
                        : get_m_padd(ithr, m, arg);
                auto n_padd = get_n_padd(ithr, n, k, arg);
                auto k_padd = get_k_padd(ithr, k, arg);

                do_a ? pack_dst->set_blocking(ithr, m, k, m_padd, k_padd)
                     : pack_dst->set_blocking(ithr, k, n, k_padd, n_padd);
            }
        } else {
            auto ld = do_a ? gemm_utils::get_ld_padd<a_type>(arg->m)
                           : gemm_utils::get_ld_padd<b_type>(arg->k);

            pack_dst->set_nocopy(0, no_trans, ld, do_a ? arg->k : arg->n);
        }

        do_a ? pack_dst->finalize<a_type, c_type>()
             : pack_dst->finalize<b_type, c_type>();

        if (arg->measure_only) return dnnl_success;
    }

    if (nocopy_checker(nthr_goal, arg)) return call_no_copy_sgemm(arg);

    if (nthr_goal == 1)
        return gemm_kernel_driver(0, arg->m, arg->n, arg->k, arg->a, arg->b,
                arg->beta, arg->c, arg->ldc, arg->offsetc, arg->co, arg);

    bool k_blocking = force_threading && (force_threading->nthrs_k > 1);
    bool k_summing = k_blocking && !packing;

    auto *thread_arg = (gemm_per_thread_t<c_type> *)malloc(
            sizeof(gemm_per_thread_t<c_type>) * nthr_max, PAGE_4K);

    if (!thread_arg) return dnnl_out_of_memory;

    dim_t max_mt = 0, max_nt = 0;
    for (int ithr = 0; ithr < nthr_max; ithr++) {
        thread_arg[ithr].result = dnnl_success;
        thread_arg[ithr].compute_done = false;
        thread_arg[ithr].c_local = thread_arg[ithr].c_global = nullptr;
        thread_arg[ithr].ldc_global = arg->ldc;
        thread_arg[ithr].ldc_local = 0;

        if (force_threading) {
            thread_arg[ithr].slice = force_threading->get_thread_slice(
                    ithr, arg->m, arg->n, arg->k);
            thread_arg[ithr].nthr_k = force_threading->nthrs_k;
            thread_arg[ithr].thr_k_stride = force_threading->thr_k_stride();
            max_mt = nstl::max(max_mt, thread_arg[ithr].slice.m);
            max_nt = nstl::max(max_nt, thread_arg[ithr].slice.n);
        } else {
            thread_arg[ithr].slice = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            thread_arg[ithr].nthr_k = 1;
            thread_arg[ithr].thr_k_stride = 0;
        }
    }

    // Create temporary C buffers for k blocking if needed.
    c_type *c_local_storage = nullptr;
    if (k_summing) {
        const dim_t BAD_LD_MULT = 256;
        dim_t ldc_local = max_mt % BAD_LD_MULT
                ? max_mt
                : gemm_utils::get_ld_padd<c_type>(max_mt);
        dim_t c_local_stride = ldc_local * max_nt;
        c_local_storage = (c_type *)malloc(
                sizeof(c_type) * c_local_stride * nthr_goal, PAGE_4K);

        if (!c_local_storage) {
            free(thread_arg);
            return dnnl_out_of_memory;
        }

        for (int ithr = 0; ithr < nthr_goal; ithr++) {
            thread_arg[ithr].c_local = c_local_storage + ithr * c_local_stride;
            thread_arg[ithr].ldc_local = ldc_local;
        }
    }

    char *shared_mem = nullptr;

    // Always use the maximum number of threads to avoid OMP overhead that can
    // occur due to change thread counts.
    int nthr_spawn = dnnl_thr_syncable() ? nthr_max : nthr_goal;

    parallel(nthr_spawn, [&](int ithr, int nthr) {
        int nthr_eff = force_threading ? nthr_goal : nstl::min(nthr_goal, nthr);

        if (nthr_eff == 1) {
            thread_arg[0].result = gemm_kernel_driver(0, arg->m, arg->n, arg->k,
                    arg->a, arg->b, arg->beta, arg->c, arg->ldc, arg->offsetc,
                    arg->co, arg);
        } else {
            gemm_threading_t thread_info;

            if (force_threading)
                thread_info = *force_threading;
            else {
                nthr_eff = set_thread_opts(nthr_eff, nthr, thread_info, arg);
                if (ithr < nthr_eff)
                    thread_arg[ithr].slice = thread_info.get_thread_slice(
                            ithr, arg->m, arg->n, arg->k);
            }

            for (; ithr < nthr_eff; ithr += nthr) {
                // Get submatrices and parameters for this thread's GEMM.
                const a_type *a = nullptr;
                const b_type *b = nullptr;
                c_type *c = nullptr;
                const c_type *co = nullptr;
                std::tie(a, b, c, co)
                        = decompose_matrices(thread_arg[ithr].slice, arg);

                auto m = thread_arg[ithr].slice.m;
                auto n = thread_arg[ithr].slice.n;
                auto k = thread_arg[ithr].slice.k;
                thread_arg[ithr].c_global = c;
                auto c_eff = c;
                auto ldc_eff = arg->ldc;
                auto beta_eff = arg->beta;
                auto offsetc_eff = arg->offsetc;

                // For all but first k block: substitute local C matrix and
                // disable postops.
                if (k_summing && thread_arg[ithr].slice.ithr_k > 0) {
                    c_eff = thread_arg[ithr].c_local;
                    ldc_eff = thread_arg[ithr].ldc_local;
                    beta_eff = 0;
                    offsetc_eff = offset_type::none;
                }

                // Dispatch appropriate GEMM driver.
                switch (thread_info.copy) {
                    case copy_type::shared_a:
                        thread_arg[ithr].result = parallel_a_copy(ithr,
                                nthr_eff, m, n, k, a, b, beta_eff, c_eff,
                                ldc_eff, offsetc_eff, co, arg, &shared_mem);
                        break;

                    default:
                    case copy_type::nonshared:
                        thread_arg[ithr].result = gemm_kernel_driver(ithr, m, n,
                                k, a, b, beta_eff, c_eff, ldc_eff, offsetc_eff,
                                co, arg);
                        break;

                    case copy_type::no_copy:
                        // This route is taken only if we realize we need no-copy
                        //  after launching the parallel section, due to less
                        //  threads being spawned than expected.
                        assert(data_traits<a_type>::data_type
                                == data_type::f32);
                        assert(arg->packing == pack_type::none);

                        if (mayiuse(avx512_core)) {
                            avx512_common_gemm_f32::sgemm_nocopy_driver(
                                    arg->transa == no_trans ? "N" : "T",
                                    arg->transb == no_trans ? "N" : "T", m, n,
                                    k, &arg->alpha, (float *)a, arg->lda,
                                    (float *)b, arg->ldb, &beta_eff,
                                    (float *)c_eff, ldc_eff, nullptr, nullptr);
                        } else {
                            avx_gemm_f32::sgemm_nocopy_driver(
                                    arg->transa == no_trans ? "N" : "T",
                                    arg->transb == no_trans ? "N" : "T", m, n,
                                    k, &arg->alpha, (float *)a, arg->lda,
                                    (float *)b, arg->ldb, &beta_eff,
                                    (float *)c_eff, ldc_eff, nullptr, nullptr);
                        }
                        thread_arg[ithr].result = dnnl_success;
                        break;
                }

                    // Sum thread results along k dimension, parallelized in the n
                    // dimension. To avoid deadlocks, results are summed later if
                    // not all threads are running concurrently. We can only detect
                    // if this is safe when using OpenMP.
#if DNNL_THR_SYNC == 1
                if (k_summing && (nthr >= nthr_eff)) {
                    thread_arg[ithr].compute_done = true;
                    sum_k_blocks(ithr, thread_arg, true);
                }
#endif
            }
        }
    });

    dnnl_status_t result = dnnl_success; // Initialize to success
    for (int ithr = 0; ithr < nthr_max; ithr++) {
        if (thread_arg[ithr].result != dnnl_success) {
            result = static_cast<dnnl_status_t>(thread_arg[ithr].result);
            break;
        }
    }

    // Sum thread results along k dimension if this wasn't done earlier.
    if (k_summing && !thread_arg[0].compute_done) {
        parallel(nthr_goal, [&](int ithr, int nthr) {
            for (; ithr < nthr_goal; ithr += nthr)
                sum_k_blocks(ithr, thread_arg, false);
        });
    }

    if (c_local_storage) dnnl::impl::free(c_local_storage);
    dnnl::impl::free(thread_arg);

    return result;
}

template <typename a_type, typename b_type, typename c_type>
dnnl_status_t gemm_driver(const char *transA, const char *transB,
        const char *offsetC, const dim_t *m, const dim_t *n, const dim_t *k,
        const float *alpha, const a_type *a, const dim_t *lda, const a_type *oa,
        const b_type *b, const dim_t *ldb, const b_type *ob, const float *beta,
        c_type *c, const dim_t *ldc, const c_type *oc, const bool force_nocopy,
        pack_type packing, gemm_pack_storage_t *pack_dst, bool measure_only) {

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_type>::data_type, data_type::s8, data_type::u8);
    MAYBE_UNUSED(is_int8);

    // gemm_driver supports bfloat16 gemm for Intel AVX512 and
    // Intel AVX512 BF16.
    assert(IMPLICATION(data_traits<a_type>::data_type == data_type::bf16,
            mayiuse(avx512_core) && !force_nocopy));

    // gemm_driver supports 8-bit integer Intel AVX512, Intel AVX2, Intel AVX,
    // Intel SSE4.1 and Intel DL Boost.
    assert(IMPLICATION(is_int8, mayiuse(sse41) && !mayiuse(avx512_mic)));

    // gemm_driver supports sgemm for Intel AVX512, Intel AVX2, Intel AVX,
    // and Intel SSE4.1
    assert(IMPLICATION(
            data_traits<a_type>::data_type == data_type::f32, mayiuse(sse41)));

    // 8-bit integer gemm doesn't support nocopy kernels.
    assert(IMPLICATION(is_int8, !force_nocopy));

    // gemm_driver can only dispatch nocopy for avx and above.
    assert(IMPLICATION(force_nocopy, mayiuse(avx)));

    gemm_info_t<a_type, b_type, c_type> args(transA, transB, offsetC, m, n, k,
            alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc, force_nocopy,
            packing, pack_dst, measure_only);

    // Check if copy algorithm kernels were generated on supported ISAs.
    if (!args.hasKernels()) return dnnl_unimplemented;

    return gemm_threading_driver(&args);
}

template // Instantiate gemm_bf16bf16f32
        dnnl_status_t
        gemm_driver<bfloat16_t, bfloat16_t, float>(const char *transA,
                const char *transB, const char *offsetC, const dim_t *m,
                const dim_t *n, const dim_t *k, const float *alpha,
                const bfloat16_t *a, const dim_t *lda, const bfloat16_t *oa,
                const bfloat16_t *b, const dim_t *ldb, const bfloat16_t *ob,
                const float *beta, float *c, const dim_t *ldc, const float *oc,
                const bool force_nocopy, pack_type packing,
                gemm_pack_storage_t *pack_dst, bool measure_only);

template // Instantiate gemm_s8s8s32
        dnnl_status_t
        gemm_driver<int8_t, int8_t, int32_t>(const char *transA,
                const char *transB, const char *offsetC, const dim_t *m,
                const dim_t *n, const dim_t *k, const float *alpha,
                const int8_t *a, const dim_t *lda, const int8_t *oa,
                const int8_t *b, const dim_t *ldb, const int8_t *ob,
                const float *beta, int32_t *c, const dim_t *ldc,
                const int32_t *oc, const bool force_nocopy, pack_type packing,
                gemm_pack_storage_t *pack_dst, bool measure_only);

template // Instantiate gemm_s8u8s32
        dnnl_status_t
        gemm_driver<int8_t, uint8_t, int32_t>(const char *transA,
                const char *transB, const char *offsetC, const dim_t *m,
                const dim_t *n, const dim_t *k, const float *alpha,
                const int8_t *a, const dim_t *lda, const int8_t *oa,
                const uint8_t *b, const dim_t *ldb, const uint8_t *ob,
                const float *beta, int32_t *c, const dim_t *ldc,
                const int32_t *oc, const bool force_nocopy, pack_type packing,
                gemm_pack_storage_t *pack_dst, bool measure_only);

template // Instantiate sgemm
        dnnl_status_t
        gemm_driver<float, float, float>(const char *transA, const char *transB,
                const char *offsetC, const dim_t *m, const dim_t *n,
                const dim_t *k, const float *alpha, const float *a,
                const dim_t *lda, const float *oa, const float *b,
                const dim_t *ldb, const float *ob, const float *beta, float *c,
                const dim_t *ldc, const float *oc, const bool force_nocopy,
                pack_type packing, gemm_pack_storage_t *pack_dst,
                bool measure_only);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
