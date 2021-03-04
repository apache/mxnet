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

#include <type_traits>

#include "common/bfloat16.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/gemm_msan_unpoison.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/gemm_info.hpp"
#include "cpu/x64/gemm/gemm_utils.hpp"

#include "cpu/x64/gemm/s8x8s32/common_u8.hpp"
#include "cpu/x64/gemm/s8x8s32/jit_avx512_core_gemv_s8x8s32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

template <typename b_type>
void gemv_kernel_driver(gemm_info_t<int8_t, b_type, int32_t> *arg) {
    if (std::is_same<b_type, int8_t>::value) {
        arg->gemv_s8s8s32_kernel(arg->m, arg->n, 1.0f, (const int8_t *)arg->a,
                arg->lda, (const int8_t *)arg->b, arg->beta, arg->c);
    } else if (arg->swap) {
        arg->gemv_u8s8s32_kernel(arg->m, arg->n, 1.0f, (const uint8_t *)arg->a,
                arg->lda, (const int8_t *)arg->b, arg->beta, arg->c);
    } else {
        arg->gemv_s8u8s32_kernel(arg->m, arg->n, 1.0f, (const int8_t *)arg->a,
                arg->lda, (const uint8_t *)arg->b, arg->beta, arg->c);
    }

    if (arg->beta == 0)
        msan_unpoison_matrix(arg->c, arg->m, 1, arg->m, sizeof(int32_t));
}

template <typename b_type>
int gemv_threading_driver(gemm_info_t<int8_t, b_type, int32_t> *arg) {
    dim_t nthr_m, nthr_n = 1;
    dim_t MB, NB, UM = 16, UN = 64;
    dim_t BLOCKM = 192, BLOCKN = 3072;
    dim_t i;

    dim_t nthr = (dnnl_in_parallel()) ? 1 : dnnl_get_max_threads();

    b_type *new_x = nullptr;
    int32_t *tmp_y = nullptr, *new_y = nullptr;

    dim_t m = arg->m, n = arg->n;

    gemm_info_t<int8_t, b_type, int32_t> arg_seq = *arg;
    float zero = 0.0f;

    nthr_m = nstl::min(nstl::max(m / BLOCKM, (dim_t)1), nthr);
    MB = m / nthr_m;
    MB = (((MB / UM) * UM) == MB) ? MB : (MB / UM) * UM + UM;
    nthr_m = (((m / MB) * MB) == m) ? m / MB : m / MB + 1;
    nthr_m = nstl::min(nstl::max(nthr_m, (dim_t)1), nthr);

    while ((nthr_m * (nthr_n + 1) <= nthr) && ((n / (nthr_n + 1)) >= BLOCKN)) {
        nthr_n++;
    }

    NB = n / nthr_n;
    NB = (((NB / UN) * UN) == NB) ? NB : (NB / UN) * UN + UN;
    nthr_n = (((n / NB) * NB) == n) ? n / NB : n / NB + 1;
    nthr_n = nstl::min(nstl::max(nthr_n, (dim_t)1), nthr / nthr_m);

    nthr = nthr_m * nthr_n;

    if (arg->ldb != 1) {
        new_x = (decltype(new_x))malloc(n, 64);
        if (new_x == nullptr) return 0;
        for (i = 0; i < n; i++) {
            new_x[i] = (arg->b)[i * arg->ldb];
        }
        arg_seq.b = new_x;
        arg_seq.ldb = 1;
    } else
        new_x = (b_type *)arg->b;

    if (arg->ldc != 1) {
        new_y = (int32_t *)malloc(
                nthr_m * PADD_BYTESIZE_ONPAGE(MB, sizeof(int32_t)), 64);
        if (new_y == nullptr) {
            if (arg->ldb != 1) { free(new_x); }
            return 0;
        }
        arg_seq.c = new_y;
        arg_seq.ldc = 1;
    }

    // GEMV computation
    if (nthr == 1) {
        if (arg->ldc != 1) {
            if (arg->beta != 0.0f) {
                for (i = 0; i < m; i++) {
                    new_y[i] = arg->c[i * arg->ldc];
                }
            }
        }

        gemv_kernel_driver(&arg_seq);

        if (arg->ldc != 1) {
            for (i = 0; i < m; i++) {
                arg->c[i * arg->ldc] = new_y[i];
            }
        }

        if (arg->ldb != 1) { free(new_x); }
        if (arg->ldc != 1) { free(new_y); }

        return 1;
    }

    if (nthr_n > 1) {
        tmp_y = (int32_t *)malloc(
                (nthr_n - 1) * PADD_BYTESIZE_ONPAGE(m, sizeof(int32_t)),
                PAGE_4K);
        if (tmp_y == nullptr) {
            if (arg->ldb != 1) { free(new_x); }
            return 0;
        }
    }

    parallel_nd((int)nthr, [&](const dim_t ithr) {
        dim_t m_from, m_to, myM;
        dim_t n_from, n_to, myN;

        dim_t n_id, m_id;
        dim_t loc_incy = 1;
        int32_t *loc_y;

        gemm_info_t<int8_t, b_type, int32_t> arg_loc = arg_seq;
        dim_t j;

        m_id = ithr / nthr_n;
        n_id = ithr % nthr_n;

        m_from = MB * m_id;
        m_to = MB * (m_id + 1);
        if ((m_to > m) || (m_id == nthr_m - 1)) m_to = m;

        myM = m_to - m_from;

        n_from = NB * n_id;
        n_to = NB * (n_id + 1);
        if ((n_to > n) || (n_id == nthr_n - 1)) n_to = n;

        myN = n_to - n_from;

        if (n_id != 0) {
            arg_loc.beta = zero;
            loc_y = tmp_y + (NEXT_THR_STRIDE(m, sizeof(int32_t))) * (n_id - 1)
                    + m_from;
        } else {
            if (arg->ldc == 1) {
                loc_y = arg_seq.c + m_from;
            } else {
                // need to copy the block of c in new_y
                loc_y = new_y + m_id * NEXT_THR_STRIDE(MB, sizeof(int32_t));
                if (arg->beta != 0.0f) {
                    for (j = 0; j < myM; j++) {
                        loc_y[j] = arg->c[(m_from + j) * arg->ldc];
                    }
                }
            }
        }

        arg_loc.m = myM;
        arg_loc.n = myN;
        arg_loc.a = arg_seq.a + m_from * arg_seq.lda + n_from;
        arg_loc.b = arg_seq.b + n_from;
        arg_loc.c = loc_y;
        arg_loc.ldc = loc_incy;

        gemv_kernel_driver(&arg_loc);

        if ((n_id == 0) && (arg->ldc != 1)) {
            for (j = 0; j < myM; j++) {
                arg->c[(m_from + j) * arg->ldc] = loc_y[j];
            }
        }
    });

    if (nthr_n > 1) {
        parallel_nd((int)nthr_m, [&](const dim_t ithr) {
            dim_t j, j_from, j_to, ii;
            int32_t acc;

            j_from = MB * ithr;
            j_to = MB * (ithr + 1);
            if ((j_to > m) || (ithr == nthr - 1)) j_to = m;

            for (j = j_from; j < j_to; j++) {
                acc = 0;
                for (ii = 0; ii < nthr_n - 1; ii++) {
                    acc += tmp_y[ii * NEXT_THR_STRIDE(m, sizeof(int32_t)) + j];
                }
                (arg->c)[j * arg->ldc] += acc;
            }
        });
        free(tmp_y);
    }

    if (arg->ldb != 1) { free(new_x); }
    if (arg->ldc != 1) { free(new_y); }

    return 1;
}

template <typename b_type>
typename std::enable_if<std::is_same<b_type, uint8_t>::value
                || std::is_same<b_type, int8_t>::value,
        int>::type
jump_to_gemv_s8x8s32_impl(gemm_info_t<int8_t, b_type, int32_t> *arg) {
    gemm_info_t<int8_t, b_type, int32_t> arg_gemv = *arg;

    gemm_pack_storage_t *pack_dst = arg->pack_dst;
    auto do_a = (arg->packing == pack_type::pack_a);
    auto packing = (arg->packing != pack_type::none);
    bool supported = mayiuse(avx512_core);
    bool bo_ok
            = IMPLICATION((std::is_same<b_type, int8_t>::value), arg->bo == 128)
            && IMPLICATION(
                    (std::is_same<b_type, uint8_t>::value), arg->bo == 0);

    bool applicable = (arg->offsetc == offset_type::fixed || packing)
            && // Fix offset
            (arg->ao == 0) && bo_ok && ((arg->co && arg->co[0] == 0) || packing)
            && (arg->alpha == 1.0f) && (arg->beta == 1.0f || arg->beta == 0.0f);

    if (!applicable || !supported) return 0;

    if (arg->n == 1 && (arg->transa == do_trans || packing)) {
        if (!packing) {
            arg_gemv.n = arg->k;
            arg_gemv.ldc = 1;
            arg_gemv.swap = 0;
            if (arg->transb == no_trans) { arg_gemv.ldb = 1; }
            // B transpose arg_gemv.ldb = arg->ldb
            return gemv_threading_driver(&arg_gemv);
        } else {
            if (do_a) {
                gemm_utils::prep_gemm_pack<int8_t, int32_t>(
                        do_a, do_trans, arg->m, arg->k, pack_dst);
            } else {
                gemm_utils::prep_gemm_pack<b_type, int32_t>(
                        do_a, no_trans, arg->k, arg->n, pack_dst);
            }

            if (arg->measure_only) return 1;

            if (do_a) {
                gemm_utils::pack_no_copy(arg->a, arg->lda, arg->m, arg->k,
                        arg->transa, arg->alpha, arg->pack_dst);
            } else {
                gemm_utils::pack_no_copy(arg->b, arg->ldb, arg->k, arg->n,
                        arg->transb, arg->alpha, arg->pack_dst);
            }
            return 1;
        }
    }

    if (arg->m == 1 && (arg->transb == no_trans || packing)) {
        if (!packing) {
            arg_gemv.transa = do_trans;
            arg_gemv.m = arg->n;
            arg_gemv.n = arg->k;
            arg_gemv.a = (decltype(arg_gemv.a))arg->b;
            arg_gemv.lda = arg->ldb;
            arg_gemv.b = (decltype(arg_gemv.b))arg->a;
            arg_gemv.swap = 1;
            if (arg->transa == no_trans) {
                arg_gemv.ldb = arg->lda;
            } else { // A transpose
                arg_gemv.ldb = 1;
            }
            return gemv_threading_driver(&arg_gemv);
        } else {
            if (do_a) {
                gemm_utils::prep_gemm_pack<int8_t, int32_t>(
                        do_a, do_trans, arg->m, arg->k, pack_dst);
            } else {
                gemm_utils::prep_gemm_pack<b_type, int32_t>(
                        do_a, no_trans, arg->k, arg->n, pack_dst);
            }

            if (arg->measure_only) return 1;

            if (do_a) {
                gemm_utils::pack_no_copy(arg->a, arg->lda, arg->m, arg->k,
                        arg->transa, arg->alpha, arg->pack_dst);
            } else {
                gemm_utils::pack_no_copy(arg->b, arg->ldb, arg->k, arg->n,
                        arg->transb, arg->alpha, arg->pack_dst);
            }
            return 1;
        }
    }

    return 0;
}

} // namespace

template <>
int jump_to_gemv_s8x8s32(gemm_info_t<float, float, float> *arg) {
    return 0;
}

template <>
int jump_to_gemv_s8x8s32(gemm_info_t<bfloat16_t, bfloat16_t, float> *arg) {
    return 0;
}

template <>
int jump_to_gemv_s8x8s32(gemm_info_t<int8_t, int8_t, int32_t> *arg) {
    return jump_to_gemv_s8x8s32_impl(arg);
}

template <>
int jump_to_gemv_s8x8s32(gemm_info_t<int8_t, uint8_t, int32_t> *arg) {
    return jump_to_gemv_s8x8s32_impl(arg);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
