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

#include "oneapi/dnnl/dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm/gemm_pack.hpp"
#include "cpu/gemm/os_blas.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu/x64/gemm/gemm_driver.hpp"
#include "cpu/x64/gemm/gemm_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

bool pack_sgemm_supported() {
#if USE_MKL_PACKED_GEMM
    return true;
#else
    return mayiuse(sse41);
#endif
}

bool pack_gemm_bf16bf16f32_supported() {
    return mayiuse(avx512_core);
}

#if USE_MKL_PACKED_GEMM
static inline CBLAS_IDENTIFIER cblas_identifier(const char *identifier) {
    return utils::one_of(*identifier, 'a', 'A') ? CblasAMatrix : CblasBMatrix;
}

static inline CBLAS_TRANSPOSE cblas_transpose(const char *trans) {
    return utils::one_of(*trans, 'n', 'N') ? CblasNoTrans : CblasTrans;
}

static inline MKL_INT cblas_storage(const char *trans) {
    switch (*trans) {
        case 'N':
        case 'n': return CblasNoTrans;
        case 'T':
        case 't': return CblasTrans;
        default: return CblasPacked;
    }
}

static inline CBLAS_OFFSET cblas_offset(const char *offset) {
    switch (*offset) {
        case 'R':
        case 'r': return CblasRowOffset;
        case 'C':
        case 'c': return CblasColOffset;
        default: return CblasFixOffset;
    }
}
#endif

#if !USE_MKL_PACKED_GEMM
template <typename a_dt, typename b_dt>
static inline bool use_reference_igemm(void) {
    constexpr bool is_s8u8 = true
            && data_traits<a_dt>::data_type == data_type::s8
            && data_traits<b_dt>::data_type == data_type::u8;
    if (is_s8u8)
        return !mayiuse(sse41) || mayiuse(avx512_mic);
    else
        return !mayiuse(avx512_core);
}

#else
template <typename a_dt, typename b_dt>
static inline bool use_reference_igemm(void) {
    return true;
}
#endif

template <typename T>
static bool is_good_ld(dim_t ld) {
    static constexpr auto align = 64 / sizeof(T);
    static constexpr auto no_align = 2048 / sizeof(T);

    return ((ld % align) == 0) && ((ld % no_align) != 0);
}

static dnnl_status_t check_pack_get_size_input(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb) {

    if (utils::any_null(identifier, transa, transb, M, N, K, lda, ldb))
        return dnnl_invalid_arguments;

    bool is_transa = utils::one_of(*transa, 'T', 't');
    bool is_transb = utils::one_of(*transb, 'T', 't');

    bool ok = true && utils::one_of(*transa, 'T', 't', 'N', 'n')
            && utils::one_of(*transb, 'T', 't', 'N', 'n')
            && utils::one_of(*identifier, 'A', 'a', 'B', 'b') && *M >= 0
            && *N >= 0 && *K >= 0
            && *lda >= nstl::max(dim_t(1), !is_transa ? *M : *K)
            && *ldb >= nstl::max(dim_t(1), !is_transb ? *K : *N);

    if (!ok) return dnnl_invalid_arguments;

    return dnnl_success;
}

static dnnl_status_t check_pack_input(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const float *alpha, const dim_t *lda, const dim_t *ldb,
        const void *src, void *dst) {
    if (utils::any_null(src, dst, alpha)) return dnnl_invalid_arguments;

    return check_pack_get_size_input(
            identifier, transa, transb, M, N, K, lda, ldb);
}

template <typename a_dt, typename b_dt, typename c_dt>
static dnnl_status_t gemm_pack_driver(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const float *alpha, const dim_t *lda, const dim_t *ldb,
        const void *src, gemm_pack_storage_t *pack_dst, bool measure_only) {

    a_dt oa = 0;
    b_dt ob = 0;

    const a_dt *a = nullptr;
    const b_dt *b = nullptr;
    pack_type packing;

    if (utils::one_of(*identifier, 'a', 'A')) {
        a = (const a_dt *)src;
        packing = pack_type::pack_a;
    } else {
        b = (const b_dt *)src;
        packing = pack_type::pack_b;
    }

    return gemm_driver<a_dt, b_dt, c_dt>(transa, transb, "N", M, N, K, alpha, a,
            lda, &oa, b, ldb, &ob, nullptr, nullptr, nullptr, nullptr, false,
            packing, pack_dst, measure_only);
}

dnnl_status_t sgemm_pack_get_size(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, size_t *size, bool *pack) {

    if (!pack_sgemm_supported()) return dnnl_unimplemented;

    dnnl_status_t result;
    *size = 0;
    if (pack) *pack = true;

    result = check_pack_get_size_input(
            identifier, transa, transb, M, N, K, lda, ldb);
    if (result != dnnl_success) return result;

#if USE_MKL_PACKED_GEMM
    *size = cblas_sgemm_pack_get_size(cblas_identifier(identifier), *M, *N, *K);
#else
    bool do_a = utils::one_of(*identifier, 'a', 'A');
    float alpha = 1.0f;
    gemm_pack_storage_shell_t shell {dnnl_get_max_threads()};
    if (!shell.get()) return dnnl_out_of_memory;

    result = gemm_pack_driver<float, float, float>(identifier, transa, transb,
            M, N, K, &alpha, lda, ldb, nullptr, &shell, true);
    if (result != dnnl_success) return result;

    *size = shell.size();
    if (pack) {
        *pack = !(shell.single_nocopy()
                && utils::one_of(do_a ? *transa : *transb, 'n', 'N')
                && is_good_ld<float>(do_a ? *lda : *ldb));
    }
#endif

    return dnnl_success;
}

dnnl_status_t gemm_bf16bf16f32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack) {

    if (!pack_gemm_bf16bf16f32_supported()) return dnnl_unimplemented;

    dnnl_status_t result;
    *size = 0;
    if (pack) *pack = true;

    result = check_pack_get_size_input(
            identifier, transa, transb, M, N, K, lda, ldb);
    if (result != dnnl_success) return result;

    float alpha = 1.0f;
    gemm_pack_storage_shell_t shell {dnnl_get_max_threads()};
    if (!shell.get()) return dnnl_out_of_memory;

    result = gemm_pack_driver<bfloat16_t, bfloat16_t, float>(identifier, transa,
            transb, M, N, K, &alpha, lda, ldb, nullptr, &shell, true);
    if (result != dnnl_success) return result;

    *size = shell.size();

    return dnnl_success;
}

template <typename a_dt, typename b_dt>
dnnl_status_t gemm_x8x8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack) {

    dnnl_status_t result;
    *size = 0;
    if (pack) *pack = true;

    result = check_pack_get_size_input(
            identifier, transa, transb, M, N, K, lda, ldb);
    if (result != dnnl_success) return result;

#if USE_MKL_PACKED_GEMM
    constexpr bool is_s8u8 = true
            && data_traits<a_dt>::data_type == data_type::s8
            && data_traits<b_dt>::data_type == data_type::u8;

    if (is_s8u8) {
        *size = cblas_gemm_s8u8s32_pack_get_size(
                cblas_identifier(identifier), *M, *N, *K);
        return dnnl_success;
    }
#endif

    bool do_a = utils::one_of(*identifier, 'a', 'A');
    float alpha = 1.0f;
    gemm_pack_storage_shell_t shell {dnnl_get_max_threads(), do_a, !do_a};
    if (!shell.get()) return dnnl_out_of_memory;

    if (!use_reference_igemm<a_dt, b_dt>()) {
        result = gemm_pack_driver<a_dt, b_dt, int32_t>(identifier, transa,
                transb, M, N, K, &alpha, lda, ldb, nullptr, &shell, true);
        if (result != dnnl_success) return result;
    } else {
        auto rows = do_a ? *M : *K;
        auto cols = do_a ? *K : *N;
        if (do_a) {
            gemm_utils::prep_gemm_pack<int8_t, int32_t>(
                    do_a, no_trans, rows, cols, &shell);
        } else {
            gemm_utils::prep_gemm_pack<uint8_t, int32_t>(
                    do_a, no_trans, rows, cols, &shell);
        }
    }

    *size = shell.size();
    if (pack) {
        *pack = !(shell.single_nocopy()
                && utils::one_of(do_a ? *transa : *transb, 'n', 'N')
                && is_good_ld<float>(do_a ? *lda : *ldb));
    }

    return dnnl_success;
}

dnnl_status_t gemm_s8u8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack) {

    return gemm_x8x8s32_pack_get_size<int8_t, uint8_t>(
            identifier, transa, transb, M, N, K, lda, ldb, size, pack);
}

dnnl_status_t gemm_s8s8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack) {

    return gemm_x8x8s32_pack_get_size<int8_t, int8_t>(
            identifier, transa, transb, M, N, K, lda, ldb, size, pack);
}

dnnl_status_t sgemm_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const float *src, float *dst) {
    float one = 1.f, *alpha = &one;

    if (!pack_sgemm_supported()) return dnnl_unimplemented;

    auto result = check_pack_input(
            identifier, transa, transb, M, N, K, alpha, lda, ldb, src, dst);
    if (result != dnnl_success) return result;

#if USE_MKL_PACKED_GEMM
    auto cblas_id = cblas_identifier(identifier);
    auto ld = (cblas_id == CblasAMatrix) ? *lda : *ldb;
    auto trans = (cblas_id == CblasAMatrix) ? transa : transb;
    cblas_sgemm_pack(CblasColMajor, cblas_id, cblas_transpose(trans), *M, *N,
            *K, *alpha, src, ld, dst);
    return dnnl_success;
#else
    gemm_pack_storage_t pack_dst {dst};

    return gemm_pack_driver<float, float, float>(identifier, transa, transb, M,
            N, K, alpha, lda, ldb, src, &pack_dst, false);
#endif
}

dnnl_status_t gemm_bf16bf16f32_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const bfloat16_t *src,
        bfloat16_t *dst) {
    float one = 1.f, *alpha = &one;

    if (!pack_gemm_bf16bf16f32_supported()) return dnnl_unimplemented;

    auto result = check_pack_input(
            identifier, transa, transb, M, N, K, alpha, lda, ldb, src, dst);
    if (result != dnnl_success) return result;

    gemm_pack_storage_t pack_dst {dst};

    return gemm_pack_driver<bfloat16_t, bfloat16_t, float>(identifier, transa,
            transb, M, N, K, alpha, lda, ldb, src, &pack_dst, false);
}

template <typename a_dt, typename b_dt>
dnnl_status_t gemm_x8x8s32_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const void *src_void, void *dst) {

    float alpha = 1.0f; // Not used with igemm.
    auto result = check_pack_input(identifier, transa, transb, M, N, K, &alpha,
            lda, ldb, src_void, dst);
    if (result != dnnl_success) return result;

#if USE_MKL_PACKED_GEMM
    constexpr bool is_s8u8 = true
            && data_traits<a_dt>::data_type == data_type::s8
            && data_traits<b_dt>::data_type == data_type::u8;

    if (is_s8u8) {
        auto cblas_id = cblas_identifier(identifier);
        auto ld = (cblas_id == CblasAMatrix) ? *lda : *ldb;
        auto trans = (cblas_id == CblasAMatrix) ? transa : transb;
        cblas_gemm_s8u8s32_pack(CblasColMajor, cblas_id, cblas_transpose(trans),
                *M, *N, *K, src_void, ld, dst);
        return dnnl_success;
    }
#endif
    gemm_pack_storage_t pack_dst {dst};

    if (!use_reference_igemm<a_dt, b_dt>()) {
        return gemm_pack_driver<a_dt, b_dt, int32_t>(identifier, transa, transb,
                M, N, K, &alpha, lda, ldb, src_void, &pack_dst, false);
    } else {
        bool do_a = utils::one_of(*identifier, 'a', 'A');
        bool is_trans = utils::one_of(do_a ? *transa : *transb, 't', 'T');
        auto ld = do_a ? *lda : *ldb;
        auto rows = do_a ? *M : *K;
        auto cols = do_a ? *K : *N;

        if (do_a) {
            gemm_utils::prep_gemm_pack<int8_t, int32_t>(
                    do_a, no_trans, rows, cols, &pack_dst);
            auto src = reinterpret_cast<const int8_t *>(src_void);
            return gemm_utils::pack_no_copy(
                    src, ld, rows, cols, is_trans, alpha, &pack_dst);
        } else {
            gemm_utils::prep_gemm_pack<uint8_t, int32_t>(
                    do_a, no_trans, rows, cols, &pack_dst);
            auto src = reinterpret_cast<const uint8_t *>(src_void);
            return gemm_utils::pack_no_copy(
                    src, ld, rows, cols, is_trans, alpha, &pack_dst);
        }
    }
}

dnnl_status_t gemm_s8u8s32_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const void *src, void *dst) {

    return gemm_x8x8s32_pack<int8_t, uint8_t>(
            identifier, transa, transb, M, N, K, lda, ldb, src, dst);
}

dnnl_status_t gemm_s8s8s32_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const void *src, void *dst) {

    return gemm_x8x8s32_pack<int8_t, int8_t>(
            identifier, transa, transb, M, N, K, lda, ldb, src, dst);
}

dnnl_status_t sgemm_compute(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *A,
        const dim_t *lda, const float *B, const dim_t *ldb, const float *beta,
        float *C, const dim_t *ldc) {

#if USE_MKL_PACKED_GEMM
    if (utils::any_null(transa, transb, M, N, K, A, lda, B, ldb, beta, C, ldc))
        return dnnl_invalid_arguments;
    cblas_sgemm_compute(CblasColMajor, cblas_storage(transa),
            cblas_storage(transb), *M, *N, *K, A, *lda, B, *ldb, *beta, C,
            *ldc);
    return dnnl_success;
#else
    if (!pack_sgemm_supported()) return dnnl_unimplemented;

    float one = 1.0f;

    return extended_sgemm(
            transa, transb, M, N, K, &one, A, lda, B, ldb, beta, C, ldc);
#endif
}

dnnl_status_t gemm_bf16bf16f32_compute(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const bfloat16_t *A,
        const dim_t *lda, const bfloat16_t *B, const dim_t *ldb,
        const float *beta, float *C, const dim_t *ldc) {

    if (!pack_gemm_bf16bf16f32_supported()) return dnnl_unimplemented;

    float one = 1.0f;

    return gemm_bf16bf16f32(
            transa, transb, M, N, K, &one, A, lda, B, ldb, beta, C, ldc);
}

template <typename a_dt, typename b_dt>
dnnl_status_t gemm_x8x8s32_compute(const char *transa, const char *transb,
        const char *offsetc, const dim_t *M, const dim_t *N, const dim_t *K,
        const a_dt *A, const dim_t *lda, const b_dt *B, const dim_t *ldb,
        const float *beta, int32_t *C, const dim_t *ldc, const int32_t *co) {

    const float one = 1.f, *alpha = &one;
    const a_dt zero_a_dt = 0, *ao = &zero_a_dt;
    const b_dt zero_b_dt = 0, *bo = &zero_b_dt;

#if USE_MKL_PACKED_GEMM
    constexpr bool is_s8u8 = true
            && data_traits<a_dt>::data_type == data_type::s8
            && data_traits<b_dt>::data_type == data_type::u8;

    if (is_s8u8) {
        if (utils::any_null(transa, transb, offsetc, M, N, K, alpha, A, lda, ao,
                    B, ldb, bo, beta, C, ldc, co))
            return dnnl_invalid_arguments;
        cblas_gemm_s8u8s32_compute(CblasColMajor, cblas_storage(transa),
                cblas_storage(transb), cblas_offset(offsetc), *M, *N, *K,
                *alpha, A, *lda, *ao, B, *ldb, *bo, *beta, C, *ldc, co);
        return dnnl_success;
    }
#endif
    auto lda_eff = *lda, ldb_eff = *ldb;
    auto transa_eff = *transa, transb_eff = *transb;

    if (!use_reference_igemm<a_dt, b_dt>()) {
        return gemm_s8x8s32(&transa_eff, &transb_eff, offsetc, M, N, K, alpha,
                A, &lda_eff, ao, B, &ldb_eff, bo, beta, C, ldc, co);
    } else {
        dim_t ld, td;

        if (transa_eff == 'p' || transa_eff == 'P') {
            gemm_pack_storage_t a_packed {A};
            int trans;
            if (!a_packed.get_nocopy(trans, ld, td))
                return dnnl_invalid_arguments;
            A = a_packed.matrix<a_dt>();
            lda_eff = ld;
            transa_eff = trans == no_trans ? 'N' : 'T';
        }

        if (transb_eff == 'p' || transb_eff == 'P') {
            gemm_pack_storage_t b_packed {B};
            int trans;
            if (!b_packed.get_nocopy(trans, ld, td))
                return dnnl_invalid_arguments;
            B = b_packed.matrix<b_dt>();
            ldb_eff = ld;
            transb_eff = trans == no_trans ? 'N' : 'T';
        }

        return gemm_s8x8s32(&transa_eff, &transb_eff, offsetc, M, N, K, alpha,
                A, &lda_eff, ao, B, &ldb_eff, bo, beta, C, ldc, co);
    }
}

dnnl_status_t gemm_s8u8s32_compute(const char *transa, const char *transb,
        const char *offsetc, const dim_t *M, const dim_t *N, const dim_t *K,
        const int8_t *A, const dim_t *lda, const uint8_t *B, const dim_t *ldb,
        const float *beta, int32_t *C, const dim_t *ldc, const int32_t *co) {

    return gemm_x8x8s32_compute(
            transa, transb, offsetc, M, N, K, A, lda, B, ldb, beta, C, ldc, co);
}

dnnl_status_t gemm_s8s8s32_compute(const char *transa, const char *transb,
        const char *offsetc, const dim_t *M, const dim_t *N, const dim_t *K,
        const int8_t *A, const dim_t *lda, const int8_t *B, const dim_t *ldb,
        const float *beta, int32_t *C, const dim_t *ldc, const int32_t *co) {

    return gemm_x8x8s32_compute(
            transa, transb, offsetc, M, N, K, A, lda, B, ldb, beta, C, ldc, co);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
