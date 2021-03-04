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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_batch_normalization_utils.hpp"
#include "cpu/platform.hpp"

#include "cpu/ncsp_batch_normalization.hpp"

// clang 6 and 7 generate incorrect code with OMP_SIMD in some particular cases
#if (defined __clang_major__) && (__clang_major__ >= 6)
#define SAFE_TO_USE_OMP_SIMD 0
#else
#define SAFE_TO_USE_OMP_SIMD 1
#endif

namespace dnnl {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;
using namespace data_type;

template <data_type_t d_type>
void ncsp_batch_normalization_fwd_t<d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    const bool calculate_stats = !pd()->stats_is_src();
    const bool save_stats = pd()->is_training();
    const bool is_training = pd()->is_training();
    const bool fuse_norm_relu = pd()->fuse_norm_relu();

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto scaleshift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE_SHIFT);

    auto scratchpad = ctx.get_scratchpad_grantor();
    auto *ws_reduce = scratchpad.template get<acc_data_t>(key_bnorm_reduction);

    acc_data_t *mean, *variance;
    if (!calculate_stats) {
        mean = const_cast<acc_data_t *>(
                CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN));
        variance = const_cast<acc_data_t *>(
                CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE));
    } else {
        if (save_stats) {
            mean = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_MEAN);
            variance = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_VARIANCE);
        } else {
            mean = scratchpad.template get<acc_data_t>(key_bnorm_tmp_mean);
            variance = scratchpad.template get<acc_data_t>(key_bnorm_tmp_var);
        }
    }

    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, DNNL_ARG_WORKSPACE);
    acc_data_t *bf16_src_cvt_wsp
            = scratchpad.template get<acc_data_t>(key_bnorm_bf16cvt);

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    const bool with_relu = pd()->with_relu_post_op();
    auto maybe_post_op
            = [&](acc_data_t res) { return (with_relu && res < 0) ? 0 : res; };
    const bool has_spatial = utils::one_of(pd()->ndims(), 4, 5);
    const dim_t SP = (has_spatial) ? pd()->H() * pd()->W() * pd()->D() : 1;
    const dim_t simd_w = 16;
    const dim_t SP_cl_align = utils::rnd_up(SP, simd_w);
    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();

    int nthr = dnnl_get_max_threads();
    size_t l3_size_ = platform::get_per_core_cache_size(3) * nthr / 2;
    size_t data_size = N * C * SP * sizeof(data_t);
    bool do_blocking = (data_size >= l3_size_ / 2 && l3_size_ > 0);

    parallel(0, [&](const int ithr, const int nthr) {
        int C_ithr = 0, C_nthr = 0;
        int N_ithr = 0, N_nthr = 0;
        int S_ithr = 0, S_nthr = 0;

        dim_t C_blk_gl_s = 0, C_blk_gl_e = 0, C_blk_s = 0, C_blk_e = 0;
        dim_t N_s = 0, N_e = 0;
        dim_t S_s = 0, S_e = 0;

        dim_t C_blks_per_iter = 1;
        int64_t iters = 1;

        if (do_blocking) {
            size_t working_set_size = N * SP * sizeof(data_t);
            bnorm_utils::cache_balance(
                    working_set_size, C, N, nthr, C_blks_per_iter, iters);
        } else
            C_blks_per_iter = C;
        int64_t last_iter_blks = C - (iters - 1) * C_blks_per_iter;
        bool spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking,
                true, false, ithr, nthr, N, C_blks_per_iter, SP, C_ithr, C_nthr,
                C_blk_s, C_blk_e, N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s,
                S_e);
        balance211(C_blks_per_iter, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
        int SP_N_ithr = N_ithr * S_nthr + S_ithr;
        int SP_N_nthr = N_nthr * S_nthr;
        for (int64_t it = 0; it < iters; ++it) {
            size_t C_off = it * C_blks_per_iter;
            if (it == iters - 1 && iters > 1) {
                // On the last iteration the access pattern to ws_reduce
                // might change (due to re-balance on C). So sync the
                // threads if they are not synced by the algorithm.
                if (SP_N_nthr == 1 && dnnl_thr_syncable()) dnnl_thr_barrier();

                S_s = S_e = C_blk_s = C_blk_e = N_s = N_e = 0;
                spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking,
                        spatial_thr_allowed, false, ithr, nthr, N,
                        last_iter_blks, SP, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s, S_e);
                C_blks_per_iter = last_iter_blks;
                balance211(last_iter_blks, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
                SP_N_ithr = N_ithr * S_nthr + S_ithr;
                SP_N_nthr = N_nthr * S_nthr;
            }
            const auto S_chunk = nstl::max(dim_t(0), S_e - S_s);
            // On the last iteration the access pattern to ws_reduce
            // might change (due to re-balance on C). Since sync is not always
            // possible (in case of TBB) use different parts of ws for each
            // iteration if threads are not synced by the algorithm.
            size_t ws_iter_off = (dnnl_thr_syncable() ? 0 : 1) * C_off;

            if (calculate_stats) {
                acc_data_t *mean_blk = mean + C_off;
                acc_data_t *variance_blk = variance + C_off;
                for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                    size_t off = (c + C_off) * SP;
                    acc_data_t sum = 0;
                    for (dim_t n = N_s; n < N_e; ++n) {
                        const acc_data_t *scr_fp32;
                        size_t soff = off + n * C * SP;
                        if (d_type == bf16) {
                            // convert src from bf16 to f32
                            acc_data_t *tmp_src
                                    = bf16_src_cvt_wsp + ithr * SP_cl_align;
                            /*TODO: remove this conversion if performance
                            doesn't degrade, since bfloat16_t supports +=
                            operator with implicit conversions from bf16 to
                            float */
                            cvt_bfloat16_to_float(tmp_src + S_s,
                                    (bfloat16_t *)src + soff + S_s, S_chunk);
                            scr_fp32 = tmp_src;
                        } else {
                            scr_fp32 = reinterpret_cast<const acc_data_t *>(
                                    src + soff);
                        }
                        PRAGMA_OMP_SIMD(reduction(+ : sum))
                        for (dim_t sp = S_s; sp < S_e; ++sp) {
                            sum += scr_fp32[sp];
                        }
                    }
                    ws_reduce[ws_iter_off + SP_N_ithr * C_blks_per_iter + c]
                            = sum;
                }

                if (dnnl_thr_syncable()) dnnl_thr_barrier();

                for (dim_t c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                    mean_blk[c] = 0.;
                    for (dim_t n = 0; n < SP_N_nthr; n++)
                        mean_blk[c] += ws_reduce[ws_iter_off
                                + n * C_blks_per_iter + c];
                    mean_blk[c] /= (N * SP);
                }

                if (dnnl_thr_syncable()) dnnl_thr_barrier();

                for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                    size_t off = c + C_off;
                    acc_data_t sum = 0.;
                    for (dim_t n = N_s; n < N_e; ++n) {
                        const acc_data_t *_src;
                        size_t soff = off * SP + n * C * SP;
                        if (d_type == bf16) {
                            // convert src from bf16 to f32
                            acc_data_t *tmp_src
                                    = bf16_src_cvt_wsp + ithr * SP_cl_align;
                            /*TODO: remove this conversion if performance
                            doesn't degrade, since bfloat16_t supports +=
                            operator with implicit conversions from bf16 to
                            float */
                            cvt_bfloat16_to_float(tmp_src + S_s,
                                    (bfloat16_t *)src + soff + S_s, S_chunk);
                            _src = tmp_src;
                        } else {
                            _src = reinterpret_cast<const acc_data_t *>(
                                    src + soff);
                        }
                        PRAGMA_OMP_SIMD(reduction(+ : sum))
                        for (dim_t sp = S_s; sp < S_e; ++sp) {
                            acc_data_t m = _src[sp] - mean[off];
                            sum += m * m;
                        }
                    }
                    ws_reduce[ws_iter_off + SP_N_ithr * C_blks_per_iter + c]
                            = sum;
                }

                if (dnnl_thr_syncable()) dnnl_thr_barrier();

                for (dim_t c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                    variance_blk[c] = 0.;
                    for (dim_t n = 0; n < SP_N_nthr; n++)
                        variance_blk[c] += ws_reduce[ws_iter_off
                                + n * C_blks_per_iter + c];
                    variance_blk[c] /= (N * SP);
                }

                if (dnnl_thr_syncable()) dnnl_thr_barrier();
            }

            for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                size_t off = c + C_off;
                acc_data_t sqrt_variance
                        = static_cast<acc_data_t>(sqrtf(variance[off] + eps));
                acc_data_t sm = (use_scaleshift ? (acc_data_t)scaleshift[off]
                                                : (acc_data_t)1.0f)
                        / sqrt_variance;
                acc_data_t sv = use_scaleshift ? (acc_data_t)scaleshift[C + off]
                                               : (acc_data_t)0;
                for (dim_t n = N_s; n < N_e; ++n) {
                    acc_data_t *_dst;
                    const acc_data_t *_src;
                    size_t s_off = off * SP + n * C * SP;
                    if (d_type == bf16) {
                        // store dst to f32 buffer
                        _dst = bf16_src_cvt_wsp + ithr * SP_cl_align;
                        // convert src from bf16 to f32
                        acc_data_t *tmp_src = bf16_src_cvt_wsp
                                + (nthr + ithr) * SP_cl_align;
                        /*TODO: remove this conversion if performance
                        doesn't degrade, since bfloat16_t supports +=
                        operator with implicit conversions from bf16 to
                        float */
                        cvt_bfloat16_to_float(tmp_src + S_s,
                                (bfloat16_t *)src + s_off + S_s, S_chunk);
                        _src = tmp_src;
                    } else {
                        _dst = reinterpret_cast<acc_data_t *>(dst + s_off);
                        _src = reinterpret_cast<const acc_data_t *>(
                                src + s_off);
                    }
#if SAFE_TO_USE_OMP_SIMD
                    PRAGMA_OMP_SIMD()
#endif
                    for (dim_t sp = S_s; sp < S_e; ++sp) {
                        size_t d_off = s_off + sp;
                        acc_data_t bn_res = sm * (_src[sp] - mean[off]) + sv;
                        if (fuse_norm_relu) {
                            if (bn_res <= 0) {
                                bn_res = 0;
                                if (is_training) ws[d_off] = 0;
                            } else {
                                if (is_training) ws[d_off] = 1;
                            }
                        }
                        _dst[sp] = maybe_post_op(bn_res);
                    }
                    if (d_type == bf16) {
                        // convert dst from f32 to bf16
                        cvt_float_to_bfloat16((bfloat16_t *)dst + s_off + S_s,
                                _dst + S_s, S_chunk);
                    }
                }
            }
        }
    });
}

template struct ncsp_batch_normalization_fwd_t<f32>;
template struct ncsp_batch_normalization_fwd_t<bf16>;

template <data_type_t d_type>
void ncsp_batch_normalization_bwd_t<d_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN);
    auto variance = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto scaleshift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE_SHIFT);
    auto ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);
    auto diff_scaleshift = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SCALE_SHIFT);

    auto scratchpad = ctx.get_scratchpad_grantor();
    auto *ws_reduce = scratchpad.template get<acc_data_t>(key_bnorm_reduction);
    acc_data_t *tmp_data_
            = scratchpad.template get<acc_data_t>(key_bnorm_bf16cvt);

    if (diff_scaleshift == nullptr)
        diff_scaleshift
                = scratchpad.template get<acc_data_t>(key_bnorm_tmp_diff_ss);

    const bool has_spatial = utils::one_of(pd()->ndims(), 4, 5);
    const dim_t SP = (has_spatial) ? pd()->H() * pd()->W() * pd()->D() : 1;
    const dim_t simd_w = 16; //??
    const dim_t SP_cl_align = utils::rnd_up(SP, simd_w);
    const dim_t C = pd()->C(), N = pd()->MB();
    const bool use_scaleshift = pd()->use_scaleshift();
    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool calculate_diff_stats = !pd()->use_global_stats();
    const bool fuse_norm_relu = pd()->fuse_norm_relu();

    int nthr = dnnl_get_max_threads();
    size_t l3_size_ = platform::get_per_core_cache_size(3) * nthr / 2;
    size_t data_size = N * C * SP * sizeof(data_t);
    bool do_blocking = (data_size >= l3_size_ / 2 && l3_size_ > 0);

    parallel(0, [&](const int ithr, const int nthr) {
        int C_ithr = 0, C_nthr = 0;
        int N_ithr = 0, N_nthr = 0;
        int S_ithr = 0, S_nthr = 0;

        dim_t C_blk_gl_s = 0, C_blk_gl_e = 0, C_blk_s = 0, C_blk_e = 0;
        dim_t N_s = 0, N_e = 0;
        dim_t S_s = 0, S_e = 0;

        dim_t C_blks_per_iter = 1;
        int64_t iters = 1;

        if (do_blocking) {
            size_t working_set_size = 2 * N * SP * sizeof(data_t);
            bnorm_utils::cache_balance(
                    working_set_size, C, N, nthr, C_blks_per_iter, iters);
        } else
            C_blks_per_iter = C;
        int64_t last_iter_blks = C - (iters - 1) * C_blks_per_iter;
        bool spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking,
                true, false, ithr, nthr, N, C_blks_per_iter, SP, C_ithr, C_nthr,
                C_blk_s, C_blk_e, N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s,
                S_e);
        balance211(C_blks_per_iter, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
        int SP_N_ithr = N_ithr * S_nthr + S_ithr;
        int SP_N_nthr = N_nthr * S_nthr;

        for (int64_t it = 0; it < iters; ++it) {
            size_t C_off = it * C_blks_per_iter;
            if (it == iters - 1 && iters > 1) {
                // On the last iteration the access pattern to ws_reduce
                // might change (due to re-balance on C). So sync the
                // threads if they are not synced by the algorithm.
                if (SP_N_nthr == 1 && dnnl_thr_syncable()) dnnl_thr_barrier();

                C_blk_s = C_blk_e = N_s = N_e = 0;
                spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking,
                        spatial_thr_allowed, false, ithr, nthr, N,
                        last_iter_blks, SP, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s, S_e);
                balance211(last_iter_blks, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
                C_blks_per_iter = last_iter_blks;
                SP_N_ithr = N_ithr * S_nthr + S_ithr;
                SP_N_nthr = N_nthr * S_nthr;
            }
            const auto S_chunk = nstl::max(dim_t(0), S_e - S_s);
            // On the last iteration the access pattern to ws_reduce
            // might change (due to re-balance on C). Since sync is not always
            // possible (in case of TBB) use different parts of ws for each
            // iteration if threads are not synced by the algorithm.
            size_t ws_iter_off = (dnnl_thr_syncable() ? 0 : 1) * 2 * C_off;

            acc_data_t *diff_gamma_blk = diff_scaleshift + C_off;
            acc_data_t *diff_beta_blk = diff_scaleshift + C + C_off;
            for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                size_t off = c + C_off;
                acc_data_t diff_gamma = 0.0, diff_beta = 0.0;
                acc_data_t v_mean = mean[off];
                for (dim_t n = N_s; n < N_e; ++n) {
                    const acc_data_t *_diff_dst;
                    const acc_data_t *_src;
                    dim_t s_off = off * SP + n * C * SP;
                    if (d_type == bf16) {
                        // convert diff_dst from bf16 to f32
                        acc_data_t *tmp_diff_dst
                                = tmp_data_ + ithr * SP_cl_align;
                        cvt_bfloat16_to_float(tmp_diff_dst + S_s,
                                (bfloat16_t *)diff_dst + s_off + S_s, S_chunk);
                        _diff_dst = tmp_diff_dst;
                        // convert src from bf16 to f32
                        acc_data_t *tmp_src
                                = tmp_data_ + (nthr + ithr) * SP_cl_align;
                        cvt_bfloat16_to_float(tmp_src + S_s,
                                (bfloat16_t *)src + s_off + S_s, S_chunk);
                        _src = tmp_src;
                    } else {
                        _diff_dst = reinterpret_cast<const acc_data_t *>(
                                diff_dst + s_off);
                        _src = reinterpret_cast<const acc_data_t *>(
                                src + s_off);
                    }
#if SAFE_TO_USE_OMP_SIMD
                    PRAGMA_OMP_SIMD(reduction(+ : diff_gamma, diff_beta))
#endif
                    for (dim_t sp = S_s; sp < S_e; ++sp) {
                        const dim_t d_off = s_off + sp;
                        acc_data_t dd;
                        if (fuse_norm_relu && !ws[d_off])
                            dd = 0;
                        else
                            dd = _diff_dst[sp];
                        diff_gamma += (_src[sp] - v_mean) * dd;
                        diff_beta += dd;
                    }
                }
                ws_reduce[ws_iter_off + SP_N_ithr * C_blks_per_iter + c]
                        = diff_gamma;
                ws_reduce[ws_iter_off + SP_N_nthr * C_blks_per_iter
                        + SP_N_ithr * C_blks_per_iter + c]
                        = diff_beta;
            }

            if (dnnl_thr_syncable()) dnnl_thr_barrier();

            for (dim_t c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                acc_data_t sqrt_variance = static_cast<acc_data_t>(
                        1.0f / sqrtf(variance[c + C_off] + eps));
                diff_gamma_blk[c] = 0.;
                diff_beta_blk[c] = 0.;
                for (dim_t n = 0; n < SP_N_nthr; n++) {
                    diff_gamma_blk[c]
                            += ws_reduce[ws_iter_off + n * C_blks_per_iter + c];
                    diff_beta_blk[c] += ws_reduce[ws_iter_off
                            + SP_N_nthr * C_blks_per_iter + n * C_blks_per_iter
                            + c];
                }
                diff_gamma_blk[c] *= sqrt_variance;
            }

            if (dnnl_thr_syncable()) dnnl_thr_barrier();

            for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                size_t off = c + C_off;
                acc_data_t gamma = use_scaleshift ? scaleshift[off] : 1;
                acc_data_t sqrt_variance = static_cast<acc_data_t>(
                        1.0f / sqrtf(variance[off] + eps));
                acc_data_t v_mean = mean[off];
                for (dim_t n = N_s; n < N_e; ++n) {
                    acc_data_t *_diff_src;
                    const acc_data_t *_diff_dst;
                    const acc_data_t *_src;
                    dim_t s_off = off * SP + n * C * SP;
                    if (d_type == bf16) {
                        // store diff_src to f32 buffer
                        _diff_src = tmp_data_ + ithr * SP_cl_align;
                        // convert diff_dst from bf16 to f32
                        acc_data_t *tmp_diff_dst
                                = tmp_data_ + ithr * SP_cl_align;
                        cvt_bfloat16_to_float(tmp_diff_dst + S_s,
                                (bfloat16_t *)diff_dst + s_off + S_s, S_chunk);
                        _diff_dst = tmp_diff_dst;
                        if (calculate_diff_stats) {
                            // convert src from bf16 to f32
                            acc_data_t *tmp_src = tmp_data_
                                    + (2 * nthr + ithr) * SP_cl_align;
                            cvt_bfloat16_to_float(tmp_src + S_s,
                                    (bfloat16_t *)src + s_off + S_s, S_chunk);
                            _src = tmp_src;
                        } else
                            _src = nullptr; // to avoid compiler warning w/
                                    // gcc483
                    } else {
                        _diff_src = reinterpret_cast<acc_data_t *>(
                                diff_src + s_off);
                        _diff_dst = reinterpret_cast<const acc_data_t *>(
                                diff_dst + s_off);
                        _src = reinterpret_cast<const acc_data_t *>(
                                src + s_off);
                    }
#if SAFE_TO_USE_OMP_SIMD
                    PRAGMA_OMP_SIMD()
#endif
                    for (dim_t sp = S_s; sp < S_e; ++sp) {
                        const dim_t d_off = s_off + sp;
                        acc_data_t v_diff_src;
                        if (fuse_norm_relu && !ws[d_off])
                            v_diff_src = 0;
                        else
                            v_diff_src = _diff_dst[sp];
                        if (calculate_diff_stats) {
                            v_diff_src -= diff_beta_blk[c] / (SP * N)
                                    + (_src[sp] - v_mean) * diff_gamma_blk[c]
                                            * sqrt_variance / (SP * N);
                        }
                        v_diff_src *= gamma * sqrt_variance;
                        _diff_src[sp] = v_diff_src;
                    }
                    if (d_type == bf16) {
                        // convert diff_src from f32 to bf16
                        cvt_float_to_bfloat16(
                                (bfloat16_t *)diff_src + s_off + S_s,
                                _diff_src + S_s, S_chunk);
                    }
                }
            }
        }
    });
}

template struct ncsp_batch_normalization_bwd_t<f32>;
template struct ncsp_batch_normalization_bwd_t<bf16>;
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
