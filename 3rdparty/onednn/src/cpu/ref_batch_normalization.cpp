/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "cpu/ref_batch_normalization.hpp"
#include "cpu/simple_q10n.hpp"

#define DATA_OFF(f, n, c, d, h, w) \
    (ndims == 2) ? (f).off(n, c) \
                 : ((ndims == 3) ? (f).off(n, c, w) \
                                 : ((ndims == 4) ? (f).off(n, c, h, w) \
                                                 : (f).off(n, c, d, h, w)))

namespace dnnl {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

namespace {

using acc_data_t = float;

template <typename T>
inline float maybe_up_convert(T x) {
    return x;
}

template <>
inline float maybe_up_convert<bfloat16_t>(bfloat16_t x) {
    return (float)x;
}

} // namespace

using namespace data_type;

template <impl::data_type_t d_type>
void ref_batch_normalization_fwd_t<d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    /* fast return */
    if (this->pd()->has_zero_dim_memory()) return;

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto scaleshift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE_SHIFT);

    auto mean = pd()->stats_is_src()
            ? const_cast<acc_data_t *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
            : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
    auto variance = pd()->stats_is_src()
            ? const_cast<acc_data_t *>(
                    CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
            : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);

    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, DNNL_ARG_WORKSPACE);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper scaleshift_d(pd()->weights_md());

    const auto ndims = data_d.ndims();
    const auto N = pd()->MB();
    const auto C = pd()->C();
    const auto D = pd()->D();
    const auto H = pd()->H();
    const auto W = pd()->W();

    const auto eps = pd()->desc()->batch_norm_epsilon;
    const auto use_scaleshift = pd()->use_scaleshift();
    const auto calculate_stats = !pd()->stats_is_src();
    const auto fuse_norm_relu = pd()->fuse_norm_relu();
    const auto save_stats = pd()->is_training();
    const auto is_training = pd()->is_training();

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) {
        if (calculate_stats && save_stats)
            for (dim_t c = 0; c < pd()->C(); c++) {
                mean[c] = 0;
                variance[c] = 0;
            }
        return;
    }

    const bool with_relu = pd()->with_relu_post_op();
    auto maybe_post_op = [&](acc_data_t res) {
        return (with_relu && res < 0.0f) ? 0.0f : res;
    };

    parallel_nd(C, [&](dim_t c) {
        acc_data_t v_mean = calculate_stats ? 0 : mean[c];
        acc_data_t v_variance = calculate_stats ? 0 : variance[c];

        if (calculate_stats) {
            for_(int n = 0; n < N; ++n)
            for_(int d = 0; d < D; ++d)
            for_(int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                v_mean += maybe_up_convert(
                        src[DATA_OFF(data_d, n, c, d, h, w)]);
            }
            v_mean /= W * N * H * D;

            for_(int n = 0; n < N; ++n)
            for_(int d = 0; d < D; ++d)
            for_(int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                acc_data_t m = src[DATA_OFF(data_d, n, c, d, h, w)] - v_mean;
                v_variance += m * m;
            }
            v_variance /= W * H * N * D;
        }

        acc_data_t sqrt_variance = sqrtf(v_variance + eps);
        acc_data_t sm
                = (use_scaleshift ? scaleshift[scaleshift_d.off(0, c)] : 1.0f)
                / sqrt_variance;
        acc_data_t sv = use_scaleshift ? scaleshift[scaleshift_d.off(1, c)] : 0;

        for_(dim_t n = 0; n < N; ++n)
        for_(dim_t d = 0; d < D; ++d)
        for_(dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            auto d_off = DATA_OFF(data_d, n, c, d, h, w);
            acc_data_t bn_res
                    = sm * (maybe_up_convert(src[d_off]) - v_mean) + sv;
            if (fuse_norm_relu) {
                if (bn_res <= 0) {
                    bn_res = 0;
                    if (is_training) ws[d_off] = 0;
                } else {
                    if (is_training) ws[d_off] = 1;
                }
            }
            if (d_type == s8)
                dst[d_off] = qz_a1b0<float, data_t>()(maybe_post_op(bn_res));
            else
                dst[d_off] = maybe_post_op(bn_res);
        }

        if (calculate_stats) {
            if (save_stats) {
                mean[c] = v_mean;
                variance[c] = v_variance;
            }
        }
    });
}

template struct ref_batch_normalization_fwd_t<s8>;
template struct ref_batch_normalization_fwd_t<f32>;
template struct ref_batch_normalization_fwd_t<bf16>;

template <impl::data_type_t d_type>
void ref_batch_normalization_bwd_t<d_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN);
    auto variance = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto scaleshift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE_SHIFT);
    auto ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);
    auto diff_scaleshift = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SCALE_SHIFT);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());
    const memory_desc_wrapper scaleshift_d(pd()->weights_md());
    const memory_desc_wrapper diff_scaleshift_d(pd()->diff_weights_md());

    const auto ndims = data_d.ndims();
    const auto N = pd()->MB();
    const auto C = pd()->C();
    const auto D = pd()->D();
    const auto H = pd()->H();
    const auto W = pd()->W();

    const auto eps = pd()->desc()->batch_norm_epsilon;
    const auto use_scaleshift = pd()->use_scaleshift();
    const auto calculate_diff_stats = !pd()->use_global_stats();
    const auto fuse_norm_relu = pd()->fuse_norm_relu();

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) {
        if (diff_scaleshift) {
            for (dim_t c = 0; c < C; ++c) {
                diff_scaleshift[diff_scaleshift_d.off(0, c)] = 0;
                diff_scaleshift[diff_scaleshift_d.off(1, c)] = 0;
            }
        }
        return;
    }

    parallel_nd(C, [&](dim_t c) {
        acc_data_t v_mean = mean[c];
        acc_data_t v_variance = variance[c];
        acc_data_t sqrt_variance
                = static_cast<acc_data_t>(1.0f / sqrtf(v_variance + eps));
        acc_data_t gamma
                = use_scaleshift ? scaleshift[scaleshift_d.off(0, c)] : 1;
        acc_data_t diff_gamma = 0;
        acc_data_t diff_beta = 0;

        for_(dim_t n = 0; n < N; ++n)
        for_(dim_t d = 0; d < D; ++d)
        for_(dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            const size_t s_off = DATA_OFF(data_d, n, c, d, h, w);
            acc_data_t dd;
            if (fuse_norm_relu && !ws[s_off])
                dd = 0;
            else
                dd = maybe_up_convert(
                        diff_dst[DATA_OFF(diff_data_d, n, c, d, h, w)]);
            diff_gamma += (maybe_up_convert(src[s_off]) - v_mean) * dd;
            diff_beta += dd;
        }
        diff_gamma *= sqrt_variance;

        if (diff_scaleshift) {
            diff_scaleshift[diff_scaleshift_d.off(0, c)] = diff_gamma;
            diff_scaleshift[diff_scaleshift_d.off(1, c)] = diff_beta;
        }

        for_(dim_t n = 0; n < N; ++n)
        for_(dim_t d = 0; d < D; ++d)
        for_(dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            const size_t s_off = DATA_OFF(data_d, n, c, d, h, w);
            const size_t dd_off = DATA_OFF(diff_data_d, n, c, d, h, w);
            acc_data_t dd;
            if (fuse_norm_relu && !ws[s_off])
                dd = 0;
            else
                dd = maybe_up_convert(diff_dst[dd_off]);
            acc_data_t v_diff_src = dd;
            if (calculate_diff_stats) {
                v_diff_src -= diff_beta / (D * W * H * N)
                        + (maybe_up_convert(src[s_off]) - v_mean) * diff_gamma
                                * sqrt_variance / (D * W * H * N);
            }
            v_diff_src *= gamma * sqrt_variance;
            diff_src[dd_off] = v_diff_src;
        }
    });
}

template struct ref_batch_normalization_bwd_t<f32>;
template struct ref_batch_normalization_bwd_t<bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
