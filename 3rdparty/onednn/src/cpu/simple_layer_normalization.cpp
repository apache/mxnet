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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_batch_normalization_utils.hpp"
#include "cpu/cpu_engine.hpp"

#include "cpu/simple_layer_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;
using namespace data_type;

namespace {
/* Stats and src here are compatible if
 * stat_strides[:] == data_strides[:] / last_data_dimension
 * i.e. abcd & abc, bacd & bac - compatible */
status_t fill_compatible_stats_md(
        const memory_desc_t &src_md, memory_desc_t &stat_md) {
    stat_md = src_md;
    stat_md.data_type = dnnl_f32;
    stat_md.ndims -= 1;
    return memory_desc_init_by_blocking_desc(
            stat_md, src_md.format_desc.blocking);
}

status_t create_reorder_pd(engine_t *engine, const memory_desc_t *from_md,
        const memory_desc_t *to_md,
        std::unique_ptr<primitive_desc_t> &reorder_pd) {
    auto r_impls = engine->get_reorder_implementation_list(from_md, to_md);
    for (auto r = r_impls; *r; ++r) {
        primitive_attr_t r_attr;
        r_attr.set_scratchpad_mode(scratchpad_mode::user);
        reorder_pd_t *r_pd = nullptr;
        if ((*r)(&r_pd, engine, &r_attr, engine, from_md, engine, to_md)
                == status::success) {
            reorder_pd.reset(r_pd);
            break;
        }
    }
    return status::success;
}
} // namespace

template <data_type_t data_type>
status_t simple_layer_normalization_fwd_t<data_type>::pd_t::init(
        engine_t *engine) {
    using namespace data_type;
    const memory_desc_wrapper src_d(src_md());

    const bool ok = is_fwd() && !has_zero_dim_memory()
            && platform::has_data_type_support(data_type)
            && utils::everyone_is(
                    data_type, src_md()->data_type, dst_md()->data_type)
            && (f32 == stat_md()->data_type) && check_scale_shift_data_type()
            && src_d.is_blocking_desc()
            && src_d.blocking_desc().strides[ndims() - 1]
                    == 1 // plain format, last logical dim is last physical
            && attr()->has_default_values() && set_default_formats_common();
    if (!ok) return status::unimplemented;

    CHECK(fill_compatible_stats_md(*src_md(), reordered_stat_md_));

    if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
        CHECK(create_reorder_pd(engine,
                stats_are_src() ? stat_md() : &reordered_stat_md_,
                stats_are_src() ? &reordered_stat_md_ : stat_md(),
                reorder_pd_));
    }

    init_scratchpad();
    return status::success;
}

template <data_type_t data_type>
void simple_layer_normalization_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto scaleshift = CTX_IN_MEM(const float *, DNNL_ARG_SCALE_SHIFT);

    float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        auto scratchpad = ctx.get_scratchpad_grantor();
        mean = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = pd()->stats_are_src()
                ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
                : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
        variance = pd()->stats_are_src()
                ? const_cast<float *>(
                        CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
                : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
    }

    const memory_desc_wrapper src_d(pd()->src_md());

    const dim_t N = pd()->across_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    const bool save_stats = pd()->is_training();
    const bool calculate_stats = !pd()->stats_are_src();

    parallel_nd(N, [&](dim_t n) {
        auto v_mean = calculate_stats ? 0 : mean[n];
        auto v_variance = calculate_stats ? 0 : variance[n];

        if (calculate_stats)
            (*stat_kernel_)(&src[n * C_padded], &v_mean, &v_variance);

        (*data_kernel_)(&src[n * C_padded], &dst[n * C_padded], scaleshift,
                &v_mean, &v_variance);

        if (calculate_stats) {
            if (save_stats) {
                mean[n] = v_mean;
                variance[n] = v_variance;
            }
        }
    });
}

template <data_type_t data_type>
status_t simple_layer_normalization_bwd_t<data_type>::pd_t::init(
        engine_t *engine) {
    using namespace data_type;
    const memory_desc_wrapper src_d(src_md());

    const bool ok = is_bwd() && !has_zero_dim_memory()
            && set_default_formats_common()
            && platform::has_data_type_support(data_type)
            && utils::everyone_is(
                    data_type, src_md()->data_type, dst_md()->data_type)
            && (f32 == stat_md()->data_type) && check_scale_shift_data_type()
            && src_d.is_blocking_desc()
            && src_d.blocking_desc().strides[ndims() - 1]
                    == 1 //plain format, last logical dim is last physical
            && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    CHECK(fill_compatible_stats_md(*src_md(), reordered_stat_md_));

    if (reordered_stat_md_ != *stat_md()) {
        CHECK(create_reorder_pd(
                engine, stat_md(), &reordered_stat_md_, reorder_pd_));
    }

    init_scratchpad();
    return status::success;
}

template <data_type_t data_type>
void simple_layer_normalization_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    auto scratchpad = ctx.get_scratchpad_grantor();
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto scaleshift = CTX_IN_MEM(const float *, DNNL_ARG_SCALE_SHIFT);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);
    auto diff_scaleshift = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SCALE_SHIFT);

    const float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        mean = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
        variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    }

    const memory_desc_wrapper src_d(pd()->src_md());

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    float *reduce = scratchpad.template get<float>(key_lnorm_reduction);
    if (diff_scaleshift == nullptr)
        diff_scaleshift = scratchpad.template get<float>(key_lnorm_tmp_diff_ss);

    int max_nthr = dnnl_get_max_threads();
    parallel(max_nthr, [&](int ithr, int nthr) {
        assert(nthr == max_nthr);
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        float *my_diff_gamma = reduce + C * ithr;
        float *my_diff_beta = reduce + C * nthr + C * ithr;
        for (dim_t c = 0; c < C; c++) {
            my_diff_gamma[c] = 0.;
            my_diff_beta[c] = 0.;
        }
        for (dim_t n = N_s; n < N_e; n++) {
            (*diff_ss_kernel_)(&src[n * C_padded], &diff_dst[n * C_padded],
                    my_diff_gamma, my_diff_beta, &mean[n], &variance[n]);
        }
    });

    parallel_nd(C, [&](dim_t c) {
        float diff_gamma = 0, diff_beta = 0;
        for (dim_t n = 0; n < max_nthr; n++) {
            diff_gamma += reduce[C * n + c];
            diff_beta += reduce[C * max_nthr + C * n + c];
        }
        diff_scaleshift[c] = diff_gamma;
        diff_scaleshift[C + c] = diff_beta;
    });

    parallel(max_nthr, [&](int ithr, int nthr) {
        assert(nthr == max_nthr);
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        for (dim_t n = N_s; n < N_e; n++) {
            (*diff_data_kernel_)(&src[n * C_padded], &diff_dst[n * C_padded],
                    &diff_src[n * C_padded], scaleshift, &mean[n],
                    &variance[n]);
        }
    });
}

template struct simple_layer_normalization_fwd_t<bf16>;
template struct simple_layer_normalization_fwd_t<f32>;
template struct simple_layer_normalization_bwd_t<bf16>;
template struct simple_layer_normalization_bwd_t<f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
