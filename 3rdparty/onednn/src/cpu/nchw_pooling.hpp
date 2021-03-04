/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef CPU_NCHW_POOLING_HPP
#define CPU_NCHW_POOLING_HPP

#include <assert.h>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t d_type>
struct nchw_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nchw:any", nchw_pooling_fwd_t);

        status_t init(engine_t *engine) {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            bool ok = is_fwd()
                    && utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                            alg_kind::pooling_avg_include_padding,
                            alg_kind::pooling_avg_exclude_padding)
                    && utils::everyone_is(
                            d_type, src_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(d_type)
                    && !has_zero_dim_memory()
                    && set_default_params() == status::success
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*dst_md(), desired_fmt_tag)
                    && !is_dilated();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (src_md()->data_type == data_type::bf16) {
                size_t src_sz_ = ID() * IH() * IW() * C() * MB();
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<float>(key_pool_src_bf16cvt, src_sz_);
            }
        }
    };

    nchw_pooling_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<d_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <data_type_t d_type>
struct nchw_pooling_bwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nchw:any", nchw_pooling_bwd_t);

        status_t init(engine_t *engine) {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            using namespace prop_kind;
            using namespace alg_kind;
            bool ok = !is_fwd()
                    && utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                            alg_kind::pooling_avg_include_padding,
                            alg_kind::pooling_avg_exclude_padding)
                    && utils::everyone_is(d_type, diff_dst_md()->data_type,
                            diff_src_md()->data_type)
                    && platform::has_data_type_support(d_type)
                    && !has_zero_dim_memory()
                    && set_default_params() == status::success
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*diff_dst_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*diff_src_md(), desired_fmt_tag)
                    && !is_dilated();
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max) {
                bool ws_ok
                        = true && hint_fwd_pd_ && hint_fwd_pd_->workspace_md();
                if (!ws_ok) return status::unimplemented;

                const auto &ws_blk
                        = hint_fwd_pd_->workspace_md()->format_desc.blocking;
                ws_ok = ws_ok && ws_blk.inner_nblks <= 1
                        && IMPLICATION(ws_blk.inner_nblks == 1,
                                ws_blk.inner_idxs[0] == 1);
                if (!ws_ok) return status::unimplemented;

                ws_md_ = *hint_fwd_pd_->workspace_md();
            }

            calculate_channel_block_size();
            init_scratchpad();

            return status::success;
        }

        dim_t channel_block_size_;

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (diff_dst_md()->data_type == data_type::bf16) {
                size_t dst_sz_ = OD() * OH() * OW();
                size_t src_sz_ = ID() * IH() * IW();
                size_t nthrs = dnnl_get_max_threads();
                auto scratchpad = scratchpad_registry().registrar();

                scratchpad.template book<float>(key_pool_src_bf16cvt,
                        src_sz_ * nthrs * channel_block_size_);
                scratchpad.template book<float>(key_pool_dst_bf16cvt,
                        dst_sz_ * nthrs * channel_block_size_);
            }
        }

        void calculate_channel_block_size() {
            // calculate channels block size at which the data fits into half
            // of L1, it allows to improve performance for problems with small
            // spatial
            dim_t dst_sz_ = OD() * OH() * OW();
            dim_t src_sz_ = ID() * IH() * IW();
            dim_t nthrs = dnnl_get_max_threads();
            dim_t C_per_thr = nstl::min(MB() * C() / nthrs, C());
            const dim_t max_block_size
                    = platform::get_per_core_cache_size(1) / 2;
            dim_t data_size_per_ch = (dst_sz_ + src_sz_) * 6; // f32 + bf16
            channel_block_size_ = nstl::max(
                    nstl::min(C_per_thr, max_block_size / data_size_per_ch),
                    (dim_t)1);
        }
    };

    nchw_pooling_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<d_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
