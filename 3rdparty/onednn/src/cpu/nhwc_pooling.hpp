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

#ifndef CPU_NHWC_POOLING_HPP
#define CPU_NHWC_POOLING_HPP

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

namespace nhwc_pooling {
size_t strided_offset(const int _n, const size_t _sn, const int _d,
        const size_t _sd, const int _h, const size_t _sh, const int _w,
        const size_t _sw);
}

template <data_type_t d_type>
struct nhwc_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nhwc:any", nhwc_pooling_fwd_t);

        status_t init(engine_t *engine) {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

            using namespace prop_kind;
            using namespace alg_kind;
            bool ok = is_fwd()
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && utils::everyone_is(
                            d_type, src_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(d_type)
                    && set_default_params() == status::success
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*dst_md(), desired_fmt_tag)
                    && !is_dilated();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                init_default_ws();
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (src_md()->data_type == data_type::bf16) {
                size_t bf16cvt_sz_ = C() * dnnl_get_max_threads();
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<float>(
                        key_pool_src_bf16cvt, bf16cvt_sz_);
                scratchpad.template book<float>(
                        key_pool_dst_bf16cvt, bf16cvt_sz_);
            }
        }
    };

    nhwc_pooling_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<d_type>::type data_t;
    typedef typename prec_traits<data_type::f32>::type ker_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void array_div_by_const(const int n, const ker_data_t *src,
            const size_t num, ker_data_t *dst) const;
    void array_add(const int n, const ker_data_t *src, ker_data_t *dst) const;
    void array_nhwc_max(const int n, ker_data_t *dst, const ker_data_t *src,
            unsigned char *ws, const size_t ws_offset, const data_type_t ws_dt,
            const int index) const;
    void array_nhwc_initialize(const int n, ker_data_t *dst, unsigned char *ws,
            const size_t ws_offset, const data_type_t ws_dt) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <impl::data_type_t d_type>
struct nhwc_pooling_bwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nhwc:any", nhwc_pooling_bwd_t);

        status_t init(engine_t *engine) {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

            using namespace prop_kind;
            using namespace alg_kind;
            bool ok = !is_fwd()
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && utils::everyone_is(d_type, diff_dst_md()->data_type,
                            diff_src_md()->data_type)
                    && platform::has_data_type_support(d_type)
                    && set_default_params() == status::success && !is_fwd()
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*diff_dst_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*diff_src_md(), desired_fmt_tag)
                    && !is_dilated();
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max) {
                init_default_ws();
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (diff_src_md()->data_type == data_type::bf16) {
                size_t bf16cvt_sz_ = C() * dnnl_get_max_threads();
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<float>(
                        key_pool_src_bf16cvt, bf16cvt_sz_);
                scratchpad.template book<float>(
                        key_pool_dst_bf16cvt, bf16cvt_sz_);
            }
        }
    };

    nhwc_pooling_bwd_t(const pd_t *apd) : primitive_t(apd) {}
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
