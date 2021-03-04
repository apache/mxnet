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

#ifndef CPU_REF_LRN_HPP
#define CPU_REF_LRN_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_lrn_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t d_type>
struct ref_lrn_fwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_fwd_pd_t {
        using cpu_lrn_fwd_pd_t::cpu_lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("lrn_ref:any", ref_lrn_fwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            bool ok = is_fwd() && src_md()->data_type == d_type
                    && platform::has_data_type_support(d_type)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            dat_tag_ = memory_desc_matches_one_of_tag(
                    *src_md(), nChw16c, nChw8c, nchw, nhwc);

            return status::success;
        }

        format_tag_t dat_tag_;
    };

    ref_lrn_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<d_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace format_tag;
        switch (pd()->dat_tag_) {
            case nChw16c: execute_forward<nChw16c>(ctx); break;
            case nChw8c: execute_forward<nChw8c>(ctx); break;
            case nchw: execute_forward<nchw>(ctx); break;
            case nhwc: execute_forward<nhwc>(ctx); break;
            default: execute_forward<any>(ctx);
        }
        return status::success;
    }

private:
    template <format_tag_t tag>
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <impl::data_type_t d_type>
struct ref_lrn_bwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_bwd_pd_t {
        using cpu_lrn_bwd_pd_t::cpu_lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("lrn_ref:any", ref_lrn_bwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            bool ok = !is_fwd() && set_default_formats_common()
                    && utils::everyone_is(d_type, src_md()->data_type,
                            diff_src_md()->data_type)
                    && platform::has_data_type_support(d_type)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            dat_tag_ = memory_desc_matches_one_of_tag(
                    *src_md(), nChw16c, nChw8c, nchw, nhwc);

            return status::success;
        }

        format_tag_t dat_tag_;
    };

    ref_lrn_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<d_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace format_tag;
        switch (pd()->dat_tag_) {
            case nChw16c: execute_backward<nChw16c>(ctx); break;
            case nChw8c: execute_backward<nChw8c>(ctx); break;
            case nchw: execute_backward<nchw>(ctx); break;
            case nhwc: execute_backward<nhwc>(ctx); break;
            default: execute_backward<any>(ctx);
        }
        return status::success;
    }

private:
    template <format_tag_t tag>
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
