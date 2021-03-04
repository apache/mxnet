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

#ifndef CPU_REF_INNER_PRODUCT_HPP
#define CPU_REF_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src_type, data_type_t wei_type = src_type,
        data_type_t dst_type = src_type, data_type_t acc_type = dst_type>
struct ref_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            bool ok = is_fwd()
                    && expect_data_types(src_type, wei_type, data_type::undef,
                            dst_type, acc_type)
                    && platform::has_data_type_support(src_type)
                    && platform::has_data_type_support(wei_type)
                    && platform::has_data_type_support(dst_type)
                    && IMPLICATION(with_bias(),
                            IMPLICATION(src_type == u8,
                                    utils::one_of(bias_md_.data_type, f32, s32,
                                            s8, u8))
                                    && IMPLICATION(src_type == f32,
                                            bias_md_.data_type == f32))
                    && set_default_params() == status::success
                    && attr()->has_default_values(
                            smask_t::oscale | smask_t::post_ops)
                    && output_scales_mask_ok();
            return ok ? status::success : status::unimplemented;
        }

    private:
        bool output_scales_mask_ok() const {
            using namespace data_type;
            const auto &mask = attr()->output_scales_.mask_;
            return IMPLICATION(!utils::one_of(src_type, s8, u8),
                           attr()->output_scales_.has_default_values())
                    && (mask == 0 || mask == 1 << 1);
        }
    };

    ref_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        return status::success;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

template <data_type_t diff_src_type, data_type_t wei_type,
        data_type_t diff_dst_type, data_type_t acc_type = diff_src_type>
struct ref_inner_product_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_data_pd_t {
        using cpu_inner_product_bwd_data_pd_t::cpu_inner_product_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && diff_src_md()->data_type == diff_src_type
                    && weights_md()->data_type == wei_type
                    && desc()->accum_data_type == acc_type
                    && diff_dst_md()->data_type == diff_dst_type
                    && attr()->has_default_values()
                    && set_default_params() == status::success;
            return ok ? status::success : status::unimplemented;
        }
    };

    ref_inner_product_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <data_type_t data_type>
struct ref_inner_product_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_weights_pd_t {
        using cpu_inner_product_bwd_weights_pd_t::
                cpu_inner_product_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && utils::everyone_is(data_type, src_md()->data_type,
                            diff_dst_md()->data_type,
                            diff_weights_md()->data_type)
                    && IMPLICATION(with_bias(),
                            data_type == diff_weights_md(1)->data_type)
                    && attr()->has_default_values()
                    && set_default_params() == status::success;
            return ok ? status::success : status::unimplemented;
        }
    };

    ref_inner_product_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
