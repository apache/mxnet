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

#ifndef CPU_GEMM_X8S8S32X_CONVOLUTION_HPP
#define CPU_GEMM_X8S8S32X_CONVOLUTION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/gemm_convolution_utils.hpp"
#include "cpu/gemm_x8s8s32x_convolution_utils.hpp"

#include "cpu/gemm/gemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src_type, data_type_t dst_type>
struct _gemm_x8s8s32x_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_ISA_STR,
                _gemm_x8s8s32x_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(
                            src_type, s8, data_type::undef, dst_type, s32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type, f32, s32,
                                    s8, u8))
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale
                                    | primitive_attr_t::skip_mask_t::post_ops,
                            dst_type)
                    && output_scales_mask_ok() && post_ops_ok();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, *attr(),
                    dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;

    protected:
        bool output_scales_mask_ok() const {
            const auto &mask = attr()->output_scales_.mask_;
            return mask == 0 || mask == 1 << 1;
        }

        bool post_ops_ok() const {
            using namespace dnnl::impl::primitive_kind;
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(); };

            switch (po.len()) {
                case 0: return true;
                case 1: return is_eltwise(0) || po.contain(sum, 0);
                case 2:
                    return (po.contain(sum, 0) && is_eltwise(1))
                            || (po.contain(sum, 1) && is_eltwise(0));
                default: return false;
            }
            return false;
        }
    };

    _gemm_x8s8s32x_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(pp_ker_, pp_ker_t::create(pd(), pd()->jcp_)));
        return (pp_ker_) ? pp_ker_->create_kernel() : status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;
    status_t execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src_base, const wei_data_t *wei_base,
            const char *bia_base, dst_data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;

    int nthr_ = 0;

    using pp_ker_t = gemm_x8s8s32x_convolution_utils::pp_ker_t;
    std::unique_ptr<pp_ker_t> pp_ker_;
};

template <data_type_t dst_type>
struct _gemm_u8s8s32x_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_ISA_STR,
                _gemm_u8s8s32x_convolution_bwd_data_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(
                            dst_type, s8, data_type::undef, u8, s32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type, f32, s32,
                                    s8, u8))
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale)
                    && output_scales_mask_ok();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_md_, weights_md_, diff_dst_md_, bias_md_,
                    *attr(), dnnl_get_max_threads());
        }

        bool support_bias() const override { return true; }

        conv_gemm_conf_t jcp_;

    protected:
        bool output_scales_mask_ok() const {
            const auto &mask = attr()->output_scales_.mask_;
            return mask == 0 || mask == 1 << 1;
        }
    };

    _gemm_u8s8s32x_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::u8>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    status_t execute_backward_data_thr(const int ithr, const int nthr,
            const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
            const char *bia_base, diff_src_data_t *diff_src_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
