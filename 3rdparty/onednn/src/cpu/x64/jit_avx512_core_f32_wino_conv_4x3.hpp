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

#ifndef CPU_X64_JIT_AVX512_CORE_F32_WINO_CONV_4X3_HPP
#define CPU_X64_JIT_AVX512_CORE_F32_WINO_CONV_4X3_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/jit_avx512_core_f32_wino_conv_4x3_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace winograd_avx512_core {
inline void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_conv_winograd_conf_t &jcp) {
    using namespace utils;
    using namespace memory_tracking::names;

    size_t U_sz = (size_t)alpha * alpha * jcp.ic * jcp.oc;
    size_t V_sz
            = (size_t)alpha * alpha * jcp.mb * jcp.ic * jcp.itiles * jcp.jtiles;
    size_t M_sz
            = (size_t)alpha * alpha * jcp.mb * jcp.oc * jcp.itiles * jcp.jtiles;

    switch (jcp.sched_policy) {
        case WSCHED_DATA_W_SGD:
            V_sz = (size_t)jcp.nthr * alpha * alpha * jcp.nb_tile_block_ur
                    * jcp.tile_block_ur * jcp.ic;
            M_sz = (size_t)jcp.nthr * alpha * alpha * jcp.nb_tile_block_ur
                    * jcp.tile_block_ur * jcp.oc;
            break;
        case WSCHED_WEI_SDGtWo:
            U_sz = (size_t)jcp.nthr
                    * (alpha * alpha * jcp.oc * (jcp.ic / jcp.nb_ic)
                            + jcp.ic * jcp.oc * jcp.kh * jcp.kw);
            M_sz = (size_t)jcp.nthr * alpha * alpha
                    * (jcp.ntiles / jcp.tile_block) * (jcp.oc / jcp.nb_oc);
            V_sz = (size_t)jcp.nthr * alpha * alpha
                    * (jcp.ntiles / jcp.tile_block) * (jcp.ic / jcp.nb_ic);
            break;
        case WSCHED_WEI_S_D_Giot_W:
            U_sz = (jcp.nthr + static_cast<size_t>(1)) * alpha * alpha * jcp.ic
                    * jcp.oc;
            M_sz = (size_t)alpha * alpha * jcp.oc * jcp.ntiles;
            V_sz = (size_t)alpha * alpha * jcp.ic * jcp.ntiles;
            break;
        default: break;
    }

    scratchpad.book<float>(key_wino_U, U_sz, PAGE_2M);
    scratchpad.book<float>(key_wino_V, V_sz, PAGE_2M);
    scratchpad.book<float>(key_wino_M, M_sz, PAGE_2M);

    if (one_of(jcp.sched_policy, WSCHED_WEI_SDGtWo, WSCHED_WEI_S_D_Giot_W)) {
        size_t br_sz = (size_t)jcp.nthr * jcp.oc;
        scratchpad.book<float>(key_conv_bia_reduction, br_sz, PAGE_2M);
    }
}
} // namespace winograd_avx512_core

template <bool is_fwd>
struct _jit_avx512_core_f32_wino_conv_4x3_t {

    _jit_avx512_core_f32_wino_conv_4x3_t(
            const jit_conv_winograd_conf_t &jcp, const primitive_attr_t *attr)
        : attr_(attr) {}

protected:
    void weight_transform_data(
            const jit_conv_winograd_conf_t &jcp, float *wp, float *twp) const;
    void input_transform_data(int image, const jit_conv_winograd_conf_t &jcp,
            float *inp, float *tinp) const;
    void input_transform_tileblock_data(int tile_block,
            const jit_conv_winograd_conf_t &jcp, float *inp, float *tinp) const;
    void output_transform_data(int image, const jit_conv_winograd_conf_t &jcp,
            const post_ops_t &p_ops, float *toutp, float *pout_b,
            float *bias) const;
    void output_transform_tileblock_data(int tile_block,
            const jit_conv_winograd_conf_t &jcp, const post_ops_t &p_ops,
            float *toutp, float *outp, float *bias) const;
    void _execute_data_W_S_G_D(float *inp_ptr, float *out_ptr, float *wei_ptr,
            float *bias_ptr,
            const memory_tracking::grantor_t &scratchpad) const;
    void _execute_data_W_SGD(float *inp_ptr, float *out_ptr, float *wei_ptr,
            float *bias_ptr,
            const memory_tracking::grantor_t &scratchpad) const;
    std::unique_ptr<_jit_avx512_core_f32_wino_conv_4x3_data_kernel> kernel_;
    const primitive_attr_t *attr_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(_jit_avx512_core_f32_wino_conv_4x3_t);
};

struct jit_avx512_core_f32_wino_conv_4x3_fwd_t
    : _jit_avx512_core_f32_wino_conv_4x3_t<true>,
      public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino_4x3:", avx512_core, ""),
                jit_avx512_core_f32_wino_conv_4x3_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::convolution_auto,
                            alg_kind::convolution_winograd)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops,
                            data_type::f32)
                    && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status
                    = jit_avx512_core_f32_wino_conv_4x3_fwd_kernel::init_conf(
                            jcp_, *desc(), src_md_, weights_md_, dst_md_,
                            *attr());
            if (status != status::success) return status;
            set_default_alg_kind(alg_kind::convolution_winograd);

            auto scratchpad = scratchpad_registry().registrar();
            winograd_avx512_core::init_scratchpad(scratchpad, jcp_);

            return status;
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto wei_fmt = desc()->prop_kind == prop_kind::forward_training
                    ? (with_groups() ? gOIhw16i16o : OIhw16i16o)
                    : any;
            return set_default_formats_common(nChw16c, wei_fmt, nChw16c);
        }
    };

    jit_avx512_core_f32_wino_conv_4x3_fwd_t(const pd_t *apd)
        : _jit_avx512_core_f32_wino_conv_4x3_t<true>(apd->jcp_, apd->attr())
        , primitive_t(apd) {}
    ~jit_avx512_core_f32_wino_conv_4x3_fwd_t() = default;

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new _jit_avx512_core_f32_wino_conv_4x3_data_kernel(
                        pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
        auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
        auto bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

        auto scratchpad = ctx.get_scratchpad_grantor();

        switch ((pd()->jcp_).sched_policy) {
            case WSCHED_DATA_W_S_G_D:
                this->_execute_data_W_S_G_D((float *)src, dst, (float *)weights,
                        (float *)bias, scratchpad);
                break;
            case WSCHED_DATA_W_SGD:
                this->_execute_data_W_SGD((float *)src, dst, (float *)weights,
                        (float *)bias, scratchpad);
                break;
            default: break;
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct jit_avx512_core_f32_wino_conv_4x3_bwd_data_t
    : _jit_avx512_core_f32_wino_conv_4x3_t<false>,
      public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino_4x3:", avx512_core, ""),
                jit_avx512_core_f32_wino_conv_4x3_bwd_data_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && dnnl_thr_syncable()
                    && desc()->prop_kind == prop_kind::backward_data
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::convolution_auto,
                            alg_kind::convolution_winograd)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::undef, data_type::f32, data_type::f32)
                    && attr()->has_default_values() && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status
                    = jit_avx512_core_f32_wino_conv_4x3_bwd_data_kernel ::
                            init_conf(jcp_, *desc(), *diff_src_md(),
                                    *weights_md(), *diff_dst_md());
            if (status != status::success) return status;
            set_default_alg_kind(alg_kind::convolution_winograd);

            auto scratchpad = scratchpad_registry().registrar();
            winograd_avx512_core::init_scratchpad(scratchpad, jcp_);

            return status;
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto wei_fmt = with_groups() ? gOIhw16i16o : OIhw16i16o;
            return set_default_formats_common(nChw16c, wei_fmt, nChw16c);
        }
    };

    jit_avx512_core_f32_wino_conv_4x3_bwd_data_t(const pd_t *apd)
        : _jit_avx512_core_f32_wino_conv_4x3_t<false>(apd->jcp_, apd->attr())
        , primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new _jit_avx512_core_f32_wino_conv_4x3_data_kernel(
                        pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto diff_dst = CTX_IN_MEM(const float *, DNNL_ARG_DIFF_DST);
        auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
        auto diff_src = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SRC);

        auto scratchpad = ctx.get_scratchpad_grantor();

        switch ((pd()->jcp_).sched_policy) {
            case WSCHED_DATA_W_S_G_D:
                this->_execute_data_W_S_G_D((float *)diff_dst, diff_src,
                        (float *)weights, NULL, scratchpad);
                break;

            case WSCHED_DATA_W_SGD:
                this->_execute_data_W_SGD((float *)diff_dst, diff_src,
                        (float *)weights, NULL, scratchpad);
                break;

            default: break;
        }

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct jit_avx512_core_f32_wino_conv_4x3_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino_4x3:", avx512_core, ""),
                jit_avx512_core_f32_wino_conv_4x3_bwd_weights_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && dnnl_thr_syncable()
                    && desc()->prop_kind == prop_kind::backward_weights
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::convolution_auto,
                            alg_kind::convolution_winograd)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && attr()->has_default_values() && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status
                    = jit_avx512_core_f32_wino_conv_4x3_bwd_weights_kernel::
                            init_conf(jcp_, *desc(), *src_md(), *diff_dst_md(),
                                    *diff_weights_md());
            if (status != status::success) return status;
            set_default_alg_kind(alg_kind::convolution_winograd);

            auto scratchpad = scratchpad_registry().registrar();
            winograd_avx512_core::init_scratchpad(scratchpad, jcp_);

            return status;
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto wei_fmt = with_groups() ? gOIhw16i16o : OIhw16i16o;
            return set_default_formats_common(nChw16c, wei_fmt, nChw16c);
        }
    };

    jit_avx512_core_f32_wino_conv_4x3_bwd_weights_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx512_core_f32_wino_conv_4x3_bwd_weights_kernel(
                        pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto diff_dst = CTX_IN_MEM(const float *, DNNL_ARG_DIFF_DST);
        auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
        auto diff_weights = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_WEIGHTS);
        auto diff_bias = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);

        switch (kernel_->jcp.sched_policy) {
            case WSCHED_WEI_SDGtWo:
                _execute_backward_weights_SDGtWo(src, diff_dst, diff_weights,
                        diff_bias, ctx.get_scratchpad_grantor());
                break;
            case WSCHED_WEI_S_D_Giot_W:
                _execute_backward_weights_S_D_Giot_W(src, diff_dst,
                        diff_weights, diff_bias, ctx.get_scratchpad_grantor());
                break;
            default: assert(kernel_->jcp.sched_policy != WSCHED_INVALID); break;
        }
        return status::success;
    }

private:
    void _execute_backward_weights_SDGtWo(const float *src,
            const float *diff_dst, float *diff_weights, float *diff_bias,
            const memory_tracking::grantor_t &scratchpad) const;
    void _execute_backward_weights_S_D_Giot_W(const float *src,
            const float *diff_dst, float *diff_weights, float *diff_bias,
            const memory_tracking::grantor_t &scratchpad) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_avx512_core_f32_wino_conv_4x3_bwd_weights_kernel>
            kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
