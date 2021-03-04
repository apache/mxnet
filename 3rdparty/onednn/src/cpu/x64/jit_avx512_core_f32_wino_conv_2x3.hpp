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

#ifndef CPU_X64_JIT_AVX512_CORE_F32_WINO_CONV_2X3_HPP
#define CPU_X64_JIT_AVX512_CORE_F32_WINO_CONV_2X3_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_f32_wino_conv_2x3_fwd_ker_t;
struct jit_avx512_core_f32_wino_conv_2x3_src_trans_t;
struct jit_avx512_core_f32_wino_conv_2x3_dst_trans_t;

struct jit_avx512_core_f32_wino_conv_2x3_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_fp32_wino_2x3:", avx512_core, ""),
                jit_avx512_core_f32_wino_conv_2x3_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::forward_inference
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

            memory_desc_t expect_wei_md = *weights_md();
            status_t jit_conf_result = jit_conf(expect_wei_md);
            if (jit_conf_result != status::success) return jit_conf_result;
            set_default_alg_kind(alg_kind::convolution_winograd);

            if (weights_md_.format_kind == format_kind::any)
                weights_md_ = expect_wei_md;
            if (weights_md_ != expect_wei_md) return status::unimplemented;

            init_scratchpad();

            return status::success;
        }

        jit_conv_conf_2x3_wino_t jcp_;

    protected:
        status_t jit_conf(memory_desc_t &expect_wei_md);

        void init_scratchpad() {
            using namespace memory_tracking::names;

            auto scratchpad = scratchpad_registry().registrar();

            int wino_size_offset = (jcp_.yb / 2) * (jcp_.xb / 2) + jcp_.xb;

            size_t V_sz = (size_t)jcp_.ic * 16 * wino_size_offset * jcp_.nthr;
            scratchpad.book<float>(key_wino_V, V_sz, PAGE_4K);

            size_t M_sz = (size_t)jcp_.oc * 16 * wino_size_offset * jcp_.nthr;
            scratchpad.book<float>(key_wino_M, M_sz, PAGE_4K);

            if (wants_padded_bias()) {
                assert(jcp_.ngroups == 1);
                scratchpad.book<float>(key_conv_padded_bias, jcp_.oc);
            }
        }

        bool set_default_formats() {
            using namespace format_tag;
            return set_default_formats_common(nChw16c, any, nChw16c);
        }
    };

    jit_avx512_core_f32_wino_conv_2x3_fwd_t(const pd_t *apd);
    ~jit_avx512_core_f32_wino_conv_2x3_fwd_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
        auto bia = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

        if (pd()->jcp_.small_mb)
            execute_forward_small_mb(
                    src, wei, bia, dst, ctx.get_scratchpad_grantor());
        else
            execute_forward_mbN(
                    src, wei, bia, dst, ctx.get_scratchpad_grantor());

        return status::success;
    }

private:
    void execute_forward_small_mb(const float *src, const float *wei,
            const float *bia, float *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    void execute_forward_mbN(const float *src, const float *wei,
            const float *bia, float *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx512_core_f32_wino_conv_2x3_fwd_ker_t> kernel_;
    std::unique_ptr<jit_avx512_core_f32_wino_conv_2x3_src_trans_t> src_trans_;
    std::unique_ptr<jit_avx512_core_f32_wino_conv_2x3_dst_trans_t> dst_trans_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
