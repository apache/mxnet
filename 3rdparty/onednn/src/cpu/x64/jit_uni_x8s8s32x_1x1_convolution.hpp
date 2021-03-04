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

#ifndef CPU_X64_JIT_UNI_X8S8S32X_1X1_CONVOLUTION_HPP
#define CPU_X64_JIT_UNI_X8S8S32X_1X1_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_hashing.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/dw_convolution_utils.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_1x1_conv_kernel.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_uni_x8s8s32x_1x1_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using dw_conv_pd_type = cpu_convolution_fwd_pd_t;
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_()
            , jcp_dw_(nullptr) {}

        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            if (copy(other) != status::success) is_initialized_ = false;
        }

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_uni_int8_1x1:", isa, ""),
                jit_uni_x8s8s32x_1x1_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using smask_t = primitive_attr_t::skip_mask_t;
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, data_type::s8,
                            data_type::undef, dst_type, data_type::s32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type,
                                    data_type::f32, data_type::s32,
                                    data_type::s8, data_type::u8))
                    && attr()->has_default_values(smask_t::oscale
                                    | smask_t::zero_points_runtime
                                    | smask_t::post_ops,
                            dst_type)
                    && !has_zero_dim_memory() && zero_points_ok()
                    && set_default_formats_common(
                            dat_tag(), format_tag::any, dat_tag())
                    && set_or_check_wei_format();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md(), weights_md());

            status_t status = jit_uni_x8s8s32x_1x1_conv_kernel<isa>::init_conf(
                    jcp_, *conv_d, *src_d, *weights_md(), *dst_md(),
                    with_bias() ? *weights_md(1) : types::zero_md(), *attr(),
                    dnnl_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            if (jcp_.with_dw_conv) {
                status = depthwise_po_init(engine);
                if (status != status::success) return status;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_x8s8s32x_1x1_conv_kernel<isa>::init_scratchpad(
                    scratchpad, jcp_, *attr());

            rtus_prepare_space_info(this, scratchpad, jcp_.nthr);

            return status::success;
        }

        const memory_desc_t *dst_md(int index = 0) const override {
            return jcp_.with_dw_conv ? dw_conv_pd_->dst_md(index) : &dst_md_;
        }

        const memory_desc_t *arg_md(int index = 0) const override {
            if (jcp_.with_dw_conv) {
                switch (index) {
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS:
                        return dw_conv_pd_->weights_md(0);
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS:
                        return dw_conv_pd_->weights_md(1);
                    default: break;
                }
            }
            return convolution_fwd_pd_t::arg_md(index);
        }

        arg_usage_t arg_usage(int arg) const override {

            if (utils::one_of(arg, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                        DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS))
                return arg_usage_t::input;

            return convolution_fwd_pd_t::arg_usage(arg);
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;
        jit_conv_conf_t *jcp_dw_; // doesn't own a resource
        std::unique_ptr<cpu_convolution_fwd_pd_t> dw_conv_pd_;
        template <data_type_t sdt, data_type_t ddt>
        using dw_pd_t = typename jit_uni_x8s8s32x_convolution_fwd_t<isa, sdt,
                ddt>::pd_t;

    protected:
        bool zero_points_ok() const {
            using namespace data_type;
            int mask_src = 0, mask_dst = 0;
            const int c_mask = 0x1,
                      g_mask = 0x3; // mask for i/o-channel and ngroups
            attr()->zero_points_.get(DNNL_ARG_SRC, nullptr, &mask_src, nullptr);
            attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &mask_dst, nullptr);
            return attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
                    && utils::one_of(mask_src, 0, c_mask, g_mask)
                    && utils::one_of(mask_dst, 0, c_mask, g_mask);
        }

        status_t copy(const pd_t &other) {
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            jcp_dw_ = nullptr;
            if (other.dw_conv_pd_) {
                dw_conv_pd_.reset(static_cast<cpu_convolution_fwd_pd_t *>(
                        other.dw_conv_pd_->clone()));
                if (!dw_conv_pd_) return status::out_of_memory;
#define CASE(sdt, ddt) \
    case ddt: \
        jcp_dw_ = &( \
                static_cast<dw_pd_t<sdt, ddt> *>(dw_conv_pd_.get())->jcp_); \
        break;

                auto dw_dst_dt = dw_conv_pd_->dst_md()->data_type;
                if (jcp_.dst_dt == data_type::u8) {
                    switch (dw_dst_dt) {
                        CASE(data_type::u8, data_type::u8);
                        CASE(data_type::u8, data_type::s8);
                        CASE(data_type::u8, data_type::s32);
                        CASE(data_type::u8, data_type::f32);
                        default: assert(!"unreachable");
                    }
                } else if (jcp_.dst_dt == data_type::s8) {
                    switch (dw_dst_dt) {
                        CASE(data_type::s8, data_type::u8);
                        CASE(data_type::s8, data_type::s8);
                        CASE(data_type::s8, data_type::s32);
                        CASE(data_type::s8, data_type::f32);
                        default: assert(!"unreachable");
                    }
                } else {
                    assert(!"unreachable");
                }

#undef CASE
            }
            return status::success;
        }

        format_tag_t dat_tag() const {
            return utils::pick(ndims() - 3, format_tag::nwc, format_tag::nhwc,
                    format_tag::ndhwc);
        }

        bool set_or_check_wei_format() {
            using namespace format_tag;
            using namespace memory_extra_flags;
            const auto zp = attr()->zero_points_;
            const int c_mask = 0x1,
                      g_mask = 0x3; // mask for i/o-channel and ngroups

            const bool is_src_s8 = src_md_.data_type == data_type::s8;
            const bool is_src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
            format_tag_t wei_tag;
            switch (isa) {
                case avx2:
                    wei_tag = with_groups()
                            ? utils::pick(ndims() - 3, gOIw2i8o4i, gOIhw2i8o4i,
                                    gOIdhw2i8o4i)
                            : utils::pick(ndims() - 3, OIw2i8o4i, OIhw2i8o4i,
                                    OIdhw2i8o4i);
                    break;
                case sse41:
                    wei_tag = with_groups() ? utils::pick(ndims() - 3, gOIw4o4i,
                                      gOIhw4o4i, gOIdhw4o4i)
                                            : utils::pick(ndims() - 3, OIw4o4i,
                                                    OIhw4o4i, OIdhw4o4i);
                    break;
                default: assert(!"Current ISA is not supported!"); break;
            }

            memory_desc_t want_wei_md = weights_md_;
            memory_desc_init_by_tag(want_wei_md, wei_tag);
            if (is_src_s8) {
                want_wei_md.extra.flags
                        = 0 | compensation_conv_s8s8 | scale_adjust;
                want_wei_md.extra.compensation_mask
                        = with_groups() ? g_mask : c_mask;
                want_wei_md.extra.scale_adjust = 0.5f;
            }
            if (is_src_zero_point) {
                want_wei_md.extra.flags |= compensation_conv_asymmetric_src;
                want_wei_md.extra.asymm_compensation_mask
                        = with_groups() ? g_mask : c_mask;
            }

            if (weights_md_.format_kind == format_kind::any) {
                weights_md_ = want_wei_md;
                return true;
            }

            return weights_md_ == want_wei_md;
        }

        status_t depthwise_po_init(engine_t *engine) {
            using namespace memory_tracking;
            auto &jcp_1x1 = jcp_;
            primitive_attr_t attr_1x1(*attr());
            if (!attr_1x1.is_initialized()) return status::out_of_memory;
            attr_1x1.set_scratchpad_mode(scratchpad_mode::user);

            const auto &src_md = dst_md_;
            const memory_desc_wrapper src_d(src_md);
            const auto nthr = dnnl_get_max_threads();
            auto l2_cache = platform::get_per_core_cache_size(2) * nthr;

            // Note: A robust fusion implementation would be to check if both
            // 1x1 conv and dw conv that are considered here for fusion are
            // optimal independently. This would require creating a new
            // primitive_desc through primitive_iterator & check if they match.
            // Due to concern that these creations and/or checks could be heavy,
            // for 1x1: Check that no better ISA is available.
            // for dw: Always fuse with same ISA.
            // Caveat: May be a better dw conv exists.

            bool ok = true && (!mayiuse(isa == avx2 ? avx512_core : avx2))
                    && (attr_1x1.post_ops_.find(primitive_kind::sum) == -1)
                    // TODO: Below may be further tuned.
                    && (l2_cache < src_d.size())
                    // load_grp_count check can be redundant due to l2 check
                    // above. Adding it explicitly as the current driver doesn't
                    // work if this condition fails.
                    && (jcp_1x1.load_grp_count < 2);
            if (!ok) return status::unimplemented;

            int dw_po_index
                    = attr_1x1.post_ops_.find(primitive_kind::convolution);

            convolution_desc_t cd_dw;
            primitive_attr_t attr_dw;
            CHECK(get_depthwise_conv_desc(
                    cd_dw, src_md, attr_1x1, attr_dw, dw_po_index));

            auto dw_dst_dt = cd_dw.dst_desc.data_type;

#define CASE(sdt, ddt) \
    case ddt: { \
        std::unique_ptr<dw_pd_t<sdt, ddt>> fusable_pd( \
                new dw_pd_t<sdt, ddt>(&cd_dw, &attr_dw, nullptr)); \
        CHECK(fusable_pd->init(engine)); \
        jcp_dw_ = &(fusable_pd->jcp_); \
        dw_conv_pd_ = std::move(fusable_pd); \
        break; \
    }
            if (jcp_1x1.dst_dt == data_type::u8) {
                switch (dw_dst_dt) {
                    CASE(data_type::u8, data_type::u8);
                    CASE(data_type::u8, data_type::s8);
                    CASE(data_type::u8, data_type::s32);
                    CASE(data_type::u8, data_type::f32);
                    default: return status::unimplemented;
                }
            } else if (jcp_1x1.dst_dt == data_type::s8) {
                switch (dw_dst_dt) {
                    CASE(data_type::s8, data_type::u8);
                    CASE(data_type::s8, data_type::s8);
                    CASE(data_type::s8, data_type::s32);
                    CASE(data_type::s8, data_type::f32);
                    default: return status::unimplemented;
                }
            } else
                return status::unimplemented;
#undef CASE

            ok = true
                    && (dnnl_memory_desc_equal(&src_md, dw_conv_pd_->src_md(0)))
                    && (jcp_1x1.oc_without_padding % jcp_1x1.oc_block == 0)
                    && IMPLICATION(jcp_dw_->ow_block,
                            jcp_dw_->ow_block == jcp_dw_->ow);
            if (!ok) return status::unimplemented;

            assert(jcp_dw_);
            assert(dw_conv_pd_->dst_md(0)->format_kind != format_kind::any);
            assert(dw_conv_pd_->weights_md(0)->format_kind != format_kind::any);
            assert(IMPLICATION(
                    dw_conv_pd_->weights_md(1)->data_type != data_type::undef,
                    dw_conv_pd_->weights_md(1)->format_kind
                            != format_kind::any));

            jcp_dw_->is_fused_conv = true;
            // TODO: Support/experiment arbitary oc_work in dw conv.
            // Until then we keep ch_work perfectly divisible.
            while (jcp_1x1.nb_load % jcp_1x1.nb_load_blocking != 0)
                --jcp_1x1.nb_load_blocking;
            jcp_1x1.nb_load_blocking_max = jcp_1x1.nb_load_blocking;

            while (jcp_1x1.nb_load_blocking % jcp_dw_->nb_ch_blocking != 0)
                --jcp_dw_->nb_ch_blocking;

            jcp_dw_->dw_conv_buffer_oc
                    = jcp_1x1.nb_load_blocking * jcp_1x1.oc_block;
            jcp_1x1.bcast_loop_output_step = jcp_1x1.ur
                    * (jcp_1x1.nb_load_blocking * jcp_1x1.oc_block)
                    * jcp_1x1.typesize_out;

            registrar_t scratchpad(scratchpad_registry_);
            registrar_t dw_scratchpad(scratchpad, names::prefix_fusion);

            size_t dw_conv_buffer_size_ = (size_t)nthr * jcp_dw_->kh
                    * jcp_dw_->iw * jcp_dw_->dw_conv_buffer_oc;
            assert(dw_conv_buffer_size_);
            dw_scratchpad.book(memory_tracking::names::key_fusion_inout_buffer,
                    dw_conv_buffer_size_,
                    types::data_type_size(dw_conv_pd_->src_md()->data_type));

            dw_conv_kernel_t::init_scratchpad(
                    dw_scratchpad, *jcp_dw_, *(dw_conv_pd_->attr()));

            return status::success;
        }
    };

    template <cpu_isa_t _isa, typename conv_t>
    friend status_t init_rtus_driver(conv_t *self);

    template <impl::data_type_t _sdt, impl::data_type_t _ddt>
    using fusable_pd_type =
            typename jit_uni_x8s8s32x_convolution_fwd_t<isa, _sdt, _ddt>::pd_t;

    jit_uni_x8s8s32x_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    // Note: In case of fused depthwise convolution, the final output datatype
    // after fusion may not be dst_data_t.
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_uni_x8s8s32x_1x1_conv_kernel<isa>(
                        pd()->jcp_, *pd()->attr())));
        CHECK(kernel_->create_kernel());

        if (pd()->jcp_.with_dw_conv) {
            CHECK(safe_ptr_assign(kernel_dw_,
                    new dw_conv_kernel_t(
                            *(pd()->jcp_dw_), *(pd()->dw_conv_pd_->attr()))));
            CHECK(kernel_dw_->create_kernel());
        }

        CHECK(init_rtus_driver<isa>(this));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights, const char *bias,
            const wei_data_t *weights_dw, const char *bias_dw, dst_data_t *dst,
            const int32_t *src_zero_point, const int32_t *dst_zero_point,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_x8s8s32x_1x1_conv_kernel<isa>> kernel_;
    std::unique_ptr<rtus_driver_t<isa>> rtus_driver_;
    using dw_conv_kernel_t = jit_uni_x8s8s32x_fwd_kernel<isa>;
    std::unique_ptr<dw_conv_kernel_t> kernel_dw_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
