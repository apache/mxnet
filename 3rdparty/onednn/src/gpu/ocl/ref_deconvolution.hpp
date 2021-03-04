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

#ifndef GPU_OCL_REF_DECONVOLUTION_HPP
#define GPU_OCL_REF_DECONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_deconvolution_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t weights_axes_permutation(
        memory_desc_t *o_md, const memory_desc_t *i_md, bool with_groups) {
    int perm[DNNL_MAX_NDIMS] {}; // deconv to conv weight permutation
    for (int d = 0; d < DNNL_MAX_NDIMS; ++d)
        perm[d] = d;
    nstl::swap(perm[0 + with_groups], perm[1 + with_groups]);

    return dnnl_memory_desc_permute_axes(o_md, i_md, perm);
}

static status_t conv_descr_create(
        const deconvolution_desc_t *dd, convolution_desc_t *cd) {
    using namespace prop_kind;
    alg_kind_t alg_kind = alg_kind::convolution_direct;

    const memory_desc_t *src_md, *dst_md, *d_weights_d;
    prop_kind_t prop_kind;

    switch (dd->prop_kind) {
        case forward:
        case forward_inference:
            prop_kind = backward_data;
            src_md = &dd->dst_desc;
            dst_md = &dd->src_desc;
            d_weights_d = &dd->weights_desc;
            break;
        case backward_data:
            prop_kind = forward_training;
            src_md = &dd->diff_dst_desc;
            dst_md = &dd->diff_src_desc;
            d_weights_d = &dd->weights_desc;
            break;
        case backward_weights:
            prop_kind = dd->prop_kind;
            src_md = &dd->diff_dst_desc;
            dst_md = &dd->src_desc;
            d_weights_d = &dd->diff_weights_desc;
            break;
        default: assert(!"unknown prop kind"); return status::invalid_arguments;
    }

    // Create weights desc for convolution
    memory_desc_t c_weights_d;
    const bool with_groups = d_weights_d->ndims == src_md->ndims + 1;
    CHECK(weights_axes_permutation(&c_weights_d, d_weights_d, with_groups));

    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &c_weights_d,
            prop_kind != backward_weights ? &dd->bias_desc : nullptr, dst_md,
            dd->strides, dd->dilates, dd->padding[0], dd->padding[1]);
}

struct ref_deconvolution_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_deconvolution_fwd_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : gpu_deconvolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : gpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_fwd_t);

        status_t init_convolution(engine_t *engine) {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            conv_attr.set_scratchpad_mode(scratchpad_mode::user);
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            ++it;
            conv_pd_.reset(it.fetch_once());
            return status::success;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = is_fwd()
                    && desc()->alg_kind == alg_kind::deconvolution_direct
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_with_binary_ok(
                            attr(), desc()->dst_desc.data_type)
                    && (utils::everyone_is(data_type::f32,
                                desc()->src_desc.data_type,
                                desc()->weights_desc.data_type,
                                desc()->dst_desc.data_type)
                            || utils::everyone_is(data_type::f16,
                                    desc()->src_desc.data_type,
                                    desc()->weights_desc.data_type,
                                    desc()->dst_desc.data_type)
                            || (utils::everyone_is(data_type::bf16,
                                        desc()->src_desc.data_type,
                                        desc()->weights_desc.data_type)
                                    && utils::one_of(desc()->dst_desc.data_type,
                                            data_type::f32, data_type::bf16))
                            || (desc()->weights_desc.data_type == data_type::s8
                                    && utils::one_of(desc()->src_desc.data_type,
                                            data_type::u8, data_type::s8)
                                    && utils::one_of(desc()->dst_desc.data_type,
                                            data_type::u8, data_type::s8,
                                            data_type::s32, data_type::f32)));
            if (ok) {
                CHECK(init_convolution(engine));
                if (weights_md_.format_kind == format_kind::any)
                    CHECK(weights_axes_permutation(&weights_md_,
                            conv_pd_->weights_md(), with_groups()));
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (dst_md_.format_kind == format_kind::any)
                    dst_md_ = *conv_pd_->diff_src_md();
                if (bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(bias_md_, x));
                init_scratchpad();

                return status::success;
            }

            return status::unimplemented;
        }

        std::unique_ptr<primitive_desc_t> conv_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    ref_deconvolution_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t conv_status
                = pd()->conv_pd_->create_primitive(conv_p_, engine);
        return conv_status;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        conv_args[DNNL_ARG_DIFF_SRC] = args.at(DNNL_ARG_DST);
        if (pd()->with_bias())
            conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);

        for (int idx = 0; idx < pd()->attr()->post_ops_.len(); ++idx) {
            if (pd()->attr()->post_ops_.entry_[idx].is_binary()) {
                conv_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1]
                        = args.at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | DNNL_ARG_SRC_1);
            }
        }
        exec_ctx_t conv_ctx(ctx, std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());
        // Executing the convolution kernel
        return conv_p_->execute(conv_ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {conv_p_.get()};
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
};

struct ref_deconvolution_bwd_data_t : public gpu_primitive_t {
    struct pd_t : public gpu_deconvolution_bwd_data_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : gpu_deconvolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : gpu_deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_data_t);

        status_t init_convolution(engine_t *engine) {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            conv_attr.set_scratchpad_mode(scratchpad_mode::user);
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            ++it;
            conv_pd_.reset(it.fetch_once());
            return status::success;
        }

        status_t init(engine_t *engine) {
            bool ok = desc()->prop_kind == prop_kind::backward_data
                    && (utils::everyone_is(data_type::f32,
                                desc()->diff_src_desc.data_type,
                                desc()->weights_desc.data_type,
                                desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->weights_desc.data_type,
                                    desc()->diff_dst_desc.data_type))
                    && utils::one_of(desc()->diff_src_desc.data_type,
                            data_type::bf16, data_type::f32)
                    && desc()->alg_kind == alg_kind::deconvolution_direct
                    && attr()->has_default_values();

            if (ok) {
                CHECK(init_convolution(engine));
                if (weights_md_.format_kind == format_kind::any)
                    CHECK(weights_axes_permutation(&weights_md_,
                            conv_pd_->weights_md(), with_groups()));
                if (diff_src_md_.format_kind == format_kind::any)
                    diff_src_md_ = *conv_pd_->dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                init_scratchpad();

                return status::success;
            }

            return status::unimplemented;
        }

        std::unique_ptr<primitive_desc_t> conv_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    ref_deconvolution_bwd_data_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t conv_status
                = pd()->conv_pd_->create_primitive(conv_p_, engine);
        return conv_status;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        conv_args[DNNL_ARG_DST] = args.at(DNNL_ARG_DIFF_SRC);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t conv_ctx(ctx, std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());
        // Executing the convolution kernel
        return conv_p_->execute(conv_ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {conv_p_.get()};
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
};

struct ref_deconvolution_bwd_weights_t : public gpu_primitive_t {
    struct pd_t : public gpu_deconvolution_bwd_weights_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : gpu_deconvolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : gpu_deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_weights_t);

        status_t init_convolution(engine_t *engine) {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            conv_attr.set_scratchpad_mode(scratchpad_mode::user);
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            ++it;
            conv_pd_.reset(it.fetch_once());
            return status::success;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;
            bool ok = desc()->prop_kind == prop_kind::backward_weights
                    && (utils::everyone_is(data_type::f32,
                                desc()->src_desc.data_type,
                                desc()->diff_weights_desc.data_type,
                                desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->diff_dst_desc.data_type,
                                    desc()->src_desc.data_type))
                    && utils::one_of(
                            desc()->alg_kind, alg_kind::deconvolution_direct)
                    && attr()->has_default_values()
                    && utils::one_of(desc()->diff_weights_desc.data_type,
                            data_type::bf16, data_type::f32);
            if (ok) {
                CHECK(init_convolution(engine));
                if (diff_weights_md_.format_kind == format_kind::any)
                    CHECK(weights_axes_permutation(&diff_weights_md_,
                            conv_pd_->diff_weights_md(), with_groups()));
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                if (diff_bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(diff_bias_md_, x));
                init_scratchpad();

                return status::success;
            }

            return status::unimplemented;
        }

        std::unique_ptr<primitive_desc_t> conv_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    ref_deconvolution_bwd_weights_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        // Creating convolution primitve
        status_t conv_status
                = pd()->conv_pd_->create_primitive(conv_p_, engine);
        if (conv_status != status::success) return conv_status;

        if (!pd()->with_bias()) return conv_status;
        // Initializing values for the deconv bias kernel
        compute::kernel_ctx_t kernel_ctx;

        memory_desc_wrapper diff_dst_mdw(pd()->diff_dst_md());
        kernel_ctx.set_data_type(pd()->diff_dst_md()->data_type);
        offsets_t off;
        set_offsets(diff_dst_mdw, off.dst_off);
        def_offsets(off.dst_off, kernel_ctx, "DST",
                pd()->desc()->diff_dst_desc.ndims);

        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("OH", pd()->OH());
        kernel_ctx.define_int("OW", pd()->OW());
        kernel_ctx.define_int("OD", pd()->OD());
        kernel_ctx.define_int("OC", pd()->OC() / pd()->G());
        kernel_ctx.define_int("NDIMS", pd()->desc()->src_desc.ndims);

        gws[0] = pd()->OC() * pd()->G();
        gws[1] = 1;
        gws[2] = 1;

        dst_data_type = pd()->diff_dst_md()->data_type;
        bias_data_type = pd()->diff_weights_md(1)->data_type;
        accum_data_type = pd()->desc()->accum_data_type;

        def_data_type(kernel_ctx, dst_data_type, "DST");
        def_data_type(kernel_ctx, bias_data_type, "BIA");
        def_data_type(kernel_ctx, accum_data_type, "ACC");

        create_kernel(
                engine, &bias_kernel_, "ref_deconv_backward_bias", kernel_ctx);
        if (!bias_kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;

        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        conv_args[DNNL_ARG_DIFF_WEIGHTS] = args.at(DNNL_ARG_DIFF_WEIGHTS);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t conv_ctx(ctx, std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());

        status_t status = conv_p_->execute(conv_ctx);
        if (status != status::success) return status;

        if (pd()->with_bias()) {
            // Calling the bias kernel if bias=1
            auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);
            auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);

            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, diff_dst);
            arg_list.set(1, diff_bias);

            // Setting up global work-space to {OC*G, 1, 1}
            auto nd_range = compute::nd_range_t({gws[0], gws[1], gws[2]});
            status = parallel_for(ctx, nd_range, bias_kernel_, arg_list);
        }
        return status::success;
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {conv_p_.get()};
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
    compute::kernel_t bias_kernel_;
    size_t gws[3];
    data_type_t dst_data_type = data_type::undef;
    data_type_t bias_data_type = data_type::undef;
    data_type_t accum_data_type = data_type::undef;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
