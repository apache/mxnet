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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/ref_deconvolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
dim_t get_data_off(const memory_desc_wrapper &mdw, int ndims, dim_t mb, dim_t c,
        dim_t id, dim_t ih, dim_t iw) {
    switch (ndims) {
        case 5: return mdw.off(mb, c, id, ih, iw);
        case 4: return mdw.off(mb, c, ih, iw);
        case 3: return mdw.off(mb, c, iw);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }
}

} // namespace

template <data_type_t dst_type>
void ref_deconvolution_fwd_t::compute_fwd_bias_common(const exec_ctx_t &ctx,
        typename prec_traits<dst_type>::type *dst,
        const float *conv_output) const {
    using dst_data_t = typename prec_traits<dst_type>::type;
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto OD = pd()->OD();
    const auto OC = pd()->OC() / G;
    const auto ndims = pd()->desc()->src_desc.ndims;

    parallel_nd(MB, G, OC, OD, OH, OW,
            [&](dim_t mb, dim_t g, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
                const dim_t c = g * OC + oc;
                const dim_t off = get_data_off(dst_d, ndims, mb, c, od, oh, ow);
                float b = types::get_float_value(bias_d.data_type(), bias, c);
                float d = conv_output[off];
                dst[off] = cpu::saturate_and_round<dst_data_t>(d + b);
            });
}

template <data_type_t dst_type>
void ref_deconvolution_fwd_t::compute_fwd_bias_ncdhw(const exec_ctx_t &ctx,
        typename prec_traits<dst_type>::type *dst,
        const float *conv_output) const {
    using dst_data_t = typename prec_traits<dst_type>::type;
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto SP = pd()->OW() * pd()->OH() * pd()->OD();

    parallel_nd(MB, OC, [&](dim_t mb, dim_t oc) {
        const dim_t off = (mb * OC + oc) * SP;
        float b = types::get_float_value(bias_d.data_type(), bias, oc);
        PRAGMA_OMP_SIMD()
        for (dim_t sp = 0; sp < SP; ++sp) {
            float d = conv_output[off + sp];
            dst[off + sp] = cpu::saturate_and_round<dst_data_t>(d + b);
        }
    });
}

template <data_type_t dst_type>
void ref_deconvolution_fwd_t::compute_fwd_bias_ndhwc(const exec_ctx_t &ctx,
        typename prec_traits<dst_type>::type *dst,
        const float *conv_output) const {
    using dst_data_t = typename prec_traits<dst_type>::type;
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto SP = pd()->OW() * pd()->OH() * pd()->OD();

    parallel_nd(MB, SP, [&](dim_t mb, dim_t sp) {
        const dim_t off = (mb * SP + sp) * OC;
        PRAGMA_OMP_SIMD()
        for (dim_t oc = 0; oc < OC; ++oc) {
            float b = types::get_float_value(bias_d.data_type(), bias, oc);
            float d = conv_output[off + oc];
            dst[off + oc] = cpu::saturate_and_round<dst_data_t>(d + b);
        }
    });
}

template <data_type_t dst_type, dim_t blk_size>
void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc(const exec_ctx_t &ctx,
        typename prec_traits<dst_type>::type *dst,
        const float *conv_output) const {
    using dst_data_t = typename prec_traits<dst_type>::type;
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto SP = pd()->OW() * pd()->OH() * pd()->OD();
    const auto stride_mb = dst_d.blocking_desc().strides[0];

    parallel_nd(MB, utils::div_up(OC, blk_size), SP,
            [&](dim_t mb, dim_t oc_blk, dim_t sp) {
                const dim_t oc = oc_blk * blk_size;
                const dim_t off = mb * stride_mb + oc * SP + sp * blk_size;
                const dim_t blk = nstl::min(blk_size, OC - oc);

                PRAGMA_OMP_SIMD()
                for (dim_t i = 0; i < blk; ++i) {
                    float b = types::get_float_value(
                            bias_d.data_type(), bias, oc + i);
                    float d = conv_output[off + i];
                    dst[off + i] = cpu::saturate_and_round<dst_data_t>(d + b);
                }
            });
}

template <data_type_t dst_type>
void ref_deconvolution_fwd_t::compute_fwd_bias(const exec_ctx_t &ctx,
        typename prec_traits<dst_type>::type *dst,
        const float *conv_output) const {
    using namespace format_tag;
    switch (pd()->dst_tag_) {
        case ncdhw:
        case nchw:
        case ncw:
            compute_fwd_bias_ncdhw<dst_type>(ctx, dst, conv_output);
            break;
        case ndhwc:
        case nhwc:
        case nwc:
            compute_fwd_bias_ndhwc<dst_type>(ctx, dst, conv_output);
            break;
        case nCdhw8c:
        case nChw8c:
        case nCw8c:
            assert(dst_type != data_type::bf16);
            compute_fwd_bias_nCdhwXc<dst_type, 8>(ctx, dst, conv_output);
            break;
        case nCdhw16c:
        case nChw16c:
        case nCw16c:
            compute_fwd_bias_nCdhwXc<dst_type, 16>(ctx, dst, conv_output);
            break;
        default:
            compute_fwd_bias_common<dst_type>(ctx, dst, conv_output);
            break;
    }
}

template <data_type_t dst_type>
void ref_deconvolution_fwd_t::compute_ref_attrs(const exec_ctx_t &ctx,
        const float *conv_output,
        typename prec_traits<dst_type>::type *original_dst) const {
    using dst_data_t = typename prec_traits<dst_type>::type;
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto OD = pd()->OD();
    const auto OC = pd()->OC() / G;
    const auto ndims = pd()->desc()->src_desc.ndims;

    auto maybe_oscale = [=](float &d, dim_t g, dim_t oc) {
        // scale_idx_mult = 1 for per_oc scales and 0, otherwise
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        d *= scales[(g * OC + oc) * scale_idx_mult];
    };

    parallel_nd(MB, G, OC, OD, OH, OW,
            [&](dim_t mb, dim_t g, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
                auto dst_off = get_data_off(
                        dst_d, ndims, mb, g * OC + oc, od, oh, ow);
                dim_t dst_l_off = (mb * OC * G + g * OC + oc) * OD * OH * OW
                        + od * OH * OW + oh * OW + ow;

                float a = conv_output[dst_off];
                maybe_oscale(a, g, oc);

                ref_post_ops_t::args_t args;
                if (pd()->attr()->post_ops_.find(primitive_kind::sum) != -1)
                    args.dst_val = (float)original_dst[dst_off];
                args.ctx = &ctx;
                args.l_offset = dst_l_off;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(a, args);

                dst[dst_off] = cpu::saturate_and_round<dst_data_t>(a);
            });
}

status_t ref_deconvolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    const auto scratchpad = ctx.get_scratchpad_grantor();
    const bool ref_bias = pd()->with_bias() && !pd()->conv_supports_bias_;
    const bool non_default_attr = !pd()->attr()->has_default_values();

    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
    if (pd()->with_bias() && pd()->conv_supports_bias_)
        conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);

    // Create intermediate memory for f32 output if needed.
    auto dst = args.at(DNNL_ARG_DST);
    memory_t tmp_memory(dst.mem->engine(), pd()->conv_pd_->diff_src_md(),
            scratchpad.get_memory_storage(key_deconv_bias), false);
    memory_arg_t tmp_conv_output = {&tmp_memory, false};

    conv_args[DNNL_ARG_DIFF_SRC]
            = ref_bias || non_default_attr ? tmp_conv_output : dst;

    // When sum post-op happens, we need to copy original destination memory
    // prior call to external convolution happens.
    if (pd()->attr()->post_ops_.find(primitive_kind::sum) != -1) {
        void *original_dst = scratchpad.get(key_deconv_sum);
        const memory_desc_wrapper dst_d(pd()->dst_md());
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        const auto dt_size = dst_d.data_type_size();

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start {0}, end {0};
            balance211(dst_d.nelems(true), nthr, ithr, start, end);
            auto o_dst_start = (char *)original_dst + start * dt_size;
            auto dst_start = (char *)dst + start * dt_size;
            const auto size = (end - start) * dt_size;

            std::memcpy(o_dst_start, dst_start, size);
        });
    }

    exec_ctx_t conv_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(ctx, key_nested, conv_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());
    auto status = conv_p_->execute(conv_ctx);
    if (status != status::success) return status;

    using namespace data_type;
    auto dst_type = pd()->dst_md()->data_type;

    if (ref_bias) {
        float *conv_output = scratchpad.get<float>(key_deconv_bias);
        if (non_default_attr) {
            // Overwrite conv_output since further attr computations still need
            // f32 output.
            compute_fwd_bias<f32>(ctx, conv_output, conv_output);
        } else {

#define CASE(DT) \
    case (DT): { \
        using dst_data_t = typename prec_traits<DT>::type; \
        dst_data_t *dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST); \
        compute_fwd_bias<DT>(ctx, dst, conv_output); \
    } break

            switch (dst_type) {
                CASE(f32);
                CASE(bf16);
                CASE(s32);
                CASE(s8);
                CASE(u8);
                default: assert(!"unsupported data type");
            }
#undef CASE
        }
    }

    if (non_default_attr) {
        float *conv_output = scratchpad.get<float>(key_deconv_bias);

#define CASE(DT) \
    case (DT): { \
        using dst_data_t = typename prec_traits<DT>::type; \
        dst_data_t *original_dst = scratchpad.get<dst_data_t>(key_deconv_sum); \
        compute_ref_attrs<DT>(ctx, conv_output, original_dst); \
    } break

        switch (dst_type) {
            CASE(f32);
            CASE(bf16);
            CASE(s32);
            CASE(s8);
            CASE(u8);
            default: assert(!"unsupported data type");
        }
#undef CASE
    }

    return status::success;
}

status_t ref_deconvolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
    conv_args[DNNL_ARG_DST] = args.at(DNNL_ARG_DIFF_SRC);
    exec_ctx_t conv_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(ctx, key_nested, conv_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());
    conv_p_->execute(conv_ctx);
    return status::success;
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias(
        float *diff_bias, const float *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto OC = pd()->OC() / G;
    const auto OD = pd()->OD();
    const auto ndims = pd()->desc()->src_desc.ndims;

    parallel_nd(G, OC, [&](dim_t g, dim_t oc) {
        float db = 0;
        for_(dim_t mb = 0; mb < MB; ++mb)
        for_(dim_t od = 0; od < OD; ++od)
        for_(dim_t oh = 0; oh < OH; ++oh)
        for (dim_t ow = 0; ow < OW; ++ow) {
            const auto d_dst_off = get_data_off(
                    diff_dst_d, ndims, mb, g * OC + oc, od, oh, ow);
            db += diff_dst[d_dst_off];
        }
        diff_bias[g * OC + oc] = db;
    });
}

template <data_type_t dbia_type, data_type_t ddst_type>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_ncdhw(
        typename prec_traits<dbia_type>::type *diff_bias,
        const typename prec_traits<ddst_type>::type *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto OC = pd()->OC();
    const auto MB = pd()->MB();
    const auto SP = pd()->OH() * pd()->OW() * pd()->OD();

    parallel_nd(OC, [&](dim_t oc) {
        float db = 0;
        for (dim_t mb = 0; mb < MB; ++mb) {
            PRAGMA_OMP_SIMD(reduction(+ : db))
            for (dim_t sp = 0; sp < SP; ++sp) {
                auto offset = (size_t)(mb * OC + oc) * SP + sp;
                db += diff_dst[offset];
            }
        }
        diff_bias[oc] = db;
    });
}

template <data_type_t dbia_type, data_type_t ddst_type>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_ndhwc(
        typename prec_traits<dbia_type>::type *diff_bias,
        const typename prec_traits<ddst_type>::type *diff_dst) const {
    const auto MB = pd()->MB();
    const auto SP = pd()->OW() * pd()->OH() * pd()->OD();
    const auto OC = pd()->OC();

    parallel_nd(OC, [&](dim_t oc) {
        float db = 0;
        for (dim_t mb = 0; mb < MB; ++mb) {
            PRAGMA_OMP_SIMD(reduction(+ : db))
            for (dim_t sp = 0; sp < SP; ++sp) {
                const dim_t offset = (mb * SP + sp) * OC + oc;
                db += diff_dst[offset];
            }
        }
        diff_bias[oc] = static_cast<typename prec_traits<dbia_type>::type>(db);
    });
}

template <data_type_t dbia_type, data_type_t ddst_type, dim_t blksize>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc(
        typename prec_traits<dbia_type>::type *diff_bias,
        const typename prec_traits<ddst_type>::type *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto OC = pd()->OC();
    const auto MB = pd()->MB();
    const auto SP = pd()->OH() * pd()->OW() * pd()->OD();

    const ptrdiff_t stride_mb = diff_dst_d.blocking_desc().strides[0];

    parallel_nd(utils::div_up(OC, blksize), [&](dim_t ocb) {
        float db[blksize] = {0};

        for (dim_t mb = 0; mb < MB; ++mb) {
            for (dim_t sp = 0; sp < SP; ++sp) {
                auto offset = mb * stride_mb + (ocb * SP + sp) * blksize;

                PRAGMA_OMP_SIMD()
                for (dim_t i = 0; i < blksize; ++i)
                    db[i] += diff_dst[offset + i];
            }
        }

        const dim_t blk = nstl::min(blksize, OC - ocb * blksize);

        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < blk; ++i)
            diff_bias[ocb * blksize + i] = db[i];
    });
}

template <data_type_t dbia_type, data_type_t ddst_type>
void ref_deconvolution_bwd_weights_t::compute_bias(
        const exec_ctx_t &ctx) const {
    using dbia_data_t = typename prec_traits<dbia_type>::type;
    using ddst_data_t = typename prec_traits<ddst_type>::type;

    auto diff_bias = CTX_OUT_MEM(dbia_data_t *, DNNL_ARG_DIFF_BIAS);
    auto diff_dst = CTX_IN_MEM(const ddst_data_t *, DNNL_ARG_DIFF_DST);

    using namespace format_tag;
    switch (pd()->dst_tag_) {
        case ncdhw:
        case nchw:
        case ncw:
            compute_bwd_bias_ncdhw<dbia_type, ddst_type>(diff_bias, diff_dst);
            break;
        case ndhwc:
        case nhwc:
        case nwc:
            compute_bwd_bias_ndhwc<dbia_type, ddst_type>(diff_bias, diff_dst);
            break;
        case nCdhw8c:
        case nChw8c:
        case nCw8c:
            assert(!utils::one_of(data_type::bf16, dbia_type, ddst_type));
            compute_bwd_bias_nCdhwXc<dbia_type, ddst_type, 8>(
                    diff_bias, diff_dst);
            break;
        case nCdhw16c:
        case nChw16c:
        case nCw16c:
            compute_bwd_bias_nCdhwXc<dbia_type, ddst_type, 16>(
                    diff_bias, diff_dst);
            break;
        default:
            assert(!utils::one_of(data_type::bf16, dbia_type, ddst_type));
            compute_bwd_bias((float *)diff_bias, (const float *)diff_dst);
            break;
    }
}

status_t ref_deconvolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
    conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
    conv_args[DNNL_ARG_DIFF_WEIGHTS] = args.at(DNNL_ARG_DIFF_WEIGHTS);
    exec_ctx_t conv_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(ctx, key_nested, conv_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());
    status_t status = conv_p_->execute(conv_ctx);
    if (status != status::success) return status;

    if (pd()->with_bias()) {
        using namespace data_type;

        auto dbia_type = pd()->diff_weights_md(1)->data_type;
        auto ddst_type = pd()->diff_dst_md()->data_type;
        if (utils::everyone_is(f32, dbia_type, ddst_type))
            compute_bias<f32, f32>(ctx);
        else if (utils::everyone_is(bf16, dbia_type, ddst_type))
            compute_bias<bf16, bf16>(ctx);
        else if (dbia_type == f32 && ddst_type == bf16)
            compute_bias<f32, bf16>(ctx);
        else {
            assert(!"unsupported data type");
            return status::runtime_error;
        }
    }
    return status::success;
}

using namespace data_type;

template void ref_deconvolution_bwd_weights_t::compute_bias<f32, f32>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_bwd_weights_t::compute_bias<f32, bf16>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_bwd_weights_t::compute_bias<bf16, bf16>(
        const exec_ctx_t &ctx) const;
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
