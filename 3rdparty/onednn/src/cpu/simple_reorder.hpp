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

#ifndef CPU_SIMPLE_REORDER_HPP
#define CPU_SIMPLE_REORDER_HPP

#include <assert.h>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/cpu_reorder_pd.hpp"

#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using bd = block_dim_t;
using ib = inner_blk_t;

template <impl::data_type_t type>
using data_t = typename prec_traits<type>::type;

template <impl::data_type_t type_i, impl::data_type_t type_o>
using _qz_a1b0 = qz_a1b0<data_t<type_i>, data_t<type_o>>;

template <impl::data_type_t type_i, impl::data_type_t type_o>
using _qz = qz<data_t<type_i>, data_t<type_o>>;

namespace fmt_order {
const bool keep = true;
const bool reverse = false;
const bool any = keep;
} // namespace fmt_order

namespace spec {
struct direct_copy {};
struct direct_copy_except_dim_0 {};
struct reference {};
struct conv_req_comp {}; // {s8, u8: asymmetric quantization}
} // namespace spec

#define SIMPLE_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, impl::format_tag_t tag_i, \
            impl::data_type_t type_o, impl::format_tag_t tag_o, \
            bool order_keep
#define SIMPLE_REORDER_TEMPL_CALL type_i, tag_i, type_o, tag_o, order_keep

#define DECLARE_COMMON_PARAMS() \
    auto input = CTX_IN_MEM(const data_t<type_i> *, DNNL_ARG_FROM); \
    auto output = CTX_OUT_MEM(data_t<type_o> *, DNNL_ARG_TO); \
    const auto &scratchpad = ctx.get_scratchpad_grantor(); \
    MAYBE_UNUSED(scratchpad); \
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd->src_md()); \
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd->dst_md()); \
    const float alpha = pd->alpha(); \
    MAYBE_UNUSED(alpha); \
    const float beta = pd->beta(); \
    MAYBE_UNUSED(beta);

#define GET_SCRATCHPAD_SIZE_ZERO() \
    static size_t get_scratchpad_size(const memory_desc_wrapper &input_d, \
            const memory_desc_wrapper &output_d) { \
        return 0; \
    }

/* specific reorders: common template */
template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_impl {};

namespace {
bool simple_fmt_check(bool order_keep, impl::format_tag_t tag_i,
        impl::format_tag_t tag_o, const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d) {
    if (input_d.has_runtime_dims_or_strides()) return false;
    return input_d.matches_tag(order_keep ? tag_i : tag_o)
            && output_d.matches_tag(order_keep ? tag_o : tag_i);
}
bool simple_po_check(const primitive_attr_t *attr) {
    const auto &po = attr->post_ops_;
    return po.len() == 0
            || (po.len() == 1 && po.contain(primitive_kind::sum, 0));
}
bool simple_attr_check(const primitive_attr_t *attr, bool many_scales_support,
        bool sum_support) {
    using smask_t = primitive_attr_t::skip_mask_t;
    smask_t skip_mask = smask_t::oscale;
    if (sum_support) skip_mask = skip_mask | smask_t::post_ops;
    if (!attr->has_default_values(skip_mask)) return false;
    if (!attr->defined()) return false;
    if (sum_support) simple_po_check(attr);
    if (many_scales_support) return true;
    return attr->output_scales_.mask_ == 0;
}
} // namespace

/* specific reorders: implementation */
template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && utils::one_of(tag_o, format_tag::wio,
                                format_tag::wigo, format_tag::hwio,
                                format_tag::hwigo, format_tag::dhwio,
                                format_tag::dhwigo),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        const size_t D_mask = array_product(
                input_d.dims(), math::ilog2q(attr->output_scales_.mask_ + 1));
        static constexpr bool w_groups = one_of(
                tag_o, format_tag::wigo, format_tag::hwigo, format_tag::dhwigo);
        const int oc_idx = w_groups ? 1 : 0;
        const int oc = input_d.dims()[oc_idx];
        const int g = w_groups ? (input_d.dims()[0]) : 1;

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        auto mask_ok = [&](bool check, int mask) {
            return IMPLICATION(check, mask == (w_groups ? 0x3 : 0x1));
        };

        return simple_attr_check(attr, true, false)
                && output_d.matches_tag(tag_o) && input_d.is_plain()
                && (req_comp || req_asymmetric_comp)
                && mask_ok(req_comp, output_d.extra().compensation_mask)
                && mask_ok(req_asymmetric_comp,
                        output_d.extra().asymm_compensation_mask)
                && IMPLICATION(
                        req_comp, one_of(D_mask, (size_t)1, (size_t)g * oc))
                && one_of(input_d.data_type(), f32, s8, bf16)
                && output_d.data_type() == s8;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        static constexpr bool w_groups = utils::one_of(
                tag_o, format_tag::wigo, format_tag::hwigo, format_tag::dhwigo);
        static constexpr bool w_height
                = !utils::one_of(tag_o, format_tag::wio, format_tag::wigo);
        static constexpr bool w_depth
                = utils::one_of(tag_o, format_tag::dhwio, format_tag::dhwigo);

        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();

        const int G = w_groups ? dims[0] : 1;
        const int OC = dims[w_groups + 0];
        const int IC = dims[w_groups + 1];
        const int D = w_depth ? dims[w_groups + 2] : 1;
        const int H = w_height ? dims[w_groups + w_depth + 2] : 1;
        const int W = dims[w_groups + w_depth + w_height + 2];

        const float *scales = pd->attr()->output_scales_.scales_;
        const size_t D_mask = utils::array_product(input_d.dims(),
                math::ilog2q(pd->attr()->output_scales_.mask_ + 1));
        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        assert(req_comp || has_asymmetric_comp);

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        size_t offset
                = G * pdims[w_groups + 0] * pdims[w_groups + 1] * D * H * W;
        size_t zp_offset = offset
                + (req_comp ? G * pdims[w_groups + 0] * sizeof(int32_t) : 0);
        int32_t *cp = req_comp ? reinterpret_cast<int32_t *>(output + offset)
                               : nullptr;
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + zp_offset)
                : nullptr;

        parallel_nd(G, OC, [&](int g, int oc) {
            if (req_comp) cp[g * OC + oc] = 0;
            if (has_asymmetric_comp) zp[g * OC + oc] = 0;
            for_(int ic = 0; ic < IC; ic++)
            for_(int d = 0; d < D; d++)
            for_(int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                auto i = w_depth
                        ? input[input_d.blk_off<!w_groups>(g, oc, ic, d, h, w)]
                        : w_height ? input[input_d.blk_off<!w_groups>(
                                  g, oc, ic, h, w)]
                                   : input[input_d.blk_off<!w_groups>(
                                           g, oc, ic, w)];
                auto &o = w_depth
                        ? output[output_d.blk_off<!w_groups>(
                                g, oc, ic, d, h, w)]
                        : w_height ? output[output_d.blk_off<!w_groups>(
                                  g, oc, ic, h, w)]
                                   : output[output_d.blk_off<!w_groups>(
                                           g, oc, ic, w)];
                const float s = scales[(D_mask == 1) ? 0 : g * OC + oc];

                o = qz_b0<data_t<type_i>, data_t<type_o>>()(i, s * adj_scale);
                if (req_comp) cp[g * OC + oc] -= (int32_t)o;
                if (has_asymmetric_comp) zp[g * OC + oc] -= (int32_t)o;
            }
            if (req_comp) cp[g * OC + oc] *= 128;
        });
        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<
                (utils::one_of(tag_i, format_tag::oiw, format_tag::wio)
                        && utils::one_of(tag_o, format_tag::OIw4i16o4i,
                                format_tag::OIw2i8o4i, format_tag::OIw4o4i))
                        || (utils::one_of(
                                    tag_i, format_tag::goiw, format_tag::wigo)
                                && utils::one_of(tag_o, format_tag::gOIw4i16o4i,
                                        format_tag::gOIw2i8o4i,
                                        format_tag::gOIw4o4i))
                        || (utils::one_of(
                                    tag_i, format_tag::hwio, format_tag::oihw)
                                && utils::one_of(tag_o, format_tag::OIhw4i16o4i,
                                        format_tag::OIhw2i8o4i,
                                        format_tag::OIhw4o4i))
                        || (utils::one_of(
                                    tag_i, format_tag::dhwio, format_tag::oidhw)
                                && utils::one_of(tag_o,
                                        format_tag::OIdhw4i16o4i,
                                        format_tag::OIdhw2i8o4i,
                                        format_tag::OIdhw4o4i))
                        || (utils::one_of(
                                    tag_i, format_tag::goihw, format_tag::hwigo)
                                && utils::one_of(tag_o, format_tag::gOIhw4o4i,
                                        format_tag::gOIhw2i8o4i,
                                        format_tag::gOIhw4i16o4i))
                        || (utils::one_of(tag_i, format_tag::goidhw)
                                && (utils::one_of(tag_o,
                                        format_tag::gOIdhw4i16o4i,
                                        format_tag::gOIdhw2i8o4i,
                                        format_tag::gOIdhw4o4i))),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace format_tag;
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        const size_t D_mask = array_product(
                input_d.dims(), math::ilog2q(attr->output_scales_.mask_ + 1));
        const bool w_groups = !one_of(tag_o, OIw4i16o4i, OIw2i8o4i, OIw4o4i,
                OIhw4i16o4i, OIhw2i8o4i, OIhw4o4i, OIdhw4i16o4i, OIdhw2i8o4i,
                OIdhw4o4i);
        const int oc = (input_d.dims()[w_groups ? 1 : 0]);
        const int g = w_groups ? input_d.dims()[0] : 1;

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        auto mask_ok = [&](bool check, int mask) {
            return IMPLICATION(check, mask == (w_groups ? 0x3 : 0x1));
        };

        return simple_attr_check(attr, true, false)
                && input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && (req_comp || req_asymmetric_comp)
                && mask_ok(req_comp, output_d.extra().compensation_mask)
                && mask_ok(req_asymmetric_comp,
                        output_d.extra().asymm_compensation_mask)
                && IMPLICATION(
                        req_comp, one_of(D_mask, (size_t)1, (size_t)g * oc))
                && one_of(input_d.data_type(), f32, s8, bf16)
                && output_d.data_type() == s8;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        static constexpr bool w_groups = !utils::one_of(tag_o, OIw4o4i,
                OIw4i16o4i, OIhw4i16o4i, OIdhw4i16o4i, OIhw4o4i, OIw2i8o4i,
                OIhw2i8o4i, OIdhw2i8o4i, OIdhw4o4i);

        constexpr int is_1d = utils::one_of(tag_o, gOIw4i16o4i, OIw4i16o4i,
                gOIw2i8o4i, OIw2i8o4i, gOIw4o4i, OIw4o4i);
        constexpr int is_3d = utils::one_of(tag_o, gOIdhw4i16o4i, OIdhw4i16o4i,
                gOIdhw2i8o4i, OIdhw2i8o4i, gOIdhw4o4i, OIdhw4o4i);
        constexpr int blksize = utils::one_of(tag_traits<tag_o>::inner_blks,
                                        ib::_4a4b, ib::_4b4c)
                ? 4
                : utils::one_of(tag_traits<tag_o>::inner_blks, ib::_2c8b4c,
                          ib::_2b8a4b)
                        ? 8
                        : 16;

        const auto &plain_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        const int G = w_groups ? dims[0] : 1;
        const int OC = dims[w_groups + 0];
        const int NB_OC = pdims[w_groups + 0] / blksize;
        const int IC = dims[w_groups + 1];
        const int NB_IC = pdims[w_groups + 1] / blksize;
        const int D = is_3d ? dims[2 + w_groups] : 1;
        const int H = is_1d ? 1 : dims[2 + w_groups + is_3d];
        const int W = dims[w_groups + is_3d + 3 - is_1d];

        const float *scales = pd->attr()->output_scales_.scales_;
        const size_t D_mask = utils::array_product(input_d.dims(),
                math::ilog2q(pd->attr()->output_scales_.mask_ + 1));
        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        assert(req_comp || has_asymmetric_comp);

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        auto ker = [&](const data_t<type_i> *inp, data_t<type_o> *out,
                           int32_t *c, int32_t *zp, const float *s,
                           const int oc_block, const int ic_block) {
#define index AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>
            for_(int ic = 0; ic < ic_block; ++ic)
            for (int oc = 0; oc < oc_block; ++oc) {
                const auto plain_off
                        = oc * plain_d.blocking_desc().strides[w_groups + 0]
                        + ic * plain_d.blocking_desc().strides[w_groups + 1];
                out[index(oc, ic)] = qz_b0<data_t<type_i>, data_t<type_o>>()(
                        inp[plain_off], s[oc] * adj_scale);
                if (req_comp) c[oc] -= (128 * (int32_t)(out[index(oc, ic)]));
                if (has_asymmetric_comp)
                    zp[oc] -= (int32_t)(out[index(oc, ic)]);
            }
#undef index
        };

        constexpr int i_mult = blksize;
        constexpr int o_mult = 1;

        size_t offset
                = G * pdims[w_groups + 0] * pdims[w_groups + 1] * D * H * W;
        size_t zp_offset = offset
                + (req_comp ? G * pdims[w_groups + 0] * sizeof(int32_t) : 0);
        int32_t *cp = req_comp ? reinterpret_cast<int32_t *>(output + offset)
                               : nullptr;
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + zp_offset)
                : nullptr;
        parallel_nd(G * NB_OC * blksize, [&](int i) {
            if (req_comp) cp[i] = 0;
            if (has_asymmetric_comp) zp[i] = 0;
        });

#define wei_blk_off(md, g, o, i, d, h, w) \
    (is_1d ? (md).blk_off<!w_groups>(g, o, i, w) \
           : is_3d ? (md).blk_off<!w_groups>(g, o, i, d, h, w) \
                   : (md).blk_off<!w_groups>(g, o, i, h, w))

        parallel_nd(G, NB_OC, [&](int g, int O) {
            for_(int I = 0; I < NB_IC; I++)
            for_(int d = 0; d < D; d++)
            for_(int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                auto i = &input[wei_blk_off(
                        input_d, g, i_mult * O, i_mult * I, d, h, w)];
                auto o = &output[wei_blk_off(
                        output_d, g, o_mult * O, o_mult * I, d, h, w)];
                const int oc_block = nstl::min(blksize, OC - O * blksize);
                const int ic_block = nstl::min(blksize, IC - I * blksize);

                int _offset = (g * NB_OC + O) * blksize;
                ker(i, o, (order_keep && req_comp) ? &cp[_offset] : nullptr,
                        (order_keep && has_asymmetric_comp) ? &zp[_offset]
                                                            : nullptr,
                        &scales[(D_mask == 1) ? 0 : _offset], oc_block,
                        ic_block);
            }
        });

#undef wei_blk_off

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<false
                        || (utils::one_of(
                                    tag_i, format_tag::goiw, format_tag::wigo)
                                && utils::one_of(tag_o, format_tag::Goiw16g,
                                        format_tag::Goiw8g, format_tag::Goiw4g))
                        || (utils::one_of(
                                    tag_i, format_tag::goihw, format_tag::hwigo)
                                && utils::one_of(tag_o, format_tag::Goihw16g,
                                        format_tag::Goihw8g,
                                        format_tag::Goihw4g)),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        const size_t D_mask = array_product(
                input_d.dims(), math::ilog2q(attr->output_scales_.mask_ + 1));
        const dim_t g = input_d.dims()[0];
        const dim_t oc = input_d.dims()[1];
        const dim_t ic = input_d.dims()[2];

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        return order_keep && oc == 1 && ic == 1 // depth-wise case
                && simple_attr_check(attr, true, false)
                && (req_comp || req_asymmetric_comp)
                && input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && IMPLICATION(
                        req_comp, one_of(D_mask, (size_t)1, (size_t)g * oc))
                && one_of(input_d.data_type(), f32, s8, bf16)
                && output_d.data_type() == s8;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        constexpr bool is_1d
                = utils::one_of(tag_i, format_tag::goiw, format_tag::wigo);
        constexpr int blksize
                = utils::one_of(tag_o, format_tag::Goihw4g, format_tag::Goiw4g)
                ? 4
                : utils::one_of(tag_o, format_tag::Goihw8g, format_tag::Goiw8g)
                        ? 8
                        : 16;

        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();
        const int G = dims[0];
        const int Gp = pdims[0];
        const int OC = dims[1];
        const int IC = dims[2];
        const int H = is_1d ? 1 : dims[3];
        const int W = dims[4 - is_1d];

        const size_t D_mask = utils::array_product(input_d.dims(),
                math::ilog2q(pd->attr()->output_scales_.mask_ + 1));
        const float *scales = pd->attr()->output_scales_.scales_;
        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        assert(req_comp || has_asymmetric_comp);

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        auto ker_out = [&](const data_t<type_i> *inp, data_t<type_o> *out,
                               const float *s, const int g_block) {
            PRAGMA_OMP_SIMD()
            for (int g = 0; g < g_block; g++) {
                const auto i_off = g * input_d.blocking_desc().strides[0];
                out[g] = qz_b0<data_t<type_i>, data_t<type_o>>()(
                        inp[i_off], s[g * OC] * adj_scale);
            }
        };

        /* Note: having separate kernels for s8 and zero-point fixes a
         * compiler-generated bug which results in seg-fault. */
        auto ker_s8 = [&](const data_t<type_o> *out, int32_t *cp,
                              const int g_block) {
            PRAGMA_OMP_SIMD()
            for (int g = 0; g < g_block; g++) {
                cp[g * OC] -= 128 * (int32_t)(out[g]);
            }
        };
        auto ker_zp = [&](const data_t<type_o> *out, int32_t *zp,
                              const int g_block) {
            PRAGMA_OMP_SIMD()
            for (int g = 0; g < g_block; g++) {
                zp[g * OC] -= (int32_t)(out[g]);
            }
        };

        size_t offset = output_d.size() - output_d.additional_buffer_size();
        size_t zp_offset = offset + (req_comp ? Gp * OC * sizeof(int32_t) : 0);
        int32_t *cp = req_comp ? reinterpret_cast<int32_t *>(output + offset)
                               : nullptr;
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + zp_offset)
                : nullptr;

        parallel_nd((Gp / blksize) * OC, [&](int ib) {
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < blksize; i++) {
                if (req_comp) cp[ib * blksize + i] = 0;
                if (has_asymmetric_comp) zp[ib * blksize + i] = 0;
            }
        });

#define wei_blk_off(md, g, o, i, h, w) \
    (is_1d ? (md).blk_off(g, o, i, w) : (md).blk_off(g, o, i, h, w))

        parallel_nd(Gp / blksize, OC, [&](int gb, int O) {
            for (int I = 0; I < IC; I++) {
                for_(int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    const int g_block = nstl::min(G - gb * blksize, blksize);
                    const auto inp = &input[wei_blk_off(
                            input_d, gb * blksize, O, I, h, w)];
                    const auto out
                            = &output[wei_blk_off(output_d, gb, O, I, h, w)];
                    int offset = gb * blksize + O;

                    ker_out(inp, out, &scales[(D_mask == 1) ? 0 : offset],
                            g_block);
                    if (req_comp) ker_s8(out, &cp[offset], g_block);
                    if (has_asymmetric_comp) ker_zp(out, &zp[offset], g_block);
                }
            }
        });

#undef wei_blk_off

        return status::success;
    }
};

/* bf16 reorders */
template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<(
                (tag_i == format_tag::goihw || tag_i == format_tag::oihw)
                && (tag_o == format_tag::gOIhw16i16o
                        || tag_o == format_tag::OIhw16i16o
                        || tag_o == format_tag::gOIhw8i16o2i
                        || tag_o == format_tag::OIhw8i16o2i
                        || tag_o == format_tag::gOIhw8o16i2o
                        || tag_o == format_tag::OIhw8o16i2o
                        || tag_o == format_tag::gIOhw8o16i2o
                        || tag_o == format_tag::IOhw8o16i2o)
                && type_i == data_type::f32
                && type_o == data_type::bf16)>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;

        if (input_d.has_runtime_dims_or_strides()) return false;

        return order_keep && input_d.matches_tag(tag_i)
                && output_d.matches_tag(tag_o) && input_d.data_type() == f32
                && output_d.data_type() == bf16 && attr->has_default_values();
    }

    static size_t get_scratchpad_size(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        const int blksize = 16;
        return sizeof(float) * blksize * blksize * dnnl_get_max_threads();
    }

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        static constexpr bool w_groups = tag_i == goihw;
        const int blksize = 16;
        const int sblk = 2;

        const auto &plain_d = input_d;
        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();

        const int G = w_groups ? dims[0] : 1;
        const int OC = dims[w_groups + 0];
        const int NB_OC = pdims[w_groups + 0] / blksize;
        const int IC = dims[w_groups + 1];
        const int NB_IC = pdims[w_groups + 1] / blksize;
        const int H = dims[w_groups + 2];
        const int W = dims[w_groups + 3];

        const size_t wsp_size = blksize * blksize;
        float *wspace = scratchpad.template get<float>(
                memory_tracking::names::key_reorder_space);

        auto index = [&](const int ic, const int oc) {
            if (utils::one_of(tag_o, gOIhw16i16o, OIhw16i16o))
                return (ic * blksize + oc);
            else if (utils::one_of(tag_o, gOIhw8i16o2i, OIhw8i16o2i))
                return ((ic / sblk) * blksize * sblk + sblk * oc + ic % sblk);
            else if (utils::one_of(tag_o, gOIhw8o16i2o, gIOhw8o16i2o,
                             OIhw8o16i2o, IOhw8o16i2o))
                return ((oc / sblk) * blksize * sblk + sblk * ic + oc % sblk);
            else
                assert(!"Invalid weight format");
            return 0;
        };

        auto ker = [&](const data_t<type_i> *inp, data_t<type_i> *out,
                           const int curr_oc_block, const int oc_block,
                           const int curr_ic_block, const int ic_block) {
            int ic = 0;
            for (ic = 0; ic < curr_ic_block; ++ic) {
                int oc = 0;
                for (oc = 0; oc < curr_oc_block; ++oc) {
                    const auto plain_off
                            = oc * plain_d.blocking_desc().strides[w_groups + 0]
                            + ic
                                    * plain_d.blocking_desc()
                                              .strides[w_groups + 1];
                    out[index(ic, oc)] = inp[plain_off];
                }
                for (/* continue */; oc < oc_block; ++oc) {
                    out[index(ic, oc)] = (data_t<type_i>)0;
                }
            }
            for (/* continue */; ic < ic_block; ++ic) {
                for (int oc = 0; oc < oc_block; ++oc) {
                    out[index(ic, oc)] = (data_t<type_i>)0;
                }
            }
        };

        constexpr int i_mult = blksize;
        constexpr int o_mult = 1;

        parallel_nd_ext(0, G, NB_OC, NB_IC, H, W,
                [&](int ithr, int, int g, int O, int I, int h, int w) {
                    float *_wspace = wspace + wsp_size * ithr;
                    auto i = &input[input_d.blk_off<!w_groups>(
                            g, i_mult * O, i_mult * I, h, w)];
                    auto o = &output[output_d.blk_off<!w_groups>(
                            g, o_mult * O, o_mult * I, h, w)];
                    const int oc_block = nstl::min(blksize, OC - O * blksize);
                    const int ic_block = nstl::min(blksize, IC - I * blksize);
                    ker(i, _wspace, oc_block, blksize, ic_block, blksize);
                    cvt_float_to_bfloat16(o, _wspace, wsp_size);
                });

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<(tag_i == format_tag::nchw
                                          && tag_o == format_tag::nChw16c)
                && type_i == data_type::f32
                && type_o == data_type::bf16>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;

        if (input_d.has_runtime_dims_or_strides()) return false;

        return input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && input_d.data_type() == f32 && output_d.data_type() == bf16
                && attr->has_default_values();
    }

    static size_t get_scratchpad_size(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        const size_t blksize = 16;
        const size_t W = input_d.dims()[3];
        return sizeof(float) * blksize * W * dnnl_get_max_threads();
    }

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        constexpr int blksize = 16;

        const auto &flat_d = input_d;
        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();

        const int C = dims[1];
        const int H = dims[2];
        const int W = dims[3];

        const int wsp_size = W * blksize;
        float *wspace = scratchpad.template get<float>(
                memory_tracking::names::key_reorder_space);

        auto ker = [&](const data_t<type_i> *i, data_t<type_i> *o,
                           const int curr_c_block, const int c_block) {
            for (int w = 0; w < W; ++w) {
                int c = 0;
                for (c = 0; c < curr_c_block; ++c) {
                    const ptrdiff_t flat_off = 0
                            + c * flat_d.blocking_desc().strides[1]
                            + w * flat_d.blocking_desc().strides[3];
                    o[w * blksize + c] = i[flat_off];
                }
                for (/* continue */; c < c_block; ++c) {
                    o[w * blksize + c] = (data_t<type_i>)0;
                }
            }
        };

        constexpr int i_c_mult = blksize;
        constexpr int o_c_mult = 1;

        parallel_nd_ext(0, dims[0], pdims[1] / blksize, H,
                [&](int ithr, int, int n, int nb_c, int h) {
                    float *_wspace = wspace + wsp_size * ithr;
                    auto i = &input[input_d.blk_off(n, i_c_mult * nb_c, h)];
                    auto o = &output[output_d.blk_off(n, o_c_mult * nb_c, h)];
                    const int c_block = nstl::min(blksize, C - nb_c * blksize);
                    ker(i, _wspace, c_block, blksize);
                    cvt_float_to_bfloat16(o, _wspace, wsp_size);
                });

        return status::success;
    }
};

/* reorders with tail support */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<false
                || (utils::one_of(
                            tag_i, format_tag::nCdhw4c, format_tag::nCdhw8c)
                        && tag_o == format_tag::nCdhw16c)
                || (utils::one_of(tag_i, format_tag::nChw4c, format_tag::nChw8c)
                        && tag_o == format_tag::nChw16c)
                || (utils::one_of(tag_i, format_tag::nCw4c, format_tag::nCw8c)
                        && tag_o == format_tag::nCw16c)>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return simple_fmt_check(order_keep, tag_i, tag_o, input_d, output_d)
                && simple_attr_check(attr, false, true);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        constexpr int is_1d = utils::one_of(tag_i, nCw4c, nCw8c);
        constexpr int is_3d = utils::one_of(tag_i, nCdhw4c, nCdhw8c);

        constexpr int blksize_i
                = tag_traits<tag_i>::inner_blks == ib::_4b ? 4 : 8;
        constexpr int blksize_16 = 16;

        constexpr int ic_mult = order_keep ? blksize_16 / blksize_i : 1;
        constexpr int oc_mult = order_keep ? 1 : blksize_16 / blksize_i;

        const auto &dims = input_d.dims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        const auto &d_i = order_keep ? input_d : output_d;
        const auto stride_C_in_blk_i = d_i.blocking_desc().strides[1];

        const int C = dims[1];
        const int D = is_3d ? dims[2] : 1;
        const int H = is_1d ? 1 : dims[2 + is_3d];
        const int W = dims[3 + is_3d - is_1d];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                           const int block) {
            const int nb = utils::div_up(block, blksize_i);
            if (alpha == 1.0 && beta == 0.0) {
                for (int b = 0; b < nb; ++b) {
                    const ptrdiff_t i_off
                            = b * (order_keep ? stride_C_in_blk_i : blksize_i);
                    const ptrdiff_t o_off
                            = b * (order_keep ? blksize_i : stride_C_in_blk_i);
                    const int block_i
                            = nstl::min(blksize_i, block - b * blksize_i);
                    for (int c = 0; c < block_i; ++c) {
                        o[o_off + c] = _qz_a1b0<type_i, type_o>()(i[i_off + c]);
                    }
                }
            } else {
                for (int b = 0; b < nb; ++b) {
                    const ptrdiff_t i_off
                            = b * (order_keep ? stride_C_in_blk_i : blksize_i);
                    const ptrdiff_t o_off
                            = b * (order_keep ? blksize_i : stride_C_in_blk_i);
                    const int block_i
                            = nstl::min(blksize_i, block - b * blksize_i);
                    for (int c = 0; c < block_i; ++c) {
                        o[o_off + c] = _qz<type_i, type_o>()(
                                i[i_off + c], o[o_off + c], alpha, beta);
                    }
                }
            }
        };

#define data_blk_off(md, n, c, d, h, w) \
    (is_1d ? (md).blk_off(n, c, w) \
           : is_3d ? (md).blk_off(n, c, d, h, w) : (md).blk_off(n, c, h, w))

        parallel_nd(dims[0], pdims[1] / blksize_16, D, H, W,
                [&](int n, int nb_c, int d, int h, int w) {
                    auto i = &input[data_blk_off(
                            input_d, n, ic_mult * nb_c, d, h, w)];
                    auto o = &output[data_blk_off(
                            output_d, n, oc_mult * nb_c, d, h, w)];
                    const int block
                            = nstl::min(blksize_16, C - nb_c * blksize_16);
                    ker(i, o, block);
                });

#undef data_blk_off

        return status::success;
    }
};

#define PLAIN_TO_BLOCKED_IS_APPLICABLE() \
    static bool is_applicable(const memory_desc_wrapper &input_d, \
            const memory_desc_wrapper &output_d, \
            const primitive_attr_t *attr) { \
        return !input_d.has_runtime_dims_or_strides() \
                && simple_attr_check(attr, false, true) \
                && (order_keep ? output_d.matches_tag(tag_o) \
                                        && input_d.is_plain() \
                               : input_d.matches_tag(tag_o) \
                                        && output_d.is_plain()); \
    }

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                && (tag_traits<tag_o>::block_dims == bd::_A
                        || tag_traits<tag_o>::block_dims == bd::_B)
                && tag_traits<tag_o>::ndims >= 3
                && tag_traits<tag_o>::ndims <= 6>::type> {
    PLAIN_TO_BLOCKED_IS_APPLICABLE();

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &block_d = order_keep ? output_d : input_d;
        const auto &dims = input_d.dims();
        const auto &pdims = block_d.padded_dims();

        constexpr int ndims = tag_traits<tag_o>::ndims;
        constexpr int blk_idx = tag_traits<tag_o>::block_dims == bd::_A ? 0 : 1;

        const dim_t H0 = dims[0];
        const dim_t H1 = dims[1];
        const dim_t M0 = ndims >= 6 ? dims[ndims - 4] : 1;
        const dim_t M1 = ndims >= 5 ? dims[ndims - 3] : 1;
        const dim_t M2 = ndims >= 4 ? dims[ndims - 2] : 1;
        const dim_t L = dims[ndims - 1];
        const dim_t l_blk_stride = block_d.blocking_desc().strides[ndims - 1];
        const dim_t l_flat_stride = flat_d.blocking_desc().strides[ndims - 1];
        const dim_t blk_flat_stride = flat_d.blocking_desc().strides[blk_idx];
        using namespace data_type;
        using namespace utils;

        constexpr int blksize = false
                ? 0
                : one_of(tag_traits<tag_o>::inner_blks, ib::_4a, ib::_4b)
                        ? 4
                        : one_of(tag_traits<tag_o>::inner_blks, ib::_8a,
                                  ib::_8b)
                                ? 8
                                : 16;

        constexpr bool f32bf16
                = one_of(type_i, f32, bf16) && one_of(type_o, f32, bf16);

        auto wrap_qz_a1b0 = [=](data_t<type_o> &out, data_t<type_i> inp) {
            if (f32bf16)
                out = inp;
            else
                out = _qz_a1b0<type_i, type_o>()(inp);
        };

        auto wrap_qz = [=](data_t<type_o> &out, data_t<type_i> inp, float alpha,
                               float beta) {
            if (f32bf16)
                out = alpha * inp + (beta ? beta * out : 0);
            else
                out = _qz<type_i, type_o>()(inp, out, alpha, beta);
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o, int block) {
            if (alpha == 1.0 && beta == 0.0) {
                for_(int l = 0; l < L; ++l)
                for (int blk = 0; blk < block; ++blk) {
                    const dim_t flat_off
                            = blk * blk_flat_stride + l * l_flat_stride;
                    const dim_t blk_offset = l * l_blk_stride + blk;
                    if (order_keep)
                        wrap_qz_a1b0(o[blk_offset], i[flat_off]);
                    else
                        wrap_qz_a1b0(o[flat_off], i[blk_offset]);
                }
            } else {
                for_(int l = 0; l < L; ++l)
                for (int blk = 0; blk < block; ++blk) {
                    const dim_t flat_off
                            = blk * blk_flat_stride + l * l_flat_stride;
                    const dim_t blk_offset = l * l_blk_stride + blk;
                    if (order_keep)
                        wrap_qz(o[blk_offset], i[flat_off], alpha, beta);
                    else
                        wrap_qz(o[flat_off], i[blk_offset], alpha, beta);
                }
            }
        };

#define off(md, h0, h1, m0, m1, m2) \
    (ndims >= 6 ? (md).blk_off(h0, h1, m0, m1, m2) \
                : ndims >= 5 ? (md).blk_off(h0, h1, m1, m2) \
                             : ndims >= 4 \
                                    ? (md).blk_off(h0, h1, m2) \
                                    : /* ndims >= 3 ? */ (md).blk_off(h0, h1))

        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        if (blk_idx == 0) {
            const dim_t BH0 = pdims[0] / blksize;
            parallel_nd(BH0, H1, M0, M1, M2,
                    [&](dim_t bh0, dim_t h1, dim_t m0, dim_t m1, dim_t m2) {
                        auto i = &input[off(
                                input_d, bh0 * i_mult, h1, m0, m1, m2)];
                        auto o = &output[off(
                                output_d, bh0 * o_mult, h1, m0, m1, m2)];
                        const int block
                                = nstl::min<int>(blksize, H0 - bh0 * blksize);
                        ker(i, o, block);
                    });
        } else if (blk_idx == 1) {
            const dim_t BH1 = pdims[1] / blksize;
            parallel_nd(H0, BH1, M0, M1, M2,
                    [&](dim_t h0, dim_t bh1, dim_t m0, dim_t m1, dim_t m2) {
                        auto i = &input[off(
                                input_d, h0, bh1 * i_mult, m0, m1, m2)];
                        auto o = &output[off(
                                output_d, h0, bh1 * o_mult, m0, m1, m2)];
                        const int block
                                = nstl::min<int>(blksize, H1 - bh1 * blksize);
                        ker(i, o, block);
                    });
        } else {
            assert(!"unimplemented");
        }

#undef off

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                && (tag_traits<tag_o>::block_dims == bd::_AB
                        || tag_traits<tag_o>::block_dims == bd::_BC)
                && IMPLICATION(tag_traits<tag_o>::block_dims == bd::_AB,
                        tag_traits<tag_o>::ndims >= 3
                                && tag_traits<tag_o>::ndims <= 5)
                && IMPLICATION(tag_traits<tag_o>::block_dims == bd::_BC,
                        tag_traits<tag_o>::ndims >= 4
                                && tag_traits<tag_o>::ndims <= 6)>::type> {
    PLAIN_TO_BLOCKED_IS_APPLICABLE();

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        constexpr int ndims = tag_traits<tag_o>::ndims;

        static constexpr bool with_g = tag_traits<tag_o>::block_dims == bd::_BC;
        const dim_t G = with_g ? dims[0] : 1;

        const dim_t H0 = dims[0 + with_g];
        const dim_t H1 = dims[1 + with_g];

        const dim_t M0 = ndims >= 5 + with_g ? dims[ndims - 3] : 1;
        const dim_t M1 = ndims >= 4 + with_g ? dims[ndims - 2] : 1;
        const dim_t M2 = ndims >= 3 + with_g ? dims[ndims - 1] : 1;

        const dim_t h0_flat_stride = flat_d.blocking_desc().strides[with_g + 0];
        const dim_t h1_flat_stride = flat_d.blocking_desc().strides[with_g + 1];
        using namespace data_type;
        using namespace utils;

        constexpr int blksize_0 = false
                ? 0
                : one_of(tag_traits<tag_o>::inner_blks, ib::_4b4a, ib::_4b4c,
                          ib::_4c4b)
                        ? 4
                        : one_of(tag_traits<tag_o>::inner_blks, ib::_8a8b,
                                  ib::_8b8a, ib::_8b8c, ib::_8c8b, ib::_2c8b4c)
                                ? 8
                                : one_of(tag_traits<tag_o>::inner_blks,
                                          ib::_16a16b, ib::_16b16a, ib::_16b16c,
                                          ib::_16c16b, ib::_8a16b2a,
                                          ib::_4b16a4b, ib::_8b16a2b,
                                          ib::_8b16c2b, ib::_4c16b4c,
                                          ib::_8c16b2c)
                                        ? 16
                                        : INT_MIN;

        constexpr int blksize_1
                = one_of(tag_traits<tag_o>::inner_blks, ib::_8a8b, ib::_8b8a,
                          ib::_8b8c, ib::_8c8b, ib::_2c8b4c)
                ? 8
                : one_of(tag_traits<tag_o>::inner_blks, ib::_16a16b,
                          ib::_16b16a, ib::_16b16c, ib::_16c16b, ib::_8a16b2a,
                          ib::_4b16a4b, ib::_8b16a2b, ib::_8b16c2b,
                          ib::_4c16b4c, ib::_8c16b2c)
                        ? 16
                        : one_of(tag_traits<tag_o>::inner_blks, ib::_4b4a,
                                  ib::_4b4c, ib::_4c4b)
                                ? 4
                                : INT_MIN;

        const dim_t NB_H0 = pdims[0 + with_g] / blksize_0;
        const dim_t NB_H1 = pdims[1 + with_g] / blksize_1;

        constexpr bool f32bf16
                = one_of(type_i, f32, bf16) && one_of(type_o, f32, bf16);

        auto wrap_qz_a1b0 = [=](data_t<type_o> &out, data_t<type_i> inp) {
            if (f32bf16)
                out = inp;
            else
                out = _qz_a1b0<type_i, type_o>()(inp);
        };

        auto wrap_qz = [=](data_t<type_o> &out, data_t<type_i> inp, float alpha,
                               float beta) {
            if (f32bf16)
                out = alpha * inp + (beta ? beta * out : 0);
            else
                out = _qz<type_i, type_o>()(inp, out, alpha, beta);
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                           const int block_h0, const int block_h1) {
#define blk_off AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>
            if (alpha == 1.0 && beta == 0.0) {
                for_(int h0 = 0; h0 < block_h0; ++h0)
                for (int h1 = 0; h1 < block_h1; ++h1) {
                    const dim_t flat_off
                            = h0 * h0_flat_stride + h1 * h1_flat_stride;
                    if (order_keep)
                        wrap_qz_a1b0(o[blk_off(h0, h1)], i[flat_off]);
                    else
                        wrap_qz_a1b0(o[flat_off], i[blk_off(h0, h1)]);
                }
            } else {
                for_(int h0 = 0; h0 < block_h0; ++h0)
                for (int h1 = 0; h1 < block_h1; ++h1) {
                    const dim_t flat_off
                            = h0 * h0_flat_stride + h1 * h1_flat_stride;
                    if (order_keep)
                        wrap_qz(o[blk_off(h0, h1)], i[flat_off], alpha, beta);
                    else
                        wrap_qz(o[flat_off], i[blk_off(h0, h1)], alpha, beta);
                }
            }

#undef blk_off
        };

        constexpr int i_mult_0 = order_keep ? blksize_0 : 1;
        constexpr int o_mult_0 = order_keep ? 1 : blksize_0;

        constexpr int i_mult_1 = order_keep ? blksize_1 : 1;
        constexpr int o_mult_1 = order_keep ? 1 : blksize_1;

#define off(md, g, h0, h1, m0, m1, m2) \
    (ndims >= 5 + with_g ? (md).blk_off<!with_g>(g, h0, h1, m0, m1, m2) \
                         : ndims >= 4 + with_g \
                            ? (md).blk_off<!with_g>(g, h0, h1, m1, m2) \
                            : /* ndims >= 3 + with_g ? */ (md) \
                                      .blk_off<!with_g>(g, h0, h1, m2))

        parallel_nd(G, NB_H0, NB_H1, M0, M1, M2,
                [&](dim_t g, dim_t nb_h0, dim_t nb_h1, dim_t m0, dim_t m1,
                        dim_t m2) {
                    auto i = &input[off(input_d, g, i_mult_0 * nb_h0,
                            i_mult_1 * nb_h1, m0, m1, m2)];
                    auto o = &output[off(output_d, g, o_mult_0 * nb_h0,
                            o_mult_1 * nb_h1, m0, m1, m2)];
                    const int block_h0
                            = nstl::min<int>(blksize_0, H0 - nb_h0 * blksize_0);
                    const int block_h1
                            = nstl::min<int>(blksize_1, H1 - nb_h1 * blksize_1);
                    ker(i, o, block_h0, block_h1);
                });

#undef off

        return status::success;
    }
};

/* generic and direct-copy reorders */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && tag_o == format_tag::any
                        && order_keep == fmt_order::any,
                spec::direct_copy>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        /* FIXME: is the formula correct? */
        return !input_d.has_runtime_dims_or_strides()
                && input_d.similar_to(output_d, true, false, 0)
                && input_d.is_dense() && output_d.is_dense()
                && simple_attr_check(attr, false, true);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        assert(input_d.is_dense());

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const size_t nelems = input_d.nelems();

        constexpr int block_size = 16;
        const auto num_blocks = nelems / block_size;
        const auto rem_elems = nelems % block_size;

        parallel(0, [&](const int ithr, const int nthr) {
            size_t start {0}, end {0};
            balance211(num_blocks, nthr, ithr, start, end);
            start = start * block_size;
            end = end * block_size;

            if (alpha == 1.0 && beta == 0.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_a1b0<data_t<type_i>, data_t<type_o>>()(
                            input[e]);
                }
            } else if (alpha == 1.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_a1<data_t<type_i>, data_t<type_o>>()(
                            input[e], output[e], beta);
                }
            } else if (beta == 0.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_b0<data_t<type_i>, data_t<type_o>>()(
                            input[e], alpha);
                }
            } else {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz<data_t<type_i>, data_t<type_o>>()(
                            input[e], output[e], alpha, beta);
                }
            }

            if (rem_elems != 0 && ithr == nthr - 1) {
                if (alpha == 1.0 && beta == 0.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_a1b0<data_t<type_i>, data_t<type_o>>()(
                                input[e]);
                    }
                } else if (alpha == 1.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_a1<data_t<type_i>, data_t<type_o>>()(
                                input[e], output[e], beta);
                    }
                } else if (beta == 0.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_b0<data_t<type_i>, data_t<type_o>>()(
                                input[e], alpha);
                    }
                } else {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz<data_t<type_i>, data_t<type_o>>()(
                                input[e], output[e], alpha, beta);
                    }
                }
            }
        });
        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && tag_o == format_tag::any
                        && order_keep == fmt_order::any,
                spec::direct_copy_except_dim_0>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        auto is_dense_no_0 = [](const memory_desc_wrapper &data_d) {
            return nelems_no_dim_0(data_d) == _size_no_dim_0(data_d);
        };
        /* FIXME: is the formula correct? */
        return !input_d.has_runtime_dims_or_strides()
                && input_d.similar_to(output_d, true, false, 1)
                && is_dense_no_0(input_d) && is_dense_no_0(output_d)
                && simple_attr_check(attr, false, true);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace utils;

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const int N = input_d.dims()[0];
        const dim_t is = input_d.blocking_desc().strides[0];
        const dim_t os = output_d.blocking_desc().strides[0];
        const dim_t nelems_no_d0 = nelems_no_dim_0(input_d);
        const dim_t work_amount = N * nelems_no_d0;

        if (alpha == 1.0 && beta == 0.0) {
            parallel(0, [&](const int ithr, const int nthr) {
                dim_t n {0}, dim1_s {0};
                dim_t start {0}, end {0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while (start < end) {
                    dim_t work_rem = end - start;
                    dim_t dim1_e = dim1_s + work_rem > nelems_no_d0
                            ? nelems_no_d0
                            : dim1_s + work_rem;
                    PRAGMA_OMP_SIMD()
                    for (dim_t e = dim1_s; e < dim1_e; ++e) {
                        output[os * n + e]
                                = _qz_a1b0<type_i, type_o>()(input[is * n + e]);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            });
        } else {
            parallel(0, [&](const int ithr, const int nthr) {
                dim_t n {0}, dim1_s {0};
                dim_t start {0}, end {0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while (start < end) {
                    dim_t work_rem = end - start;
                    dim_t dim1_e = dim1_s + work_rem > nelems_no_d0
                            ? nelems_no_d0
                            : dim1_s + work_rem;
                    PRAGMA_OMP_SIMD()
                    for (dim_t e = dim1_s; e < dim1_e; ++e) {
                        output[os * n + e]
                                = _qz<type_i, type_o>()(input[is * n + e],
                                        output[os * n + e], alpha, beta);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            });
        }

        return status::success;
    }

private:
    static dim_t nelems_no_dim_0(const memory_desc_wrapper &data_d) {
        const int ndims = data_d.ndims();
        if (ndims <= 1) return 1;
        return utils::array_product(data_d.dims() + 1, data_d.ndims() - 1);
    }

    static dim_t _size_no_dim_0(const memory_desc_wrapper &data_d) {
        dims_t blocks;
        data_d.compute_blocks(blocks);

        const auto &blk = data_d.blocking_desc();

        dim_t blk_size = 1;
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk)
            blk_size *= blk.inner_blks[iblk];

        dim_t max_size = blk_size;
        for (int d = 1; d < data_d.ndims(); ++d) {
            max_size = nstl::max(max_size,
                    data_d.padded_dims()[d] / blocks[d] * blk.strides[d]);
        }

        return max_size;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && tag_o == format_tag::any
                        && order_keep == fmt_order::any,
                spec::reference>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        /* supported smask: 0x0...011..10...0,
         * i.e. 1 should be contiguous */
        int smask = attr ? attr->output_scales_.mask_ : 0;
        for (; smask > 0 && !(smask & 0x1); smask >>= 1)
            ;
        for (; smask > 0 && smask & 0x1; smask >>= 1)
            ;
        return input_d.is_blocking_desc() && output_d.is_blocking_desc()
                && !output_d.is_additional_buffer()
                && !input_d.is_additional_buffer() && smask == 0
                && attr->has_default_values(
                        dnnl_primitive_attr::skip_mask_t::oscale_runtime
                        | dnnl_primitive_attr::skip_mask_t::zero_points_runtime
                        | dnnl_primitive_attr::skip_mask_t::post_ops)
                && simple_po_check(attr);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(
            const cpu_reorder_pd_t *pd_object, const exec_ctx_t &ctx) {
        // DEFINE_SCALES_BUFFER and DEFINE_ZERO_POINT_VALUE macro use pd() to
        // query properties, hence wrapping the primitive descriptor into a
        // function.
        auto pd = [pd_object]() { return pd_object; };

        auto input = CTX_IN_MEM(const data_t<type_i> *, DNNL_ARG_FROM);
        auto output = CTX_OUT_MEM(data_t<type_o> *, DNNL_ARG_TO);

        const float beta = pd()->beta();
        DEFINE_SCALES_BUFFER(scales);
        DEFINE_ZERO_POINT_VALUE(i0, DNNL_ARG_FROM);
        DEFINE_ZERO_POINT_VALUE(o0, DNNL_ARG_TO);

        const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
        const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());

        const size_t nelems = input_d.nelems();

        int ndims_start = 0, ndims_mask = 0;
        int smask = pd()->attr()->output_scales_.mask_;
        for (; smask > 0 && !(smask & 0x1); smask >>= 1)
            ++ndims_start;
        for (; smask > 0 && smask & 0x1; smask >>= 1)
            ++ndims_mask;
        assert(smask == 0);

        const ptrdiff_t D_start
                = utils::array_product(input_d.dims(), ndims_start);
        const ptrdiff_t D_mask = utils::array_product(
                input_d.dims() + ndims_start, ndims_mask);
        const ptrdiff_t D_rest = nelems / D_start / D_mask;

        parallel_nd(D_start, D_mask, D_rest,
                [&](ptrdiff_t ds, ptrdiff_t dm, ptrdiff_t dr) {
                    const float scale = scales[dm];

                    const size_t e = (ds * D_mask + dm) * D_rest + dr;
                    const auto &i = input[input_d.off_l(e)];
                    auto &o = output[output_d.off_l(e)];

                    float f = scale * ((float)i - i0) + o0;
                    o = _qz<data_type::f32, type_o>()(f, o, 1.f, beta);
                });

        return status::success;
    }
};

/* high level class declaration */

template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            bool args_ok = true && src_md->data_type == type_i
                    && dst_md->data_type == type_o
                    && attr->has_default_values(
                            dnnl_primitive_attr::skip_mask_t::oscale_runtime
                            | dnnl_primitive_attr::skip_mask_t::zero_points
                            | dnnl_primitive_attr::skip_mask_t::
                                    zero_points_runtime
                            | dnnl_primitive_attr::skip_mask_t::post_ops)
                    && simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
                            spec>::is_applicable(src_md, dst_md, attr);
            if (!args_ok) return status::invalid_arguments;

            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }

            const size_t scratchpad_sz_
                    = simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
                            spec>::get_scratchpad_size(src_md, dst_md);
            auto scratchpad = _pd->scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_reorder_space,
                    scratchpad_sz_, 1, 16);
            _pd->init_scratchpad_md();
            return safe_ptr_assign(*reorder_pd, _pd);
        }
    };

    simple_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::execute(
                pd(), ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

#undef SIMPLE_REORDER_TEMPL_DECL
#undef SIMPLE_REORDER_TEMPL_CALL

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
