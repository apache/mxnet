/*******************************************************************************
* Copyright 2020 Intel Corporation
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
* See theb_ License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_amx_1x1_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_core_amx_1x1_convolution_fwd_t<src_type, wei_type,
        dst_type>::prepare_padded_bias(const char *&bias,
        const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;
    auto padded_bias = scratchpad.template get<char>(
            memory_tracking::names::key_conv_padded_bias);
    utils::array_copy(
            padded_bias, bias, bia_dt_size * pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + bia_dt_size * pd()->jcp_.oc_without_padding,
            0.f, bia_dt_size * (pd()->jcp_.oc - pd()->jcp_.oc_without_padding));
    bias = padded_bias;
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_core_amx_1x1_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward(const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;
    const size_t dst_dt_size
            = types::data_type_size(pd()->desc()->dst_desc.data_type);
    const size_t src_dt_size
            = types::data_type_size(pd()->desc()->src_desc.data_type);
    const size_t wei_dt_size
            = types::data_type_size(pd()->desc()->weights_desc.data_type);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    const float *oscales = pd()->attr()->output_scales_.scales_;

    const bool is_1d = pd()->ndims() == 3;

    const bool is_ic_tail = jcp.ic_without_padding % jcp.ic_block_int;
    auto wsp = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_amx_wsp_buffer);
    int32_t *wsp_tile = (is_ic_tail)
            ? ctx.get_scratchpad_grantor().template get<int32_t>(
                    key_conv_amx_tile_buffer)
            : nullptr;
    auto tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);

    const size_t wei_oc_shift = (size_t)jcp.nb_ic_int * jcp.ic_block_int
            * jcp.oc_block * jcp.nb_oc_blocking;

    int nb_os = (jcp.tile_tail) ? jcp.nb_os + 1 : jcp.nb_os;
    int os_step = jcp.nb_os2_blocking * jcp.nb_os_blocking;
    int os_chunks = div_up(nb_os, os_step);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;

    int work_amount = jcp.mb * jcp.ngroups * os_chunks * oc_chunks;
    kernel_->tile_configure(tcfg);

    parallel(0, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_s();
        p.tile_cfg = tcfg;
        p.tile_cfg_tail = tcfg + 64;

        amx_tile_configure(tcfg);

        int mb {0}, g {0}, _osb {0}, _ocb {0};
        nd_iterator_init(start, mb, jcp.mb, g, jcp.ngroups, _osb, os_chunks,
                _ocb, oc_chunks);

        while (start < end) {
            int osb = _osb * os_step;
            int ocb = _ocb * jcp.nb_oc_blocking;
            auto bias_w = bias
                    ? bias + (bias_d.blk_off(ocb * jcp.oc_block) * bia_dt_size)
                    : nullptr;

            int oc = g * jcp.oc_without_padding + ocb * jcp.oc_block;
            int ic = g * jcp.ic_without_padding;

            p.acc_s32 = wsp + ithr * jcp.wsp_buffer_size;
            p.src_prf = wsp_tile + ithr * (jcp.wsp_buffer_size / 2);
            p.filt = weights + wei_dt_size * _ocb * wei_oc_shift;
            p.bias = bias_w;
            p.scales = &oscales[jcp.is_oc_scale * oc];
            p.oc_blocks = ocb;

            const bool check_last_sp = is_ic_tail && !(nb_os % 2);
            const bool is_overflow = (osb + os_step >= nb_os);
            if (is_overflow
                    && (os_chunks > 1 || (os_chunks == 1 && is_ic_tail))) {
                int step = (check_last_sp) ? 1 : jcp.nb_os_blocking;
                for (int osi = 0; osi < nb_os - osb; osi += step) {
                    int osb_i = osi + osb;
                    int oh = (osb_i * jcp.tile_width) / jcp.ow;
                    int ow = (osb_i * jcp.tile_width) % jcp.ow;
                    size_t dst_offset = is_1d ? dst_d.blk_off(mb, oc, ow)
                                              : dst_d.blk_off(mb, oc, oh, ow);
                    p.dst = dst + dst_dt_size * dst_offset;

                    int ih = oh * jcp.stride_h;
                    int iw = ow * jcp.stride_w;
                    size_t inp_offset = is_1d ? src_d.blk_off(mb, ic, iw)
                                              : src_d.blk_off(mb, ic, ih, iw);
                    p.src = src + src_dt_size * inp_offset;

                    bool l_overflow = osb_i + jcp.nb_os_blocking >= nb_os;
                    p.last_h = (check_last_sp || (nb_os % 2 && l_overflow)) ? 1
                                                                            : 0;
                    p.is_osb = 0;
                    (*kernel_)(&p);
                }
            } else {
                int oh = (osb * jcp.tile_width) / jcp.ow;
                int ow = (osb * jcp.tile_width) % jcp.ow;
                size_t dst_offset = is_1d ? dst_d.blk_off(mb, oc, ow)
                                          : dst_d.blk_off(mb, oc, oh, ow);
                p.dst = dst + dst_dt_size * dst_offset;

                int ih = oh * jcp.stride_h;
                int iw = ow * jcp.stride_w;
                size_t inp_offset = is_1d ? src_d.blk_off(mb, ic, iw)
                                          : src_d.blk_off(mb, ic, ih, iw);
                p.src = src + src_dt_size * inp_offset;

                p.last_h = 0;
                p.is_osb = 1;

                (*kernel_)(&p);
            }
            ++start;
            nd_iterator_step(mb, jcp.mb, g, jcp.ngroups, _osb, os_chunks, _ocb,
                    oc_chunks);
        }
    });
}

template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::s8,
        data_type::s8, data_type::u8>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::u8,
        data_type::s8, data_type::u8>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::s8,
        data_type::s8, data_type::s8>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::u8,
        data_type::s8, data_type::s8>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::s8,
        data_type::s8, data_type::s32>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::u8,
        data_type::s8, data_type::s32>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::s8,
        data_type::s8, data_type::f32>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::u8,
        data_type::s8, data_type::f32>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::bf16,
        data_type::bf16, data_type::bf16>;
template struct jit_avx512_core_amx_1x1_convolution_fwd_t<data_type::bf16,
        data_type::bf16, data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
