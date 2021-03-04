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

#include "gpu/ocl/gen12lp_x8s8s32x_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gen12lp_x8s8s32x_convolution_fwd_t::pd_t::init_conf() {
    using namespace format_tag;

    const memory_desc_t *src = src_md();
    const memory_desc_t *dst = dst_md();
    const memory_desc_t *wei = weights_md();
    const memory_desc_t *bia = weights_md(1);

    memory_desc_t r_src, r_wei, r_dst;

    int ndims = src_md()->ndims;

    // XXX: try reduce number of spatial dims when iw/ow/kw=1,
    // memory tags will be selected based on the number of input dimensions
    bool use_reshaped_mem = ndims > 3;
    if (dnnl_memory_desc_reshape(&r_src, src, src->ndims - 1, src->dims)
            != status::success)
        use_reshaped_mem = false;
    if (dnnl_memory_desc_reshape(&r_dst, dst, dst->ndims - 1, dst->dims)
            != status::success)
        use_reshaped_mem = false;
    if (dnnl_memory_desc_reshape(&r_wei, wei, wei->ndims - 1, wei->dims)
            != status::success)
        use_reshaped_mem = false;

    if (use_reshaped_mem) {
        src = &r_src;
        dst = &r_dst;
        wei = &r_wei;
    }

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src);
    const memory_desc_wrapper weights_mdw(wei);
    const memory_desc_wrapper dst_mdw(dst);

    set_default_conf(conf, cd, *src, *wei, *dst, *bia, *attr());

    const bool is_src_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef;
    const bool is_dst_nhwc
            = dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef;
    const bool is_nhwc = is_src_nhwc || is_dst_nhwc;
    const bool is_1stconv = conf.ic_without_padding <= 4 && !conf.is_depthwise;

    conf.is_nhwc = is_nhwc;
    conf.is_dst_nhwc = is_dst_nhwc;
    // TODO: Add group convolution support in NHWC kernel.
    if (!conf.is_depthwise && conf.with_groups && conf.ngroups > 1
            && (conf.oc % 32 != 0 || conf.ic % 32 != 0))
        return status::unimplemented;

    conf.dst_data_type = dst_mdw.data_type();
    conf.src_data_type = src_mdw.data_type();

    conf.oc_block = 32;

    if (conf.is_nhwc) {
        conf.ver = ver_nhwc;
        if (conf.is_depthwise) {
            conf.sub_group_size = 16;
            conf.ic_block = 32;
            if (!(conf.kw <= 4 && conf.stride_w <= 2 && conf.dilate_w == 0
                        && conf.l_pad < 4)) {
                conf.mb_block = 32;
                conf.ow_block = 1;
            } else {
                conf.mb_block = 1;
                conf.ow_block = 1;

                int off = conf.kw == 4 ? 1 : 0;
                if (conf.ow < 15 - off) {
                    conf.ow_block = conf.ow;
                } else {
                    for (int i = 0; i < 7; ++i) {
                        conf.ow_block = utils::max_div(conf.ow + i, 14 - off);
                        if (conf.ow_block > 4) break;
                    }
                }
            }
            conf.lws_d[0] = 16;
            conf.lws_d[1] = 1;
            conf.lws_d[2] = 1;

            conf.gws_d[0] = utils::div_up(conf.ngroups, 32) * conf.lws_d[0];
            conf.gws_d[1]
                    = conf.od * conf.oh * utils::div_up(conf.ow, conf.ow_block);
            conf.gws_d[2]
                    = utils::div_up(conf.mb, utils::div_up(conf.mb_block, 4));

        } else {
            conf.sub_group_size = 8;
            int ow_group = 1;
            int ow_nchunk;

            conf.mb_block = 1;
            conf.nchunk = utils::div_up(conf.oc * conf.ngroups, conf.oc_block);

            int max_oc = 4;
            int oc_group = utils::max_div(
                    utils::div_up(conf.oc, conf.oc_block), max_oc);
            int max_subgroups = 32;
            int max_ow_group = max_subgroups / oc_group;

            if (!is_1stconv) {
                conf.ic_block = 32;

                conf.ow_block
                        = (conf.mb * conf.oc * conf.oh * conf.ow < 49 * 1024)
                        ? 4
                        : 8;
                ow_nchunk = utils::div_up(conf.ow, conf.ow_block);
                ow_group = utils::max_div(ow_nchunk, max_ow_group);
                if (ow_group == 1) utils::max_div(ow_nchunk + 1, max_ow_group);
            } else { // 1st
                conf.ic_block = 4;
                conf.ow_block = (conf.kw * conf.kh <= 49 && conf.ow % 16 < 8)
                        ? 16
                        : 12;
                ow_nchunk = utils::div_up(conf.ow, conf.ow_block);
                ow_group = utils::max_div(ow_nchunk, max_ow_group);
                if (ow_group == 1) utils::max_div(ow_nchunk + 1, max_ow_group);
                if (conf.mb == 8 || conf.mb % 16 == 0) { conf.mb_block = 32; }
            }

            conf.lws_d[0] = 8 * oc_group;
            conf.lws_d[1] = ow_group;
            conf.lws_d[2] = 1;

            conf.src_slm_size = conf.ic_block / 4
                    * (conf.lws_d[1] * conf.stride_w * conf.ow_block
                            + (conf.kw - 1) * (1 + conf.dilate_w));

            conf.gws_d[0] = utils::rnd_up(conf.nchunk * 8, conf.lws_d[0]);
            conf.gws_d[1] = conf.od * conf.oh
                    * utils::rnd_up(utils::div_up(conf.ow, conf.ow_block),
                            conf.lws_d[1]);
            conf.gws_d[2] = is_1stconv
                    ? conf.mb
                    : utils::div_up(conf.mb, utils::div_up(conf.mb_block, 2));
        }

    } else if (conf.is_depthwise) {
        conf.sub_group_size = 16;
        conf.ic_block = 32;

        if (conf.mb == 8 || conf.mb % 16 == 0
                || !(conf.kw <= 4 && conf.stride_w <= 2 && conf.dilate_w == 0
                        && conf.l_pad < 4)) {
            conf.ver = ver_mb_block;
            conf.mb_block = 32;
            conf.ow_block = 1;
        } else {
            conf.ver = ver_ow_block;
            conf.mb_block = 1;
            conf.ow_block = 1;
            int off = conf.kw == 4 ? 1 : 0;
            if (conf.ow < 15 - off) {
                conf.ow_block = conf.ow;
            } else {
                for (int i = 0; i < 7; ++i) {
                    conf.ow_block = utils::max_div(conf.ow + i, 14 - off);
                    if (conf.ow_block > 4) break;
                }
            }
        }

        conf.lws_d[0] = 16;
        conf.lws_d[1] = 1;
        conf.lws_d[2] = 1;

        conf.gws_d[0] = utils::div_up(conf.ngroups, 32) * conf.lws_d[0];
        conf.gws_d[1]
                = conf.od * conf.oh * utils::div_up(conf.ow, conf.ow_block);
        conf.gws_d[2] = utils::div_up(conf.mb, utils::div_up(conf.mb_block, 4));

    } else {
        conf.sub_group_size = 8;
        int ow_group = 1;
        int ow_nchunk;

        if (conf.mb == 8 || conf.mb % 16 == 0) {
            conf.ver = ver_mb_block;
            conf.mb_block = 32;
        } else {
            conf.ver = ver_ow_block;
            conf.mb_block = 1;
        }
        if (conf.ic <= 4) conf.ver = ver_1stconv;

        conf.nchunk = utils::div_up(conf.oc * conf.ngroups, conf.oc_block);

        int max_oc = 4;
        int oc_group
                = utils::max_div(utils::div_up(conf.oc, conf.oc_block), max_oc);
        int max_subgroups = 32;
        int max_ow_group = max_subgroups / oc_group;
        switch (conf.ver) {
            case ver_mb_block:
                oc_group = 1;
                conf.ic_block = 32;
                conf.ow_block = 1;
                ow_group = 1;
                break;
            case ver_ow_block:
                conf.ic_block = 32;
                conf.ow_block
                        = (conf.mb * conf.oc * conf.oh * conf.ow < 49 * 1024)
                        ? 4
                        : 8;
                ow_nchunk = utils::div_up(conf.ow, conf.ow_block);
                ow_group = utils::max_div(ow_nchunk, max_ow_group);
                if (ow_group == 1) utils::max_div(ow_nchunk + 1, max_ow_group);
                break;
            case ver_1stconv:
                conf.ic_block = 4;
                conf.ow_block = (conf.kw * conf.kh <= 49 && conf.ow % 16 < 8)
                        ? 16
                        : 12;
                ow_nchunk = utils::div_up(conf.ow, conf.ow_block);
                ow_group = utils::max_div(ow_nchunk, max_ow_group);
                if (ow_group == 1) utils::max_div(ow_nchunk + 1, max_ow_group);
                break;
        }

        conf.lws_d[0] = 8 * oc_group;
        conf.lws_d[1] = ow_group;
        conf.lws_d[2] = 1;

        conf.src_slm_size = conf.ic_block / 4
                * (conf.lws_d[1] * conf.stride_w * conf.ow_block
                        + (conf.kw - 1) * (1 + conf.dilate_w));

        conf.gws_d[0] = utils::rnd_up(conf.nchunk * 8, conf.lws_d[0]);
        conf.gws_d[1] = conf.od * conf.oh
                * utils::rnd_up(
                        utils::div_up(conf.ow, conf.ow_block), conf.lws_d[1]);
        conf.gws_d[2] = utils::div_up(conf.mb, utils::div_up(conf.mb_block, 2));
        if (conf.ver == ver_1stconv) {
            conf.gws_d[2] = conf.mb;
            // Save opportunity to use this implementation with nchw formats,
            // which will result in worse performance, but prevent us using reorder.
            // That can be efficient in some cases.
            conf.is_nchw = src_mdw.matches_one_of_tag(ncw, nchw, ncdhw);
            // decrease src ic_block in case of input nchw
            if (conf.is_nchw) conf.ic_block = 1;
        }
    }

    // TODO: add support for nhwc and dw ow_block
    const bool has_compensation = conf.attr_info.with_src_zpoints
            || conf.attr_info.with_dst_zpoints;
    if (has_compensation)
        if (conf.is_nhwc || (conf.is_depthwise && conf.mb_block != 32))
            return status::unimplemented;

    conf.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    format_tag_t src_tag, dst_tag, wei_tag;

    if (conf.is_nhwc) {
        src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
        dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);

        if (is_1stconv) {
            wei_tag = conf.with_groups
                    ? utils::pick(ndims - 3, gOIw8o4i, gOIhw8o4i, gOIdhw8o4i)
                    : utils::pick(ndims - 3, OIw8o4i, OIhw8o4i, OIdhw8o4i);
            if (!conf.is_dst_nhwc) {
                if (conf.mb_block == 32) {
                    dst_tag = utils::pick(
                            ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
                } else {
                    dst_tag = utils::pick(ndims - 3, nCw32c, nChw32c, nCdhw32c);
                }
            }
        } else if (conf.is_depthwise) {
            wei_tag = utils::pick(ndims - 3, Goiw32g, Goihw32g, Goidhw32g);
        } else {
            wei_tag = conf.with_groups ? utils::pick(ndims - 3, gOIw4o8i8o4i,
                              gOIhw4o8i8o4i, gOIdhw4o8i8o4i)
                                       : utils::pick(ndims - 3, OIw4o8i8o4i,
                                               OIhw4o8i8o4i, OIdhw4o8i8o4i);
        }

    } else {

        if (conf.mb_block == 32) {
            src_tag = utils::pick(
                    ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
            dst_tag = utils::pick(
                    ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        } else {
            src_tag = utils::pick(ndims - 3, nCw32c, nChw32c, nCdhw32c);
            dst_tag = utils::pick(ndims - 3, nCw32c, nChw32c, nCdhw32c);
        }

        if (!conf.is_depthwise && conf.ver == ver_1stconv) {
            src_tag = (conf.is_nchw)
                    ? utils::pick(ndims - 3, ncw, nchw, ncdhw)
                    : utils::pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
        }
        if (conf.is_depthwise) {
            wei_tag = utils::pick(ndims - 3, Goiw32g, Goihw32g, Goidhw32g);
        } else {
            if (conf.ver == ver_1stconv) {
                wei_tag = conf.with_groups
                        ? utils::pick(
                                ndims - 3, gOIw8o4i, gOIhw8o4i, gOIdhw8o4i)
                        : utils::pick(ndims - 3, OIw8o4i, OIhw8o4i, OIdhw8o4i);
            } else {
                wei_tag = conf.with_groups ? utils::pick(ndims - 3,
                                  gOIw4o8i8o4i, gOIhw4o8i8o4i, gOIdhw4o8i8o4i)
                                           : utils::pick(ndims - 3, OIw4o8i8o4i,
                                                   OIhw4o8i8o4i, OIdhw4o8i8o4i);
            }
        }
    }

    conf.src_tag = src_mdw.format_kind() == format_kind::any
            ? src_tag
            : src_mdw.matches_one_of_tag(src_tag);
    conf.wei_tag = weights_mdw.format_kind() == format_kind::any
            ? wei_tag
            : weights_mdw.matches_one_of_tag(wei_tag);
    conf.dst_tag = dst_mdw.format_kind() == format_kind::any
            ? dst_tag
            : dst_mdw.matches_one_of_tag(dst_tag);

    if (conf.src_tag != src_tag || conf.wei_tag != wei_tag
            || conf.dst_tag != dst_tag)
        return status::unimplemented;

    return status::success;
}

status_t gen12lp_x8s8s32x_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("NCHW", conf.is_nchw);
    kernel_ctx.define_int("DST_NHWC", conf.is_dst_nhwc);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);

    kernel_ctx.define_int("OW_PADDED",
            utils::rnd_up(
                    utils::div_up(conf.ow, conf.ow_block), conf.lws_d[1]));
    kernel_ctx.define_int("G_PADDED",
            utils::div_up(conf.ngroups, conf.oc_block) * conf.oc_block);
    int ow = nstl::max(
            1, utils::div_up(conf.iw + 2 * conf.l_pad, conf.stride_w));
    kernel_ctx.define_int("OWX", ow);
    kernel_ctx.define_int("OWB", utils::div_up(conf.ow, conf.ow_block));

    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);

    kernel_ctx.define_int("OC_GROUP", utils::div_up(conf.lws_d[0], 8));
    kernel_ctx.define_int("MB_GROUP", 1);
    kernel_ctx.define_int("SP_GROUP", conf.lws_d[1]);
    kernel_ctx.define_int("OW_NCHUNK", utils::div_up(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OC_NCHUNK", utils::div_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("IC_NCHUNK", utils::div_up(conf.ic, conf.ic_block));

    kernel_ctx.define_int("SLM_WORKING_GROUPS",
            nstl::min(utils::div_up(conf.ow, conf.ow_block),
                    utils::div_up(conf.iw, conf.ow_block * conf.stride_w)));

    int divx
            = conf.iw + conf.l_pad < conf.stride_w * conf.ow_block + conf.kw - 1
            ? conf.iw + conf.l_pad - conf.ow_block * conf.stride_w
            : (conf.iw + conf.l_pad) % (conf.stride_w * conf.ow_block);
    kernel_ctx.define_int("OW_TAIL", conf.ow % conf.ow_block);
    kernel_ctx.define_int("IW_TAIL",
            divx < conf.kw - 1 ? conf.stride_w * conf.ow_block + divx : divx);
    kernel_ctx.define_int("OW_SLM_TAIL",
            conf.iw
                    - conf.stride_w * conf.ow_block
                            * (nstl::min(utils::div_up(conf.ow, conf.ow_block),
                                       utils::div_up(conf.iw,
                                               conf.ow_block * conf.stride_w))
                                    - 1));
    kernel_ctx.define_int("ZERO_TAIL",
            utils::rnd_up(conf.ow, conf.ow_block) * conf.stride_w - conf.iw
                    + (conf.kw - 1) * (1 + conf.dilate_w) - conf.l_pad);

    kernel_ctx.define_int("SRC_SLM_SIZE", conf.src_slm_size);

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    def_attr_info(kernel_ctx, conf.attr_info);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    if (conf.is_depthwise)
        kernel_ctx.define_int("WEI_32G", 1);
    else
        kernel_ctx.define_int("WEI_4O8I8O4I", 1);

    kernel_ctx.set_data_type(conf.dst_data_type);
    def_data_type(kernel_ctx, conf.src_data_type, "SRC");
    def_data_type(kernel_ctx, conf.dst_data_type, "DST");
    def_data_type(kernel_ctx,
            conf.attr_info.sum_data_type == dnnl_data_type_undef
                    ? conf.dst_data_type
                    : conf.attr_info.sum_data_type,
            "SUM");

    kernel_ctx.add_option("-Dcl_intel_subgroups_char");
    kernel_ctx.add_option("-Dcl_intel_subgroups_long");

    return status::success;
}

void gen12lp_x8s8s32x_convolution_fwd_t::pd_t::init_scratchpad() {
    if (conf.attr_info.with_src_zpoints) {
        size_t size = conf.is_depthwise
                ? utils::rnd_up(conf.ngroups, 32)
                : conf.ngroups * utils::rnd_up(conf.oc, 32);

        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_conv_wei_reduction, size,
                types::data_type_size(data_type::s32), OCL_BUFFER_ALIGNMENT);
    }
}

status_t gen12lp_x8s8s32x_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &oscales = CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &src_zpoints
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    auto &dst_zpoints
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    // XXX: first convolution calculates compensation in-place
    const bool precompute_compensation = conf.is_depthwise || conf.ic > 4;

    std::unique_ptr<memory_storage_t> temp_src_compensation;
    if (conf.attr_info.with_src_zpoints && precompute_compensation) {
        temp_src_compensation = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_wei_reduction);

        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src_zpoints);
        arg_list.set(1, weights);
        arg_list.set(2, *temp_src_compensation);

        auto nd_range = conf.is_depthwise
                ? compute::nd_range_t(
                        {16, utils::div_up(conf.ngroups, 32), 1}, {16, 1, 1})
                : compute::nd_range_t(
                        {8, utils::div_up(conf.oc, 32), conf.ngroups},
                        {8, 1, 1});
        status_t status = parallel_for(
                ctx, nd_range, src_compensation_kernel_, arg_list);
        if (status != status::success) return status::runtime_error;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 4, conf.attr_info.all_post_ops);

    if (conf.attr_info.common_oscales) {
        float scales = pd()->attr()->output_scales_.scales_[0];
        arg_list.set(arg_idx++, scales);
    } else {
        arg_list.set(arg_idx++, 1.0f);
    }

    if (conf.attr_info.with_per_oc_oscales) {
        if (conf.attr_info.with_runtime_oscales)
            arg_list.set(arg_idx++, oscales);
        else
            arg_list.set(arg_idx++, CTX_GPU_RES_STORAGE(SCALES_));
    } else {
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());
    }

    if (conf.attr_info.with_src_zpoints) {
        if (precompute_compensation)
            arg_list.set(arg_idx++, *temp_src_compensation);
        else
            arg_list.set(arg_idx++, memory_storage_t::empty_storage());
        arg_list.set(arg_idx++, src_zpoints);
    } else {
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());
    }

    if (conf.attr_info.with_dst_zpoints)
        arg_list.set(arg_idx++, dst_zpoints);
    else
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (!post_ops_preserves_zeroes(ctx, conf.attr_info.all_post_ops)) {
        ctx.memory(DNNL_ARG_DST)->zero_pad(ctx.stream());
    }
    return status;
}

status_t gen12lp_x8s8s32x_convolution_bwd_data_t::pd_t::init_conf() {
    using namespace format_tag;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(diff_src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *diff_src_md(), *weights_md(), *diff_dst_md(),
            *weights_md(1), *attr());

    status_t status = status::success;

    if (conf.mb < 8) return status::unimplemented;

    if (conf.with_groups && conf.ngroups > 1
            && (conf.oc % 32 != 0 || conf.ic % 32 != 0))
        return status::unimplemented;

    conf.dst_data_type = dst_mdw.data_type();
    conf.src_data_type = src_mdw.data_type();

    conf.sub_group_size = 8;
    conf.mb_block = 32;
    conf.oc_block = 32;
    conf.ic_block = 32;
    conf.nchunk = utils::div_up(conf.ic * conf.ngroups, conf.ic_block);
    int ic_group = nstl::min(conf.nchunk, 2);

    conf.lws_d[0] = 8 * ic_group;
    conf.lws_d[1] = 8;
    conf.lws_d[2] = 1;

    conf.gws_d[0] = utils::rnd_up(conf.nchunk * 8, conf.lws_d[0]);
    conf.gws_d[1] = conf.id * conf.ih * utils::rnd_up(conf.iw, conf.lws_d[1]);
    conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block);

    conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block / 2);

    conf.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    format_tag_t src_tag, dst_tag, wei_tag;

    src_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
    dst_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
    wei_tag = conf.with_groups ? utils::pick(conf.ndims - 3, gIOw4i8o8i4o,
                      gIOhw4i8o8i4o, gIOdhw4i8o8i4o)
                               : utils::pick(conf.ndims - 3, IOw4i8o8i4o,
                                       IOhw4i8o8i4o, IOdhw4i8o8i4o);

    conf.src_tag = src_mdw.format_kind() == format_kind::any
            ? src_tag
            : src_mdw.matches_one_of_tag(src_tag);
    conf.wei_tag = weights_mdw.format_kind() == format_kind::any
            ? wei_tag
            : weights_mdw.matches_one_of_tag(wei_tag);
    conf.dst_tag = dst_mdw.format_kind() == format_kind::any
            ? dst_tag
            : dst_mdw.matches_one_of_tag(dst_tag);

    if (conf.src_tag != src_tag || conf.wei_tag != wei_tag
            || conf.dst_tag != dst_tag)
        return status::unimplemented;

    return status;
}

status_t gen12lp_x8s8s32x_convolution_bwd_data_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);

    kernel_ctx.define_int("IW_PADDED", utils::rnd_up(conf.iw, conf.lws_d[1]));

    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);

    kernel_ctx.define_int("IC_GROUP", utils::div_up(conf.lws_d[0], 8));
    kernel_ctx.define_int("MB_GROUP", 1);
    kernel_ctx.define_int("SP_GROUP", conf.lws_d[1]);

    kernel_ctx.define_int("OC_NCHUNK", utils::div_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("IC_NCHUNK", utils::div_up(conf.ic, conf.ic_block));

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.set_data_type(conf.dst_data_type);
    def_data_type(kernel_ctx, conf.src_data_type, "SRC");
    kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    return status::success;
}

status_t gen12lp_x8s8s32x_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {

    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, diff_dst);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
