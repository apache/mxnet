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

#include "gpu/ocl/gen9_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

static void fwd_compute_block_sizes(
        conv_conf_t &conf, const convolution_pd_t *pd) {

    int max_ow_block = (conf.src_data_type == data_type::f16 ? 20 : 16);
    if (conf.ver == ver_16mb16c) {
        max_ow_block = 1;
    } else if (conf.is_depthwise || conf.ver == ver_1stconv) {
        max_ow_block = 8;
    }
    max_ow_block = nstl::min(conf.ow, max_ow_block);

    if (conf.ver == ver_16mb16c) {
        conf.mb_block
                = (conf.src_data_type == data_type::f16 && !conf.is_depthwise)
                ? (conf.mb % 32 == 0 ? 32 : 16)
                : 16;
    } else {
        conf.mb_block = 1;
    }

    conf.ow_block = utils::max_div(conf.ow, max_ow_block);

    if (conf.ow_block < max_ow_block / 2) {
        float min_tail_ratio = 1;
        int best_ow_block = -1;
        for (int ow_block = 8; ow_block <= max_ow_block; ow_block++) {
            float tail_ratio
                    = (ow_block - (conf.ow % ow_block)) / (float)conf.ow;
            if (tail_ratio <= min_tail_ratio) {
                min_tail_ratio = tail_ratio;
                best_ow_block = ow_block;
            }
        }
        assert(best_ow_block > 0);
        conf.ow_block = best_ow_block;
    }

    if (conf.is_depthwise) {
        conf.oc_block = 16;
        conf.ic_block = 16;
        conf.omb = conf.mb_block;
        return;
    }

    if (conf.ver == ver_1stconv && conf.mb_block == 1 && conf.oc % 32 == 0) {
        conf.oc_block = 32;
    } else {
        conf.oc_block = 16;
    }
    conf.ic_block = nstl::min(conf.ic, 16);

    conf.omb = (conf.mb_block == 1 && conf.mb % 16 == 0) ? 16 : conf.mb_block;
    conf.ocb = utils::max_div(conf.oc / 16, 8) * 16;
}

status_t gen9_convolution_fwd_t::pd_t::init_conf() {

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    const bool is_src_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef;
    const bool is_dst_nhwc
            = dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef;
    const bool is_nhwc = is_src_nhwc || is_dst_nhwc;

    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);

    conf.is_nhwc = is_1stconv ? is_dst_nhwc : is_nhwc;
    conf.is_depthwise = is_depthwise;

    if (is_1stconv || (conf.with_groups && conf.ngroups > 1)) {
        conf.ic = conf.ic_without_padding;
        conf.oc = is_1stconv ? utils::rnd_up(conf.oc_without_padding, 16)
                             : conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }

    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise) conf.ngroups = utils::rnd_up(conf.ngroups, 16);

    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);
    const bool is_16oc = conf.oc % 16 == 0;
    const bool is_16ic = conf.ic % 16 == 0;

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.omb = 1;
    conf.ocb = 1;

    if (conf.is_nhwc) {
        if (!utils::one_of(src_mdw.data_type(), f32, f16))
            return status::unimplemented;
        // TODO: Add group convolution support in NHWC kernel.
        if (conf.ngroups > 1 && !(is_16oc && is_16ic)) {
            return status::unimplemented;
        }
        conf.ver = ver_nhwc;
    } else if (is_1stconv) {
        if (!is_16oc) return status::unimplemented;
        conf.ver = ver_1stconv;
    } else if ((is_16oc && is_16ic) || is_dw_16g) {
        conf.ver = (conf.mb % 16 == 0) ? ver_16mb16c : ver_8ow16c;
    } else {
        return status::unimplemented;
    }

    switch (conf.ver) {
        case ver_nhwc: {
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = is_1stconv ? 1 : 16;

            int max_ow_block = (conf.kw > 1) ? 8 : 16;
            if (conf.oc <= 64 && conf.ic <= 64) max_ow_block = 8;

            conf.ow_block = utils::max_div(conf.ow, max_ow_block);

            if (conf.ow_block <= 8) {
                int max_tail = 0;
                for (int j = 8; j < max_ow_block; j++) {
                    if (conf.ow % j > max_tail) {
                        max_tail = conf.ow % j;
                        conf.ow_block = j;
                    }
                }
            }
            if (conf.ow_block <= 8) conf.ow_block = 8;
            if (conf.ow <= 8 || conf.oc <= 32) conf.ow_block = 8;

            conf.oh_block = 1;
            conf.sub_group_size = 16;
            conf.lws_d[0] = 16;
            conf.lws_d[1] = 1;
            conf.lws_d[2] = 1;

            int max_oc_block = 8;
            if (conf.is_depthwise) {
                conf.ocb = conf.ngroups;
            } else {
                conf.ocb = conf.oc_block
                        * utils::max_div(utils::div_up(conf.oc, conf.oc_block),
                                max_oc_block);
            }

            conf.gws_d[0] = conf.ocb;
            conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                    * utils::div_up(conf.ow, conf.ow_block) * conf.od;
            if (conf.is_depthwise) {
                conf.gws_d[2] = conf.mb;
            } else {
                conf.gws_d[2] = conf.mb * utils::div_up(conf.oc, conf.ocb)
                        * conf.ngroups;
            }
        } break;
        case ver_1stconv:
        case ver_8ow16c:
        case ver_16mb16c: {
            fwd_compute_block_sizes(conf, this);
            conf.sub_group_size = 16;
            conf.lws_d[0] = 16;
            conf.lws_d[1] = 1;
            conf.lws_d[2] = 1;
            conf.gws_d[0] = conf.ngroups * conf.ocb / (conf.oc_block / 16);
            conf.gws_d[1] = conf.od * conf.oh
                    * utils::div_up(conf.ow, conf.ow_block)
                    * (conf.omb / conf.mb_block);
            conf.gws_d[2] = (conf.oc / conf.ocb) * (conf.mb / conf.omb);
            break;
        }
        default: return status::unimplemented;
    }

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_nhwc:
            src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            if (is_1stconv) {
                wei_tag = conf.with_groups ? utils::pick(
                                  conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                                           : utils::pick(conf.ndims - 3, Owi16o,
                                                   Ohwi16o, Odhwi16o);
            } else if (conf.is_depthwise) {
                wei_tag = utils::pick(
                        conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g);
            } else {
                wei_tag = conf.with_groups
                        ? utils::pick(conf.ndims - 3, gOIw16i16o, gOIhw16i16o,
                                gOIdhw16i16o)
                        : utils::pick(conf.ndims - 3, OIw16i16o, OIhw16i16o,
                                OIdhw16i16o);
            }
            break;
        case ver_1stconv:
            if (is_src_nhwc)
                src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            else
                src_tag = utils::pick(conf.ndims - 3, ncw, nchw, ncdhw);
            dst_tag = conf.mb % 16 == 0
                    ? utils::pick(
                            conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c)
                    : utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.with_groups
                    ? utils::pick(conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : utils::pick(conf.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            break;
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
                                        : utils::pick(conf.ndims - 3, OIw16i16o,
                                                OIhw16i16o, OIdhw16i16o));
            break;
        default: return status::unimplemented;
    }

    if (src_mdw.format_kind() == format_kind::any) {
        conf.src_tag = src_tag;
    } else {
        conf.src_tag = src_mdw.matches_one_of_tag(src_tag);
    }
    if (conf.src_tag != src_tag) return status::unimplemented;

    if (weights_mdw.format_kind() == format_kind::any) {
        conf.wei_tag = wei_tag;
    } else {
        conf.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
    }
    if (conf.wei_tag != wei_tag) return status::unimplemented;

    if (dst_mdw.format_kind() == format_kind::any) {
        conf.dst_tag = dst_tag;
    } else {
        conf.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
    }
    if (conf.dst_tag != dst_tag) return status::unimplemented;

    conf.is_src_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
    conf.is_src_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

    return status::success;
}

status_t gen9_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
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
    kernel_ctx.define_int("PD_R", conf.back_pad);
    kernel_ctx.define_int("PH_R", conf.b_pad);
    kernel_ctx.define_int("PW_R", conf.r_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("OW_PADDED", utils::rnd_up(conf.ow, 4));
    kernel_ctx.define_int("OC_PADDED", conf.oc);
    kernel_ctx.define_int("OMB", conf.omb);
    kernel_ctx.define_int("OCB", conf.ocb);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OH_BLOCK", conf.oh_block);
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);
    kernel_ctx.define_int("OW_LAST", utils::rnd_dn(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OWB", utils::div_up(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OHB", utils::div_up(conf.oh, conf.oh_block));
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("G_WO_PADDING", conf.ngroups_without_padding);
    kernel_ctx.define_int("IC_WO_PADDING", conf.ic_without_padding);
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("OC_GROUP", conf.lws_d[0] / 8);
    kernel_ctx.define_int("MB_GROUP", 1);
    kernel_ctx.define_int("SP_GROUP", conf.lws_d[1]);
    if (conf.kw == 1)
        kernel_ctx.define_int("SRC_SP_GROUP", conf.lws_d[1] + conf.kw - 1);
    else
        kernel_ctx.define_int(
                "SRC_SP_GROUP", conf.stride_w * (conf.lws_d[1] - 1) + conf.kw);

    kernel_ctx.set_data_type(conf.src_data_type);

    kernel_ctx.define_int("VER_1STCONV", conf.ver == ver_1stconv);
    kernel_ctx.define_int("VER_8OW16C", conf.ver == ver_8ow16c);
    kernel_ctx.define_int("VER_16MB16C", conf.ver == ver_16mb16c);

    kernel_ctx.define_int("SRC_NCHW", conf.is_src_nchw);
    kernel_ctx.define_int("SRC_NHWC", conf.is_src_nhwc);
    kernel_ctx.define_int("SRC_16N16C",
            utils::one_of(conf.src_tag, NCw16n16c, NChw16n16c, NCdhw16n16c));
    kernel_ctx.define_int(
            "SRC_W16C", utils::one_of(conf.src_tag, nCw16c, nChw16c, nCdhw16c));

    kernel_ctx.define_int("WEI_I16O",
            utils::one_of(conf.wei_tag, gOwi16o, gOhwi16o, gOdhwi16o, Owi16o,
                    Ohwi16o, Odhwi16o));
    kernel_ctx.define_int("WEI_16I16O",
            utils::one_of(conf.wei_tag, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o,
                    OIw16i16o, OIhw16i16o, OIdhw16i16o));
    kernel_ctx.define_int("WEI_16I16O_FLIPPED",
            utils::one_of(conf.wei_tag, gIOw16i16o, gIOhw16i16o, gIOdhw16i16o,
                    IOw16i16o, IOhw16i16o, IOdhw16i16o));

    kernel_ctx.define_int("DST_16N16C",
            utils::one_of(conf.dst_tag, NCw16n16c, NChw16n16c, NCdhw16n16c));
    kernel_ctx.define_int(
            "DST_W16C", utils::one_of(conf.dst_tag, nCw16c, nChw16c, nCdhw16c));

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    def_attr_info(kernel_ctx, conf.attr_info);

    kernel_ctx.print_options();
    return status::success;
}

status_t gen9_convolution_bwd_data_t::pd_t::init_conf() {
    using namespace dnnl::impl::format_tag;
    using namespace data_type;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(diff_src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *diff_src_md(), *weights_md(), *diff_dst_md(),
            *weights_md(1), *attr());
    const bool is_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef
            || dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != format_tag::undef;
    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);
    conf.is_nhwc = is_nhwc;
    conf.is_depthwise = is_depthwise;

    if (is_nhwc && (is_depthwise || is_1stconv)) return status::unimplemented;

    if (is_1stconv || (conf.with_groups && conf.ngroups > 1)) {
        conf.ic = conf.ic_without_padding;
        conf.oc = is_1stconv ? utils::rnd_up(conf.oc_without_padding, 16)
                             : conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }
    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise) conf.ngroups = utils::rnd_up(conf.ngroups, 16);
    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);

    const bool is_16ic = conf.ic % 16 == 0;
    const bool is_16oc = conf.oc % 16 == 0;
    const bool use_16mb_unroll = !is_nhwc
            && !(conf.mb == 1 || conf.mb % 16 != 0) && !is_1stconv
            && ((is_16ic && is_16oc) || is_dw_16g);
    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.icb = 1;
    if (is_nhwc)
        conf.ver = ver_nhwc;
    else if (use_16mb_unroll)
        conf.ver = ver_16mb16c;
    else if (conf.mb % 16 != 0 && ((is_16oc && is_16ic) || is_dw_16g))
        conf.ver = ver_8ow16c;
    else
        return status::unimplemented;

    status_t status = status::success;
    switch (conf.ver) {
        case ver_16mb16c:
            conf.mb_block = 16;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.od_block = 1;
            conf.ih_block = 1;
            conf.iw_block = 1;
            conf.sub_group_size = 16;
            if (conf.is_depthwise) {
                conf.icb = conf.ngroups;
                conf.lws_d[0] = 1;
                conf.lws_d[1] = 16;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.ih * conf.iw * conf.id;
                conf.gws_d[1] = conf.ic * conf.ngroups;
                conf.gws_d[2] = conf.mb / 16;
            } else {
                conf.icb = 64;
                while (conf.icb > 16) {
                    if (conf.ic % conf.icb == 0) break;
                    conf.icb /= 2;
                }
                conf.lws_d[0] = 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.icb;
                conf.gws_d[1] = conf.ih * conf.iw * conf.id;
                conf.gws_d[2]
                        = conf.mb / 16 * (conf.ic / conf.icb) * conf.ngroups;
            }
            break;
        case ver_8ow16c:
        case ver_nhwc: {
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.od_block = 1;
            conf.ih_block = 1;
            int max_iw_block = 16;
            if (conf.ver == ver_nhwc) { max_iw_block = (conf.kw > 1) ? 8 : 16; }
            conf.iw_block = nstl::max(8, utils::max_div(conf.iw, max_iw_block));
            conf.sub_group_size = 16;
            if (conf.is_depthwise) {
                conf.icb = conf.ngroups;
                conf.lws_d[0] = 1;
                conf.lws_d[1] = 16;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.ih * utils::div_up(conf.iw, conf.iw_block)
                        * conf.id;
                conf.gws_d[1] = conf.ic * conf.ngroups;
                conf.gws_d[2] = conf.mb;
            } else {
                conf.icb = 64;
                while (conf.icb > 16) {
                    if (utils::rnd_up(conf.ic, conf.ic_block) % conf.icb == 0)
                        break;
                    conf.icb /= 2;
                }
                conf.lws_d[0] = 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.icb;
                conf.gws_d[1] = conf.ih * utils::div_up(conf.iw, conf.iw_block)
                        * conf.id;
                conf.gws_d[2] = conf.mb
                        * (utils::rnd_up(conf.ic, conf.ic_block) / conf.icb)
                        * conf.ngroups;
            }
            break;
        }
        default: status = status::unimplemented;
    }

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_nhwc:
            src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            wei_tag = conf.with_groups ? utils::pick(conf.ndims - 3, gOIw16o16i,
                              gOIhw16o16i, gOIdhw16o16i)
                                       : utils::pick(conf.ndims - 3, OIw16o16i,
                                               OIhw16o16i, OIdhw16o16i);
            break;
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
                                        : utils::pick(conf.ndims - 3, OIw16o16i,
                                                OIhw16o16i, OIdhw16o16i));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
                                        : utils::pick(conf.ndims - 3, OIw16o16i,
                                                OIhw16o16i, OIdhw16o16i));
            break;
        default: status = status::unimplemented;
    }
    if (status != status::success) return status;

    if (src_mdw.format_kind() == format_kind::any) {
        conf.src_tag = src_tag;
    } else {
        conf.src_tag = src_mdw.matches_one_of_tag(src_tag);
    }
    if (conf.src_tag != src_tag) return status::unimplemented;

    if (weights_mdw.format_kind() == format_kind::any) {
        conf.wei_tag = wei_tag;
    } else {
        conf.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
    }
    if (conf.wei_tag != wei_tag) return status::unimplemented;

    if (dst_mdw.format_kind() == format_kind::any) {
        conf.dst_tag = dst_tag;
    } else {
        conf.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
    }
    if (conf.dst_tag != dst_tag) return status::unimplemented;

    conf.is_src_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
    conf.is_src_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

    return status::success;
}

status_t gen9_convolution_bwd_data_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
    kernel_ctx.define_int("BWD_DATA", 1);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ICB", conf.icb);
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
    kernel_ctx.define_int("PD_R", conf.back_pad);
    kernel_ctx.define_int("PH_R", conf.b_pad);
    kernel_ctx.define_int("PW_R", conf.r_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("OC_PADDED", utils::rnd_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("IC_PADDED", utils::rnd_up(conf.ic, conf.ic_block));
    kernel_ctx.define_int("G_WO_PADDING", conf.ngroups_without_padding);
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("IC_WO_PADDING", conf.ic_without_padding);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("IH_BLOCK", conf.ih_block);
    kernel_ctx.define_int("IW_BLOCK", conf.iw_block);
    kernel_ctx.define_int("IWB", utils::div_up(conf.iw, conf.iw_block));
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.set_data_type(conf.src_data_type);

    switch (conf.ver) {
        case ver_16mb16c: kernel_ctx.define_int("VER_16MB16C", 1); break;
        case ver_8ow16c: kernel_ctx.define_int("VER_8OW16C", 1); break;
        default: break;
    }

    return status::success;
}

status_t gen9_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {

    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);
    arg_list.set(3, bias);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

static void bwd_w_compute_block_sizes(
        conv_conf_t &conf, const convolution_pd_t *pd, engine_t *engine) {
    const bool is_1stconv = conf.ic_without_padding == 3;

    if (conf.is_depthwise) {
        conf.odb = 1;
        conf.ohb = 1;
        conf.owb = utils::rnd_up(conf.ow, conf.ow_block);
        conf.ocb = 1;
        conf.icb = 1;
        conf.osp_chunk = utils::div_up(conf.od, conf.odb)
                * utils::div_up(conf.oh, conf.ohb)
                * utils::div_up(conf.ow, conf.owb);

        conf.mb_chunk = utils::div_up(conf.mb, conf.mb_block);
        conf.nchunk = conf.osp_chunk * conf.mb_chunk;
        return;
    }
    auto *dev_info = utils::downcast<compute::compute_engine_t *>(engine)
                             ->device_info();
    int hw_threads = dev_info->hw_threads();
    size_t llc_bytes = dev_info->llc_cache_size();

    auto next_candidate = [](int size, int block) {
        if (size == block) return block;
        // If size is big enough, then do not care about the remainder.
        if (block * 16 < size) return block + 1;
        // Otherwise search for the next divisor.
        block++;
        while (size % block != 0)
            block++;
        return block;
    };

    int mb_nb = 1;
    conf.odb = 1;
    conf.ohb = 1;
    conf.owb = 1;

    int mb_nblk = utils::div_up(conf.mb, conf.mb_block);
    int ic_nblk = utils::div_up(conf.ic, conf.ic_block);
    int oc_nblk = utils::div_up(conf.oc, conf.oc_block);

    int ic_nb_max = is_1stconv ? 1 : nstl::min(ic_nblk, 16);
    int oc_nb_max = nstl::min(oc_nblk, 16);
    int ic_nb = is_1stconv ? 1 : utils::max_div(ic_nblk, ic_nb_max);
    int oc_nb = utils::max_div(oc_nblk, oc_nb_max);

    int mb_nb_max = 1;
    if (!is_1stconv && (conf.mb_block == 1) && (conf.ic % 1024 != 0)
            && (conf.oc % 1024 != 0)) {
        mb_nb_max = 4;
    }

    auto get_nthr = [&]() {
        int nthr = utils::div_up(mb_nblk, mb_nb)
                * utils::div_up(conf.od, conf.odb)
                * utils::div_up(conf.oh, conf.ohb)
                * utils::div_up(conf.ow, conf.owb) * conf.kh * conf.kw * conf.kd
                * oc_nblk * (is_1stconv ? 1 : ic_nblk) * conf.ngroups;
        return nthr;
    };

    auto get_src_dst_size = [&]() {
        int iwb = conf.ndims < 3 ? 1 : conf.owb + 2 * (conf.kw - 1);
        int ihb = conf.ndims < 4 ? 1 : conf.ohb + 2 * (conf.kh - 1);
        int idb = conf.ndims < 5 ? 1 : conf.odb + 2 * (conf.kd - 1);

        size_t ispb = iwb * ihb * idb;
        size_t ospb = conf.owb * conf.ohb * conf.odb;
        size_t src_size = sizeof(float) * conf.mb_block
                * (is_1stconv ? conf.ic : ic_nb * conf.ic_block) * ispb;
        size_t dst_size = sizeof(float) * conf.mb_block
                * (oc_nb * conf.oc_block) * ospb;

        int nthr_per_spb
                = conf.kh * conf.kw * conf.kd * ic_nb * oc_nb * conf.ngroups;
        size_t sz = (size_t)(src_size + dst_size);
        if (nthr_per_spb < hw_threads) sz = sz * hw_threads / nthr_per_spb;
        return sz;
    };

    auto try_next = [&](int &v, int next) {
        if (next <= v) return false;
        int v_old = v;
        v = next;
        // Heuristics:
        // - src and dst size accessed in the inner loops should fit LLC
        // - Require at least (3 * hw_threads) to spawn to have enough
        //   parallelism
        if (get_src_dst_size() > llc_bytes || get_nthr() < 3 * hw_threads) {
            v = v_old;
            return false;
        }
        return true;
    };

    if (utils::one_of(conf.ver, ver_nhwc, ver_8ow16c, ver_1stconv))
        conf.owb = conf.ow_block;

    // Increase spatial tile size as much as possible.
    for (int i = 0; i < 128; i++) {
        int owb_next;
        if (utils::one_of(conf.ver, ver_nhwc, ver_8ow16c, ver_1stconv)) {
            int ow_padded = utils::rnd_up(conf.ow, conf.ow_block);
            owb_next = conf.ow_block
                    * next_candidate(ow_padded / conf.ow_block,
                            conf.owb / conf.ow_block);
        } else {
            owb_next = next_candidate(conf.ow, conf.owb);
        }
        try_next(conf.owb, owb_next);

        int ohb_next = next_candidate(conf.oh, conf.ohb);
        try_next(conf.ohb, ohb_next);

        int odb_next = next_candidate(conf.od, conf.odb);
        try_next(conf.odb, odb_next);

        int mb_nb_next = next_candidate(mb_nb_max, mb_nb);
        try_next(mb_nb, mb_nb_next);
    }

    conf.icb = is_1stconv ? conf.ic : ic_nb * conf.ic_block;
    conf.ocb = oc_nb * conf.oc_block;

    conf.osp_chunk = utils::div_up(conf.od, conf.odb)
            * utils::div_up(conf.oh, conf.ohb)
            * utils::div_up(conf.ow, conf.owb);

    conf.mb_chunk = utils::div_up(mb_nblk, mb_nb);

    conf.nchunk = conf.mb_chunk * conf.osp_chunk;
}

status_t gen9_convolution_bwd_weights_t::pd_t::init_conf(engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    using namespace data_type;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(diff_weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());
    const memory_desc_wrapper bias_mdw(diff_weights_md(1));

    set_default_conf(conf, cd, *src_md(), *diff_weights_md(), *diff_dst_md(),
            *diff_weights_md(1), *attr());

    const bool is_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef
            || dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != format_tag::undef;

    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);

    conf.is_nhwc = is_nhwc;
    conf.is_depthwise = is_depthwise;

    if (is_1stconv || (conf.with_groups && conf.ngroups > 1) || is_nhwc) {
        conf.ic = conf.ic_without_padding;
        conf.oc = is_1stconv ? utils::rnd_up(conf.oc_without_padding, 16)
                             : conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }

    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise && !is_nhwc)
        conf.ngroups = utils::rnd_up(conf.ngroups, 16);
    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);

    const bool is_16ic = conf.ic % 16 == 0;
    const bool is_16oc = conf.oc % 16 == 0;
    const bool use_16mb_unroll = !is_nhwc
            && !(conf.mb == 1 || conf.mb % 16 != 0) && !is_1stconv
            && ((is_16ic && is_16oc) || is_dw_16g);

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.osp_chunk = 1;
    conf.mb_chunk = 1;
    if (is_nhwc)
        conf.ver = ver_nhwc;
    else if (use_16mb_unroll)
        conf.ver = ver_16mb16c;
    else if (conf.mb % 16 != 0 && ((is_16oc && is_16ic) || is_dw_16g))
        conf.ver = ver_8ow16c;
    else if (is_1stconv && is_16oc)
        conf.ver = ver_1stconv;
    else
        return status::unimplemented;

    switch (conf.ver) {
        case ver_1stconv:
        case ver_8ow16c:
        case ver_nhwc:
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = is_1stconv ? 1 : 16;
            conf.ow_block = is_1stconv && !is_nhwc ? 16 : 8;
            break;
        case ver_16mb16c:
            conf.mb_block = 16;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.ow_block = 1;
            break;
    }

    bwd_w_compute_block_sizes(conf, this, engine);

    conf.sub_group_size = 16;
    conf.lws_d[0] = 16;
    conf.lws_d[1] = 1;
    conf.lws_d[2] = 1;

    if (conf.is_depthwise) {
        conf.gws_d[0] = utils::rnd_up(conf.ngroups, 16);
    } else {
        conf.gws_d[0] = is_1stconv ? conf.ocb * conf.ngroups
                                   : conf.ocb * (conf.icb / 16) * conf.ngroups;
    }
    conf.gws_d[1] = conf.kh * conf.kw * conf.kd;
    conf.gws_d[2] = conf.nchunk * utils::div_up(conf.ic, conf.icb)
            * utils::div_up(conf.oc, conf.ocb);

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_nhwc:
            src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            if (is_1stconv) {
                wei_tag = conf.with_groups ? utils::pick(
                                  conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                                           : utils::pick(conf.ndims - 3, Owi16o,
                                                   Ohwi16o, Odhwi16o);
            } else if (conf.is_depthwise) {
                wei_tag = utils::pick(
                        conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g);
            } else {
                wei_tag = conf.with_groups
                        ? utils::pick(conf.ndims - 3, gIOw16i16o, gIOhw16i16o,
                                gIOdhw16i16o)
                        : utils::pick(conf.ndims - 3, IOw16i16o, IOhw16i16o,
                                IOdhw16i16o);
            }
            break;
        case ver_1stconv:
            assert(!conf.is_depthwise);
            src_tag = utils::pick(conf.ndims - 3, ncw, nchw, ncdhw);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.with_groups
                    ? utils::pick(conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : utils::pick(conf.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            break;
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        default: return status::unimplemented;
    }

    if (src_mdw.format_kind() == format_kind::any) {
        conf.src_tag = src_tag;
    } else {
        conf.src_tag = src_mdw.matches_one_of_tag(src_tag);
    }
    if (conf.src_tag != src_tag) return status::unimplemented;

    if (weights_mdw.format_kind() == format_kind::any) {
        conf.wei_tag = wei_tag;
    } else {
        conf.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
    }
    if (conf.wei_tag != wei_tag) return status::unimplemented;

    if (dst_mdw.format_kind() == format_kind::any) {
        conf.dst_tag = dst_tag;
    } else {
        conf.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
    }
    if (conf.dst_tag != dst_tag) return status::unimplemented;

    conf.is_src_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
    conf.is_src_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

    return status::success;
}

status_t gen9_convolution_bwd_weights_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
    kernel_ctx.define_int("BWD_WEIGHTS", 1);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ICB", conf.icb);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("OCB", conf.ocb);
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
    kernel_ctx.define_int("PD_R", conf.back_pad);
    kernel_ctx.define_int("PH_R", conf.b_pad);
    kernel_ctx.define_int("PW_R", conf.r_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("OC_PADDED", conf.oc);
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("G_WO_PADDING", conf.ngroups_without_padding);

    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);
    kernel_ctx.define_int("ODB", conf.odb);
    kernel_ctx.define_int("OHB", conf.ohb);
    kernel_ctx.define_int("OWB", conf.owb);

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("NCHUNK", conf.nchunk);
    kernel_ctx.define_int("OSP_CHUNK", conf.osp_chunk);
    kernel_ctx.define_int("MB_CHUNK", conf.mb_chunk);
    kernel_ctx.define_int(
            "MB_CHUNK_SIZE", utils::div_up(conf.mb, conf.mb_chunk));
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.add_option("-cl-std=CL2.0");

    switch (conf.ver) {
        case ver_16mb16c: kernel_ctx.define_int("VER_16MB16C", 1); break;
        case ver_1stconv:
        case ver_8ow16c: kernel_ctx.define_int("VER_8OW16C", 1); break;
        default: break;
    }

    return status::success;
}

status_t gen9_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;
    const auto &attr_info = conf.attr_info;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);
    append_post_ops_to_arg_list(ctx, arg_list, 4, attr_info.all_post_ops);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (!post_ops_preserves_zeroes(ctx, attr_info.all_post_ops)) {
        ctx.memory(DNNL_ARG_DST)->zero_pad(ctx.stream());
    }
    return status;
}

status_t gen9_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const auto &conf = pd()->conf;

    const uint8_t zero = 0;
    memory_desc_wrapper wei_mdw(pd()->diff_weights_md());
    CHECK(compute_stream->fill(diff_weights, zero, wei_mdw.size()));
    if (conf.with_bias) {
        memory_desc_wrapper bia_mdw(pd()->diff_weights_md(1));
        CHECK(compute_stream->fill(diff_bias, zero, bia_mdw.size()));
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_weights);
    arg_list.set(2, diff_bias);
    arg_list.set(3, diff_dst);

    status_t status = parallel_for(ctx,
            compute::nd_range_t(conf.gws_d, conf.lws_d), kernel_, arg_list);
    if (status != status::success) return status;

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
