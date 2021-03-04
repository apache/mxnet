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
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <algorithm>
#include "gpu/ocl/gen12lp_x8s8s32x_1x1_convolution.hpp"
#include "gpu/ocl/ocl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gen12lp_x8s8s32x_1x1_convolution_fwd_t::pd_t::init_conf(
        engine_t *engine) {
    using namespace format_tag;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));
    auto dev_info = utils::downcast<compute::compute_engine_t *>(engine)
                            ->device_info();

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    conf.is_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef
            || dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != format_tag::undef;

    if (conf.is_depthwise || conf.kw != 1 || conf.kh != 1 || conf.kd != 1
            || (conf.with_groups && conf.ngroups > 1
                    && (conf.oc % 32 != 0 || conf.ic % 32 != 0)))
        return status::unimplemented;

    conf.src_data_type = src_mdw.data_type();
    conf.dst_data_type = dst_mdw.data_type();

    conf.mb_block = 32;
    conf.oc_block = 32;
    conf.ic_block = 32;
    conf.nchunk = utils::div_up(conf.oc * conf.ngroups, conf.oc_block);

    const bool is_stride1
            = conf.stride_d == 1 && conf.stride_h == 1 && conf.stride_w == 1;
    if (is_stride1) {
        // reshape to nCx32c
        conf.iw = conf.iw * conf.ih * conf.id;
        conf.ow = conf.ow * conf.oh * conf.od;
        conf.ih = conf.id = 1;
        conf.oh = conf.od = 1;
    }

    if ((conf.mb == 8 || conf.mb % 16 == 0) && !conf.is_nhwc) {
        conf.mb_block = 32;
        conf.sp_block = 1;
    } else {
        conf.mb_block = 1;
        conf.sp_block = 4;
        auto approx_clocks = [&](const int block) {
            int ic_chunks = utils::div_up(conf.ic, conf.ic_block);
            bool use_slm = utils::div_up(conf.ow, block) % 8 == 0;
            int mem_clocks = ic_chunks * (16 - use_slm * 6)
                    + block / 2 * (ic_chunks + 1);
            int compute_clocks = 32 * block * ic_chunks;
            int num_threads = conf.nchunk * conf.mb * conf.od * conf.oh
                    * utils::div_up(conf.ow, block);
            return utils::div_up(num_threads, dev_info->hw_threads())
                    * (compute_clocks + mem_clocks);
        };
        auto clock_compare = [&](const int &block1, const int &block2) {
            return approx_clocks(block1) < approx_clocks(block2);
        };
        std::vector<int> sorted_blocks = {4, 8, 12, 16};
        std::sort(sorted_blocks.begin(), sorted_blocks.end(), clock_compare);
        conf.sp_block = sorted_blocks[0];
    }
    conf.src_data_type = src_mdw.data_type();
    conf.dst_data_type = dst_mdw.data_type();

    const int ow_group = (utils::div_up(conf.ow, conf.sp_block) % 8) ? 1 : 8;

    conf.sub_group_size = 8;
    conf.lws_d[0] = conf.sub_group_size;
    conf.lws_d[1] = ow_group;
    conf.lws_d[2] = 1;

    const int num_sp_threads
            = utils::div_up(conf.ow, conf.sp_block) * conf.oh * conf.od;

    conf.gws_d[0] = utils::rnd_up(conf.nchunk * 8, conf.lws_d[0]);
    conf.gws_d[1] = utils::rnd_up(num_sp_threads, conf.lws_d[1]);
    conf.gws_d[2] = utils::div_up(conf.mb, utils::div_up(conf.mb_block, 2));

    conf.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    format_tag_t src_tag, dst_tag, wei_tag;

    if (conf.is_nhwc) {
        src_tag = utils::pick(conf.ndims - 3, nwc, nhwc);
        dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc);
    } else {
        if (conf.mb_block == 32) {
            src_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c);
            dst_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c);
        } else {
            src_tag = utils::pick(conf.ndims - 3, nCw32c, nChw32c);
            dst_tag = utils::pick(conf.ndims - 3, nCw32c, nChw32c);
        }
    }

    wei_tag = conf.with_groups
            ? utils::pick(conf.ndims - 3, gOIw4o8i8o4i, gOIhw4o8i8o4i)
            : utils::pick(conf.ndims - 3, OIw4o8i8o4i, OIhw4o8i8o4i);

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

status_t gen12lp_x8s8s32x_1x1_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic_without_padding);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc_without_padding);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);

    kernel_ctx.define_int("SP_BLOCK", conf.sp_block);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    def_attr_info(kernel_ctx, conf.attr_info);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.define_int("OC_NCHUNK", utils::div_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("IC_NCHUNK", utils::div_up(conf.ic, conf.ic_block));

    kernel_ctx.define_int(
            "INT8_WEI_SLM", utils::div_up(conf.ow, conf.sp_block) % 8 == 0);
    kernel_ctx.define_int("SP_TAIL",
            utils::div_up(conf.ow, conf.sp_block) % conf.lws_d[1] == 0);
    kernel_ctx.define_int("OUT_SP_TAIL", conf.ow % conf.sp_block);

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

    return status::success;
}

void gen12lp_x8s8s32x_1x1_convolution_fwd_t::pd_t::init_scratchpad() {
    if (conf.attr_info.with_src_zpoints) {
        size_t size = conf.ngroups * utils::rnd_up(conf.oc, 32);

        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_conv_wei_reduction, size,
                types::data_type_size(data_type::s32), OCL_BUFFER_ALIGNMENT);
    }
}

status_t gen12lp_x8s8s32x_1x1_convolution_fwd_t::execute_forward(
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

    std::unique_ptr<memory_storage_t> temp_src_compensation;
    if (conf.attr_info.with_src_zpoints) {
        temp_src_compensation = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_wei_reduction);

        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src_zpoints);
        arg_list.set(1, weights);
        arg_list.set(2, *temp_src_compensation);

        auto nd_range = compute::nd_range_t(
                {8, utils::div_up(conf.oc, 32), conf.ngroups}, {8, 1, 1});
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

    if (conf.attr_info.with_src_zpoints)
        arg_list.set(arg_idx++, *temp_src_compensation);
    else
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());

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

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
