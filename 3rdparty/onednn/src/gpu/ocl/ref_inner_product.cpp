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

#include "gpu/ocl/ref_inner_product.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(inner_product_conf_t &conf, offsets_t &off,
        const inner_product_pd_t *pd, engine_t *engine) {
    const inner_product_desc_t &ipd = *pd->desc();
    const memory_desc_wrapper src_d(pd->invariant_src_md());
    const memory_desc_wrapper wei_d(pd->invariant_wei_md());
    const memory_desc_wrapper dst_d(pd->invariant_dst_md());
    data_type_t acc_data_type = pd->desc()->accum_data_type;

    const int ndims = src_d.ndims();

    conf.ndims = ndims;
    conf.src_ndims = src_d.ndims();
    conf.wei_ndims = wei_d.ndims();
    conf.dst_ndims = dst_d.ndims();

    conf.has_spatial = utils::one_of(conf.ndims, 3, 4, 5);

    conf.mb = pd->MB();
    conf.ic = pd->IC();

    conf.id = pd->ID();
    conf.ih = pd->IH();
    conf.iw = pd->IW();

    const auto &src_dims = src_d.padded_dims();
    conf.ic_total = utils::array_product(&src_dims[1], conf.ndims - 1);

    conf.oc = pd->OC();

    conf.od = pd->OD();
    conf.oh = pd->OH();
    conf.ow = pd->OW();

    conf.kd = pd->KD();
    conf.kh = pd->KH();
    conf.kw = pd->KW();

    conf.src_dt = src_d.data_type();
    conf.wei_dt = wei_d.data_type();
    conf.dst_dt = dst_d.data_type();
    conf.acc_dt = acc_data_type;

    conf.is_forward = utils::one_of(
            ipd.prop_kind, prop_kind::forward, prop_kind::forward_inference);
    conf.is_backward_data = ipd.prop_kind == prop_kind::backward_data;
    conf.is_backward_weights = ipd.prop_kind == prop_kind::backward_weights;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    if (conf.is_forward) {
        conf.with_bias = ipd.bias_desc.format_kind != format_kind::undef;
        conf.bia_dt = conf.with_bias ? ipd.bias_desc.data_type : data_type::f32;
        conf.dispatch = compute_engine->create_dispatch(dst_d.md_);
        conf.dispatch.define_dim("MB", 0, conf.mb);
        conf.dispatch.define_dim("OC", 1, conf.oc);
        conf.dispatch.generate();
    } else if (conf.is_backward_weights) {
        conf.with_bias = ipd.diff_bias_desc.format_kind != format_kind::undef;
        conf.bia_dt = conf.with_bias ? ipd.diff_bias_desc.data_type
                                     : data_type::f32;
        conf.dispatch = compute_engine->create_dispatch(wei_d.md_);
        conf.dispatch.define_dim("OC", 0, conf.oc);
        conf.dispatch.define_dim("IC", 1, conf.ic);
        conf.dispatch.define_dim("KD", nstl::max(1, ndims - 3), conf.kd);
        conf.dispatch.define_dim("KH", nstl::max(1, ndims - 2), conf.kh);
        conf.dispatch.define_dim("KW", nstl::max(1, ndims - 1), conf.kw);
        conf.dispatch.generate();
    } else {
        conf.with_bias = false;
        conf.bia_dt = data_type::f32;
        conf.dispatch = compute_engine->create_dispatch(src_d.md_);
        conf.dispatch.define_dim("MB_IC", 0, conf.mb * conf.ic);
        conf.dispatch.define_dim("KD", nstl::max(1, ndims - 3), conf.kd);
        conf.dispatch.define_dim("KH", nstl::max(1, ndims - 2), conf.kh);
        conf.dispatch.define_dim("KW", nstl::max(1, ndims - 1), conf.kw);
        conf.dispatch.generate();
    }

    set_offsets(src_d, off.src_off);
    set_offsets(wei_d, off.wei_off);
    set_offsets(dst_d, off.dst_off);

    conf.attr_info = attr_info_t::create(pd->attr());

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const inner_product_conf_t &conf, const offsets_t &off) {
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("IC_TOTAL", conf.ic_total);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    if (conf.with_bias) kernel_ctx.define_int("WITH_BIAS", 1);
    if (conf.has_spatial) kernel_ctx.define_int("HAS_SPATIAL", 1);

    if (conf.is_forward)
        kernel_ctx.define_int("IS_FWD", 1);
    else if (conf.is_backward_data)
        kernel_ctx.define_int("IS_BWD_D", 1);
    else if (conf.is_backward_weights)
        kernel_ctx.define_int("IS_BWD_W", 1);

    def_attr_info(kernel_ctx, conf.attr_info);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.src_ndims);
    def_offsets(off.wei_off, kernel_ctx, "WEI", conf.wei_ndims);
    def_offsets(off.dst_off, kernel_ctx, "DST", conf.dst_ndims);

    if (conf.src_dt == data_type::f16)
        kernel_ctx.set_data_type(data_type::f16);
    else
        kernel_ctx.set_data_type(data_type::f32);

    def_data_type(kernel_ctx, conf.src_dt, "SRC");
    def_data_type(kernel_ctx, conf.wei_dt, "WEI");
    def_data_type(kernel_ctx, conf.bia_dt, "BIA");
    def_data_type(kernel_ctx, conf.dst_dt, "DST");
    def_data_type(kernel_ctx, conf.acc_dt, "ACC");

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_inner_product_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t ref_inner_product_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_inner_product_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    const float *output_scales = pd()->attr()->output_scales_.scales_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 4, conf.attr_info.all_post_ops);

    arg_list.set(arg_idx, output_scales[0]);

    auto nd_range = conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

status_t ref_inner_product_bwd_data_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t ref_inner_product_bwd_data_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_inner_product_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {

    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);

    auto nd_range = conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

status_t ref_inner_product_bwd_weights_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t ref_inner_product_bwd_weights_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_inner_product_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_weights);
    arg_list.set(2, diff_bias);
    arg_list.set(3, diff_dst);

    auto nd_range = conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
