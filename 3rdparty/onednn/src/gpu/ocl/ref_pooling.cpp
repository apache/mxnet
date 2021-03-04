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

#include "gpu/ocl/ref_pooling.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(pool_conf_t &conf, offsets_t &off,
        const pooling_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    const memory_desc_wrapper src_mdw(pd->invariant_src_md());
    const memory_desc_wrapper dst_mdw(pd->invariant_dst_md());

    set_default_pool_conf(conf, *pd->desc(), *pd->invariant_src_md(),
            *pd->invariant_dst_md(), *pd->attr());

    set_offsets(src_mdw, off.src_off);
    set_offsets(dst_mdw, off.dst_off);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(
            conf.is_backward ? src_mdw.md_ : dst_mdw.md_);
    conf.dispatch.define_dim("MB", 0, conf.mb);
    conf.dispatch.define_dim("OC", 1, conf.c);
    conf.dispatch.generate();

    conf.attr_info = attr_info_t::create(pd->attr());

    return status::success;
};

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const pool_conf_t &conf, const offsets_t &off) {
    using namespace dnnl::impl::alg_kind;
    kernel_ctx.set_data_type(conf.src_dt);

    kernel_ctx.define_int("SUB_GROUP_SIZE", 1);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("OC_WO_PADDING", conf.c);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("DD", conf.dd);
    kernel_ctx.define_int("DH", conf.dh);
    kernel_ctx.define_int("DW", conf.dw);
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("IS_BWD", conf.is_backward);
    kernel_ctx.define_int("IS_FWD", !conf.is_backward);

    kernel_ctx.define_int("ALG_MAX", (conf.alg == pooling_max));
    kernel_ctx.define_int(
            "ALG_AVG_NP", (conf.alg == pooling_avg_exclude_padding));
    kernel_ctx.define_int(
            "ALG_AVG_P", (conf.alg == pooling_avg_include_padding));

    def_attr_info(kernel_ctx, conf.attr_info);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(off.dst_off, kernel_ctx, "DST", conf.ndims);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_pooling_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t ref_pooling_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_pooling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, ws);
    arg_list.set(2, dst);
    append_post_ops_to_arg_list(ctx, arg_list, 3, conf.attr_info.all_post_ops);

    auto nd_range = pd()->conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

status_t ref_pooling_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t ref_pooling_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_pooling_bwd_t::execute_backward(const exec_ctx_t &ctx) const {

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, ws);
    arg_list.set(2, diff_dst);

    auto nd_range = pd()->conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
