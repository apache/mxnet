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

#include "gpu/ocl/ref_layer_normalization.hpp"

#include "common/primitive_exec_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(lnorm_conf_t &conf,
        const layer_normalization_pd_t *pd, engine_t *engine) {
    memory_desc_wrapper src_mdw(pd->src_md());
    memory_desc_wrapper stat_mdw(pd->stat_md());
    memory_desc_wrapper dst_mdw(pd->dst_md());

    int ndims = src_mdw.ndims();

    conf.data_type = src_mdw.data_type();
    conf.ndims = ndims;
    conf.norm_axis = pd->norm_axis();

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.stat_md_info = memory_desc_info_t::create(stat_mdw);

    conf.is_fwd = pd->is_fwd();

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch_scaleshift = compute_engine->create_dispatch();
    conf.dispatch = compute_engine->create_dispatch(
            pd->is_fwd() ? dst_mdw.md_ : src_mdw.md_);
    auto &dims = (pd->is_fwd() ? src_mdw : dst_mdw).dims();
    if (pd->is_fwd()) {
        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            int dim = (i < ndims - 1) ? dims[i] : 1;
            conf.dispatch.define_dim(utils::format("X%d", i), md_hint_idx, dim);
        }
    } else {
        conf.dispatch_scaleshift.define_dim("C", pd->norm_axis());

        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            int dim = (i < ndims - 1) ? dims[i] : 1;
            conf.dispatch.define_dim(utils::format("X%d", i), md_hint_idx, dim);
        }
    }

    if (!pd->is_fwd()) {
        conf.dispatch_scaleshift.set_kernel_attr_suffix("SCALESHIFT");
        conf.dispatch_scaleshift.generate();
    }

    conf.dispatch.generate();

    conf.use_scaleshift = pd->use_scaleshift();
    conf.calculate_stats = !pd->stats_are_src();
    conf.save_stats = pd->is_training();
    conf.eps = pd->desc()->layer_norm_epsilon;

    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const lnorm_conf_t &conf) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("C", conf.norm_axis);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("USE_SCALESHIFT", conf.use_scaleshift);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_FWD", conf.is_fwd);
    kernel_ctx.define_int("IS_BWD", !conf.is_fwd);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
    def_memory_desc_info(kernel_ctx, conf.stat_md_info, "STAT");

    def_dispatch(kernel_ctx, conf.dispatch);
    if (!conf.is_fwd) def_dispatch(kernel_ctx, conf.dispatch_scaleshift);

    return status::success;
}

status_t ref_layer_normalization_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_layer_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

status_t ref_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
                                       : CTX_OUT_STORAGE(DNNL_ARG_MEAN);

    auto &variance = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
                                           : CTX_OUT_STORAGE(DNNL_ARG_VARIANCE);

    auto &scaleshift = CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, conf.eps);

    auto nd_range_kernel = conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

status_t ref_layer_normalization_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_layer_normalization_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

status_t ref_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scaleshift = CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT);

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_scaleshift = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE_SHIFT);

    const auto &conf = pd()->conf;

    if (conf.use_scaleshift) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, mean);
        arg_list.set(2, variance);
        arg_list.set(3, diff_dst);
        arg_list.set(4, diff_scaleshift);
        arg_list.set(5, conf.eps);

        auto nd_range = conf.dispatch_scaleshift.nd_range();
        status_t status
                = parallel_for(ctx, nd_range, kernel_scaleshift_, arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, diff_src);
    arg_list.set(6, conf.eps);

    auto nd_range_kernel = conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
