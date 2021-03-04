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

#include "gpu/ocl/rnn/rnn_reorders.hpp"

#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t rnn_weights_reorder_t::pd_t::init_conf(engine_t *engine) {
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    status_t status = status::success;

    const auto &dims = dst_mdw.padded_dims();
    conf.with_sum_ab = (alpha() != 1.f || beta() != 0.f);
    conf.with_sum_a = conf.with_sum_ab && beta() == 0.f;
    conf.do_reorder = src_mdw != dst_mdw;
    conf.has_padding = !src_mdw.is_dense() || !dst_mdw.is_dense();
    conf.ndims = src_mdw.ndims();
    conf.nelems = utils::array_product(dims, conf.ndims);

    conf.use_ref_impl = true;
    conf.with_group = false;
    conf.sub_group_size = 1;

    // only for LDIGO
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    conf.dispatch = compute_engine->create_dispatch(dst_mdw.md_);
    conf.dispatch.define_dim("D0", 0, dims[0]);
    conf.dispatch.define_dim("D1", 1, dims[1]);
    conf.dispatch.define_dim("D3", 3, dims[3]);
    conf.dispatch.define_dim("D4", 4, dims[4]);
    conf.dispatch.generate();

    conf.mask = attr()->rnn_weights_qparams_.mask_;
    const auto &input_dims = src_mdw.dims();
    conf.scales_count = conf.mask ? input_dims[3] * input_dims[4] : 1;

    return status;
}

status_t rnn_weights_reorder_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("NDIMS", conf.ndims);
    if (conf.with_sum_a)
        kernel_ctx.define_int("WITH_SUM_A", 1);
    else if (conf.with_sum_ab)
        kernel_ctx.define_int("WITH_SUM_AB", 1);
    kernel_ctx.define_int("WITH_GROUP", conf.with_group);

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    auto input_type = src_mdw.data_type();
    auto output_type = dst_mdw.data_type();

    switch (input_type) {
        case dnnl_u8: kernel_ctx.define_int("IN_TYPE_U8", 1); break;
        case dnnl_s8: kernel_ctx.define_int("IN_TYPE_S8", 1); break;
        case dnnl_f16: kernel_ctx.define_int("IN_TYPE_F16", 1); break;
        case dnnl_s32: kernel_ctx.define_int("IN_TYPE_S32", 1); break;
        case dnnl_f32: kernel_ctx.define_int("IN_TYPE_F32", 1); break;
        case dnnl_bf16: kernel_ctx.define_int("IN_TYPE_BF16", 1); break;
        default: return status::invalid_arguments;
    }
    switch (output_type) {
        case dnnl_u8: kernel_ctx.define_int("OUT_TYPE_U8", 1); break;
        case dnnl_s8: kernel_ctx.define_int("OUT_TYPE_S8", 1); break;
        case dnnl_f16: kernel_ctx.define_int("OUT_TYPE_F16", 1); break;
        case dnnl_s32: kernel_ctx.define_int("OUT_TYPE_S32", 1); break;
        case dnnl_f32: kernel_ctx.define_int("OUT_TYPE_F32", 1); break;
        case dnnl_bf16: kernel_ctx.define_int("OUT_TYPE_BF16", 1); break;
        default: return status::invalid_arguments;
    }

    kernel_ctx.define_int("REF_REORDER", conf.use_ref_impl);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    set_offsets(kernel_ctx, src_mdw, "SRC");
    set_offsets(kernel_ctx, dst_mdw, "DST");

    const auto &in_dims = src_mdw.dims();
    const auto &out_dims = dst_mdw.padded_dims();

    kernel_ctx.define_int("PAD_FILL_ZERO", conf.has_padding);
    for (int d = 0; d < MAX_NDIMS; ++d)
        kernel_ctx.define_int(utils::format("SRC_D%d", d),
                (d < src_mdw.ndims()) ? in_dims[d] : 1);
    for (int d = 0; d < MAX_NDIMS; ++d)
        kernel_ctx.define_int(utils::format("DST_D%d", d),
                (d < dst_mdw.ndims()) ? out_dims[d] : 1);
    kernel_ctx.define_int("MASK", conf.mask);
    def_dispatch(kernel_ctx, conf.dispatch);
    return status::success;
}

status_t rnn_weights_reorder_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &input = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &output = CTX_OUT_STORAGE(DNNL_ARG_TO);

    const auto &conf = pd()->conf;
    const bool do_reorder = conf.do_reorder;

    auto ocl_reorder = [&](const memory_storage_t &in_storage,
                               const memory_storage_t &scales_storage,
                               const memory_storage_t &out_storage) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, in_storage);
        arg_list.set(1, scales_storage);
        arg_list.set(2, out_storage);

        auto nd_range = conf.dispatch.nd_range();

        return parallel_for(ctx, nd_range, kernel_, arg_list);
    };

    status_t status = status::success;

    const memory_storage_t *scales_buf = nullptr;
    std::unique_ptr<memory_storage_t> wspace;
    if (do_reorder) {
        wspace = ctx.get_scratchpad_grantor().get_memory_storage(
                key_reorder_rnn_space);
        scales_buf = &CTX_GPU_RES_STORAGE(SCALES_);
    }

    // Copy to gpu
    memory_desc_wrapper src_mdw(pd()->src_md());
    status = compute_stream->copy(
            input, do_reorder ? *wspace : output, src_mdw.size());

    if (status == status::success && do_reorder)
        status = ocl_reorder(*wspace, *scales_buf, output);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
