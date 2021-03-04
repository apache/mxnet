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

#include "gpu/ocl/ref_binary.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_binary_t::pd_t::init_conf(engine_t *engine) {
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper dst_d(dst_md());

    alg_kind_t alg = desc()->alg_kind;

    const int ndims = src0_d.ndims();
    conf.src0_md_info = memory_desc_info_t::create(src0_d);
    conf.src1_md_info = memory_desc_info_t::create(src1_d);
    conf.dst_md_info = memory_desc_info_t::create(dst_d);
    conf.src0_data_type = src0_d.data_type();
    conf.src1_data_type = src1_d.data_type();
    conf.dst_data_type = dst_d.data_type();
    conf.ndims = ndims;
    for (int i = 0; i < MAX_NDIMS; ++i) {
        conf.bcast_dims[i] = i < ndims ? broadcast_dims()[i] : 1;
    }
    conf.is_add = (alg == alg_kind::binary_add);
    conf.is_mul = (alg == alg_kind::binary_mul);
    conf.is_max = (alg == alg_kind::binary_max);
    conf.is_min = (alg == alg_kind::binary_min);
    conf.is_div = (alg == alg_kind::binary_div);
    conf.is_tensor_op = is_tensor_op();
    conf.is_dense = dst_d.is_dense();
    conf.same_src_dt = (src0_d.data_type() == src1_d.data_type());
    conf.is_same_md = (src0_d == dst_d) && (src1_d == dst_d);
    conf.attr_info = attr_info_t::create(attr());
    conf.with_binary_post_op
            = conf.attr_info.all_post_ops.find(primitive_kind::binary) != -1;
    int ic_block_sz = 1;
    conf.use_unroll_16b = false;
    conf.src0_unroll_16b = false;

    auto &blk0 = src0_d.blocking_desc();
    auto &blk1 = src1_d.blocking_desc();
    bool is_16b_blk0 = (blk0.inner_nblks >= 1)
            && (blk0.inner_idxs[blk0.inner_nblks - 1] == 1)
            && (blk0.inner_blks[blk0.inner_nblks - 1] == 16);
    bool is_16b_blk1 = (blk1.inner_nblks >= 1)
            && (blk1.inner_idxs[blk1.inner_nblks - 1] == 1)
            && (blk1.inner_blks[blk1.inner_nblks - 1] == 16);

    if (!conf.is_tensor_op) {
        // If: in case when both are blocked
        // Else: only src0 is blocked
        if (is_16b_blk0 && is_16b_blk1) {
            ic_block_sz = 16;
            conf.use_unroll_16b = true;
        } else if (is_16b_blk0 && blk1.inner_nblks == 0) {
            ic_block_sz = 16;
            conf.src0_unroll_16b = true;
        }
    }

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_d.md_);
    if (conf.is_tensor_op && conf.is_dense && conf.is_same_md
            && !conf.with_binary_post_op) {
        conf.dispatch.define_dim("IDX", 0, dst_d.nelems());
    } else {
        for (int i = 0; i < MAX_NDIMS; ++i) {
            if (i == 1 && (conf.use_unroll_16b || conf.src0_unroll_16b)) {
                // changing value for broadcasting offsets
                // division by IC for enabling blocking within kernel
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1),
                        i < ndims ? dst_d.padded_dims()[i] : 1, ic_block_sz);
            } else {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1),
                        i < ndims ? dst_d.dims()[i] : 1);
            }
        }
    }
    conf.dispatch.generate();
    return status::success;
}

status_t ref_binary_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(conf.src0_data_type);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("IS_MUL", conf.is_mul);
    kernel_ctx.define_int("IS_ADD", conf.is_add);
    kernel_ctx.define_int("IS_MAX", conf.is_max);
    kernel_ctx.define_int("IS_MIN", conf.is_min);
    kernel_ctx.define_int("IS_DIV", conf.is_div);
    kernel_ctx.define_int("IS_TENSOR_OP", conf.is_tensor_op);
    kernel_ctx.define_int("IS_DENSE", conf.is_dense);
    kernel_ctx.define_int("IS_SAME_MD", conf.is_same_md);
    kernel_ctx.define_int("WITH_BINARY_POST_OP", conf.with_binary_post_op);
    kernel_ctx.define_int("SAME_SRC_DT", conf.same_src_dt);
    kernel_ctx.define_int("BCAST_DIM0", conf.bcast_dims[0]);
    kernel_ctx.define_int("BCAST_DIM1", conf.bcast_dims[1]);
    kernel_ctx.define_int("BCAST_DIM2", conf.bcast_dims[2]);
    kernel_ctx.define_int("BCAST_DIM3", conf.bcast_dims[3]);
    kernel_ctx.define_int("BCAST_DIM4", conf.bcast_dims[4]);
    kernel_ctx.define_int("BCAST_DIM5", conf.bcast_dims[5]);
    kernel_ctx.define_int("USE_UNROLL_16B", conf.use_unroll_16b);
    kernel_ctx.define_int("SRC0_UNROLL_16B", conf.src0_unroll_16b);
    kernel_ctx.define_int("SUB_GROUP_SIZE", 1);

    def_memory_desc_info(kernel_ctx, conf.src0_md_info, "SRC0");
    def_memory_desc_info(kernel_ctx, conf.src1_md_info, "SRC1");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_attr_info(kernel_ctx, conf.attr_info);

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_binary_t::execute_ref(const exec_ctx_t &ctx) const {

    auto &src0 = CTX_IN_STORAGE(DNNL_ARG_SRC_0);
    auto &src1 = CTX_IN_STORAGE(DNNL_ARG_SRC_1);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    auto src0_scale = conf.attr_info.src0_scale;
    auto src1_scale = conf.attr_info.src1_scale;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src0);
    arg_list.set(1, src1);
    arg_list.set(2, dst);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 3, conf.attr_info.all_post_ops);

    arg_list.set(arg_idx++, src0_scale);
    arg_list.set(arg_idx, src1_scale);

    auto nd_range = conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
