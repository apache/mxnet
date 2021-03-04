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

#include "gpu/ocl/gen9_binary.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gen9_binary_t::pd_t::init_conf(engine_t *engine) {
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper dst_d(dst_md());

    alg_kind_t alg = desc()->alg_kind;

    const int ndims = src0_d.ndims();
    conf.src0_md_info = memory_desc_info_t::create(src0_d);
    conf.src1_md_info = memory_desc_info_t::create(src1_d);
    conf.dst_md_info = memory_desc_info_t::create(dst_d);
    conf.attr_info = attr_info_t::create(attr());
    conf.src0_data_type = src0_d.data_type();
    conf.src1_data_type = src1_d.data_type();
    conf.dst_data_type = dst_d.data_type();
    conf.ndims = ndims;
    conf.is_add = (alg == alg_kind::binary_add);
    conf.is_mul = (alg == alg_kind::binary_mul);
    conf.is_max = (alg == alg_kind::binary_max);
    conf.is_min = (alg == alg_kind::binary_min);
    conf.is_div = (alg == alg_kind::binary_div);
    conf.is_tensor_op = is_tensor_op();
    conf.is_dense = dst_d.is_dense();
    conf.same_src_dt = (src0_d.data_type() == src1_d.data_type());
    conf.is_same_md = (src0_d == dst_d) && (src1_d == dst_d);

    for (int i = 0; i < MAX_NDIMS; ++i) {
        conf.bcast_dims[i] = i < ndims ? broadcast_dims()[i] : 1;
    }

    if (conf.bcast_dims[1] && !conf.bcast_dims[ndims - 1]) {
        conf.nvect = 1;
    } else {
        conf.nvect = 8;
        while (dst_d.dims()[ndims - 1] % conf.nvect != 0) {
            conf.nvect /= 2;
        }
    }

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_d.md_);

    using namespace dnnl::impl::format_tag;
    conf.is_ncX_layout = dst_d.matches_one_of_tag(nc, ncw, nchw, ncdhw);

    if (!conf.is_ncX_layout) {
        const auto blocking = src_md(0)->format_desc.blocking;
        if (!(blocking.inner_nblks == 1 && blocking.inner_idxs[0] == 1
                    && blocking.inner_blks[0] == 16
                    && src_md(0)->dims[1] % 16 == 0))
            return status::unimplemented;
        // Setting the MB as the innermost dim for optimized performance
        // Hence starting i = 1, ignoring MB
        conf.dispatch.define_dim_with_nesting_level(
                "D0", ndims, dst_d.dims()[0], 1);
        for (int i = 1; i < MAX_NDIMS; ++i) {
            int dim = i < ndims ? dst_d.dims()[i] : 1;
            if (i == 1) {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, 1);
                conf.dispatch.vectorize_dim("D1", 16);
            } else if (i == ndims - 1) {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, conf.nvect);
            } else {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, 1);
            }
        }
    } else {
        if (dst_md()->dims[dst_md()->ndims - 1] % 16 != 0)
            return status::unimplemented;
        conf.nvect = 16;
        while ((dst_d.dims()[ndims - 1] / 16) % conf.nvect != 0) {
            --conf.nvect;
        }

        int mixed_dim = 1;
        for (int i = 0; i < ndims; ++i) {
            mixed_dim *= dst_d.dims()[i];
        }
        conf.dispatch.define_dim("MIXED_DIM", 0, mixed_dim, conf.nvect);
        conf.dispatch.vectorize_dim("MIXED_DIM", 16);
    }

    conf.dispatch.generate();
    return status::success;
}

status_t gen9_binary_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(conf.src0_data_type);
    kernel_ctx.define_int("SUB_GROUP_SIZE", 16);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("IS_NCX_LAYOUT", conf.is_ncX_layout);
    kernel_ctx.define_int("IS_MUL", conf.is_mul);
    kernel_ctx.define_int("IS_ADD", conf.is_add);
    kernel_ctx.define_int("IS_MAX", conf.is_max);
    kernel_ctx.define_int("IS_MIN", conf.is_min);
    kernel_ctx.define_int("IS_DIV", conf.is_div);
    kernel_ctx.define_int("SAME_SRC_DT", conf.same_src_dt);
    kernel_ctx.define_int("BCAST_DIM0", conf.bcast_dims[0]);
    kernel_ctx.define_int("BCAST_DIM1", conf.bcast_dims[1]);
    kernel_ctx.define_int("BCAST_DIM2", conf.bcast_dims[2]);
    kernel_ctx.define_int("BCAST_DIM3", conf.bcast_dims[3]);
    kernel_ctx.define_int("BCAST_DIM4", conf.bcast_dims[4]);
    kernel_ctx.define_int("BCAST_DIM5", conf.bcast_dims[5]);
    kernel_ctx.define_int(
            "BCAST_AT_INNERMOST_DIM", conf.bcast_dims[conf.ndims - 1]);
    kernel_ctx.define_int("NVECT", conf.nvect);
    kernel_ctx.add_option("-Dcl_intel_subgroups_char");
    kernel_ctx.add_option("-Dcl_intel_subgroups_uchar");

    def_memory_desc_info(kernel_ctx, conf.src0_md_info, "SRC0");
    def_memory_desc_info(kernel_ctx, conf.src1_md_info, "SRC1");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_attr_info(kernel_ctx, conf.attr_info);

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
