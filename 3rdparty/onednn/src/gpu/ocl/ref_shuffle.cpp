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

#include "gpu/ocl/ref_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace format_tag;

status_t ref_shuffle_t::pd_t::init_conf() {
    const memory_desc_wrapper input_md(is_fwd() ? src_md() : diff_dst_md());
    conf.data_type = input_md.data_type();

    conf.axis = axis();
    conf.axis_size = axis_size();
    conf.group_size = group_size();

    conf.transpose_row
            = is_fwd() ? conf.group_size : conf.axis_size / conf.group_size;
    conf.transpose_col
            = is_fwd() ? conf.axis_size / conf.group_size : conf.group_size;

    auto dims = desc()->data_desc.dims;
    auto ndims = desc()->data_desc.ndims;
    const size_t outer_size = utils::array_product(dims, conf.axis);
    const size_t inner_size
            = utils::array_product(dims + conf.axis + 1, ndims - conf.axis - 1);
    const size_t dim = conf.axis_size * inner_size;
    conf.outer_size = outer_size;
    conf.inner_size = inner_size;
    conf.dim = dim;
    conf.ndims = ndims;

    conf.gws_d[0] = nstl::max(size_t(1), inner_size);
    conf.gws_d[1] = nstl::max(1, conf.axis_size);
    conf.gws_d[2] = nstl::max(size_t(1), outer_size);

    set_offsets(input_md, off.src_off);

    return status::success;
}

status_t ref_shuffle_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(conf.data_type);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("AXIS", conf.axis);
    kernel_ctx.define_int("AXIS_SIZE", conf.axis_size);
    kernel_ctx.define_int("GROUP_SIZE", conf.group_size);
    kernel_ctx.define_int("TRANSPOSE_ROW", conf.transpose_row);
    kernel_ctx.define_int("TRANSPOSE_COL", conf.transpose_col);
    kernel_ctx.define_int("INNER_SIZE", conf.inner_size);
    kernel_ctx.define_int("OUTER_SIZE", conf.outer_size);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);
    return status::success;
}

template <dnnl_format_tag_t tag>
status_t ref_shuffle_t::execute_(const exec_ctx_t &ctx) const {
    auto &src = pd()->is_fwd() ? CTX_IN_STORAGE(DNNL_ARG_SRC)
                               : CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &dst = pd()->is_fwd() ? CTX_OUT_STORAGE(DNNL_ARG_DST)
                               : CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);

    auto nd_range = compute::nd_range_t(conf.gws_d);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}
template status_t ref_shuffle_t::execute_<any>(const exec_ctx_t &ctx) const;

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
