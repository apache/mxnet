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

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "gpu/compute/dispatch.hpp"

#include "common/utils.hpp"
#include "gpu/compute/compute_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

// Compute optimal local work size for the given global work size.
void get_optimal_lws(const size_t *gws, size_t *lws, size_t n) {
    const size_t lws_max = 256;
    // Factors in descending order, prefer bigger sizes for local work size.
    const size_t optimal_lws_values[]
            = {256, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1};
    size_t total_lws = 1;

    size_t gws_copy[3];
    for (size_t i = 0; i < n; ++i) {
        lws[i] = 1;
        gws_copy[i] = gws[i];
    }

    // Iterate through global work size and calculate max divisor from
    // the array optimal_lws_values.
    for (size_t i = 0; i < n; ++i) {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = 0;
        while (rest_lws < optimal_lws_values[lws_idx])
            lws_idx++;

        while (gws_copy[i] % optimal_lws_values[lws_idx])
            lws_idx++;

        lws[i] *= optimal_lws_values[lws_idx];
        total_lws *= optimal_lws_values[lws_idx];
        gws_copy[i] /= optimal_lws_values[lws_idx];
    }
}

dispatch_t::dispatch_t(const compute_engine_t *engine, const memory_desc_t *md)
    : engine_(engine) {

    if (md && md->format_kind == dnnl_blocked) {
        md_ndims_ = md->ndims;
        auto &blocking = md->format_desc.blocking;
        auto *strides = blocking.strides;
        std::pair<int, dim_t> sorted_strides[DNNL_MAX_NDIMS];
        for (int i = 0; i < md->ndims; ++i) {
            sorted_strides[i] = {i, strides[i]};
            for (int j = 0; j < blocking.inner_nblks; j++) {
                if (blocking.inner_idxs[j] == i) {
                    int str = 1;
                    for (int k = blocking.inner_nblks - 1; k > j; k--)
                        str *= blocking.inner_blks[k];
                    sorted_strides[i] = {i, str};
                    break;
                }
            }
        }
        std::sort(sorted_strides, sorted_strides + md->ndims,
                [](const std::pair<int, dim_t> &a,
                        const std::pair<int, dim_t> &b) {
                    return a.second < b.second;
                });
        for (int i = 0; i < md->ndims; i++) {
            md_nesting_levels_[sorted_strides[i].first] = md->ndims - i - 1;
        }
    }
}

std::string dispatch_t::str() const {
    std::ostringstream oss;
    for (int i = 0; i < ndims_; ++i) {
        auto &d = dims_[i];
        oss << "    "
            << "dim #" << i << " name: " << std::setw(10) << d.name
            << " size: " << std::setw(6) << d.size << " block: " << std::setw(4)
            << d.block << " nesting_level: " << std::setw(4) << d.nesting_level
            << " vsize: " << std::setw(4) << d.vector_size
            << " gws_idx: " << d.gws_index << std::endl;
    }
    return oss.str();
}

void dispatch_t::define_dim_with_nesting_level(
        const std::string &name, int nesting_level, dim_t size, dim_t block) {
#ifndef NDEBUG
    for (int i = 0; i < ndims_; ++i)
        assert(dims_[i].name != name && "Name is not unique.");
#endif

    dim_info_t di;
    di.name = name;
    di.size = size;
    di.block = block;
    di.nesting_level = nesting_level;
    di.vector_size = 1;
    di.gws_index = -1;
    dims_[ndims_] = di;

    ++ndims_;
}

void dispatch_t::vectorize_dim(const std::string &name, int vector_size) {
    assert(vector_size > 1);
    for (int i = 0; i < ndims_; ++i) {
        if (dims_[i].name == name) {
            assert(dims_[i].size % vector_size == 0);
            dims_[i].vector_size = vector_size;
            return;
        }
    }
    assert(!"not found");
}

void dispatch_t::def_kernel_macros(kernel_ctx_t &kernel_ctx) const {
    assert(generate_called && "generate() must be called.");

    // Find a unique prefix (in case there are many kernels in a file).
    std::string gws_prefix;
    for (int i = 0; i < 3; i++) {
        if (!kernel_ctx.has_macro(utils::format("GWS%d_DEF", i))) {
            gws_prefix = "GWS" + std::to_string(i);
            break;
        }
    }

    assert(!gws_prefix.empty());

    kernel_ctx.define_int(utils::format("%s_DEF", gws_prefix.c_str()), 1);

    for (int i = 0; i < ndims_; ++i) {
        auto get_dim_str = utils::format("-DGWS_GET_%s=%s_GET_ID%d",
                dims_[i].name.c_str(), gws_prefix.c_str(), i);
        kernel_ctx.add_option(get_dim_str);

        auto get_block_str = utils::format("-DGWS_GET_%s_BLOCK=%s_GET_BLOCK%d",
                dims_[i].name.c_str(), gws_prefix.c_str(), i);
        kernel_ctx.add_option(get_block_str);
        kernel_ctx.define_int(utils::format("%s_IDX%d", gws_prefix.c_str(), i),
                dims_[i].gws_index);
        kernel_ctx.define_int(
                utils::format("%s_STRIDE%d", gws_prefix.c_str(), i),
                get_gws_stride(i));

        bool is_zero = (dims_[i].size == 1);
        bool is_outermost = (i == ndims_ - 1)
                || dims_[i + 1].gws_index != dims_[i].gws_index;
        const char *op_name = is_zero
                ? "GWS_OP_ZERO"
                : is_outermost ? "GWS_OP_FIRST" : "GWS_OP_MOD";
        kernel_ctx.add_option(
                utils::format("-D%s_OP%d=%s", gws_prefix.c_str(), i, op_name));
        kernel_ctx.define_int(utils::format("%s_DIM%d", gws_prefix.c_str(), i),
                dims_[i].size);
        kernel_ctx.define_int(
                utils::format("%s_VEC_SIZE%d", gws_prefix.c_str(), i),
                dims_[i].vector_size);
        kernel_ctx.define_int(
                utils::format("%s_BLOCK%d", gws_prefix.c_str(), i),
                dims_[i].block);
    }

    // Local work size and subgroup sizes.
    int vec_dim_idx = find_vectorized_dim();
    kernel_ctx.define_int(
            utils::format("GWS_WITH_SG_%s", attr_suffix_), vec_dim_idx != -1);

    if (vec_dim_idx != -1)
        kernel_ctx.define_int(utils::format("GWS_SGS_%s", attr_suffix_),
                dims_[vec_dim_idx].vector_size);

    auto r = nd_range();
    for (int i = 0; i < 3; i++) {
        auto *lws = r.local_range();
        // lws may be NULL only when dispatch_info is default-initialized.
        kernel_ctx.define_int(utils::format("GWS_LWS%d_%s", i, attr_suffix_),
                lws ? lws[i] : 1);
    }
}

void dispatch_t::generate(bool generate_lws) {
    // Keep order of elements with the same nesting level unchanged.
    std::stable_sort(dims_, dims_ + ndims_,
            [](const dim_info_t &a, const dim_info_t &b) {
                return a.nesting_level > b.nesting_level;
            });

    // XXX: Move dimensions with size = 1 to the end.
    for (int i = ndims_ - 2; i >= 0; --i) {
        if (dims_[i].size == 1) {
            for (int j = i; j < ndims_ - 1; ++j) {
                if (dims_[j + 1].size == 1) break;
                std::swap(dims_[j], dims_[j + 1]);
            }
        }
    }

    // Find vectorized dimension (if any).
    int vec_dim_idx = find_vectorized_dim();

    // Compute GWS indices.
    for (int i = 0; i < ndims_; ++i) {
        if (vec_dim_idx == -1) {
            // Keep up to 4 dims in gws[0] to have bigger choice for work group
            // size.
            dims_[i].gws_index = std::min(2, std::max(0, i - 3));
        } else {
            // With vectorized dimension, work group size choices are more
            // limited so no need to group dimensions together.
            dims_[i].gws_index = std::min(2, i);
        }
    }

    size_t gws[3] = {1, 1, 1};
    for (int i = ndims_ - 1; i >= 0; --i) {
        dim_t block = std::max(dims_[i].block, (dim_t)1);
        int gws_index = dims_[i].gws_index;
        gws[gws_index] *= utils::div_up(dims_[i].size, block);
    }

    size_t gws_size = gws[0] * gws[1] * gws[2];

    auto *dev_info = engine_->device_info();
    size_t hw_threads = dev_info->hw_threads();

    // Calculate block sizes for the dimensions with flexible blocking.
    for (int i = 0; i < ndims_; ++i) {
        if (dims_[i].block == 0) {
            int gws_index = dims_[i].gws_index;
            // Heuristic: use max blocking but keep at least eu_count work items.
            size_t max_block = std::max((size_t)1, gws_size / hw_threads);
            size_t block = utils::max_div(dims_[i].size, max_block);
            gws[gws_index] /= block;
            gws_size /= block;
            dims_[i].block = block;
        }
    }

    // Handle a vectorized dimension (if presented).
    size_t lws[3] = {1, 1, 1};
    bool with_lws = false;
    if (vec_dim_idx != -1) {
        int gws_index = dims_[vec_dim_idx].gws_index;
        int vec_size = dims_[vec_dim_idx].vector_size;
        int nblocks = dims_[vec_dim_idx].size / dims_[vec_dim_idx].block;
        // XXX: max 256 work items per group
        lws[gws_index]
                = utils::max_div(gws[gws_index] / vec_size, 256 / vec_size)
                * vec_size;
        lws[gws_index] = utils::max_div(nblocks / vec_size,
                                 (int)lws[gws_index] / vec_size)
                * vec_size;
        with_lws = true;

        // Move the vectorized dimension to the first place in the group.
        int group_beg = ndims_ - 1;
        int group_end = 0;
        for (int i = 0; i < ndims_; ++i) {
            if (dims_[i].gws_index == gws_index) {
                group_beg = std::min(group_beg, i);
                group_end = std::max(group_end, i);
            }
        }

        if (vec_dim_idx != group_beg) {
            auto vec_dim_info = dims_[vec_dim_idx];
            for (int i = vec_dim_idx - 1; i >= group_beg; --i) {
                dims_[i + 1] = dims_[i];
            }
            dims_[group_beg] = vec_dim_info;
        }
    }

    // Use a work-group size = 1 if the number of work items < HW threads.
    if (!with_lws && gws_size < hw_threads) { with_lws = true; }

    if (!with_lws) {
        // Compute the best lws.
        get_optimal_lws(gws, lws, 3);
        with_lws = true;
    }

    nd_range_ = nd_range_t(gws, with_lws && generate_lws ? lws : nullptr);
    generate_called = true;
}

void dispatch_t::define_dim_with_md_hint(
        const std::string &name, int md_hint_index, dim_t size, dim_t block) {
    int nesting_level = min_nesting_level;
    if (md_ndims_ > 0) {
        assert(md_hint_index >= 0 && md_hint_index < md_ndims_);
        nesting_level = md_nesting_levels_[md_hint_index];
    }

    define_dim_with_nesting_level(name, nesting_level, size, block);
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl
