/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"

#if DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.h"
#endif

#include "c_types_map.hpp"
#include "engine.hpp"
#include "memory.hpp"
#include "memory_desc_wrapper.hpp"
#include "stream.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::data_type;

namespace dnnl {
namespace impl {
memory_desc_t glob_zero_md = memory_desc_t();
}
} // namespace dnnl

namespace {
// Returns the size required for memory descriptor mapping.
// Caveats:
// 1. If memory descriptor with run-time parameters, the mapping cannot be done;
//    hence return DNNL_RUNTIME_SIZE_VAL
// 2. Otherwise, the size returned includes `offset0` and holes (for the case
//    of non-trivial strides). Strictly speaking, the mapping should happen only
//    for elements accessible with `md.off_l(0 .. md.nelems())`. However, for
//    the sake of simple implementation let's have such limitation hoping that
//    no one will do concurrent mapping for overlapping memory objects.
//
// XXX: remove limitation mentioned in 2nd bullet.
size_t memory_desc_map_size(const memory_desc_t *md) {
    auto mdw = memory_desc_wrapper(md);

    if (mdw.has_runtime_dims_or_strides()) return DNNL_RUNTIME_SIZE_VAL;
    if (mdw.offset0() == 0) return mdw.size();

    memory_desc_t md_no_offset0 = *md;
    md_no_offset0.offset0 = 0;
    return memory_desc_wrapper(md_no_offset0).size()
            + md->offset0 * mdw.data_type_size();
}
} // namespace

dnnl_memory::dnnl_memory(dnnl::impl::engine_t *engine,
        const dnnl::impl::memory_desc_t *md, unsigned flags, void *handle)
    : engine_(engine), md_(*md) {
    const size_t size = memory_desc_wrapper(md_).size();

    memory_storage_t *memory_storage_ptr;
    status_t status = engine->create_memory_storage(
            &memory_storage_ptr, flags, size, handle);
    if (status != success) return;

    memory_storage_.reset(memory_storage_ptr);
    if (!(flags & omit_zero_pad)) zero_pad(nullptr);
}

dnnl_memory::dnnl_memory(dnnl::impl::engine_t *engine,
        const dnnl::impl::memory_desc_t *md,
        std::unique_ptr<dnnl::impl::memory_storage_t> &&memory_storage,
        bool do_zero_pad)
    : engine_(engine), md_(*md) {
    this->reset_memory_storage(std::move(memory_storage), do_zero_pad);
}

status_t dnnl_memory::set_data_handle(void *handle, stream_t *stream) {
    using namespace dnnl::impl;

    void *old_handle;
    CHECK(memory_storage()->get_data_handle(&old_handle));

    if (handle != old_handle) {
        CHECK(memory_storage()->set_data_handle(handle));
    }
    return zero_pad(stream);
}

status_t dnnl_memory::reset_memory_storage(
        std::unique_ptr<dnnl::impl::memory_storage_t> &&memory_storage,
        bool do_zero_pad) {
    if (memory_storage) {
        memory_storage_ = std::move(memory_storage);
        if (do_zero_pad) zero_pad(nullptr);
    } else {
        memory_storage_t *memory_storage_ptr;
        status_t status = engine_->create_memory_storage(
                &memory_storage_ptr, use_runtime_ptr, 0, nullptr);
        if (status != status::success) return status;

        memory_storage_.reset(memory_storage_ptr);
    }

    return status::success;
}

status_t dnnl_memory_desc_init_by_tag(memory_desc_t *memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, format_tag_t tag) {
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0 || tag == format_tag::undef) {
        *memory_desc = types::zero_md();
        return success;
    }

    format_kind_t format_kind = types::format_tag_to_kind(tag);

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
            && memory_desc_sanity_check(ndims, dims, data_type, format_kind);
    if (!args_ok) return invalid_arguments;

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind;

    status_t status = success;
    if (tag == format_tag::undef) {
        status = invalid_arguments;
    } else if (tag == format_tag::any) {
        // nop
    } else if (format_kind == format_kind::blocked) {
        status = memory_desc_wrapper::compute_blocking(md, tag);
    } else {
        assert(!"unreachable");
        status = invalid_arguments;
    }

    if (status == success) *memory_desc = md;

    return status;
}

status_t dnnl_memory_desc_init_by_strides(memory_desc_t *memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, const dims_t strides) {
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0) {
        *memory_desc = types::zero_md();
        return success;
    }

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
            && memory_desc_sanity_check(
                    ndims, dims, data_type, format_kind::undef);
    if (!args_ok) return invalid_arguments;

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::blocked;

    dims_t default_strides = {0};
    if (strides == nullptr) {
        bool has_runtime_strides = false;
        default_strides[md.ndims - 1] = 1;
        for (int d = md.ndims - 2; d >= 0; --d) {
            if (md.padded_dims[d] == DNNL_RUNTIME_DIM_VAL)
                has_runtime_strides = true;
            default_strides[d] = has_runtime_strides
                    ? DNNL_RUNTIME_DIM_VAL
                    : default_strides[d + 1] * md.padded_dims[d + 1];
        }
        strides = default_strides;
    } else {
        /* TODO: add sanity check for the provided strides */
    }

    array_copy(md.format_desc.blocking.strides, strides, md.ndims);

    *memory_desc = md;

    return success;
}

status_t dnnl_memory_desc_init_submemory(memory_desc_t *md,
        const memory_desc_t *parent_md, const dims_t dims,
        const dims_t offsets) {
    if (any_null(md, parent_md) || !memory_desc_sanity_check(parent_md))
        return invalid_arguments;

    const memory_desc_wrapper src_d(parent_md);
    if (src_d.has_runtime_dims_or_strides()) return unimplemented;

    for (int d = 0; d < src_d.ndims(); ++d) {
        if (utils::one_of(DNNL_RUNTIME_DIM_VAL, dims[d], offsets[d]))
            return unimplemented;

        if (dims[d] < 0 || offsets[d] < 0
                || (offsets[d] + dims[d] > src_d.dims()[d]))
            return invalid_arguments;
    }

    if (src_d.format_kind() != format_kind::blocked) return unimplemented;

    dims_t blocks;
    src_d.compute_blocks(blocks);

    memory_desc_t dst_d = *parent_md;
    auto &dst_d_blk = dst_d.format_desc.blocking;

    /* TODO: put this into memory_desc_wrapper */
    for (int d = 0; d < src_d.ndims(); ++d) {
        /* very limited functionality for now */
        const bool ok = true && offsets[d] % blocks[d] == 0 /* [r1] */
                && src_d.padded_offsets()[d] == 0
                && (false || dims[d] % blocks[d] == 0 || dims[d] < blocks[d]);
        if (!ok) return unimplemented;

        const bool is_right_border = offsets[d] + dims[d] == src_d.dims()[d];

        dst_d.dims[d] = dims[d];
        dst_d.padded_dims[d] = is_right_border
                ? src_d.padded_dims()[d] - offsets[d]
                : dst_d.dims[d];
        dst_d.padded_offsets[d] = src_d.padded_offsets()[d];
        dst_d.offset0 += /* [r1] */
                offsets[d] / blocks[d] * dst_d_blk.strides[d];
    }

    *md = dst_d;

    return success;
}

int dnnl_memory_desc_equal(const memory_desc_t *lhs, const memory_desc_t *rhs) {
    if (lhs == rhs) return 1;
    if (any_null(lhs, rhs)) return 0;
    return memory_desc_wrapper(*lhs) == memory_desc_wrapper(*rhs);
}

status_t dnnl_memory_desc_reshape(memory_desc_t *out_md,
        const memory_desc_t *in_md, int ndims, const dims_t dims) {
    auto volume = [](const dim_t *dims, int ndims) -> dim_t {
        dim_t prod = 1;
        for (int i = 0; i < ndims; ++i) {
            if (dims[i] == DNNL_RUNTIME_DIM_VAL) return DNNL_RUNTIME_DIM_VAL;
            prod *= dims[i] > 0 ? dims[i] : 1;
        }
        return prod;
    };

    if (any_null(out_md, in_md) || !memory_desc_sanity_check(in_md)
            || !memory_desc_sanity_check(
                    ndims, dims, in_md->data_type, in_md->format_kind)
            || !one_of(
                    in_md->format_kind, format_kind::any, format_kind::blocked)
            || types::is_zero_md(in_md)
            || volume(in_md->dims, in_md->ndims) != volume(dims, ndims)
            || memory_desc_wrapper(in_md).has_runtime_dims_or_strides()
            || in_md->extra.flags != 0)
        return invalid_arguments;

    if (in_md->format_kind == format_kind::any)
        return dnnl_memory_desc_init_by_tag(
                out_md, ndims, dims, in_md->data_type, format_tag::any);

    assert(in_md->format_kind == format_kind::blocked);
    assert(in_md->extra.flags == 0);

    // temporary output
    auto md = *in_md;

    md.ndims = ndims;
    array_copy(md.dims, dims, md.ndims);

    const int i_ndims = in_md->ndims;
    const int o_ndims = md.ndims;

    const auto &i_dims = in_md->dims, &i_pdims = in_md->padded_dims;
    const auto &o_dims = md.dims;

    const auto &i_bd = in_md->format_desc.blocking;
    auto &o_bd = md.format_desc.blocking;

    dims_t blocks = {0};
    memory_desc_wrapper(in_md).compute_blocks(blocks);

    enum class action_t { REMOVE_1, ADD_1, KEEP_DIM, REARRANGE_DIMS, FAIL };

    // Determine groups in input and output dims starting with the given
    // positions (going backwards) that satisfy one of the conditions:
    // - REMOVE_1
    //      input_group = {1}, output_group = empty
    // - ADD_1
    //      input_group = empty, output_group = {1}
    // - KEEP_DIM
    //      input_group = {x}, output_group = {x}
    // - REARRANGE_DIMS
    //      input_group = {x1, x2, .., xk}, output_group = {y1, y2, ..., ym}
    //      and product(x_i) = product(y_j), and the groups are minimal
    // - FAIL
    //      invalid configuration (return false)
    auto find_groups
            = [&](int &i_group_begin, int i_group_end, int &o_group_begin,
                      int o_group_end) -> action_t {
        // 1st step: check for `1` in the input dims
        if (i_group_end > 0 && i_dims[i_group_end - 1] == 1) {
            i_group_begin = i_group_end - 1;
            if (i_pdims[i_group_end - 1] == 1) {
                o_group_begin = o_group_end;
                return action_t::REMOVE_1;
            } else if (o_group_end > 0 && o_dims[o_group_end - 1] == 1) {
                o_group_begin = o_group_end - 1;
                return action_t::KEEP_DIM;
            } else {
                return action_t::FAIL;
            }
        }

        // 2nd step: check for `1` in the output dims
        if (o_group_end > 0 && o_dims[o_group_end - 1] == 1) {
            i_group_begin = i_group_end;
            o_group_begin = o_group_end - 1;
            return action_t::ADD_1;
        }

        // at this moment both groups cannot be empty
        if (i_group_end == 0 || o_group_end == 0) return action_t::FAIL;

        // 3rd step: find the non-trivial groups of the same volume
        i_group_begin = i_group_end - 1;
        o_group_begin = o_group_end - 1;

        dim_t i_volume = i_dims[i_group_begin];
        dim_t o_volume = o_dims[o_group_begin];

        while (i_volume != o_volume) {
            if (i_volume < o_volume) {
                if (i_group_begin == 0) return action_t::FAIL;
                i_volume *= i_dims[--i_group_begin];

                // do not allow `0` axis in the middle
                if (i_volume == 0) return action_t::FAIL;
            } else {
                if (o_group_begin == 0) return action_t::FAIL;
                o_volume *= o_dims[--o_group_begin];

                // do not allow `0` axis in the middle
                if (o_volume == 0) return action_t::FAIL;
            }
        }

        assert(i_volume == o_volume);
        assert(i_group_begin >= 0);
        assert(o_group_begin >= 0);

        return (i_group_begin + 1 == i_group_end
                       && o_group_begin + 1 == o_group_end)
                ? action_t::KEEP_DIM
                : action_t::REARRANGE_DIMS;
    };

    int i_group_begin = i_ndims, i_group_end = i_ndims;
    int o_group_begin = o_ndims, o_group_end = o_ndims;

    while (i_group_end != 0 || o_group_end != 0) {
        action_t action = find_groups(
                i_group_begin, i_group_end, o_group_begin, o_group_end);

        if (action == action_t::REMOVE_1) {
            // nop, padding is already taken into account by `find_groups()`
        } else if (action == action_t::ADD_1) {
            // get the stride from the right
            dim_t current_stride = 1;
            if (i_group_begin == i_ndims) {
                for (int d = 0; d < i_bd.inner_nblks; ++d)
                    current_stride *= i_bd.inner_blks[d];
            } else {
                // Add `1` to the left from axes with index `i_group_begin`
                current_stride
                        = i_bd.strides[i_group_begin] * i_dims[i_group_begin];
                for (int d = 0; d < i_bd.inner_nblks; ++d)
                    if (i_bd.inner_idxs[d] == i_group_begin)
                        current_stride /= i_bd.inner_blks[d];
            }
            md.padded_dims[o_group_begin] = 1;
            md.padded_offsets[o_group_begin] = 0;
            o_bd.strides[o_group_begin] = current_stride;
        } else if (action == action_t::KEEP_DIM) {
            // change the axis index from `i_group_begin` to `o_group_begin`
            assert(i_group_begin + 1 == i_group_end);
            assert(o_group_begin + 1 == o_group_end);

            md.padded_dims[o_group_begin] = in_md->padded_dims[i_group_begin];
            md.padded_offsets[o_group_begin]
                    = in_md->padded_offsets[i_group_begin];
            o_bd.strides[o_group_begin] = i_bd.strides[i_group_begin];
            for (int d = 0; d < i_bd.inner_nblks; ++d)
                if (i_bd.inner_idxs[d] == i_group_begin)
                    o_bd.inner_idxs[d] = o_group_begin;
        } else if (action == action_t::REARRANGE_DIMS) {
            // check that input group is dense, sequential, and is not blocked
            for (int d = i_group_end - 1; d > i_group_begin; --d)
                if (i_dims[d] * i_bd.strides[d] != i_bd.strides[d - 1])
                    return invalid_arguments;

            // checked (i_group_begin, i_group_end), `i_group_begin` remains
            for (int d = 0; d < i_bd.inner_nblks; ++d)
                if (i_bd.inner_idxs[d] == i_group_begin)
                    return invalid_arguments;
            if (in_md->padded_dims[i_group_begin] != i_dims[i_group_begin])
                return invalid_arguments;
            if (in_md->padded_offsets[i_group_begin] != 0)
                return invalid_arguments;

            // oK, fill output md according to
            // o_dims[o_group_begin .. o_group_end]

            dim_t current_stride = i_bd.strides[i_group_end - 1];
            for (int d = o_group_end - 1; d >= o_group_begin; --d) {
                md.padded_dims[d] = o_dims[d];
                md.padded_offsets[d] = 0;
                o_bd.strides[d] = current_stride;
                current_stride *= md.padded_dims[d];
            }
        } else {
            assert(action == action_t::FAIL);
            return invalid_arguments;
        }

        i_group_end = i_group_begin;
        o_group_end = o_group_begin;
    }

    *out_md = md;
    return success;
}

status_t dnnl_memory_desc_permute_axes(
        memory_desc_t *out_md, const memory_desc_t *in_md, const int *perm) {
    if (any_null(out_md, in_md) || !memory_desc_sanity_check(in_md)
            || !one_of(
                    in_md->format_kind, format_kind::any, format_kind::blocked)
            || types::is_zero_md(in_md)
            || memory_desc_wrapper(in_md).has_runtime_dims_or_strides()
            || in_md->extra.flags != 0)
        return invalid_arguments;

    // verify that perm is indeed a permutation of [0 .. ndims)
    unsigned occurrence_mask = 0;
    for (int d = 0; d < in_md->ndims; ++d)
        if (0 <= perm[d] && perm[d] < in_md->ndims)
            occurrence_mask |= (1u << perm[d]);
    if (occurrence_mask + 1 != (1u << in_md->ndims)) return invalid_arguments;

    *out_md = *in_md;
    for (int d = 0; d < in_md->ndims; ++d) {
        if (perm[d] == d) continue;
        out_md->dims[perm[d]] = in_md->dims[d];
        out_md->padded_dims[perm[d]] = in_md->padded_dims[d];
        out_md->padded_offsets[perm[d]] = in_md->padded_offsets[d];
        if (in_md->format_kind == format_kind::blocked) {
            const auto &i_bd = in_md->format_desc.blocking;
            auto &o_bd = out_md->format_desc.blocking;

            o_bd.strides[perm[d]] = i_bd.strides[d];
            for (int blk = 0; blk < i_bd.inner_nblks; ++blk)
                if (i_bd.inner_idxs[blk] == d) o_bd.inner_idxs[blk] = perm[d];
        }
    }

    return success;
}

size_t dnnl_memory_desc_get_size(const memory_desc_t *md) {
    if (md == nullptr) return 0;
    return memory_desc_wrapper(*md).size();
}

status_t dnnl_memory_create(memory_t **memory, const memory_desc_t *md,
        engine_t *engine, void *handle) {
#if DNNL_WITH_SYCL
    return dnnl_sycl_interop_memory_create(
            memory, md, engine, dnnl_sycl_interop_usm, handle);
#else
    if (any_null(memory, engine)) return invalid_arguments;

    memory_desc_t z_md = types::zero_md();
    if (md == nullptr) md = &z_md;

    const auto mdw = memory_desc_wrapper(md);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return invalid_arguments;

    unsigned flags = (handle == DNNL_MEMORY_ALLOCATE)
            ? memory_flags_t::alloc
            : memory_flags_t::use_runtime_ptr;
    void *handle_ptr = (handle == DNNL_MEMORY_ALLOCATE) ? nullptr : handle;
    auto _memory = new memory_t(engine, md, flags, handle_ptr);
    if (_memory == nullptr) return out_of_memory;
    if (_memory->memory_storage() == nullptr) {
        delete _memory;
        return out_of_memory;
    }
    *memory = _memory;
    return success;
#endif
}

status_t dnnl_memory_get_memory_desc(
        const memory_t *memory, const memory_desc_t **md) {
    if (any_null(memory, md)) return invalid_arguments;
    *md = memory->md();
    return success;
}

status_t dnnl_memory_get_engine(const memory_t *memory, engine_t **engine) {
    if (any_null(memory, engine)) return invalid_arguments;
    *engine = memory->engine();
    return success;
}

status_t dnnl_memory_get_data_handle(const memory_t *memory, void **handle) {
    if (any_null(handle)) return invalid_arguments;
    if (memory == nullptr) {
        *handle = nullptr;
        return success;
    }
    return memory->get_data_handle(handle);
}

status_t dnnl_memory_set_data_handle(memory_t *memory, void *handle) {
    return dnnl_memory_set_data_handle_v2(memory, handle, nullptr);
}

status_t dnnl_memory_set_data_handle_v2(
        memory_t *memory, void *handle, stream_t *stream) {
    if (any_null(memory)) return invalid_arguments;
    if (stream) stream->before_exec_hook();
    status_t status = memory->set_data_handle(handle, stream);
    if (stream) stream->after_exec_hook();
    return status;
}

status_t dnnl_memory_map_data(const memory_t *memory, void **mapped_ptr) {
    bool args_ok = !any_null(memory, mapped_ptr);
    if (!args_ok) return invalid_arguments;

    const memory_desc_t *md = memory->md();
    // See caveats in the comment to `memory_desc_map_size()` function.
    const size_t map_size = memory_desc_map_size(md);

    if (map_size == 0) {
        *mapped_ptr = nullptr;
        return success;
    } else if (map_size == DNNL_RUNTIME_SIZE_VAL) {
        return invalid_arguments;
    }

    return memory->memory_storage()->map_data(mapped_ptr, nullptr, map_size);
}

status_t dnnl_memory_unmap_data(const memory_t *memory, void *mapped_ptr) {
    bool args_ok = !any_null(memory);
    if (!args_ok) return invalid_arguments;

    return memory->memory_storage()->unmap_data(mapped_ptr, nullptr);
}

status_t dnnl_memory_destroy(memory_t *memory) {
    delete memory;
    return success;
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
