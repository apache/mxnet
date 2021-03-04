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

#ifndef COMMON_TYPE_HELPERS_HPP
#define COMMON_TYPE_HELPERS_HPP

#include <assert.h>
#include <math.h>

#include "oneapi/dnnl/dnnl.h"

#include "bit_cast.hpp"
#include "c_types_map.hpp"
#include "dnnl_traits.hpp"
#include "math_utils.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

// Global zero memory descriptor. Mostly used for queries to return
extern memory_desc_t DNNL_API glob_zero_md;

template <typename base_type, typename derived_type>
status_t safe_ptr_assign(base_type *&lhs, derived_type *rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs = rhs;
    return status::success;
}

template <typename base_type, typename derived_type>
status_t safe_ptr_assign(std::unique_ptr<base_type> &lhs, derived_type *rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs.reset(rhs);
    return status::success;
}

template <typename T, typename U>
struct is_subset {
    static constexpr bool value = false;
};
template <typename T>
struct is_subset<T, T> {
    static constexpr bool value = true;
};
template <typename T>
struct is_subset<T,
        typename utils::enable_if<nstl::is_integral<T>::value, float>::type> {
    static constexpr bool value = true;
};
#define ISSPEC(t1, t2) \
    template <> \
    struct is_subset<t1, t2> { \
        static constexpr bool value = true; \
    }
ISSPEC(int16_t, int32_t);
ISSPEC(int8_t, int32_t);
ISSPEC(uint8_t, int32_t);
ISSPEC(int8_t, int16_t);
ISSPEC(uint8_t, int16_t);
#undef ISSPEC

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs);

namespace types {

inline size_t data_type_size(data_type_t data_type) {
    using namespace data_type;
    switch (data_type) {
        case f16: return sizeof(prec_traits<f16>::type);
        case bf16: return sizeof(prec_traits<bf16>::type);
        case f32: return sizeof(prec_traits<f32>::type);
        case s32: return sizeof(prec_traits<s32>::type);
        case s8: return sizeof(prec_traits<s8>::type);
        case u8: return sizeof(prec_traits<u8>::type);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return (size_t)-1; /* not supposed to be reachable */
}

template <typename T>
inline T max_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<T>(nstl::numeric_limits<prec_traits<x>::type>::max())
    switch (data_type) {
        CASE(f16);
        CASE(bf16);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return static_cast<T>(0); /* not supposed to be reachable */
#undef CASE
}

inline float get_float_value(data_type_t dt, const void *ptr, dim_t idx) {
#define CASE(dt) \
    case dt: \
        return static_cast<float>(((typename prec_traits<dt>::type *)ptr)[idx]);

    using namespace data_type;
    switch (dt) {
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        default: assert(!"bad data_type");
    }

#undef CASE
    return NAN;
}

inline format_kind_t format_tag_to_kind(format_tag_t tag) {
    switch (tag) {
        case format_tag::undef: return format_kind::undef;
        case format_tag::any: return format_kind::any;
        case format_tag::last: return format_kind::undef;
        default: return format_kind::blocked;
    }

    assert(!"unreachable");
    return format_kind::undef;
}

inline bool memory_extra_desc_is_equal(
        const memory_extra_desc_t &lhs, const memory_extra_desc_t &rhs) {
    using namespace memory_extra_flags;
    return true && lhs.flags == rhs.flags
            && IMPLICATION(lhs.flags & compensation_conv_s8s8,
                    lhs.compensation_mask == rhs.compensation_mask)
            && IMPLICATION(lhs.flags & gpu_rnn_u8s8_compensation,
                    lhs.compensation_mask == rhs.compensation_mask)
            && IMPLICATION(lhs.flags & scale_adjust,
                    lhs.scale_adjust == rhs.scale_adjust)
            && IMPLICATION(lhs.flags & compensation_conv_asymmetric_src,
                    lhs.asymm_compensation_mask == rhs.asymm_compensation_mask);
}

inline bool blocking_desc_is_equal(const memory_desc_t &lhs_md,
        const memory_desc_t &rhs_md, bool ignore_strides = false) {
    using dnnl::impl::utils::array_cmp;

    assert(lhs_md.format_kind == format_kind::blocked);
    assert(rhs_md.format_kind == format_kind::blocked);

    const auto &lhs = lhs_md.format_desc.blocking;
    const auto &rhs = rhs_md.format_desc.blocking;
    bool equal = lhs.inner_nblks == rhs.inner_nblks
            && array_cmp(lhs.inner_blks, rhs.inner_blks, lhs.inner_nblks)
            && array_cmp(lhs.inner_idxs, rhs.inner_idxs, lhs.inner_nblks);
    if (ignore_strides) return equal;

    // Check the strides.
    // Note: for dimensions of size `1` the stride doesn't really matter.
    for (int d = 0; d < lhs_md.ndims; ++d) {
        if (lhs_md.dims[d] == 1 && lhs_md.padded_dims[d] == 1) continue;
        equal = equal && lhs.strides[d] == rhs.strides[d];
    }

    return equal;
}

inline bool wino_desc_is_equal(const wino_desc_t &lhs, const wino_desc_t &rhs) {
    return lhs.wino_format == rhs.wino_format && lhs.alpha == rhs.alpha
            && lhs.ic == rhs.ic && lhs.oc == rhs.oc
            && lhs.ic_block == rhs.ic_block && lhs.oc_block == rhs.oc_block
            && lhs.ic2_block == rhs.ic2_block && lhs.oc2_block == rhs.oc2_block
            && lhs.r == rhs.r;
}

inline bool rnn_packed_desc_is_equal(
        const rnn_packed_desc_t &lhs, const rnn_packed_desc_t &rhs) {
    bool ok = true && lhs.format == rhs.format && lhs.ldb == rhs.ldb
            && lhs.n_parts == rhs.n_parts
            && lhs.offset_compensation == rhs.offset_compensation
            && lhs.size == rhs.size && lhs.n == rhs.n;
    if (!ok) return false;

    for (int i = 0; i < rhs.n_parts; i++)
        ok = ok && lhs.parts[i] == rhs.parts[i];
    for (int i = 0; i < rhs.n_parts; i++)
        ok = ok && lhs.part_pack_size[i] == rhs.part_pack_size[i];
    return ok;
}

inline memory_desc_t zero_md() {
    auto zero = memory_desc_t();
    return zero;
}

inline bool is_zero_md(const memory_desc_t *md) {
    return md == nullptr || *md == zero_md();
}

inline data_type_t default_accum_data_type(
        data_type_t src_dt, data_type_t dst_dt) {
    using namespace utils;
    using namespace data_type;

    if (one_of(f16, src_dt, dst_dt)) return f16;
    if (one_of(bf16, src_dt, dst_dt)) return f32;
    if (one_of(f32, src_dt, dst_dt)) return f32;
    if (one_of(s32, src_dt, dst_dt)) return s32;

    if (one_of(s8, src_dt, dst_dt) || one_of(u8, src_dt, dst_dt)) return s32;

    return data_type::undef;
}

inline data_type_t default_accum_data_type(data_type_t src_dt,
        data_type_t wei_dt, data_type_t dst_dt, prop_kind_t prop_kind) {
    using namespace utils;
    using namespace data_type;
    using namespace prop_kind;

    /* prop_kind doesn't matter */
    if (everyone_is(f16, src_dt, wei_dt, dst_dt)) return f16;
    if (one_of(bf16, src_dt, wei_dt, dst_dt)) return f32;
    if (everyone_is(f32, src_dt, wei_dt, dst_dt)) return f32;

    if (one_of(prop_kind, forward_training, forward_inference)) {
        if ((src_dt == u8 || src_dt == s8) && wei_dt == s8
                && one_of(dst_dt, f32, s32, s8, u8))
            return s32;
    } else if (prop_kind == backward_data) {
        if (one_of(src_dt, f32, s32, s8, u8) && wei_dt == s8
                && one_of(dst_dt, s8, u8))
            return s32;
    }

    return data_type::undef;
}

} // namespace types

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    using namespace dnnl::impl::utils;
    // quick path for zero_mds
    if (utils::everyone_is(0, lhs.ndims, rhs.ndims)) return true;

    bool base_equal = true && lhs.ndims == rhs.ndims
            && array_cmp(lhs.dims, rhs.dims, lhs.ndims)
            && lhs.data_type == rhs.data_type
            && array_cmp(lhs.padded_dims, rhs.padded_dims, lhs.ndims)
            && array_cmp(lhs.padded_offsets, rhs.padded_offsets, lhs.ndims)
            && lhs.offset0 == rhs.offset0 && lhs.format_kind == rhs.format_kind;
    if (!base_equal) return false;
    if (!types::memory_extra_desc_is_equal(lhs.extra, rhs.extra)) return false;
    if (lhs.format_kind == format_kind::blocked)
        return types::blocking_desc_is_equal(lhs, rhs);
    else if (lhs.format_kind == format_kind::wino)
        return types::wino_desc_is_equal(
                lhs.format_desc.wino_desc, rhs.format_desc.wino_desc);
    else if (lhs.format_kind == format_kind::rnn_packed)
        return types::rnn_packed_desc_is_equal(lhs.format_desc.rnn_packed_desc,
                rhs.format_desc.rnn_packed_desc);
    return true;
}

inline bool operator!=(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

// Comparison operators for descriptors
#define COMPARE_DESC_MEMBERS(m) lhs.m == rhs.m
#define COMPARE_DESC_ARRAY_MEMBERS(m, s) utils::array_cmp(lhs.m, rhs.m, s)

// clang-format off
inline bool operator==(const batch_normalization_desc_t &lhs,
        const batch_normalization_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(data_desc)
            && COMPARE_DESC_MEMBERS(diff_data_desc)
            && COMPARE_DESC_MEMBERS(data_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(diff_data_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(stat_desc)
            && COMPARE_DESC_MEMBERS(batch_norm_epsilon)
            && COMPARE_DESC_MEMBERS(flags);
    return ret;
}

inline bool operator==(const binary_desc_t &lhs, const binary_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc[0])
            && COMPARE_DESC_MEMBERS(src_desc[1])
            && COMPARE_DESC_MEMBERS(dst_desc);
    return ret;
}

inline bool operator==(const concat_desc_t &lhs, const concat_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(dst_md)
            && COMPARE_DESC_MEMBERS(n)
            && COMPARE_DESC_MEMBERS(concat_dimension);

    if (!ret) return ret;

    for (int i = 0; i < lhs.n; i++) {
        ret = COMPARE_DESC_MEMBERS(src_mds[i]);
        if (!ret) break;
    }
    return ret;
}

inline bool operator==(
        const convolution_desc_t &lhs, const convolution_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(strides, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(dilates, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[0], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[1], DNNL_MAX_NDIMS)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(const eltwise_desc_t &lhs, const eltwise_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(data_desc)
            && COMPARE_DESC_MEMBERS(diff_data_desc)
            && COMPARE_DESC_MEMBERS(alpha)
            && COMPARE_DESC_MEMBERS(beta);
    return ret;
}

inline bool operator==(const gemm_desc_t &lhs, const gemm_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(transa)
            && COMPARE_DESC_MEMBERS(transb)
            && COMPARE_DESC_MEMBERS(batch)
            && COMPARE_DESC_MEMBERS(m)
            && COMPARE_DESC_MEMBERS(n)
            && COMPARE_DESC_MEMBERS(k)
            && COMPARE_DESC_MEMBERS(stride_a)
            && COMPARE_DESC_MEMBERS(stride_b)
            && COMPARE_DESC_MEMBERS(stride_c)
            && COMPARE_DESC_MEMBERS(lda)
            && COMPARE_DESC_MEMBERS(ldb)
            && COMPARE_DESC_MEMBERS(ldc)
            && COMPARE_DESC_MEMBERS(bias_mask)
            && COMPARE_DESC_MEMBERS(a_type)
            && COMPARE_DESC_MEMBERS(b_type)
            && COMPARE_DESC_MEMBERS(c_type)
            && COMPARE_DESC_MEMBERS(acc_type)
            && COMPARE_DESC_MEMBERS(bias_type);
    return ret;
}

inline bool operator==(
        const inner_product_desc_t &lhs, const inner_product_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(const layer_normalization_desc_t &lhs,
        const layer_normalization_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(data_desc)
            && COMPARE_DESC_MEMBERS(diff_data_desc)
            && COMPARE_DESC_MEMBERS(data_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(diff_data_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(stat_desc)
            && COMPARE_DESC_MEMBERS(layer_norm_epsilon)
            && COMPARE_DESC_MEMBERS(flags);
    return ret;
}

inline bool operator==(const lrn_desc_t &lhs, const lrn_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(data_desc)
            && COMPARE_DESC_MEMBERS(diff_data_desc)
            && COMPARE_DESC_MEMBERS(local_size)
            && COMPARE_DESC_MEMBERS(lrn_alpha)
            && COMPARE_DESC_MEMBERS(lrn_beta)
            && COMPARE_DESC_MEMBERS(lrn_k);
    return ret;
}

inline bool operator==(const matmul_desc_t &lhs, const matmul_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(const pooling_desc_t &lhs, const pooling_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(strides, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(kernel, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[0], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[1], DNNL_MAX_NDIMS)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(const pooling_v2_desc_t &lhs, const pooling_v2_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(strides, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(kernel, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[0], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[1], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(dilation, DNNL_MAX_NDIMS)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(
        const reduction_desc_t &lhs, const reduction_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(p)
            && COMPARE_DESC_MEMBERS(eps);
    return ret;
}

inline bool operator==(const reorder_desc_t &lhs, const reorder_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(src_md)
            && COMPARE_DESC_MEMBERS(dst_md)
            && COMPARE_DESC_MEMBERS(src_engine_kind)
            && COMPARE_DESC_MEMBERS(dst_engine_kind);
    return ret;
}

inline bool operator==(
        const resampling_desc_t &lhs, const resampling_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(factors, DNNL_MAX_NDIMS);
    return ret;
}

inline bool operator==(const rnn_desc_t &lhs, const rnn_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(cell_kind)
            && COMPARE_DESC_MEMBERS(direction)
            && COMPARE_DESC_MEMBERS(src_layer_desc)
            && COMPARE_DESC_MEMBERS(src_iter_desc)
            && COMPARE_DESC_MEMBERS(src_iter_c_desc)
            && COMPARE_DESC_MEMBERS(weights_layer_desc)
            && COMPARE_DESC_MEMBERS(weights_iter_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(dst_layer_desc)
            && COMPARE_DESC_MEMBERS(dst_iter_desc)
            && COMPARE_DESC_MEMBERS(dst_iter_c_desc)
            && COMPARE_DESC_MEMBERS(weights_peephole_desc)
            && COMPARE_DESC_MEMBERS(weights_projection_desc)
            && COMPARE_DESC_MEMBERS(diff_src_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_src_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_src_iter_c_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_iter_c_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_peephole_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_projection_desc)
            && COMPARE_DESC_MEMBERS(flags)
            && COMPARE_DESC_MEMBERS(activation_kind)
            && COMPARE_DESC_MEMBERS(alpha)
            && COMPARE_DESC_MEMBERS(beta);
    return ret;
}

inline bool operator==(const shuffle_desc_t &lhs, const shuffle_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(data_desc)
            && COMPARE_DESC_MEMBERS(axis)
            && COMPARE_DESC_MEMBERS(group_size);
    return ret;
}

inline bool operator==(const softmax_desc_t &lhs, const softmax_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(data_desc)
            && COMPARE_DESC_MEMBERS(diff_desc)
            && COMPARE_DESC_MEMBERS(softmax_axis);
    return ret;
}

inline bool operator==(const sum_desc_t &lhs, const sum_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(dst_md)
            && COMPARE_DESC_MEMBERS(n)
            && COMPARE_DESC_MEMBERS(scales);
    if (!ret) return ret;

    for (int i = 0; i < lhs.n; i++) {
        ret = COMPARE_DESC_MEMBERS(src_mds[i]);
        if (!ret) break;
    }
    return ret;
}

inline bool operator==(const zero_pad_desc_t &lhs, const zero_pad_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind);
    return ret;
}
// clang-format on
#undef COMPARE_DESC_MEMBERS
#undef COMPARE_DESC_ARRAY_MEMBERS

inline status_t memory_desc_init_by_strides(
        memory_desc_t &md, const dims_t strides) {
    return dnnl_memory_desc_init_by_strides(
            &md, md.ndims, md.dims, md.data_type, strides);
}

inline status_t memory_desc_init_by_tag(
        memory_desc_t &md, format_tag_t tag, const dims_t strides = nullptr) {
    status_t status = dnnl_memory_desc_init_by_tag(
            &md, md.ndims, md.dims, md.data_type, tag);
    if (status != status::success || strides == nullptr) return status;

    /* TODO: add consistency check */

    for (int d = 0; d < md.ndims; ++d)
        md.format_desc.blocking.strides[d] = strides[d];

    return status::success;
}

/** inits memory descriptor based on logical dimensions kept in @p md, and the
 * blocking structure @p blk.
 *
 * @note blk.strides represent the order only (from smaller to bigger)
 *
 * TODO: move md related functions to one single place
 */
inline status_t memory_desc_init_by_blocking_desc(
        memory_desc_t &md, const blocking_desc_t &blk) {
    dims_t blocks = {0};
    utils::array_set(blocks, 1, md.ndims);
    dim_t block_size = 1;
    for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
        blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
        block_size *= blk.inner_blks[iblk];
    }

    for (int d = 0; d < md.ndims; ++d) {
        md.padded_dims[d] = utils::rnd_up(md.dims[d], blocks[d]);
        md.padded_offsets[d] = 0;
    }
    md.offset0 = 0;

    md.format_kind = format_kind::blocked;
    auto &mblk = md.format_desc.blocking;
    mblk = blk;

    const int ndims = nstl::min(DNNL_MAX_NDIMS, md.ndims); // make GCC 5 happy
    utils::array_copy(mblk.strides, blk.strides, ndims);

    dims_t ou_blocks = {0};
    utils::array_copy(ou_blocks, md.padded_dims, ndims);

    int perm[DNNL_MAX_NDIMS];
    for (int d = 0; d < ndims; ++d) {
        perm[d] = d;
        ou_blocks[d] /= blocks[d];
    }

    utils::simultaneous_sort(mblk.strides, ou_blocks, perm, ndims,
            [](stride_t a, stride_t b) { return b - a; });

    dim_t stride = block_size;
    for (int _d = ndims - 1; _d >= 0; --_d) {
        const int d = perm[_d];
        md.format_desc.blocking.strides[d] = stride;
        stride *= md.padded_dims[d] / blocks[d];
    }

    md.extra = utils::zero<memory_extra_desc_t>();

    return status::success;
}

/** inits memory descriptor @p md based on another one memory descriptor
 * @p md_base and given @p data_type.
 * Essentially: { md = md_base; md.dt = data_type; } */
inline status_t memory_desc_init_by_md_and_dt(memory_desc_t &md,
        const memory_desc_t &md_base, data_type_t data_type) {
    if (&md != &md_base) md = md_base;
    md.data_type = data_type;
    return status::success;
}

/** returns true if memory desc @p md corresponds to the given format tag and
 * strides.
 * If strides are not passed (or passed as nullptr) the dense structure is
 * assumed (i.e. the one that dnnl_memory_desc_init_by_tag() returns).
 * Strides might contain `0` value, indicating the stride must match the one
 * that dnnl_memory_desc_init_by_tag() returns.
 * Strides might contain `-1` values, that would be ignored during the
 * comparison. For instance, this can be used if a stride along minibatch
 * doesn't matter. */
inline bool memory_desc_matches_tag(const memory_desc_t &md, format_tag_t tag,
        const dims_t strides = nullptr) {
    if (md.format_kind != types::format_tag_to_kind(tag)) return false;

    memory_desc_t md_gold;
    status_t status = dnnl_memory_desc_init_by_tag(
            &md_gold, md.ndims, md.dims, md.data_type, tag);
    if (status != status::success) return false;

    if (md.format_kind != format_kind::blocked)
        return false; // unimplemented yet

    const auto &blk = md.format_desc.blocking;
    const auto &blk_gold = md_gold.format_desc.blocking;

    using utils::array_cmp;
    bool same_blocks = true && blk.inner_nblks == blk_gold.inner_nblks
            && array_cmp(blk.inner_blks, blk_gold.inner_blks, blk.inner_nblks)
            && array_cmp(blk.inner_idxs, blk_gold.inner_idxs, blk.inner_nblks);

    if (!same_blocks) return false;

    if (strides == nullptr)
        return array_cmp(blk.strides, blk_gold.strides, md.ndims);

    for (int d = 0; d < md.ndims; ++d) {
        dim_t stride = strides[d];
        if (stride == -1) continue;
        if (stride == 0) stride = blk_gold.strides[d];
        if (blk.strides[d] != stride) return false;
    }

    return true;
}

/** returns matching tag (or undef if match is not found)
 * XXX: This is a workaround that eventually should go away! */
template <typename... Tags>
format_tag_t memory_desc_matches_one_of_tag(
        const memory_desc_t &md, Tags... tags) {
    for (const auto tag : {tags...}) {
        if (memory_desc_matches_tag(md, tag)) return tag;
    }
    return format_tag::undef;
}

/** returns true if fp32 value denotes DNNL_RUNTIME_F32_VAL */
inline bool is_runtime_value(float val) {
    return utils::bit_cast<unsigned>(val) == DNNL_RUNTIME_F32_VAL_REP.u;
}

/** returns true if s32 value denotes DNNL_RUNTIME_S32_VAL */
inline bool is_runtime_value(int val) {
    return val == DNNL_RUNTIME_S32_VAL;
}

/** returns true if dim_t value denotes DNNL_RUNTIME_DIM_VAL */
inline bool is_runtime_value(dim_t val) {
    return val == DNNL_RUNTIME_DIM_VAL;
}

inline bool memory_desc_sanity_check(int ndims, const dims_t dims,
        data_type_t data_type, format_kind_t format_kind) {
    using namespace data_type;

    if (ndims == 0) return true;

    bool ok = dims != nullptr && 0 < ndims && ndims <= DNNL_MAX_NDIMS
            && utils::one_of(data_type, f16, bf16, f32, s32, s8, u8);
    if (!ok) return false;

    bool has_runtime_dims = false;
    for (int d = 0; d < ndims; ++d) {
        if (dims[d] != DNNL_RUNTIME_DIM_VAL && dims[d] < 0) return false;
        if (dims[d] == DNNL_RUNTIME_DIM_VAL) has_runtime_dims = true;
    }

    if (has_runtime_dims) {
        // format `any` is currently not supported for run-time dims
        if (format_kind == format_kind::any) return false;
    }

    return true;
}

inline bool memory_desc_sanity_check(const memory_desc_t *md) {
    if (md == nullptr) return false;
    return memory_desc_sanity_check(
            md->ndims, md->dims, md->data_type, format_kind::undef);
}
} // namespace impl
} // namespace dnnl

#include "memory_desc_wrapper.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
