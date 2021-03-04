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

#ifndef COMMON_PRIMITIVE_HASHING_HPP
#define COMMON_PRIMITIVE_HASHING_HPP

#include <typeindex>
#include <type_traits>

#include "c_types_map.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;
namespace primitive_hashing {

struct cached_op_desc_t {
    cached_op_desc_t(const op_desc_t *op_desc)
        : kind_(get_kind(op_desc->kind)) {
#define CASE(pkind) \
    case primitive_kind::pkind: \
        placeholder_ \
                = new pkind##_desc_t(cast_to_desc<pkind##_desc_t>(op_desc)); \
        break;

        // clang-format off
        switch ((int)kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(convolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            case primitive_kind::pooling_v2:
            CASE(pooling)
            CASE(reduction)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(concat)
            CASE(sum)
            CASE(zero_pad)
            default: assert(!"unknown primitive kind");
        }
            // clang-format on
#undef CASE
    }

    cached_op_desc_t(const cached_op_desc_t &other)
        : cached_op_desc_t(
                reinterpret_cast<const op_desc_t *>(other.placeholder_)) {}

    bool operator==(const cached_op_desc_t &other) const {
        if (kind_ != other.kind_) return false;
#define CASE(pkind) \
    case primitive_kind::pkind: \
        ret = cast_to_desc<pkind##_desc_t>(placeholder_) \
                == cast_to_desc<pkind##_desc_t>(other.placeholder_); \
        break;

        bool ret = true;
        // clang-format off
        switch ((int)kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(concat)
            CASE(convolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            case primitive_kind::pooling_v2:
            CASE(pooling)
            CASE(reduction)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(sum)
            CASE(zero_pad)
            default: assert(!"unknown primitive kind");
        }
            // clang-format on
#undef CASE
        return ret;
    }

#define DECLARE_CONVERSION_OPERATOR(pkind) \
    operator pkind##_desc_t() const { \
        assert(kind_ == primitive_kind::pkind \
                || (primitive_kind::pkind == primitive_kind::pooling \
                        && kind_ == primitive_kind::pooling_v2)); \
        return cast_to_desc<pkind##_desc_t>(placeholder_); \
    }
    DECLARE_CONVERSION_OPERATOR(batch_normalization)
    DECLARE_CONVERSION_OPERATOR(binary)
    DECLARE_CONVERSION_OPERATOR(concat)
    DECLARE_CONVERSION_OPERATOR(convolution)
    DECLARE_CONVERSION_OPERATOR(eltwise)
    DECLARE_CONVERSION_OPERATOR(gemm)
    DECLARE_CONVERSION_OPERATOR(inner_product)
    DECLARE_CONVERSION_OPERATOR(layer_normalization)
    DECLARE_CONVERSION_OPERATOR(lrn)
    DECLARE_CONVERSION_OPERATOR(matmul)
    DECLARE_CONVERSION_OPERATOR(pooling)
    DECLARE_CONVERSION_OPERATOR(pooling_v2)
    DECLARE_CONVERSION_OPERATOR(reduction)
    DECLARE_CONVERSION_OPERATOR(reorder)
    DECLARE_CONVERSION_OPERATOR(resampling)
    DECLARE_CONVERSION_OPERATOR(rnn)
    DECLARE_CONVERSION_OPERATOR(shuffle)
    DECLARE_CONVERSION_OPERATOR(softmax)
    DECLARE_CONVERSION_OPERATOR(sum)
    DECLARE_CONVERSION_OPERATOR(zero_pad)
#undef DECLARE_CONVERSION_OPERATOR

    ~cached_op_desc_t() {
#define CASE(pkind) \
    case primitive_kind::pkind: \
        delete reinterpret_cast<pkind##_desc_t *>(placeholder_); \
        break;

        // clang-format off
        switch ((int)kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(concat)
            CASE(convolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(logsoftmax)
            CASE(lrn)
            CASE(matmul)
            case primitive_kind::pooling_v2:
            CASE(pooling)
            CASE(reduction)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(sum)
            CASE(zero_pad)
            default: assert(!"unknown primitive_kind");
        }
            // clang-format on
#undef CASE
    }

private:
    cached_op_desc_t() = delete;
    cached_op_desc_t &operator=(const cached_op_desc_t &) = delete;

    template <typename desc_t>
    static const desc_t &cast_to_desc(const void *p) {
        return *(reinterpret_cast<const desc_t *>(p));
    }

    static primitive_kind_t get_kind(primitive_kind_t kind) {
        auto k = primitive_kind::undefined;
        switch (kind) {
            case primitive_kind::softmax:
            case primitive_kind::logsoftmax: k = primitive_kind::softmax; break;
            case primitive_kind::convolution:
            case primitive_kind::deconvolution:
                k = primitive_kind::convolution;
                break;
            default: k = kind;
        }
        return k;
    }

    primitive_kind_t kind_;
    void *placeholder_ = nullptr;
};

struct key_t {
    key_t(const primitive_desc_t *pd, const engine_t *engine, int impl_nthr);

    bool operator==(const key_t &other) const;

    primitive_kind_t primitive_kind_;
    cached_op_desc_t op_desc_;
    primitive_attr_t attr_;
    std::type_index impl_id_;
    int impl_nthr_;
    std::vector<memory_desc_t> mds;
    engine_kind_t engine_kind_;
    runtime_kind_t runtime_kind_;
    device_id_t device_id_;

private:
    void init_mds(const primitive_desc_t *pd);
};

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
template <typename T>
static size_t hash_combine(size_t seed, const T &v) {
    return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

size_t get_md_hash(const memory_desc_t &md);
size_t get_attr_hash(const primitive_attr_t &attr);
size_t get_desc_hash(const concat_desc_t &desc);
size_t get_desc_hash(const batch_normalization_desc_t &desc);
size_t get_desc_hash(const binary_desc_t &desc);
size_t get_desc_hash(const convolution_desc_t &desc);
size_t get_desc_hash(const eltwise_desc_t &desc);
size_t get_desc_hash(const gemm_desc_t &desc);
size_t get_desc_hash(const inner_product_desc_t &desc);
size_t get_desc_hash(const layer_normalization_desc_t &desc);
size_t get_desc_hash(const lrn_desc_t &desc);
size_t get_desc_hash(const matmul_desc_t &desc);
size_t get_desc_hash(const pooling_desc_t &desc);
size_t get_desc_hash(const pooling_v2_desc_t &desc);
size_t get_desc_hash(const reduction_desc_t &desc);
size_t get_desc_hash(const reorder_desc_t &desc);
size_t get_desc_hash(const resampling_desc_t &desc);
size_t get_desc_hash(const rnn_desc_t &desc);
size_t get_desc_hash(const shuffle_desc_t &desc);
size_t get_desc_hash(const softmax_desc_t &desc);
size_t get_desc_hash(const sum_desc_t &desc);
size_t get_desc_hash(const zero_pad_desc_t &desc);

template <typename T>
size_t get_array_hash(size_t seed, const T *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, v[i]);
    }
    return seed;
}

// Specialization for an array of mds
template <>
inline size_t get_array_hash<memory_desc_t>(
        size_t seed, const memory_desc_t *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, get_md_hash(v[i]));
    }
    return seed;
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl

// inject a specialization of std::hash for key_t in std namespace
namespace std {
template <>
struct hash<dnnl::impl::primitive_hashing::key_t> {
    using argument_type = dnnl::impl::primitive_hashing::key_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &key) const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        // Compute hash for primitive_kind_, attr_, impl_id_ and impl_nthr_
        seed = hash_combine(seed,
                hash_combine(0, static_cast<size_t>(key.primitive_kind_)));
        seed = hash_combine(seed, get_attr_hash(key.attr_));
        seed = hash_combine(seed, hash_combine(0, key.impl_id_));
        seed = hash_combine(seed, hash_combine(0, key.impl_nthr_));
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.engine_kind_)));
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.runtime_kind_)));
        seed = hash_combine(seed, hash_combine(0, std::get<0>(key.device_id_)));
        seed = hash_combine(seed, hash_combine(0, std::get<1>(key.device_id_)));
        seed = hash_combine(seed, hash_combine(0, std::get<2>(key.device_id_)));
        // Combine hash for op_desc with the computed hash
#define CASE(pkind) \
    case primitive_kind::pkind: \
        seed = hash_combine( \
                seed, get_desc_hash((pkind##_desc_t)key.op_desc_)); \
        break;

        // clang-format off
        switch ((int)key.primitive_kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(concat)
            CASE(convolution)
            CASE(deconvolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            case primitive_kind::pooling_v2:
            CASE(pooling)
            CASE(reduction)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(sum)
            CASE(zero_pad)
            default: assert(!"unknown primitive_kind");
        }
            // clang-format on
#undef CASE
        seed = get_array_hash(seed, key.mds.data(), (int)key.mds.size());

        return seed;
    }
};

} // namespace std

#endif
