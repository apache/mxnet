/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
* Copyright 2020 Arm Ltd. and affiliates
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

#ifndef CPU_CPU_ENGINE_HPP
#define CPU_CPU_ENGINE_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"

#include "cpu/platform.hpp"

#define CPU_INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>,
#define CPU_INSTANCE_X64(...) DNNL_X64_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AARCH64(...) DNNL_AARCH64_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AARCH64_ACL(...) \
    DNNL_AARCH64_ACL_ONLY(CPU_INSTANCE(__VA_ARGS__))

namespace dnnl {
namespace impl {
namespace cpu {

#define DECLARE_IMPL_LIST(kind) \
    const engine_t::primitive_desc_create_f *get_##kind##_impl_list( \
            const kind##_desc_t *desc);

DECLARE_IMPL_LIST(batch_normalization);
DECLARE_IMPL_LIST(binary);
DECLARE_IMPL_LIST(convolution);
DECLARE_IMPL_LIST(deconvolution);
DECLARE_IMPL_LIST(eltwise);
DECLARE_IMPL_LIST(inner_product);
DECLARE_IMPL_LIST(layer_normalization);
DECLARE_IMPL_LIST(lrn);
DECLARE_IMPL_LIST(logsoftmax);
DECLARE_IMPL_LIST(matmul);
DECLARE_IMPL_LIST(pooling_v2);
DECLARE_IMPL_LIST(reduction);
DECLARE_IMPL_LIST(resampling);
DECLARE_IMPL_LIST(rnn);
DECLARE_IMPL_LIST(shuffle);
DECLARE_IMPL_LIST(softmax);

#undef DECLARE_IMPL_LIST

class cpu_engine_impl_list_t {
public:
    static const engine_t::concat_primitive_desc_create_f *
    get_concat_implementation_list();
    static const engine_t::reorder_primitive_desc_create_f *
    get_reorder_implementation_list(
            const memory_desc_t *src_md, const memory_desc_t *dst_md);
    static const engine_t::sum_primitive_desc_create_f *
    get_sum_implementation_list();
    static const engine_t::primitive_desc_create_f *get_implementation_list(
            const op_desc_t *desc) {
        static const engine_t::primitive_desc_create_f empty_list[] = {nullptr};

// clang-format off
#define CASE(kind) \
    case primitive_kind::kind: \
        return get_##kind##_impl_list((const kind##_desc_t *)desc);
        switch (desc->kind) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(logsoftmax);
            CASE(matmul);
            case primitive_kind::pooling:
            CASE(pooling_v2);
            CASE(reduction);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            default: assert(!"unknown primitive kind"); return empty_list;
        }
#undef CASE
    }
    // clang-format on
};

class cpu_engine_t : public engine_t {
public:
    cpu_engine_t() : engine_t(engine_kind::cpu, get_cpu_native_runtime()) {}

    /* implementation part */

    virtual status_t create_memory_storage(memory_storage_t **storage,
            unsigned flags, size_t size, void *handle) override;

    virtual status_t create_stream(stream_t **stream, unsigned flags) override;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    virtual status_t create_stream(stream_t **stream,
            dnnl::threadpool_interop::threadpool_iface *threadpool) override;
#endif

    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const override {
        return cpu_engine_impl_list_t::get_concat_implementation_list();
    }

    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list(const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return cpu_engine_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }
    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const override {
        return cpu_engine_impl_list_t::get_sum_implementation_list();
    }
    virtual const primitive_desc_create_f *get_implementation_list(
            const op_desc_t *desc) const override {
        return cpu_engine_impl_list_t::get_implementation_list(desc);
    }

    device_id_t device_id() const override { return std::make_tuple(0, 0, 0); }
};

class cpu_engine_factory_t : public engine_factory_t {
public:
    size_t count() const override { return 1; }
    status_t engine_create(engine_t **engine, size_t index) const override {
        assert(index == 0);
        *engine = new cpu_engine_t();
        return status::success;
    };
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
