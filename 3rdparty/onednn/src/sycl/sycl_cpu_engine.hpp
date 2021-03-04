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

#ifndef SYCL_CPU_ENGINE_HPP
#define SYCL_CPU_ENGINE_HPP

#include "cpu/cpu_engine.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "sycl/sycl_engine_base.hpp"
#include "sycl/sycl_utils.hpp"

#include <CL/sycl.hpp>

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_cpu_engine_t : public sycl_engine_base_t {
public:
    using sycl_engine_base_t::context;
    using sycl_engine_base_t::create_stream;
    using sycl_engine_base_t::device;

    sycl_cpu_engine_t(const cl::sycl::device &dev, const cl::sycl::context &ctx)
        : sycl_engine_base_t(engine_kind::cpu, dev, ctx) {
        assert(dev.is_cpu() || dev.is_host());
    }

    virtual status_t create_memory_storage(memory_storage_t **storage,
            unsigned flags, size_t size, void *handle) override {
        return sycl_engine_base_t::create_memory_storage(
                storage, flags, size, handle);
    }

    virtual status_t create_stream(stream_t **stream, unsigned flags) override {
        return sycl_engine_base_t::create_stream(stream, flags);
    }

    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list(const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return cpu::cpu_engine_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }

    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const override {
        return cpu::cpu_engine_impl_list_t::get_concat_implementation_list();
    }

    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const override {
        return cpu::cpu_engine_impl_list_t::get_sum_implementation_list();
    }

    virtual const primitive_desc_create_f *get_implementation_list(
            const op_desc_t *desc) const override {
        return cpu::cpu_engine_impl_list_t::get_implementation_list(desc);
    }
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
