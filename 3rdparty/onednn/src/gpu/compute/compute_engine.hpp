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

#ifndef GPU_COMPUTE_COMPUTE_ENGINE_HPP
#define GPU_COMPUTE_COMPUTE_ENGINE_HPP

#include <cassert>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/primitive_iterator.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/compute/dispatch.hpp"
#include "gpu/compute/kernel.hpp"
#include "gpu/compute/kernel_ctx.hpp"
#include "gpu/jit/jit_generator_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class compute_engine_t : public engine_t {
public:
    compute_engine_t(engine_kind_t kind, runtime_kind_t runtime_kind,
            device_info_t *device_info)
        : engine_t(kind, runtime_kind), device_info_(device_info) {}

    status_t init() { return device_info_->init(); }

    const device_info_t *device_info() const { return device_info_.get(); }

    status_t create_kernel(kernel_t *kernel, const char *kernel_name,
            const kernel_ctx_t &kernel_ctx) const {

        std::vector<kernel_t> kernels(1);
        auto status = create_kernels(&kernels, {kernel_name}, kernel_ctx);
        if (status == status::success) *kernel = kernels[0];
        return status;
    }

    virtual status_t create_kernel(compute::kernel_t *kernel,
            jit::jit_generator_base &jitter) const = 0;

    virtual status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const = 0;

    virtual status_t create_kernels_from_ocl_source(
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const char **source_strings,
            const compute::kernel_ctx_t &kernel_ctx) const {
        assert(!"unexpected");
        return status::success;
    };

    status_t get_zero_pad_primitive(primitive_t *&result) {
        status_t status = status::success;
        if (zero_pad_primitive_ == nullptr) {
            zero_pad_desc_t desc;
            desc.primitive_kind = primitive_kind::zero_pad;
            dnnl_primitive_desc_iterator it(
                    this, (op_desc_t *)&desc, nullptr, nullptr);
            ++it;
            std::unique_ptr<primitive_desc_t> zero_pad_pd(it.fetch_once());
            status = zero_pad_pd->create_primitive(zero_pad_primitive_, this);
        }
        result = zero_pad_primitive_.get();
        return status;
    };

    bool mayiuse(device_ext_t ext) const { return device_info_->has(ext); }

    dispatch_t create_dispatch(const memory_desc_t *md = nullptr) const {
        return dispatch_t(this, md);
    }

    virtual bool mayiuse_ngen_kernels() { return false; }

    status_t get_service_stream(stream_t *&stream) override {
        status_t status = status::success;
        if (service_stream_ == nullptr) {
            const std::lock_guard<std::mutex> lock(service_stream_mutex_);
            if (service_stream_ == nullptr) {
                stream_t *service_stream_ptr;
                status = create_stream(
                        &service_stream_ptr, stream_flags::default_flags);
                if (status == status::success)
                    service_stream_.reset(service_stream_ptr);
            }
        }
        stream = service_stream_.get();
        return status;
    }

    // non-blocking query to check if service stream is already created
    bool is_service_stream_created() const { return (bool)service_stream_; }

private:
    std::unique_ptr<device_info_t> device_info_;
    std::shared_ptr<primitive_t> zero_pad_primitive_;
    std::unique_ptr<stream_t> service_stream_;
    std::mutex service_stream_mutex_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
