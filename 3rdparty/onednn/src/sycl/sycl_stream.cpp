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

#include "sycl/sycl_stream.hpp"

#include "gpu/ocl/ocl_utils.hpp"
#include "sycl/sycl_engine.hpp"

#include <map>
#include <memory>
#include <CL/cl.h>

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    // If queue_ is not set then construct it
    if (!queue_) {
        auto &sycl_engine = *utils::downcast<sycl_engine_base_t *>(engine());
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();

        // FIXME: workaround for Intel(R) oneAPI DPC++ Compiler
        // Intel(R) oneAPI DPC++ Compiler does not work with multiple queues so
        // try to reuse the service stream from the engine.
        // That way all oneDNN streams constructed without interop API are
        // mapped to the same SYCL queue.
        // If service stream is NULL then the current stream will be service
        // so construct it from scratch.
        if (!sycl_engine.is_service_stream_created()) {
#ifdef DNNL_SYCL_DPCPP
            cl::sycl::property_list props = (flags() & stream_flags::in_order)
                    ? cl::sycl::property_list {cl::sycl::property::queue::
                                    in_order {}}
                    : cl::sycl::property_list {};
            queue_.reset(new cl::sycl::queue(sycl_ctx, sycl_dev, props));
#else
            // XXX: Create a queue based on the SYCL context and engine kind.
            // This is not reliable but there is no constructor from
            // context and device in SYCL 1.2.1.
            if (sycl_dev.is_cpu()) {
                assert(sycl_engine.kind() == engine_kind::cpu);
                queue_.reset(new cl::sycl::queue(
                        sycl_ctx, cl::sycl::cpu_selector {}));
            } else if (sycl_dev.is_host()) {
                assert(sycl_engine.kind() == engine_kind::cpu);
                queue_.reset(new cl::sycl::queue(
                        sycl_ctx, cl::sycl::host_selector {}));
            } else {
                assert(sycl_engine.kind() == engine_kind::gpu);
                queue_.reset(new cl::sycl::queue(
                        sycl_ctx, cl::sycl::gpu_selector {}));
            }
#endif

        } else {
            // XXX: multiple queues support has some issues, so always re-use
            // the same queue from the service stream.
            stream_t *service_stream;
            CHECK(sycl_engine.get_service_stream(service_stream));
            auto sycl_stream = utils::downcast<sycl_stream_t *>(service_stream);
            queue_.reset(new cl::sycl::queue(sycl_stream->queue()));
        }
    } else {
        // TODO: Compare device and context of the engine with those of the
        // queue after SYCL adds support for device/context comparison.
        //
        // For now perform some simple checks.
        auto sycl_dev = queue_->get_device();
        bool args_ok = true
                && IMPLICATION(
                        engine()->kind() == engine_kind::gpu, sycl_dev.is_gpu())
                && IMPLICATION(engine()->kind() == engine_kind::cpu,
                        (sycl_dev.is_cpu() || sycl_dev.is_host()));
        if (!args_ok) return status::invalid_arguments;
    }

    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
