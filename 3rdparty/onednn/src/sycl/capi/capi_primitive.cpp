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

#include <CL/sycl.hpp>

#include "oneapi/dnnl/dnnl_sycl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "sycl/sycl_engine.hpp"
#include "sycl/sycl_stream.hpp"

using namespace dnnl::impl;

status_t dnnl_sycl_interop_primitive_execute(
        const primitive_iface_t *primitive_iface, stream_t *stream, int nargs,
        const dnnl_exec_arg_t *args, const void *deps_, void *return_event_) {
    bool ok = !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && IMPLICATION(nargs > 0, args != nullptr);
    if (!ok) return status::invalid_arguments;

    auto *sycl_stream
            = utils::downcast<dnnl::impl::sycl::sycl_stream_t *>(stream);

    if (deps_ != nullptr) {
        const auto &deps = *(const std::vector<cl::sycl::event> *)deps_;
        sycl_stream->set_deps(deps);
    }

    // run primitive
    exec_args_t exec_args;
    CHECK(cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, args, exec_args));

    exec_ctx_t ctx(sycl_stream, std::move(exec_args));
    CHECK(primitive_execute(primitive_iface, ctx));

    // return output event
    cl::sycl::event return_event = sycl_stream->get_output_event();
    if (return_event_ != nullptr)
        *(cl::sycl::event *)return_event_ = return_event;

    return status::success;
}
