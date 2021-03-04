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

#ifndef SYCL_VERBOSE_HPP
#define SYCL_VERBOSE_HPP

#include <cstdio>
#include <CL/sycl.hpp>

#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_engine.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

void print_verbose_header(
        cl::sycl::info::device_type dev_type, const char *dev_type_str) {
    auto devices = get_intel_sycl_devices(dev_type);
    for (size_t i = 0; i < devices.size(); ++i) {
        sycl_device_info_t dev_info(devices[i]);
        status_t status = dev_info.init();
        auto &name = dev_info.name();
        auto &ver = dev_info.runtime_version();
        auto s_backend = to_string(get_sycl_backend(devices[i]));
        printf("dnnl_verbose,info,%s,engine,%d,backend:%s,name:%s,driver_"
               "version:%s\n",
                dev_type_str, (int)i, s_backend.c_str(), name.c_str(),
                ver.str().c_str());
    }
}

void print_verbose_header() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    print_verbose_header(cl::sycl::info::device_type::cpu, "cpu");
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    print_verbose_header(cl::sycl::info::device_type::gpu, "gpu");
#endif
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
