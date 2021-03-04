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

#ifndef GPU_OCL_VERBOSE_HPP
#define GPU_OCL_VERBOSE_HPP

#include <cstdio>
#include <string>
#include <vector>
#include <CL/cl.h>

#include "gpu/compute/device_info.hpp"
#include "gpu/ocl/ocl_gpu_device_info.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

void print_verbose_header() {
    std::vector<cl_device_id> ocl_devices;
    auto status = get_ocl_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);
    OCL_CHECK_V(status);

    for (size_t i = 0; i < ocl_devices.size(); ++i) {
        ocl_gpu_device_info_t dev_info(ocl_devices[i]);
        status_t status = dev_info.init();
        std::string name = "unknown";
        compute::runtime_version_t ver;
        if (status == status::success) {
            name = dev_info.name();
            ver = dev_info.runtime_version();
        }
        printf("dnnl_verbose,info,gpu,engine,%d,name:%s,driver_version:%s\n",
                (int)i, name.c_str(), ver.str().c_str());
    }
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
