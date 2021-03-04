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

#include "gpu/ocl/ocl_gpu_detect.hpp"
#include "gpu/jit/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

compute::gpu_arch_t detect_gpu_arch(cl_device_id device, cl_context context) {
    using namespace ngen;

    HW hw = jit::jit_generator<HW::Unknown>::detectHW(context, device);
    switch (hw) {
        case HW::Gen9: return compute::gpu_arch_t::gen9;
        case HW::Gen12LP: return compute::gpu_arch_t::gen12lp;
        default: return compute::gpu_arch_t::unknown;
    }
}

compute::gpu_arch_t detect_gpu_arch_by_device_name(const std::string &name) {
    if (name.find("Gen9") != std::string::npos)
        return compute::gpu_arch_t::gen9;
    if (name.find("Gen12LP") != std::string::npos)
        return compute::gpu_arch_t::gen12lp;

    return compute::gpu_arch_t::unknown;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
