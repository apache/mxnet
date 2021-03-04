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

#ifndef GPU_COMPUTE_KERNEL_LIST_HPP
#define GPU_COMPUTE_KERNEL_LIST_HPP

#include <cassert>
#include <unordered_map>

#include "gpu/compute/kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class kernel_list_t {
public:
    void add(const char *name, kernel_t *kernel) {
        assert(kernels_.count(name) == 0);
        kernels_[name] = kernel;
    }

    void set(const char *name, const kernel_t &kernel) {
        assert(kernels_.count(name) > 0);
        *kernels_[name] = kernel;
    }

    const std::unordered_map<std::string, kernel_t *> &kernels() const {
        return kernels_;
    }

private:
    std::unordered_map<std::string, kernel_t *> kernels_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_KERNEL_LIST_HPP
