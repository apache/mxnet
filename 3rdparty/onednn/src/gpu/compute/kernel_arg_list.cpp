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

#include "gpu/compute/kernel_arg_list.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

kernel_arg_t kernel_arg_t::cast(
        scalar_type_t type, const kernel_arg_t &arg, void *&cast_storage) {
    assert(arg.kind() == kernel_arg_kind_t::scalar);

    if (type == arg.scalar_type()) return arg;

    // Downcast if necessary.
    kernel_arg_t ret;
    switch (type) {
        case scalar_type_t::_half:
            return ret.set_value((float16_t)arg.as<float>(), cast_storage);
        case scalar_type_t::_uchar:
            return ret.set_value((uint8_t)arg.as<int>(), cast_storage);
        case scalar_type_t::_char:
            return ret.set_value((int8_t)arg.as<int>(), cast_storage);
        default:
            assert(!"Cannot convert argument to the kernel argument type.");
            return arg;
    }
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl
