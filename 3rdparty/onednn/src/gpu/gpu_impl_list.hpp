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

#ifndef GPU_GPU_IMPL_LIST_HPP
#define GPU_GPU_IMPL_LIST_HPP

#include "common/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

class gpu_impl_list_t {
public:
    static const engine_t::concat_primitive_desc_create_f *
    get_concat_implementation_list();
    static const engine_t::reorder_primitive_desc_create_f *
    get_reorder_implementation_list(
            const memory_desc_t *src_md, const memory_desc_t *dst_md);
    static const engine_t::sum_primitive_desc_create_f *
    get_sum_implementation_list();
    static const engine_t::primitive_desc_create_f *get_implementation_list();
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_GPU_IMPL_LIST_HPP
