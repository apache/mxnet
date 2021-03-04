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

#include "cpu/cpu_engine.hpp"

#include "cpu/ref_reduction.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const pd_create_f impl_list[] = {
    CPU_INSTANCE(ref_reduction_t<f32, f32, f32>)
    CPU_INSTANCE(ref_reduction_t<bf16, bf16, f32>)
    CPU_INSTANCE(ref_reduction_t<bf16, f32, f32>)
    CPU_INSTANCE(ref_reduction_t<s8, s8, s32>)
    CPU_INSTANCE(ref_reduction_t<s8, s32, s32>)
    CPU_INSTANCE(ref_reduction_t<s8, f32, f32>)
    CPU_INSTANCE(ref_reduction_t<u8, u8, s32>)
    CPU_INSTANCE(ref_reduction_t<u8, s32, s32>)
    CPU_INSTANCE(ref_reduction_t<u8, f32, f32>)
    /* eol */
    nullptr,
};
// clang-format on
} //namespace

const pd_create_f *get_reduction_impl_list(const reduction_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl
