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

#include "cpu/cpu_engine.hpp"

#include "cpu/ref_layer_normalization.hpp"
#include "cpu/simple_layer_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const pd_create_f impl_list[] = {
        CPU_INSTANCE(simple_layer_normalization_fwd_t<f32>)
        CPU_INSTANCE(simple_layer_normalization_bwd_t<f32>)
        CPU_INSTANCE(simple_layer_normalization_fwd_t<bf16>)
        CPU_INSTANCE(simple_layer_normalization_bwd_t<bf16>)
        CPU_INSTANCE(ref_layer_normalization_fwd_t<f32>)
        CPU_INSTANCE(ref_layer_normalization_bwd_t<f32>)
        CPU_INSTANCE(ref_layer_normalization_fwd_t<bf16>)
        CPU_INSTANCE(ref_layer_normalization_bwd_t<bf16>)
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const pd_create_f *get_layer_normalization_impl_list(
        const layer_normalization_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
