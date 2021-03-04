/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "cpu/ref_sum.hpp"
#include "cpu/simple_sum.hpp"

#if DNNL_X64
#include "cpu/x64/jit_avx512_core_bf16_sum.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

using spd_create_f = dnnl::impl::engine_t::sum_primitive_desc_create_f;

namespace {
// clang-format off
#define INSTANCE(...) __VA_ARGS__::pd_t::create,
#define INSTANCE_X64(...) DNNL_X64_ONLY(INSTANCE(__VA_ARGS__))
const spd_create_f cpu_sum_impl_list[] = {
        INSTANCE_X64(jit_bf16_sum_t<data_type::bf16, data_type::bf16>)
        INSTANCE_X64(jit_bf16_sum_t<data_type::bf16, data_type::f32>)
        INSTANCE(simple_sum_t<data_type::bf16>)
        INSTANCE(simple_sum_t<data_type::bf16, data_type::f32>)
        INSTANCE(simple_sum_t<data_type::f32>)
        INSTANCE(ref_sum_t)
        nullptr,
};
#undef INSTANCE_X64
#undef INSTANCE
// clang-format on
} // namespace

const spd_create_f *cpu_engine_impl_list_t::get_sum_implementation_list() {
    return cpu_sum_impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
