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

#include "cpu/ref_eltwise.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "cpu/x64/jit_uni_eltwise_int.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const pd_create_f impl_list[] = {
        CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t<avx512_common, f32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t<avx512_common, f32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t<avx512_core, bf16>)
        CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t<avx512_core, bf16>)
        CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t<avx2, f32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t<avx2, f32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t<avx, f32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t<avx, f32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t<sse41, f32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t<sse41, f32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<avx512_common, s32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<avx512_common, s8>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<avx512_common, u8>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<avx2, s32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<avx2, s8>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<avx2, u8>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<sse41, s32>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<sse41, s8>)
        CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t<sse41, u8>)
        CPU_INSTANCE(ref_eltwise_fwd_t<f32>)
        CPU_INSTANCE(ref_eltwise_bwd_t<f32>)
        CPU_INSTANCE(ref_eltwise_fwd_t<bf16>)
        CPU_INSTANCE(ref_eltwise_bwd_t<bf16>)
        CPU_INSTANCE(ref_eltwise_fwd_t<s32>)
        CPU_INSTANCE(ref_eltwise_fwd_t<s8>)
        CPU_INSTANCE(ref_eltwise_fwd_t<u8>)
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const pd_create_f *get_eltwise_impl_list(const eltwise_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
