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

#include "cpu/ncsp_batch_normalization.hpp"
#include "cpu/nspc_batch_normalization.hpp"
#include "cpu/ref_batch_normalization.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_batch_normalization.hpp"
#include "cpu/x64/jit_uni_batch_normalization_s8.hpp"
#include "cpu/x64/jit_uni_tbb_batch_normalization.hpp"
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
        /* fp */
        CPU_INSTANCE_X64(jit_uni_batch_normalization_fwd_t<avx512_common>)
        CPU_INSTANCE_X64(jit_uni_batch_normalization_bwd_t<avx512_common>)
        CPU_INSTANCE_X64(jit_uni_batch_normalization_fwd_t<avx2>)
        CPU_INSTANCE_X64(jit_uni_batch_normalization_bwd_t<avx2>)
        CPU_INSTANCE_X64(jit_uni_batch_normalization_fwd_t<sse41>)
        CPU_INSTANCE_X64(jit_uni_batch_normalization_bwd_t<sse41>)
        CPU_INSTANCE_X64(jit_uni_tbb_batch_normalization_fwd_t<avx512_common>)
        CPU_INSTANCE_X64(jit_uni_tbb_batch_normalization_bwd_t<avx512_common>)
        CPU_INSTANCE_X64(jit_uni_tbb_batch_normalization_fwd_t<avx2>)
        CPU_INSTANCE_X64(jit_uni_tbb_batch_normalization_bwd_t<avx2>)
        CPU_INSTANCE_X64(jit_uni_tbb_batch_normalization_fwd_t<sse41>)
        CPU_INSTANCE_X64(jit_uni_tbb_batch_normalization_bwd_t<sse41>)
        CPU_INSTANCE(ncsp_batch_normalization_fwd_t<f32>)
        CPU_INSTANCE(ncsp_batch_normalization_bwd_t<f32>)
        CPU_INSTANCE(ncsp_batch_normalization_fwd_t<bf16>)
        CPU_INSTANCE(ncsp_batch_normalization_bwd_t<bf16>)
        CPU_INSTANCE(nspc_batch_normalization_fwd_t<f32>)
        CPU_INSTANCE(nspc_batch_normalization_bwd_t<f32>)
        CPU_INSTANCE(nspc_batch_normalization_fwd_t<bf16>)
        CPU_INSTANCE(nspc_batch_normalization_bwd_t<bf16>)
        CPU_INSTANCE(ref_batch_normalization_fwd_t<f32>)
        CPU_INSTANCE(ref_batch_normalization_bwd_t<f32>)
        CPU_INSTANCE(ref_batch_normalization_fwd_t<bf16>)
        CPU_INSTANCE(ref_batch_normalization_bwd_t<bf16>)
        /* int */
        CPU_INSTANCE_X64(jit_uni_batch_normalization_s8_fwd_t<avx512_core>)
        CPU_INSTANCE_X64(jit_uni_batch_normalization_s8_fwd_t<avx2>)
        CPU_INSTANCE_X64(jit_uni_batch_normalization_s8_fwd_t<sse41>)
        CPU_INSTANCE(ref_batch_normalization_fwd_t<s8>)
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const pd_create_f *get_batch_normalization_impl_list(
        const batch_normalization_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
