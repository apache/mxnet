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

#include "cpu/gemm_inner_product.hpp"
#include "cpu/gemm_x8s8s32x_inner_product.hpp"
#include "cpu/ref_inner_product.hpp"

#if DNNL_X64
#include "cpu/x64/gemm_bf16_inner_product.hpp"
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
        /* f32 */
        CPU_INSTANCE(gemm_inner_product_fwd_t<f32>)
        CPU_INSTANCE(gemm_inner_product_bwd_data_t<f32>)
        CPU_INSTANCE(gemm_inner_product_bwd_weights_t<f32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<f32>)
        CPU_INSTANCE(ref_inner_product_bwd_data_t<f32, f32, f32, f32>)
        CPU_INSTANCE(ref_inner_product_bwd_weights_t<f32>)
        /* bfloat16 */
        CPU_INSTANCE_X64(gemm_bf16_inner_product_fwd_t<f32>)
        CPU_INSTANCE_X64(gemm_bf16_inner_product_fwd_t<bf16>)
        CPU_INSTANCE_X64(gemm_bf16_inner_product_bwd_data_t<f32>)
        CPU_INSTANCE_X64(gemm_bf16_inner_product_bwd_data_t<bf16>)
        CPU_INSTANCE_X64(gemm_bf16_inner_product_bwd_weights_t<f32>)
        CPU_INSTANCE_X64(gemm_bf16_inner_product_bwd_weights_t<bf16>)
        CPU_INSTANCE(ref_inner_product_fwd_t<bf16, bf16, bf16, f32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<bf16, bf16, f32, f32>)
        /* int */
        CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<u8, u8>)
        CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<u8, s8>)
        CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<u8, s32>)
        CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<u8, f32>)
        CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<s8, u8>)
        CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<s8, s8>)
        CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<s8, s32>)
        CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t<s8, f32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<u8, s8, u8, s32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<u8, s8, s8, s32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<u8, s8, s32, s32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<u8, s8, f32, s32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<s8, s8, u8, s32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<s8, s8, s8, s32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<s8, s8, s32, s32>)
        CPU_INSTANCE(ref_inner_product_fwd_t<s8, s8, f32, s32>)
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const pd_create_f *get_inner_product_impl_list(
        const inner_product_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
