/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
* Copyright 2020 Arm Ltd. and affiliates
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

#include <map>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/gemm_convolution.hpp"
#include "cpu/gemm_x8s8s32x_convolution.hpp"
#include "cpu/ref_convolution.hpp"
#include "cpu/ref_fused_convolution.hpp"

#if DNNL_AARCH64 && DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_gemm_convolution.hpp"
using namespace dnnl::impl::cpu::aarch64;
#endif

#if DNNL_X64
#include "cpu/x64/gemm_bf16_convolution.hpp"
#include "cpu/x64/jit_avx2_1x1_convolution.hpp"
#include "cpu/x64/jit_avx2_convolution.hpp"
#include "cpu/x64/jit_avx512_common_1x1_convolution.hpp"
#include "cpu/x64/jit_avx512_common_convolution.hpp"
#include "cpu/x64/jit_avx512_common_convolution_winograd.hpp"
#include "cpu/x64/jit_avx512_core_amx_1x1_convolution.hpp"
#include "cpu/x64/jit_avx512_core_amx_convolution.hpp"
#include "cpu/x64/jit_avx512_core_bf16_1x1_convolution.hpp"
#include "cpu/x64/jit_avx512_core_bf16_convolution.hpp"
#include "cpu/x64/jit_avx512_core_f32_wino_conv_2x3.hpp"
#include "cpu/x64/jit_avx512_core_f32_wino_conv_4x3.hpp"
#include "cpu/x64/jit_avx512_core_u8s8s32x_wino_convolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_1x1_convolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_convolution.hpp"
#include "cpu/x64/jit_sse41_1x1_convolution.hpp"
#include "cpu/x64/jit_sse41_convolution.hpp"
#include "cpu/x64/jit_uni_dw_convolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_1x1_convolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_convolution.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::prop_kind;

struct conv_impl_key_t {
    prop_kind_t kind;
    data_type_t src_dt, wei_dt, dst_dt;

    bool operator<(const conv_impl_key_t &rhs) const {
        return value() < rhs.value();
    }

private:
    enum { MAX_DT_NUM = 10 };
    size_t value() const {
        return (((size_t)kind * MAX_DT_NUM + (size_t)src_dt) * MAX_DT_NUM
                       + (size_t)wei_dt)
                * MAX_DT_NUM
                + (size_t)dst_dt;
    }
};

// clang-format off
const std::map<conv_impl_key_t, std::vector<pd_create_f>> impl_list_map {
    // FWD fp
    {{forward, f32, f32, f32}, {
        CPU_INSTANCE_X64(jit_avx512_common_dw_convolution_fwd_t)
        CPU_INSTANCE_X64(jit_avx512_common_1x1_convolution_fwd_f32_t)
        CPU_INSTANCE_X64(jit_avx512_core_f32_wino_conv_2x3_fwd_t)
        CPU_INSTANCE_X64(jit_avx512_core_f32_wino_conv_4x3_fwd_t)
        CPU_INSTANCE_X64(jit_avx512_common_convolution_winograd_fwd_t)
        CPU_INSTANCE_X64(jit_avx512_common_convolution_fwd_t<f32>)
        CPU_INSTANCE_X64(jit_avx2_dw_convolution_fwd_t)
        CPU_INSTANCE_X64(jit_avx2_1x1_convolution_fwd_t)
        CPU_INSTANCE_X64(jit_sse41_dw_convolution_fwd_t)
        CPU_INSTANCE_X64(jit_sse41_1x1_convolution_fwd_t)
        CPU_INSTANCE_X64(jit_avx2_convolution_fwd_t)
        CPU_INSTANCE_X64(jit_sse41_convolution_fwd_t)
        CPU_INSTANCE_AARCH64_ACL(acl_gemm_convolution_fwd_t)
        CPU_INSTANCE(gemm_convolution_fwd_t)
        CPU_INSTANCE(ref_convolution_fwd_t<f32>)
        CPU_INSTANCE(ref_fused_convolution_fwd_t)
        nullptr,
    }},
    {{forward, bf16, bf16, f32}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<bf16, bf16, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<bf16, bf16, f32>)
        CPU_INSTANCE_X64(jit_uni_dw_convolution_fwd_t<avx512_core, bf16, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_1x1_convolution_fwd_t<f32>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_convolution_fwd_t)
        CPU_INSTANCE_X64(gemm_bf16_convolution_fwd_t<f32>)
        CPU_INSTANCE(ref_convolution_fwd_t<bf16, bf16, f32, f32>)
        nullptr,
    }},
    {{forward, bf16, bf16, bf16}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<bf16, bf16, bf16>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<bf16, bf16, bf16>)
        CPU_INSTANCE_X64(jit_uni_dw_convolution_fwd_t<avx512_core, bf16, bf16>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_1x1_convolution_fwd_t<bf16>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_convolution_fwd_t)
        CPU_INSTANCE_X64(gemm_bf16_convolution_fwd_t<bf16>)
        CPU_INSTANCE(ref_convolution_fwd_t<bf16, bf16, bf16, f32>)
        CPU_INSTANCE(ref_fused_convolution_fwd_t)
        nullptr,
    }},
    // BWD_D fp
    {{backward_data, f32, f32, f32}, {
        CPU_INSTANCE_X64(jit_avx512_common_dw_convolution_bwd_data_t)
        CPU_INSTANCE_X64(jit_avx512_common_1x1_convolution_bwd_data_f32_t)
        CPU_INSTANCE_X64(jit_avx512_core_f32_wino_conv_4x3_bwd_data_t)
        CPU_INSTANCE_X64(jit_avx512_common_convolution_winograd_bwd_data_t)
        CPU_INSTANCE_X64(jit_avx512_common_convolution_bwd_data_t<f32>)
        CPU_INSTANCE_X64(jit_avx2_dw_convolution_bwd_data_t)
        CPU_INSTANCE_X64(jit_avx2_1x1_convolution_bwd_data_t)
        CPU_INSTANCE_X64(jit_sse41_dw_convolution_bwd_data_t)
        CPU_INSTANCE_X64(jit_avx2_convolution_bwd_data_t)
        CPU_INSTANCE(gemm_convolution_bwd_data_t)
        CPU_INSTANCE(ref_convolution_bwd_data_t<f32, f32, f32, f32>)
        nullptr,
    }},
    {{backward_data, f32, bf16, bf16}, {
        CPU_INSTANCE_X64(jit_uni_dw_convolution_bwd_data_t<avx512_core, bf16, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_1x1_convolution_bwd_data_t<f32>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_convolution_bwd_data_t)
        CPU_INSTANCE_X64(gemm_bf16_convolution_bwd_data_t<f32>)
        CPU_INSTANCE(ref_convolution_bwd_data_t<f32, bf16, bf16, f32>)
        nullptr,
    }},
    {{backward_data, bf16, bf16, bf16}, {
        CPU_INSTANCE_X64(jit_uni_dw_convolution_bwd_data_t<avx512_core, bf16, bf16>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_1x1_convolution_bwd_data_t<bf16>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_convolution_bwd_data_t)
        CPU_INSTANCE_X64(gemm_bf16_convolution_bwd_data_t<bf16>)
        CPU_INSTANCE(ref_convolution_bwd_data_t<bf16, bf16, bf16, f32>)
        nullptr,
    }},
    // BWD_W fp
    {{backward_weights, f32, f32, f32}, {
        CPU_INSTANCE_X64(jit_avx512_common_dw_convolution_bwd_weights_t)
        CPU_INSTANCE_X64(jit_avx512_common_1x1_convolution_bwd_weights_t)
        CPU_INSTANCE_X64(jit_avx512_core_f32_wino_conv_4x3_bwd_weights_t)
        CPU_INSTANCE_X64(jit_avx512_common_convolution_winograd_bwd_weights_t)
        CPU_INSTANCE_X64(jit_avx512_common_convolution_bwd_weights_t<f32>)
        CPU_INSTANCE_X64(jit_avx2_dw_convolution_bwd_weights_t)
        CPU_INSTANCE_X64(jit_avx2_1x1_convolution_bwd_weights_t)
        CPU_INSTANCE_X64(jit_sse41_dw_convolution_bwd_weights_t)
        CPU_INSTANCE_X64(jit_avx2_convolution_bwd_weights_t)
        CPU_INSTANCE(gemm_convolution_bwd_weights_t)
        CPU_INSTANCE(ref_convolution_bwd_weights_t<f32, f32, f32, f32>)
        nullptr,
    }},
    {{backward_weights, bf16, f32, bf16}, {
        CPU_INSTANCE_X64(jit_uni_dw_convolution_bwd_weights_t<avx512_core, bf16, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<f32>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_convolution_bwd_weights_t)
        CPU_INSTANCE_X64(gemm_bf16_convolution_bwd_weights_t<f32>)
        CPU_INSTANCE(ref_convolution_bwd_weights_t<bf16, f32, bf16, f32>)
        nullptr,
    }},
    {{backward_weights, bf16, bf16, bf16}, {
        CPU_INSTANCE_X64(jit_uni_dw_convolution_bwd_weights_t<avx512_core, bf16, bf16>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<bf16>)
        CPU_INSTANCE_X64(jit_avx512_core_bf16_convolution_bwd_weights_t)
        CPU_INSTANCE_X64(gemm_bf16_convolution_bwd_weights_t<bf16>)
        CPU_INSTANCE(ref_convolution_bwd_weights_t<bf16, bf16, bf16, f32>)
        nullptr,
    }},
    // FWD int8 (src:s8)
    {{forward, s8, s8, f32}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<s8, s8, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<s8, s8, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, s8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<avx2, s8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, s8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41, s8, f32>)
        CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, f32>)
        CPU_INSTANCE(ref_convolution_fwd_t<s8, s8, f32, s32>)
        CPU_INSTANCE(ref_fused_convolution_fwd_t)
        nullptr,
    }},
    {{forward, s8, s8, s32}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<s8, s8, s32>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<s8, s8, s32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, s32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, s8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<avx2, s8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, s8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41, s8, s32>)
        CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, s32>)
        CPU_INSTANCE(ref_convolution_fwd_t<s8, s8, s32, s32>)
        CPU_INSTANCE(ref_fused_convolution_fwd_t)
        nullptr,
    }},
    {{forward, s8, s8, s8}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<s8, s8, s8>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<s8, s8, s8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, s8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, s8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<avx2, s8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, s8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41, s8, s8>)
        CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, s8>)
        CPU_INSTANCE(ref_convolution_fwd_t<s8, s8, s8, s32>)
        CPU_INSTANCE(ref_fused_convolution_fwd_t)
        nullptr,
    }},
    {{forward, s8, s8, u8}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<s8, s8, u8>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<s8, s8, u8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, u8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, s8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<avx2, s8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, s8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41, s8, u8>)
        CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, u8>)
        CPU_INSTANCE(ref_convolution_fwd_t<s8, s8, u8, s32>)
        CPU_INSTANCE(ref_fused_convolution_fwd_t)
        nullptr,
    }},
    // FWD int8 (src:u8)
    {{forward, u8, s8, f32}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<u8, s8, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<u8, s8, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<f32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, u8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<avx2, u8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, u8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41, u8, f32>)
        CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, f32>)
        CPU_INSTANCE(ref_convolution_fwd_t<u8, s8, f32, s32>)
        nullptr,
    }},
    {{forward, u8, s8, s32}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<u8, s8, s32>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<u8, s8, s32>)
        CPU_INSTANCE_X64(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<s32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, s32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, u8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<avx2, u8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, u8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41, u8, s32>)
        CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, s32>)
        CPU_INSTANCE(ref_convolution_fwd_t<u8, s8, s32, s32>)
        nullptr,
    }},
    {{forward, u8, s8, s8}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<u8, s8, s8>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<u8, s8, s8>)
        CPU_INSTANCE_X64(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<s8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, s8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, u8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<avx2, u8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, u8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41, u8, s8>)
        CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, s8>)
        CPU_INSTANCE(ref_convolution_fwd_t<u8, s8, s8, s32>)
        CPU_INSTANCE(ref_fused_convolution_fwd_t)
        nullptr,
    }},
    {{forward, u8, s8, u8}, {
        CPU_INSTANCE_X64(jit_avx512_core_amx_1x1_convolution_fwd_t<u8, s8, u8>)
        CPU_INSTANCE_X64(jit_avx512_core_amx_convolution_fwd_t<u8, s8, u8>)
        CPU_INSTANCE_X64(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<u8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, u8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, u8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<avx2, u8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, u8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41, u8, u8>)
        CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, u8>)
        CPU_INSTANCE(ref_convolution_fwd_t<u8, s8, u8, s32>)
        CPU_INSTANCE(ref_fused_convolution_fwd_t)
        nullptr,
    }},
    // BWD int8 (diff_dst:u8)
    {{backward_data, f32, s8, u8}, {
        CPU_INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<f32>)
        CPU_INSTANCE(ref_convolution_bwd_data_t<f32, s8, u8, s32>)
        nullptr,
    }},
    {{backward_data, s32, s8, u8}, {
        CPU_INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<s32>)
        CPU_INSTANCE(ref_convolution_bwd_data_t<s32, s8, u8, s32>)
        nullptr,
    }},
    {{backward_data, s8, s8, u8}, {
        CPU_INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<s8>)
        CPU_INSTANCE(ref_convolution_bwd_data_t<s8, s8, u8, s32>)
        nullptr,
    }},
    {{backward_data, u8, s8, u8}, {
        CPU_INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<u8>)
        CPU_INSTANCE(ref_convolution_bwd_data_t<u8, s8, u8, s32>)
        nullptr,
    }},
    // BWD int8 (diff_dst:s8)
    {{backward_data, f32, s8, s8}, {
        CPU_INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<f32>)
        CPU_INSTANCE(ref_convolution_bwd_data_t<f32, s8, s8, s32>)
        nullptr,
    }},
    {{backward_data, s32, s8, s8}, {
        CPU_INSTANCE(ref_convolution_bwd_data_t<s32, s8, s8, s32>)
        nullptr,
    }},
    {{backward_data, s8, s8, s8}, {
        CPU_INSTANCE(ref_convolution_bwd_data_t<s8, s8, s8, s32>)
        nullptr,
    }},
    {{backward_data, u8, s8, s8}, {
        CPU_INSTANCE(ref_convolution_bwd_data_t<u8, s8, s8, s32>)
        nullptr,
    }},
};
// clang-format on
} // namespace

const pd_create_f *get_convolution_impl_list(const convolution_desc_t *desc) {
    static const pd_create_f empty_list[] = {nullptr};

    prop_kind_t prop_kind = utils::one_of(desc->prop_kind, forward_training,
                                    forward_inference)
            ? forward
            : desc->prop_kind;
    conv_impl_key_t key {
            prop_kind,
            conv_prop_invariant_src_d(desc)->data_type,
            conv_prop_invariant_wei_d(desc)->data_type,
            conv_prop_invariant_dst_d(desc)->data_type,
    };

    const auto impl_list_it = impl_list_map.find(key);
    return (impl_list_it != impl_list_map.cend()) ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
