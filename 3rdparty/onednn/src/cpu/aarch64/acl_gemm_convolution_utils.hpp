/*******************************************************************************
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

#ifndef CPU_AARCH64_ACL_GEMM_CONVOLUTION_UTILS_HPP
#define CPU_AARCH64_ACL_GEMM_CONVOLUTION_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_engine.hpp"

#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace dnnl {
namespace impl {
namespace cpu {

struct acl_conv_gemm_conf_t {
    bool with_bias;
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo wei_info;
    arm_compute::TensorInfo bia_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::PadStrideInfo padstride_info;
    arm_compute::Size2D dilation_info;
    arm_compute::WeightsInfo weights_info;
    arm_compute::ActivationLayerInfo act_info;
};

namespace acl_gemm_convolution_utils {

status_t init_conf(acl_conv_gemm_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

bool acl_act_ok(alg_kind_t eltwise_activation);
arm_compute::ActivationLayerInfo get_acl_act(const primitive_attr_t &attr);

} // namespace acl_gemm_convolution_utils

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
