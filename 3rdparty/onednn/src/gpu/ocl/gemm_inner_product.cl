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

#include "gpu/ocl/ocl_types.h"

__kernel void gemm_inner_product_forward_bias(
        __global DATA_T *bias, __global DATA_T *dst) {

    const int mb = get_global_id(0) / OC;
    const int oc = get_global_id(0) % OC;

    dst[mb * OC + oc] += bias[oc];
}

__kernel void gemm_inner_product_backward_weights_bias(
        __global DATA_T *diff_dst, __global DATA_T *diff_bias) {
    const uint oc = get_global_id(0);
    DATA_T sum = DATA_ZERO;

    for (uint n = 0; n < MB; ++n)
        sum += diff_dst[n * OC + oc];

    diff_bias[oc] = sum;
}
