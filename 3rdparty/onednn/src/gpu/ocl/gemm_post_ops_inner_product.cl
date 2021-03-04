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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

__kernel void gemm_post_ops_inner_product(__global SRC_DATA_T *src,
        __global BIAS_DATA_T *bias, __global DST_DATA_T *dst POST_OP_ARGS,
        __global SPAD_DATA_T *scratchpad, global float *scales) {
    const size_t mb = get_global_id(0) / OC;
    const size_t oc = get_global_id(0) % OC;

    const size_t data_idx = mb * OC + oc;
#if USE_TEMP_DST == 1
    ACC_DATA_T acc = TO_ACC(scratchpad[data_idx]);
#else
    ACC_DATA_T acc = TO_ACC(src[data_idx]);
#endif

#if WITH_BIAS == 1
    acc += TO_ACC(bias[oc]);
#endif

#if WITH_SCALES
#if SCALES_COMMON
    const float scale = scales[0];
#elif SCALES_PER_OC
    const float scale = scales[oc];
#else
#error "Unsupported scale type"
#endif
    acc *= scale;
#endif

    // Apply postops
    float sum_src;
#if WITH_SUM
    sum_src = TO_ACC(dst[data_idx]);
#endif

    float accumulator = acc;
    APPLY_POST_OPS(accumulator, float, sum_src, float, mb, 1, oc, 1, 0, 1, 0, 1,
            0, 1, 0, 1);

    dst[data_idx] = TO_DST(accumulator);
}
