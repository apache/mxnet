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

#if WITH_ELTWISE == 1
#define POST_OP(val) \
    do { \
        if (apply_eltwise) \
            val = fwd_eltwise( \
                    val, eltwise_alpha, eltwise_beta, eltwise_scale); \
    } while (0)
#else
#define POST_OP(val)
#endif

kernel void gen12lp_gemm_scale_x8x8s32(global int *cc, global int *c, char trc,
        int offset_c, int m, int n, int ldc, float alpha, float beta,
        global int *co, int offset_co, int alpha_is_zero, int apply_eltwise,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale) {

    int idx = get_group_id(0);
    int idy = get_group_id(1);
    int lid = get_local_id(0);
    int j;
    int offset_cc = 0;
    int offset_x = 0;
    int ldcc = m;

    m -= 32 * idx;
    if (m > 32) m = 32;
    n -= 16 * idy;
    if (n > 16) n = 16;
    m -= 32 * lid / 16;
    if ((m <= 0) || (n <= 0)) return;
    offset_cc = 32 * idx + 32 * lid / 16 + 16 * idy * ldcc;
    offset_c += 32 * idx + 32 * lid / 16 + 16 * idy * ldc;
    if (trc == 'C') offset_co += 32 * idx + 32 * lid / 16;
    if (trc == 'R') offset_co += 16 * idy;
    for (j = 0; j < n; j++) {
        if (m > 0) {
            float val = (alpha_is_zero ? 0 : alpha) * cc[offset_cc + 0]
                    + beta * c[offset_c + 0];
            POST_OP(val);
            c[offset_c] = convert_int_sat_rte(
                    val + (co ? co[offset_co + offset_x] : 0));
            if (trc == 'C') { offset_x++; }
        }
        if (m > 1) {
            float val = (alpha_is_zero ? 0 : alpha) * cc[offset_cc + 1]
                    + beta * c[offset_c + 1];
            POST_OP(val);
            c[offset_c + 1] = convert_int_sat_rte(
                    val + (co ? co[offset_co + offset_x] : 0));
        }

        offset_cc += ldcc;
        offset_c += ldc;
        if (trc == 'C') offset_x = 0;
        if (trc == 'R') offset_x++;
    }
}
