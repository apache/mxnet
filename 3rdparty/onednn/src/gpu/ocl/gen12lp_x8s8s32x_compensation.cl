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
#include "gpu/ocl/ocl_zero_points.h"

#if WEI_4O8I8O4I

#define OCB ((OC + 31) / 32)
#define ICB ((IC + 31) / 32)
#define KDHW (KD * KH * KW)
#define WEI_BLOCK (32 * 32)

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1))) __kernel void
gen12lp_x8s8s32x_compensation(const __global int *src_zpoints,
        const __global char *wei, __global int *dst) {
    const int oc_block_idx = get_global_id(1);
    const int g = get_global_id(2);

    wei += g * OCB * ICB * KDHW * WEI_BLOCK;
    wei += oc_block_idx * ICB * KDHW * WEI_BLOCK;

    dst += g * OCB * 32;
    dst += oc_block_idx * 32;

#if WITH_SRC_ZPOINTS_PER_IC
    src_zpoints += g * IC;
#else
    const int z = read_src_zero_point(src_zpoints);
#endif // WITH_SRC_ZPOINTS_PER_IC

    int4 acc = 0;
    for (uint icb = 0; icb < ICB; ++icb) {
#if WITH_SRC_ZPOINTS_PER_IC
        const int4 z = read_src_zero_points_32c(src_zpoints, icb * IC_BLOCK);
#endif // WITH_SRC_ZPOINTS_PER_IC
        for (uint k = 0; k < KDHW; ++k) {
            const int8 w0 = as_int8(intel_sub_group_block_read8(
                    (__global uint *)(wei + 0 * IC_BLOCK)));
            const int8 w1 = as_int8(intel_sub_group_block_read8(
                    (__global uint *)(wei + 8 * IC_BLOCK)));
            const int8 w2 = as_int8(intel_sub_group_block_read8(
                    (__global uint *)(wei + 16 * IC_BLOCK)));
            const int8 w3 = as_int8(intel_sub_group_block_read8(
                    (__global uint *)(wei + 24 * IC_BLOCK)));

#if WITH_SRC_ZPOINTS_PER_IC
            acc.s0 += calc_src_compensation_x32(z, w0);
            acc.s1 += calc_src_compensation_x32(z, w1);
            acc.s2 += calc_src_compensation_x32(z, w2);
            acc.s3 += calc_src_compensation_x32(z, w3);
#else
            unroll_for(uint i = 0; i < 8; ++i) {
                acc.s0 = idot4(0x01010101, w0[i], acc.s0);
                acc.s1 = idot4(0x01010101, w1[i], acc.s1);
                acc.s2 = idot4(0x01010101, w2[i], acc.s2);
                acc.s3 = idot4(0x01010101, w3[i], acc.s3);
            }
#endif // WITH_SRC_ZPOINTS_PER_IC
            wei += WEI_BLOCK;
        }
    }

#if !WITH_SRC_ZPOINTS_PER_IC
    acc = z * acc;
#endif // !WITH_SRC_ZPOINTS_PER_IC

    intel_sub_group_block_write4((__global uint *)(dst), as_uint4(acc));
}

#endif // WEI_4O8I8O4I

#if WEI_32G

#define KDHW (KD * KH * KW)
#define WEI_BLOCK 32

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1))) __kernel void
gen12lp_x8s8s32x_compensation(const __global int *src_zpoints,
        const __global char *wei, __global int *dst) {
    const int g_block_idx = get_global_id(1);

    wei += g_block_idx * KDHW * WEI_BLOCK;
    dst += g_block_idx * WEI_BLOCK;

#if WITH_SRC_ZPOINTS_PER_IC
    const int2 z
            = read_src_zero_points_32g(src_zpoints, g_block_idx * WEI_BLOCK);
#else
    const int z = read_src_zero_point(src_zpoints);
#endif // WITH_SRC_ZPOINTS_PER_IC

    int2 acc = 0;
#if WITH_SRC_ZPOINTS_PER_IC

#endif // WITH_SRC_ZPOINTS_PER_IC

    for (uint k = 0; k < KDHW; ++k) {
        const int2 w0 = convert_int2(as_char2(
                intel_sub_group_block_read_uc2((const __global uchar *)(wei))));
        acc += w0;
        wei += WEI_BLOCK;
    }

    acc = z * acc;

    intel_sub_group_block_write2((__global uint *)(dst), as_uint2(acc));
}

#endif // WEI_32G
