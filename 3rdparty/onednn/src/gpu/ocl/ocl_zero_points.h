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

#ifndef GPU_OCL_OCL_ZERO_POINTS_H
#define GPU_OCL_OCL_ZERO_POINTS_H

#if WITH_SRC_ZPOINTS
#if WITH_SRC_ZPOINTS_PER_IC

// read 4 ints x 8 sg
int4 read_src_zero_points_32c(const __global int *ptr, const int ic) {
    int4 z;
    ptr += ic;
#if IC % 32 != 0
    if (ic + 32 > IC) {
        const int max_local_id = IC % 8;
        const int local_id = get_sub_group_local_id();
        if (ic + 8 > IC) {
            z.s0 = local_id < max_local_id ? ptr[0 + local_id] : 0;
            z.s1 = 0;
            z.s2 = 0;
            z.s3 = 0;
        } else if (ic + 16 > IC) {
            z.s0 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 0)));
            z.s1 = local_id < max_local_id ? ptr[8 + local_id] : 0;
            z.s2 = 0;
            z.s3 = 0;
        } else if (ic + 24 > IC) {
            z.s0 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 0)));
            z.s1 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 8)));
            z.s2 = local_id < max_local_id ? ptr[16 + local_id] : 0;
            z.s3 = 0;
        } else {
            z.s0 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 0)));
            z.s1 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 8)));
            z.s2 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 16)));
            z.s3 = local_id < max_local_id ? ptr[24 + local_id] : 0;
        }
    } else
#endif // IC % 32 != 0
    {
        z = as_int4(intel_sub_group_block_read4((const __global uint *)ptr));
    }
    return z;
}

// read 2 ints x 16 sg
int2 read_src_zero_points_32g(const __global int *ptr, const int g) {
    int2 z;
    ptr += g;
#if G % 32 != 0
    if (g + 32 > G) {
        const int max_local_id = G % 16;
        const int local_id = get_sub_group_local_id();
        if (g + 16 > G) {
            z.s0 = local_id < max_local_id ? ptr[0 + local_id] : 0;
            z.s1 = 0;
        } else {
            z.s0 = as_int(
                    intel_sub_group_block_read((const __global uint *)ptr));
            z.s1 = local_id < max_local_id ? ptr[16 + local_id] : 0;
        }
    } else
#endif // G % 32 != 0
    {
        z = as_int2(intel_sub_group_block_read2((const __global uint *)ptr));
    }
    return z;
}

int calc_src_compensation_x32(int4 z, int8 wei) {
    int sum = 0;
    __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 8; ++i) {
        char4 w = as_char4(wei[i]);
        sum += sub_group_broadcast(z[i >> 1], (i & 1) * 4 + 0) * w[0];
        sum += sub_group_broadcast(z[i >> 1], (i & 1) * 4 + 1) * w[1];
        sum += sub_group_broadcast(z[i >> 1], (i & 1) * 4 + 2) * w[2];
        sum += sub_group_broadcast(z[i >> 1], (i & 1) * 4 + 3) * w[3];
    }
    return sum;
}

#else // !WITH_SRC_ZPOINTS_PER_IC

int read_src_zero_point(const __global int *ptr) {
#if SRC_ZPOINT_COMMON != 0
    const int z = SRC_ZPOINT_COMMON;
#else
    const int z = ptr[0];
#endif
    return z;
}

#endif // WITH_SRC_ZPOINTS_PER_IC

int calc_src_compensation_x4(int4 z, int wei) {
    int sum = 0;
    {
        char4 w = as_char4(wei);
        sum += z.s0 * w.s0;
        sum += z.s1 * w.s1;
        sum += z.s2 * w.s2;
        sum += z.s3 * w.s3;
    }
    return sum;
}

#endif // WITH_SRC_ZPOINTS

#if WITH_DST_ZPOINTS

// read 4 ints x 8 sg
int4 read_dst_zero_points_32c(const __global int *ptr, const int oc) {
#if WITH_DST_ZPOINTS_PER_OC
    int4 z;
    ptr += oc;
#if OC % 32 != 0
    if (oc + 32 > OC) {
        const int max_local_id = OC % 8;
        const int local_id = get_sub_group_local_id();
        if (oc + 8 > OC) {
            z.s0 = local_id < max_local_id ? ptr[0 + local_id] : 0;
            z.s1 = 0;
            z.s2 = 0;
            z.s3 = 0;
        } else if (oc + 16 > OC) {
            z.s0 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 0)));
            z.s1 = local_id < max_local_id ? ptr[8 + local_id] : 0;
            z.s2 = 0;
            z.s3 = 0;
        } else if (oc + 24 > OC) {
            z.s0 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 0)));
            z.s1 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 8)));
            z.s2 = local_id < max_local_id ? ptr[16 + local_id] : 0;
            z.s3 = 0;
        } else {
            z.s0 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 0)));
            z.s1 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 8)));
            z.s2 = as_int(intel_sub_group_block_read(
                    (const __global uint *)(ptr + 16)));
            z.s3 = local_id < max_local_id ? ptr[24 + local_id] : 0;
        }
    } else
#endif // OC % 32 != 0
    {
        z = as_int4(intel_sub_group_block_read4((const __global uint *)ptr));
    }
#else
    const int4 z = DST_ZPOINT_COMMON != 0 ? DST_ZPOINT_COMMON : ptr[0];
#endif // WITH_DST_ZPOINTS_PER_OC
    return z;
}

int2 read_dst_zero_points_32g(const __global int *ptr, const int g) {
#if WITH_DST_ZPOINTS_PER_OC
    int2 z;
    ptr += g;
#if G % 32 != 0
    if (g + 32 > G) {
        const int max_local_id = G % 16;
        const int local_id = get_sub_group_local_id();
        if (g + 16 > G) {
            z.s0 = local_id < max_local_id ? ptr[0 + local_id] : 0;
            z.s1 = 0;
        } else {
            z.s0 = as_int(
                    intel_sub_group_block_read((const __global uint *)ptr));
            z.s1 = local_id < max_local_id ? ptr[16 + local_id] : 0;
        }
    } else
#endif // G % 32 != 0
    {
        z = as_int2(intel_sub_group_block_read2((const __global uint *)ptr));
    }
#elif DST_ZPOINT_COMMON != 0
    const int2 z = (int2)(DST_ZPOINT_COMMON);
#else
    const int2 z = (int2)(ptr[0]);
#endif // WITH_DST_ZPOINTS_PER_OC
    return z;
}

#endif // WITH_DST_ZPOINTS

#if WITH_SRC_ZPOINTS || WITH_DST_ZPOINTS

float4 zero_pad_dst_32c(float4 dst, const int oc) {
#if OC % 32 != 0
    if (oc + 32 > OC) {
        const int max_local_id = OC % 8;
        const int local_id = get_sub_group_local_id();
        if (oc + 8 > OC) {
            dst.s0 = local_id < max_local_id ? dst.s0 : 0;
            dst.s1 = 0;
            dst.s2 = 0;
            dst.s3 = 0;
        } else if (oc + 16 > OC) {
            dst.s1 = local_id < max_local_id ? dst.s1 : 0;
            dst.s2 = 0;
            dst.s3 = 0;
        } else if (oc + 24 > OC) {
            dst.s2 = local_id < max_local_id ? dst.s2 : 0;
            dst.s3 = 0;
        } else {
            dst.s3 = local_id < max_local_id ? dst.s3 : 0;
        }
    }
#endif // OC % 32 != 0
    return dst;
}

#endif // WITH_SRC_ZPOINTS || WITH_DST_ZPOINTS

#endif // GPU_OCL_OCL_ZERO_POINTS_H
