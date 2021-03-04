/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#if DATA_TYPE_SIZE == 4
#define DATA_T uint
#define DATA2_T uint2
#define DATA4_T uint4
#define DATA8_T uint8
#define BLOCK_READ intel_sub_group_block_read
#define BLOCK_WRITE intel_sub_group_block_write
#define BLOCK_READ2 intel_sub_group_block_read2
#define BLOCK_WRITE2 intel_sub_group_block_write2
#define BLOCK_READ4 intel_sub_group_block_read4
#define BLOCK_WRITE4 intel_sub_group_block_write4
#define BLOCK_READ8 intel_sub_group_block_read8
#define BLOCK_WRITE8 intel_sub_group_block_write8
#elif DATA_TYPE_SIZE == 2
#define DATA_T ushort
#define DATA2_T ushort2
#define DATA4_T ushort4
#define DATA8_T ushort8
#define BLOCK_READ intel_sub_group_block_read_us
#define BLOCK_WRITE intel_sub_group_block_write_us
#define BLOCK_READ2 intel_sub_group_block_read_us2
#define BLOCK_WRITE2 intel_sub_group_block_write_us2
#define BLOCK_READ4 intel_sub_group_block_read_us4
#define BLOCK_WRITE4 intel_sub_group_block_write_us4
#define BLOCK_READ8 intel_sub_group_block_read_us8
#define BLOCK_WRITE8 intel_sub_group_block_write_us8
#elif DATA_TYPE_SIZE == 1
#define DATA_T uchar
#define DATA2_T uchar2
#define DATA4_T uchar4
#define DATA8_T uchar8
#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ2 intel_sub_group_block_read_uc2
#define BLOCK_WRITE2 intel_sub_group_block_write_uc2
#define BLOCK_READ4 intel_sub_group_block_read_uc4
#define BLOCK_WRITE4 intel_sub_group_block_write_uc4
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#endif
#define CHECK_AND_GET(N, M) \
    if (get_global_id(2) >= OFFSET##N \
            && (M == N_INPUTS || get_global_id(2) < OFFSET##M)) { \
        src = src##N + get_global_id(1) * SRC##N##_EXT_OFFSET + x \
                - OFFSET##N * INNER_OFFSET; \
    }

#if BLOCK != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
simple_concat(__global DATA_T *dst, __global const DATA_T *src0
#if N_INPUTS > 1
        ,
        __global const DATA_T *src1
#endif
#if N_INPUTS > 2
        ,
        __global const DATA_T *src2
#endif
#if N_INPUTS > 3
        ,
        __global const DATA_T *src3
#endif
#if N_INPUTS > 4
        ,
        __global const DATA_T *src4
#endif
#if N_INPUTS > 5
        ,
        __global const DATA_T *src5
#endif
#if N_INPUTS > 6
        ,
        __global const DATA_T *src6
#endif
#if N_INPUTS > 7
        ,
        __global const DATA_T *src7
#endif
#if N_INPUTS > 8
        ,
        __global const DATA_T *src8
#endif
#if N_INPUTS > 9
        ,
        __global const DATA_T *src9
#endif
#if N_INPUTS > 10
        ,
        __global const DATA_T *src10
#endif
#if N_INPUTS > 11
        ,
        __global const DATA_T *src11
#endif
#if N_INPUTS > 12
        ,
        __global const DATA_T *src12
#endif
#if N_INPUTS > 13
        ,
        __global const DATA_T *src13
#endif
#if N_INPUTS > 14
        ,
        __global const DATA_T *src14
#endif
#if N_INPUTS > 15
        ,
        __global const DATA_T *src15
#endif
) {
    DATA8_T A0, A1, A2, A3;
    DATA_T B;
    DATA2_T C;
    DATA4_T D;
    const size_t x = get_global_id(0) * (BLOCK / SIMD)
            + get_global_id(2) * INNER_OFFSET;
    __global const DATA_T *src;

    CHECK_AND_GET(0, 1)
#if N_INPUTS > 1
    CHECK_AND_GET(1, 2)
#endif
#if N_INPUTS > 2
    CHECK_AND_GET(2, 3)
#endif
#if N_INPUTS > 3
    CHECK_AND_GET(3, 4)
#endif
#if N_INPUTS > 4
    CHECK_AND_GET(4, 5)
#endif
#if N_INPUTS > 5
    CHECK_AND_GET(5, 6)
#endif
#if N_INPUTS > 6
    CHECK_AND_GET(6, 7)
#endif
#if N_INPUTS > 7
    CHECK_AND_GET(7, 8)
#endif
#if N_INPUTS > 8
    CHECK_AND_GET(8, 9)
#endif
#if N_INPUTS > 9
    CHECK_AND_GET(9, 10)
#endif
#if N_INPUTS > 10
    CHECK_AND_GET(10, 11)
#endif
#if N_INPUTS > 11
    CHECK_AND_GET(11, 12)
#endif
#if N_INPUTS > 12
    CHECK_AND_GET(12, 13)
#endif
#if N_INPUTS > 13
    CHECK_AND_GET(13, 14)
#endif
#if N_INPUTS > 14
    CHECK_AND_GET(14, 15)
#endif
#if N_INPUTS > 15
    CHECK_AND_GET(15, 16)
#endif

#if BLOCK == 1
    B = src[0];
#elif BLOCK == SIMD
    B = BLOCK_READ(src);
#elif BLOCK == 2 * SIMD
    C = BLOCK_READ2(src);
#elif BLOCK == 3 * SIMD
    C = BLOCK_READ2(src);
    B = BLOCK_READ(&src[2 * SIMD]);
#elif BLOCK == 4 * SIMD
    D = BLOCK_READ4(src);
#elif BLOCK == 5 * SIMD
    D = BLOCK_READ4(src);
    B = BLOCK_READ(&src[4 * SIMD]);
#elif BLOCK == 6 * SIMD
    D = BLOCK_READ4(src);
    C = BLOCK_READ2(&src[4 * SIMD]);
#elif BLOCK == 7 * SIMD
    B = BLOCK_READ(src);
    C = BLOCK_READ2(&src[SIMD]);
    D = BLOCK_READ4(&src[3 * SIMD]);
#elif BLOCK >= 8 * SIMD
    A0 = BLOCK_READ8(src);
#elif BLOCK >= 16 * SIMD
    A1 = BLOCK_READ8(&src[8 * SIMD]);
#elif BLOCK >= 24 * SIMD
    A2 = BLOCK_READ8(&src[16 * SIMD]);
#elif BLOCK >= 32 * SIMD
    A3 = BLOCK_READ8(&src[24 * SIMD]);
#endif
    dst += get_global_id(1) * DST_EXT_OFFSET + x;
#if BLOCK == 1
    dst[0] = B;
#elif BLOCK == SIMD
    BLOCK_WRITE(dst, B);
#elif BLOCK == 2 * SIMD
    BLOCK_WRITE2(dst, C);
#elif BLOCK == 3 * SIMD
    BLOCK_WRITE2(dst, C);
    BLOCK_WRITE(&dst[2 * SIMD], B);
#elif BLOCK == 4 * SIMD
    BLOCK_WRITE4(dst, D);
#elif BLOCK == 5 * SIMD
    BLOCK_WRITE4(dst, D);
    BLOCK_WRITE(&dst[4 * SIMD], B);
#elif BLOCK == 6 * SIMD
    BLOCK_WRITE4(dst, D);
    BLOCK_WRITE2(&dst[4 * SIMD], C);
#elif BLOCK == 7 * SIMD
    BLOCK_WRITE(dst, B);
    BLOCK_WRITE2(&dst[SIMD], C);
    BLOCK_WRITE4(&dst[3 * SIMD], D);
#elif BLOCK >= 8 * SIMD
    BLOCK_WRITE8(dst, A0);
#elif BLOCK >= 16 * SIMD
    BLOCK_WRITE8(&dst[8 * SIMD], A1);
#elif BLOCK >= 24 * SIMD
    BLOCK_WRITE8(&dst[16 * SIMD], A2);
#elif BLOCK >= 32 * SIMD
    BLOCK_WRITE8(&dst[24 * SIMD], A3);
#endif
}
