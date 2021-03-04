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

#define GRX 8

#if DT_F32 == 1

#if UNROLL_M <= 1 * GRX
#define FLOATX float
#define SIZEX 1
#elif UNROLL_M <= 2 * GRX
#define FLOATX float2
#define SIZEX 2
#elif UNROLL_M <= 3 * GRX
#define FLOATX float3
#define SIZEX 3
#else
#define FLOATX float4
#define SIZEX 4
#endif

#if UNROLL_N <= 1 * GRX
#define FLOATY float
#define SIZEY 1
#elif UNROLL_N <= 2 * GRX
#define FLOATY float2
#define SIZEY 2
#elif UNROLL_N <= 3 * GRX
#define FLOATY float3
#define SIZEY 3
#else
#define FLOATY float4
#define SIZEY 4
#endif

#define SHUFFLE(X, Y) intel_sub_group_shuffle(X, Y)
#define SHUFFLE_DOWN(X, Y) intel_sub_group_shuffle_down(X, X, Y)
#define SHUFFLE_UP(X, Y) intel_sub_group_shuffle_up(X, X, Y)

#elif DT_F16 == 1

#if UNROLL_M <= 1 * GRX
#define FLOATX half
#define SIZEX 1
#elif UNROLL_M <= 2 * GRX
#define FLOATX half2
#define SIZEX 2
#elif UNROLL_M <= 3 * GRX
#define FLOATX half3
#define SIZEX 3
#elif UNROLL_M <= 4 * GRX
#define FLOATX half4
#define SIZEX 4
#else
#define FLOATX half8
#define SIZEX 8
#endif

#if UNROLL_N <= 1 * GRX
#define FLOATY half
#define SIZEY 1
#elif UNROLL_N <= 2 * GRX
#define FLOATY half2
#define SIZEY 2
#elif UNROLL_N <= 3 * GRX
#define FLOATY half3
#define SIZEY 3
#elif UNROLL_N <= 4 * GRX
#define FLOATY half4
#define SIZEY 4
#else
#define FLOATY half8
#define SIZEY 8
#endif

#if SIZEY == 2
#define SHUFFLE(X, Y) as_half2(intel_sub_group_shuffle(as_float(X), Y))
#elif SIZEY == 4
#define SHUFFLE(X, Y) as_half4(intel_sub_group_shuffle(as_float2(X), Y))
#else
#define SHUFFLE(X, Y) as_half8(intel_sub_group_shuffle(as_float4(X), Y))
#endif

#if SIZEX == 2
#define SHUFFLE_UP(X, Y) \
    as_half2(intel_sub_group_shuffle_up(as_float(X), as_float(X), Y))
#define SHUFFLE_DOWN(X, Y) \
    as_half2(intel_sub_group_shuffle_down(as_float(X), as_float(X), Y))
#elif SIZEX == 4
#define SHUFFLE_UP(X, Y) \
    as_half4(intel_sub_group_shuffle_up(as_float2(X), as_float2(X), Y))
#define SHUFFLE_DOWN(X, Y) \
    as_half4(intel_sub_group_shuffle_down(as_float2(X), as_float2(X), Y))
#else
#define SHUFFLE_UP(X, Y) \
    as_half8(intel_sub_group_shuffle_up(as_float4(X), as_float4(X), Y))
#define SHUFFLE_DOWN(X, Y) \
    as_half8(intel_sub_group_shuffle_down(as_float4(X), as_float4(X), Y))
#endif

#endif

#define AS_FLOATX(X, Y) *((__global FLOATX *)(X + Y))
#define AS_FLOATY(X, Y) *((__global FLOATY *)(X + Y))

#if UNROLL_M <= 1 * GRX
#define CALC_X(x, a, b, R0, R1, R2, R3) \
    bb = SHUFFLE(b, x); \
    R0##x = mad(a, bb, R0##x);
#elif UNROLL_M <= 2 * GRX
#define CALC_X(x, a, b, R0, R1, R2, R3) \
    bb = SHUFFLE(b, x); \
    R0##x = mad(a.s0, bb, R0##x); \
    R1##x = mad(a.s1, bb, R1##x);
#elif UNROLL_M <= 3 * GRX
#define CALC_X(x, a, b, R0, R1, R2, R3) \
    bb = SHUFFLE(b, x); \
    R0##x = mad(a.s0, bb, R0##x); \
    R1##x = mad(a.s1, bb, R1##x); \
    R2##x = mad(a.s2, bb, R2##x);
#else
#define CALC_X(x, a, b, R0, R1, R2, R3) \
    bb = SHUFFLE(b, x); \
    R0##x = mad(a.s0, bb, R0##x); \
    R1##x = mad(a.s1, bb, R1##x); \
    R2##x = mad(a.s2, bb, R2##x); \
    R3##x = mad(a.s3, bb, R3##x);
#endif

// This is fixed; 8 threads per core
#define CALC(a, b, R0, R1, R2, R3) \
    CALC_X(0, a, b, R0, R1, R2, R3); \
    CALC_X(1, a, b, R0, R1, R2, R3); \
    CALC_X(2, a, b, R0, R1, R2, R3); \
    CALC_X(3, a, b, R0, R1, R2, R3); \
    CALC_X(4, a, b, R0, R1, R2, R3); \
    CALC_X(5, a, b, R0, R1, R2, R3); \
    CALC_X(6, a, b, R0, R1, R2, R3); \
    CALC_X(7, a, b, R0, R1, R2, R3);

#define INIT_C(n) \
    FLOATY cc##n##0 = DATA_ZERO, cc##n##1 = DATA_ZERO; \
    FLOATY cc##n##2 = DATA_ZERO, cc##n##3 = DATA_ZERO; \
    FLOATY cc##n##4 = DATA_ZERO, cc##n##5 = DATA_ZERO; \
    FLOATY cc##n##6 = DATA_ZERO, cc##n##7 = DATA_ZERO;

#if WITH_ELTWISE == 1
#define POST_OP(val) \
    do { \
        if (last_k_block) \
            val = fwd_eltwise( \
                    val, eltwise_alpha, eltwise_beta, eltwise_scale); \
    } while (0)
#else
#define POST_OP(val)
#endif

#ifdef BETA_ZERO
#define UPDATE(c, acc) \
    do { \
        DATA_T val = acc; \
        POST_OP(val); \
        c = REF_TO_DST(val); \
    } while (0)
#else
#define UPDATE(c, acc) \
    do { \
        DATA_T val = DST_TO_REF(c) + acc; \
        POST_OP(val); \
        c = REF_TO_DST(val); \
    } while (0)
#endif

#if SIZEX == 1
#define UPDATE_YY(X, Y, R0, R1, R2, R3) \
    if (n > (Y)) { \
        if ((m > 0)) { UPDATE(c[offsetC + 0], R0); } \
        offsetC += ldc; \
    }
#elif SIZEX == 2
#define UPDATE_YY(X, Y, R0, R1, R2, R3) \
    if (n > (Y)) { \
        if ((m > 0)) { UPDATE(c[offsetC + 0], R0); } \
        if ((m > 1)) { UPDATE(c[offsetC + 1], R1); } \
        offsetC += ldc; \
    }
#elif SIZEX == 3
#define UPDATE_YY(X, Y, R0, R1, R2, R3) \
    if (n > (Y)) { \
        if ((m > 0)) { UPDATE(c[offsetC + 0], R0); } \
        if ((m > 1)) { UPDATE(c[offsetC + 1], R1); } \
        if ((m > 2)) { UPDATE(c[offsetC + 2], R2); } \
        offsetC += ldc; \
    }
#else
#define UPDATE_YY(X, Y, R0, R1, R2, R3) \
    if (n > (Y)) { \
        if ((m > 0)) { UPDATE(c[offsetC + 0], R0); } \
        if ((m > 1)) { UPDATE(c[offsetC + 1], R1); } \
        if ((m > 2)) { UPDATE(c[offsetC + 2], R2); } \
        if ((m > 3)) { UPDATE(c[offsetC + 3], R3); } \
        offsetC += ldc; \
    }
#endif

#if SIZEY == 1
#define UPDATE_Y(X, R0, R1, R2, R3) \
    UPDATE_YY(X, X *SIZEY + 0, R0##X, R1##X, R2##X, R3##X);
#elif SIZEY == 2
#define UPDATE_Y(X, R0, R1, R2, R3) \
    UPDATE_YY(X, X *SIZEY + 0, R0##X.s0, R1##X.s0, R2##X.s0, R3##X.s0); \
    UPDATE_YY(X, X *SIZEY + 1, R0##X.s1, R1##X.s1, R2##X.s1, R3##X.s1);
#elif SIZEY == 3
#define UPDATE_Y(X, R0, R1, R2, R3) \
    UPDATE_YY(X, X *SIZEY + 0, R0##X.s0, R1##X.s0, R2##X.s0, R3##X.s0); \
    UPDATE_YY(X, X *SIZEY + 1, R0##X.s1, R1##X.s1, R2##X.s1, R3##X.s1); \
    UPDATE_YY(X, X *SIZEY + 2, R0##X.s2, R1##X.s2, R2##X.s2, R3##X.s2);
#else
#define UPDATE_Y(X, R0, R1, R2, R3) \
    UPDATE_YY(X, X *SIZEY + 0, R0##X.s0, R1##X.s0, R2##X.s0, R3##X.s0); \
    UPDATE_YY(X, X *SIZEY + 1, R0##X.s1, R1##X.s1, R2##X.s1, R3##X.s1); \
    UPDATE_YY(X, X *SIZEY + 2, R0##X.s2, R1##X.s2, R2##X.s2, R3##X.s2); \
    UPDATE_YY(X, X *SIZEY + 3, R0##X.s3, R1##X.s3, R2##X.s3, R3##X.s3);
#endif

__attribute__((intel_reqd_sub_group_size(GRX))) __kernel void gen9_gemm_compute(
        long m, long n, long k, __global DATA_T *base, int offsetA, int offsetB,
        __global DST_DATA_T *c, long offsetC, long ldc, int last_k_block,
        DATA_T eltwise_alpha, DATA_T eltwise_beta, DATA_T eltwise_scale) {
    int idx, idy, lid;

    idx = get_group_id(0);
    idy = get_group_id(1) * get_enqueued_local_size(1) + get_local_id(1);

    lid = get_local_id(0); // local ID

    m -= UNROLL_M * idx;
    if (m > UNROLL_M) m = UNROLL_M;
    n -= UNROLL_N * idy;
    if (n > UNROLL_N) n = UNROLL_N;
    m -= UNROLL_M * lid / GRX;

    offsetA += UNROLL_M * k * idx + UNROLL_M * lid / GRX;
    offsetB += UNROLL_N * k * idy + UNROLL_N * lid / GRX;
    offsetC += UNROLL_M * idx + UNROLL_N * ldc * idy + UNROLL_M * lid / GRX;

    INIT_C(0);
    INIT_C(1);
    INIT_C(2);
    INIT_C(3);
    INIT_C(4);
    INIT_C(5);
    INIT_C(6);
    INIT_C(7);

    FLOATX blockA = AS_FLOATX(base, offsetA);
    offsetA += UNROLL_M;
    FLOATY blockB = AS_FLOATY(base, offsetB);
    offsetB += UNROLL_N;

    for (int l = k; l > 0; l--) {
        FLOATY bb;

        CALC(blockA, blockB, cc0, cc1, cc2, cc3);

        blockB = AS_FLOATY(base, offsetB);
        offsetB += UNROLL_N;
        blockA = AS_FLOATX(base, offsetA);
        offsetA += UNROLL_M;
    }

    UPDATE_Y(0, cc0, cc1, cc2, cc3);
#if UNROLL_N >= 2 * SIZEY
    UPDATE_Y(1, cc0, cc1, cc2, cc3);
#endif
#if UNROLL_N >= 3 * SIZEY
    UPDATE_Y(2, cc0, cc1, cc2, cc3);
#endif
#if UNROLL_N >= 4 * SIZEY
    UPDATE_Y(3, cc0, cc1, cc2, cc3);
#endif
#if UNROLL_N >= 5 * SIZEY
    UPDATE_Y(4, cc0, cc1, cc2, cc3);
#endif
#if UNROLL_N >= 6 * SIZEY
    UPDATE_Y(5, cc0, cc1, cc2, cc3);
#endif
#if UNROLL_N >= 7 * SIZEY
    UPDATE_Y(6, cc0, cc1, cc2, cc3);
#endif
#if UNROLL_N >= 8 * SIZEY
    UPDATE_Y(7, cc0, cc1, cc2, cc3);
#endif
}
