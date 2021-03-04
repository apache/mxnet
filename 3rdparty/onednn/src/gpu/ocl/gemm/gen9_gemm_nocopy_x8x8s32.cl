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

#undef GRX
#define GRX 8

#if defined(S8S8)
#define FLOATA char
#define FLOATA2 char2
#define FLOATA4 char4
#define FLOATB char
#define FLOATB4 char4
#define SHUFFLE(X, Y) as_char4(intel_sub_group_shuffle(as_int(X), Y))
#endif

#if defined(U8S8)
#define FLOATA uchar
#define FLOATA2 uchar2
#define FLOATA4 uchar4
#define FLOATB char
#define FLOATB4 char4
#define SHUFFLE(X, Y) as_char4(intel_sub_group_shuffle(as_int(X), Y))
#endif

#if defined(S8U8)
#define FLOATA char
#define FLOATA2 char2
#define FLOATA4 char4
#define FLOATB uchar
#define FLOATB4 uchar4
#define SHUFFLE(X, Y) as_uchar4(intel_sub_group_shuffle(as_int(X), Y))
#endif

#if defined(U8U8)
#define FLOATA uchar
#define FLOATA2 uchar2
#define FLOATA4 uchar4
#define FLOATB uchar
#define FLOATB4 uchar4
#define SHUFFLE(X, Y) as_uchar4(intel_sub_group_shuffle(as_int(X), Y))
#endif

#define FLOATC int
#define FLOATC4 int4

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

#define COMPUTE_C(X, Y, CO_IDX) \
    do { \
        float val = (!beta ? 0 : (X)) + (Y); \
        POST_OP(val); \
        (X) = convert_int_sat_rte(val + (apply_co ? co[CO_IDX] : 0)); \
    } while (0)

#ifdef FF
#define ADD_EACH(X, OFF) \
    do { \
        if (n > X + OFF) { \
            if (m > 0) \
                COMPUTE_C(c[0], sc[X / 4 + 0].s##OFF + xa[0] + xb[0], 0); \
            if (m > 1) \
                COMPUTE_C(c[1], sc[X / 4 + 4].s##OFF + xa[1] + xb[0], 0); \
            if (m > 2) \
                COMPUTE_C(c[2], sc[X / 4 + 8].s##OFF + xa[2] + xb[0], 0); \
            if (m > 3) \
                COMPUTE_C(c[3], sc[X / 4 + 12].s##OFF + xa[3] + xb[0], 0); \
            xb++; \
            c += ldc; \
        } \
    } while (0)
#elif defined CC
#define ADD_EACH(X, OFF) \
    do { \
        if (n > X + OFF) { \
            if (m > 0) \
                COMPUTE_C(c[0], sc[X / 4 + 0].s##OFF + xa[0] + xb[0], 0); \
            if (m > 1) \
                COMPUTE_C(c[1], sc[X / 4 + 4].s##OFF + xa[1] + xb[0], 0); \
            if (m > 2) \
                COMPUTE_C(c[2], sc[X / 4 + 8].s##OFF + xa[2] + xb[0], 0); \
            if (m > 3) \
                COMPUTE_C(c[3], sc[X / 4 + 12].s##OFF + xa[3] + xb[0], 0); \
            xb++; \
            c += ldc; \
            co++; \
        } \
    } while (0)
#else
#define ADD_EACH(X, OFF) \
    do { \
        if (n > X + OFF) { \
            if (m > 0) \
                COMPUTE_C(c[0], sc[X / 4 + 0].s##OFF + xa[0] + xb[0], 0); \
            if (m > 1) \
                COMPUTE_C(c[1], sc[X / 4 + 4].s##OFF + xa[1] + xb[0], 1); \
            if (m > 2) \
                COMPUTE_C(c[2], sc[X / 4 + 8].s##OFF + xa[2] + xb[0], 2); \
            if (m > 3) \
                COMPUTE_C(c[3], sc[X / 4 + 12].s##OFF + xa[3] + xb[0], 3); \
            xb++; \
            c += ldc; \
        } \
    } while (0)
#endif

#define ADD_SCALE(X) \
    do { \
        ADD_EACH(X, 0); \
        ADD_EACH(X, 1); \
        ADD_EACH(X, 2); \
        ADD_EACH(X, 3); \
    } while (0)

#define ACCUMULATE_1(a, b) \
    ((FLOATC)a.s0 * (FLOATC)b.s0) + ((FLOATC)a.s1 * (FLOATC)b.s1) \
            + ((FLOATC)a.s2 * (FLOATC)b.s2) + ((FLOATC)a.s3 * (FLOATC)b.s3)

#define ACCUMULATE(a, b0, b1, b2, b3) \
    (FLOATC4)(ACCUMULATE_1(a, b0), ACCUMULATE_1(a, b1), ACCUMULATE_1(a, b2), \
            ACCUMULATE_1(a, b3))

#define GROUPSIZE_M (6 * UNROLL_M)
#define GROUPSIZE_N (4 * UNROLL_N)

__attribute__((intel_reqd_sub_group_size(GRX))) kernel void
gen9_gemm_compute_x8x8s32(global FLOATA *a, global FLOATB *b, global FLOATC *c,
        long offsetA, long offsetB, long offsetC, long lda, long ldb, long ldc,
        long m, long n, long k, int beta, int ao, int bo, global int *co,
        long offsetCO, int apply_co, local FLOATA *sa, local FLOATB *sb,
        int apply_eltwise, float eltwise_alpha, float eltwise_beta,
        float eltwise_scale) {

    long kk = (k + UNROLL_K - 1) & ~(UNROLL_K - 1);
    long i, j, l, ll;
    global FLOATC *c_ori;

    int lid = get_local_id(0);
    int idx = get_local_id(1);
    int idy = get_local_id(2);
    long gdx = get_group_id(1);
    long gdy = get_group_id(2);
    long szx = get_local_size(1);
    long szy = get_local_size(2);

    a += offsetA;
    b += offsetB;
    c += offsetC + UNROLL_M * idx + GROUPSIZE_M * gdx + UNROLL_M * lid / GRX
            + (UNROLL_N * idy + GROUPSIZE_N * gdy) * ldc;
    c_ori = c;

    if (apply_co) {
        co += offsetCO;
#ifdef RR
        co += GROUPSIZE_M * gdx + UNROLL_M * idx + UNROLL_M * lid / GRX;
#endif
#ifdef CC
        co += GROUPSIZE_N * gdy + UNROLL_N * idy;
#endif
    }

    // Accumulation array for A and B
    __local FLOATC *xa = (__local FLOATC *)sa;
    sa += UNROLL_M * szx * sizeof(FLOATC);

    __local FLOATC *xb = (__local FLOATC *)sb;
    sb += UNROLL_N * szy * sizeof(FLOATC);

    int cid0 = (idy * szx + idx) * get_local_size(0) + lid;
    int ctotal = get_local_size(0) * szx * szy;

    for (int cid = cid0; cid < szx * UNROLL_M; cid += ctotal) {
        long sa_moffset = (cid & ~(UNROLL_M - 1)) * kk
                + (cid & (UNROLL_M - 1)) * UNROLL_K;
        long i = cid + GROUPSIZE_M * gdx;
        FLOATC sumA = 0;

#if defined(NN) || defined(NT)
        long a_offset = i;
#else
        long a_offset = i * lda;
#endif

        for (l = 0; l < kk; l += UNROLL_K) {
            for (ll = 0; ll < UNROLL_K; ll++) {
                FLOATA a_val = (((i < m) && (l + ll < k)) ? a[a_offset] : 0);
                sa[sa_moffset + l * UNROLL_M + ll] = a_val;
                sumA -= a_val;

#if defined(NN) || defined(NT)
                a_offset += lda;
#else
                a_offset++;
#endif
            }
        }

        xa[cid] = (FLOATC)bo * sumA;
    }

    for (int cid = cid0; cid < szy * UNROLL_N; cid += ctotal) {
        long sb_noffset = (cid & ~(UNROLL_N - 1)) * kk
                + (cid & (UNROLL_N - 1)) * UNROLL_K;
        long j = cid + GROUPSIZE_N * gdy;
        FLOATC sumB = (FLOATC)bo * k;

#if defined(NN) || defined(TN)
        long b_offset = j * ldb;
#else
        long b_offset = j;
#endif

        for (l = 0; l < kk; l += UNROLL_K) {
            for (ll = 0; ll < UNROLL_K; ll++) {
                FLOATB b_val = (((j < n) && (l + ll < k)) ? b[b_offset] : 0);
                sb[sb_noffset + l * UNROLL_N + ll] = b_val;
                sumB -= b_val;

#if defined(NN) || defined(TN)
                b_offset++;
#else
                b_offset += ldb;
#endif
            }
        }

        xb[cid] = (FLOATC)ao * sumB;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    m -= GROUPSIZE_M * gdx + UNROLL_M * idx;
    if (m > UNROLL_M) m = UNROLL_M;
    n -= GROUPSIZE_N * gdy + UNROLL_N * idy;
    if (n > UNROLL_N) n = UNROLL_N;

    if ((m <= 0) || (n <= 0)) return;
    m -= UNROLL_M * lid / GRX;

    sa += UNROLL_M * kk * idx + UNROLL_M * UNROLL_K * lid / GRX;
    sb += UNROLL_N * kk * idy + UNROLL_K * lid;

    xa += UNROLL_M * idx + UNROLL_M * lid / GRX;
    xb += UNROLL_N * idy;

    FLOATC4 sc[UNROLL_M * UNROLL_N / GRX / 4] = {0};

    for (l = 0; l < kk; l += UNROLL_K) {
        FLOATA4 a0, a1, a2, a3;
        FLOATB4 bb, b0, b1, b2, b3;

        a0 = ((__local FLOATA4 *)sa)[0];
        a1 = ((__local FLOATA4 *)sa)[1];
        a2 = ((__local FLOATA4 *)sa)[2];
        a3 = ((__local FLOATA4 *)sa)[3];

        for (ll = 0; ll < GRX / 4; ll++) {
            bb = ((__local FLOATB4 *)sb)[0];
            b0 = SHUFFLE(bb, 0);
            b1 = SHUFFLE(bb, 1);
            b2 = SHUFFLE(bb, 2);
            b3 = SHUFFLE(bb, 3);

            sc[ll * 2 + 0] += ACCUMULATE(a0, b0, b1, b2, b3);
            sc[ll * 2 + 4] += ACCUMULATE(a1, b0, b1, b2, b3);
            sc[ll * 2 + 8] += ACCUMULATE(a2, b0, b1, b2, b3);
            sc[ll * 2 + 12] += ACCUMULATE(a3, b0, b1, b2, b3);
            b0 = SHUFFLE(bb, 4);
            b1 = SHUFFLE(bb, 5);
            b2 = SHUFFLE(bb, 6);
            b3 = SHUFFLE(bb, 7);

            sc[ll * 2 + 1] += ACCUMULATE(a0, b0, b1, b2, b3);
            sc[ll * 2 + 5] += ACCUMULATE(a1, b0, b1, b2, b3);
            sc[ll * 2 + 9] += ACCUMULATE(a2, b0, b1, b2, b3);
            sc[ll * 2 + 13] += ACCUMULATE(a3, b0, b1, b2, b3);

            sb += UNROLL_N * GRX / 4;
        }
        sa += UNROLL_M * UNROLL_K;
    }

    ADD_SCALE(0);
    ADD_SCALE(4);
    ADD_SCALE(8);
    ADD_SCALE(12);
}
