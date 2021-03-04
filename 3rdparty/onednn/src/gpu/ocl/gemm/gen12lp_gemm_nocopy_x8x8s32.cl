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

#include "gpu/ocl/ocl_math_utils.h"

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#if defined(S8S8)
#define A_TYPE char
#define A_TYPE2 char2
#define A_TYPE4 char4
#define B_TYPE char
#define B_TYPE4 char4
#define AS_A_TYPE as_char
#define AS_A_TYPE2 as_char2
#define AS_A_TYPE4 as_char4
#define AS_B_TYPE as_char
#define AS_B_TYPE2 as_char2
#define AS_B_TYPE4 as_char4
#endif

#if defined(U8S8)
#define A_TYPE uchar
#define A_TYPE2 uchar2
#define A_TYPE4 uchar4
#define B_TYPE char
#define B_TYPE4 char4
#define AS_A_TYPE as_uchar
#define AS_A_TYPE2 as_uchar2
#define AS_A_TYPE4 as_uchar4
#define AS_B_TYPE as_char
#define AS_B_TYPE2 as_char2
#define AS_B_TYPE4 as_char4
#endif

#if defined(S8U8)
#define A_TYPE char
#define A_TYPE2 char2
#define A_TYPE4 char4
#define B_TYPE uchar
#define B_TYPE4 uchar4
#define AS_A_TYPE as_char
#define AS_A_TYPE2 as_char2
#define AS_A_TYPE4 as_char4
#define AS_B_TYPE as_uchar
#define AS_B_TYPE2 as_uchar2
#define AS_B_TYPE4 as_uchar4
#endif

#if defined(U8U8)
#define A_TYPE uchar
#define A_TYPE2 uchar2
#define A_TYPE4 uchar4
#define B_TYPE uchar
#define B_TYPE4 uchar4
#define AS_A_TYPE as_uchar
#define AS_A_TYPE2 as_uchar2
#define AS_A_TYPE4 as_uchar4
#define AS_B_TYPE as_uchar
#define AS_B_TYPE2 as_uchar2
#define AS_B_TYPE4 as_uchar4
#endif

#if defined(TN)
#define DO_FMA DO_FMA_TN
#endif
#if defined(NN)
#define DO_FMA DO_FMA_NN
#endif
#if defined(NT)
#define DO_FMA DO_FMA_NT
#endif
#if defined(TT)
#define DO_FMA DO_FMA_TT
#endif

#define ADD_ROW_A(z) \
    do { \
        sumRowA[z] = ai[z].s0 + ai[z].s1 + ai[z].s2 + ai[z].s3; \
    } while (0)

#define ADD_ROW_AT() \
    do { \
        sumRowA[0] = ai[0].s0 + ai[1].s0 + ai[2].s0 + ai[3].s0; \
        sumRowA[1] = ai[0].s1 + ai[1].s1 + ai[2].s1 + ai[3].s1; \
    } while (0)

#define ADD_COL_B() \
    do { \
        sumColB = bi.s0 + bi.s1 + bi.s2 + bi.s3; \
    } while (0)

#define ADD_COL_BT() \
    do { \
        sumColB = bi[0] + bi[1] + bi[2] + bi[3]; \
    } while (0)

#ifdef ALIGNED
#define VLOAD4_A(z, p) \
    do { \
        ai[z] = *((global A_TYPE4 *)p); \
    } while (0)
#else
#define VLOAD4_A(z, p) \
    do { \
        ai[z].s0 = *(p + 0); \
        ai[z].s1 = *(p + 1); \
        ai[z].s2 = *(p + 2); \
        ai[z].s3 = *(p + 3); \
    } while (0)
#endif

#ifdef ALIGNED
#define BLOCK_READ_A(h, hh) \
    do { \
        ai[hh] = AS_A_TYPE2(intel_sub_group_block_read_uc2( \
                (global uchar *)(a_ptrs[hh] + h * lda))); \
    } while (0)
#else
#define BLOCK_READ_A(h, hh) \
    do { \
        ai[hh].s0 = *((a_ptrs[hh] + h * lda) + 0); \
        ai[hh].s1 = *((a_ptrs[hh] + h * lda) + 16); \
    } while (0)
#endif

#ifdef ALIGNED
#define BLOCK_READ_B(h, hh) \
    do { \
        bi[hh] = AS_B_TYPE(intel_sub_group_block_read_uc( \
                (global uchar *)(b_ptrs[hh] + h * ldb))); \
    } while (0)
#else
#define BLOCK_READ_B(h, hh) \
    do { \
        bi[hh] = *(b_ptrs[hh] + h * ldb); \
    } while (0)
#endif

#ifdef ALIGNED
#define VLOAD4_B(p) \
    do { \
        bi = *((global B_TYPE4 *)p); \
    } while (0)
#else
#define VLOAD4_B(p) \
    do { \
        bi.s0 = *(p + 0); \
        bi.s1 = *(p + 1); \
        bi.s2 = *(p + 2); \
        bi.s3 = *(p + 3); \
    } while (0)
#endif

#define LOADA_REM(z, p) \
    do { \
        if (krem == 3) { \
            ai[z].s0 = *(p + 0); \
            ai[z].s1 = *(p + 1); \
            ai[z].s2 = *(p + 2); \
        } \
        if (krem == 2) { \
            ai[z].s0 = *(p + 0); \
            ai[z].s1 = *(p + 1); \
        } \
        if (krem == 1) { ai[z].s0 = *(p + 0); } \
    } while (0)

#define LOADB_REM(p) \
    do { \
        if (krem == 3) { \
            bi.s0 = *(p + 0); \
            bi.s1 = *(p + 1); \
            bi.s2 = *(p + 2); \
        } \
        if (krem == 2) { \
            bi.s0 = *(p + 0); \
            bi.s1 = *(p + 1); \
        } \
        if (krem == 1) { bi.s0 = *(p + 0); } \
    } while (0)

#define COPYA() \
    do { \
        ait[0].s0 = ai[0].s0; \
        ait[0].s1 = ai[1].s0; \
        ait[0].s2 = ai[2].s0; \
        ait[0].s3 = ai[3].s0; \
        ait[1].s0 = ai[0].s1; \
        ait[1].s1 = ai[1].s1; \
        ait[1].s2 = ai[2].s1; \
        ait[1].s3 = ai[3].s1; \
    } while (0)

#define COPYB() \
    do { \
        biit.s0 = bi[0]; \
        biit.s1 = bi[1]; \
        biit.s2 = bi[2]; \
        biit.s3 = bi[3]; \
    } while (0)

#define DO_FMA_TN(h, i) \
    do { \
        ci[0][i] = idot4(AS_B_TYPE4(sub_group_broadcast(as_int(bi), i)), \
                AS_A_TYPE4(ai[0]), ci[0][i]); \
        ci[1][i] = idot4(AS_B_TYPE4(sub_group_broadcast(as_int(bi), i)), \
                AS_A_TYPE4(ai[1]), ci[1][i]); \
    } while (0)

#define DO_FMA_NN(h, i) \
    do { \
        ci[0][i] = idot4(AS_B_TYPE4(sub_group_broadcast(as_int(bi), i)), \
                AS_A_TYPE4(ait[0]), ci[0][i]); \
        ci[1][i] = idot4(AS_B_TYPE4(sub_group_broadcast(as_int(bi), i)), \
                AS_A_TYPE4(ait[1]), ci[1][i]); \
    } while (0)

#define DO_FMA_NT(h, i) \
    do { \
        ci[0][i] = idot4(AS_B_TYPE4(sub_group_broadcast(as_int(biit), i)), \
                AS_A_TYPE4(ait[0]), ci[0][i]); \
        ci[1][i] = idot4(AS_B_TYPE4(sub_group_broadcast(as_int(biit), i)), \
                AS_A_TYPE4(ait[1]), ci[1][i]); \
    } while (0)

#define DO_FMA_TT(h, i) \
    do { \
        ci[0][i] = idot4(AS_B_TYPE4(sub_group_broadcast(as_int(biit), i)), \
                AS_A_TYPE4(ai[0]), ci[0][i]); \
        ci[1][i] = idot4(AS_B_TYPE4(sub_group_broadcast(as_int(biit), i)), \
                AS_A_TYPE4(ai[1]), ci[1][i]); \
    } while (0)

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

#define FMA_I_LOOP(h) \
    do { \
        DO_FMA(h, 0); \
        DO_FMA(h, 1); \
        DO_FMA(h, 2); \
        DO_FMA(h, 3); \
        DO_FMA(h, 4); \
        DO_FMA(h, 5); \
        DO_FMA(h, 6); \
        DO_FMA(h, 7); \
        DO_FMA(h, 8); \
        DO_FMA(h, 9); \
        DO_FMA(h, 10); \
        DO_FMA(h, 11); \
        DO_FMA(h, 12); \
        DO_FMA(h, 13); \
        DO_FMA(h, 14); \
        DO_FMA(h, 15); \
    } while (0)

#define ADD_BOFF(i) \
    do { \
        ci[0][i] -= bo * sumRowA[0]; \
        ci[1][i] -= bo * sumRowA[1]; \
    } while (0)

#define ADD_BOFF_LOOP() \
    do { \
        ADD_BOFF(0); \
        ADD_BOFF(1); \
        ADD_BOFF(2); \
        ADD_BOFF(3); \
        ADD_BOFF(4); \
        ADD_BOFF(5); \
        ADD_BOFF(6); \
        ADD_BOFF(7); \
        ADD_BOFF(8); \
        ADD_BOFF(9); \
        ADD_BOFF(10); \
        ADD_BOFF(11); \
        ADD_BOFF(12); \
        ADD_BOFF(13); \
        ADD_BOFF(14); \
        ADD_BOFF(15); \
    } while (0)

#define ADD_AOFF(h, i) \
    do { \
        ci[0][i] -= (ao * sub_group_broadcast(as_int(sumColB), i)) \
                - (h * ao * bo); \
        ci[1][i] -= (ao * sub_group_broadcast(as_int(sumColB), i)) \
                - (h * ao * bo); \
    } while (0)

#define ADD_AOFF_LOOP(h) \
    do { \
        ADD_AOFF(h, 0); \
        ADD_AOFF(h, 1); \
        ADD_AOFF(h, 2); \
        ADD_AOFF(h, 3); \
        ADD_AOFF(h, 4); \
        ADD_AOFF(h, 5); \
        ADD_AOFF(h, 6); \
        ADD_AOFF(h, 7); \
        ADD_AOFF(h, 8); \
        ADD_AOFF(h, 9); \
        ADD_AOFF(h, 10); \
        ADD_AOFF(h, 11); \
        ADD_AOFF(h, 12); \
        ADD_AOFF(h, 13); \
        ADD_AOFF(h, 14); \
        ADD_AOFF(h, 15); \
    } while (0)

#define UPDATE_C_COL(i, betaZero) \
    do { \
        if (jrem > i) { \
            if (irem > 0) { \
                if (c_offset_type == 0) { \
                    float val = ((betaZero) ? 0 : *c) + ci[0][i]; \
                    POST_OP(val); \
                    *c = convert_int_sat_rte(val + ((!apply_co) ? 0 : co[0])); \
                } \
                if (c_offset_type == 1) { \
                    float val = ((betaZero) ? 0 : *c) + ci[0][i]; \
                    POST_OP(val); \
                    *c = convert_int_sat_rte(val + ((!apply_co) ? 0 : co[0])); \
                } \
                if (c_offset_type == 2) { \
                    float val = ((betaZero) ? 0 : *c) + ci[0][i]; \
                    POST_OP(val); \
                    *c = convert_int_sat_rte(val + ((!apply_co) ? 0 : co[i])); \
                } \
            } \
            if (irem > 16) { \
                if (c_offset_type == 0) { \
                    float val = ((betaZero) ? 0 : *c2) + ci[1][i]; \
                    POST_OP(val); \
                    *c2 = convert_int_sat_rte( \
                            val + ((!apply_co) ? 0 : co[0])); \
                } \
                if (c_offset_type == 1) { \
                    float val = ((betaZero) ? 0 : *c2) + ci[1][i]; \
                    POST_OP(val); \
                    *c2 = convert_int_sat_rte( \
                            val + ((!apply_co) ? 0 : co[16])); \
                } \
                if (c_offset_type == 2) { \
                    float val = ((betaZero) ? 0 : *c2) + ci[1][i]; \
                    POST_OP(val); \
                    *c2 = convert_int_sat_rte( \
                            val + ((!apply_co) ? 0 : co[i])); \
                } \
            } \
        } \
        c = c + ldc; \
        c2 = c2 + ldc; \
    } while (0)

#define UPDATE_C(betaZero) \
    do { \
        UPDATE_C_COL(0, betaZero); \
        UPDATE_C_COL(1, betaZero); \
        UPDATE_C_COL(2, betaZero); \
        UPDATE_C_COL(3, betaZero); \
        UPDATE_C_COL(4, betaZero); \
        UPDATE_C_COL(5, betaZero); \
        UPDATE_C_COL(6, betaZero); \
        UPDATE_C_COL(7, betaZero); \
        UPDATE_C_COL(8, betaZero); \
        UPDATE_C_COL(9, betaZero); \
        UPDATE_C_COL(10, betaZero); \
        UPDATE_C_COL(11, betaZero); \
        UPDATE_C_COL(12, betaZero); \
        UPDATE_C_COL(13, betaZero); \
        UPDATE_C_COL(14, betaZero); \
        UPDATE_C_COL(15, betaZero); \
    } while (0)

#ifdef TN
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12lp_gemm_compute_x8x8s32(global A_TYPE *a, global B_TYPE *b, global int *c,
        int offsetA, int offsetB, int offsetC, int lda, int ldb, int ldc, int m,
        int n, int k, int beta, int ao, int bo, global int *co, int offsetCO,
        int apply_co, local A_TYPE *sa, local B_TYPE *sb, int apply_eltwise,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale) {

    // clang-format off
    A_TYPE4 ai[2];  // 32x4 block of A, 2x 16x4 scattered access
    B_TYPE4 bi;     // 4x16 block of B, 1x 4x16 scattered access
    int ci[2][16]; // 32x16 block of C, 16x1 x 2x16 scattered access
    // clang-format on

    int sumRowA[2] = {0, 0};
    int sumColB = 0;

    int idM = get_group_id(0);
    int idN = get_group_id(1);
    int idlM = get_local_id(0);
    int idlN = get_local_id(1);
    int lid = get_sub_group_local_id();
    int lsm = 32;
    int lsn = 8;

    int i0 = (idM * lsm / 16) * 32 + (get_local_id(0) / 16) * 32;
    int j0 = idlN * 16 + (idN * lsn * 16);

    int irem = m - i0 - lid;
    int jrem = n - j0;

    int irem2 = m - i0;
    int jrem2 = n - j0;

    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;

    a += offsetA + (i0 * lda) + (lid * lda);
    b += offsetB + (j0 * ldb) + (lid * ldb);
    c += offsetC + (i0) + (j0 * ldc) + lid;

    int c_offset_type = 0;

#ifdef FF
    co += offsetCO;
    c_offset_type = 0;
#endif
#ifdef RR
    co += offsetCO + i0 + lid;
    c_offset_type = 1;
#endif
#ifdef CC
    co += offsetCO + (j0);
    c_offset_type = 2;
#endif

    global A_TYPE *a_ptrs[2] = {a, a + 16 * lda};
    global B_TYPE *b_ptrs = {b};

    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 2; z++) {
            ci[z][y] = 0;
        }
    }
    int k_align = k & ~3;

#ifndef ALLOW_READ_OVERRUNS
    if (irem2 >= 32 && jrem2 >= 16) {
#endif
        for (int h = 0; h < k_align; h += 4) {
            // Load A
            for (int z = 0; z < 2; z++) {
                VLOAD4_A(z, (a_ptrs[z] + h));
#ifdef BOFFNONZERO
                ADD_ROW_A(z);
#endif
            }
            // Load B
            VLOAD4_B((b_ptrs + h));
#ifdef AOFFNONZERO
            ADD_COL_B();
#endif
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(4);
#endif
        }
        // Remainder Loop
        int krem = k & 3;
        if (krem > 0) {
            ai[0] = 0;
            ai[1] = 0;
            bi = 0;
            // Load A
            for (int z = 0; z < 2; z++) {
                LOADA_REM(z, (a_ptrs[z] + k_align));
#ifdef BOFFNONZERO
                ADD_ROW_A(z);
#endif
            }
            // Load B
            LOADB_REM((b_ptrs + k_align));
#ifdef AOFFNONZERO
            ADD_COL_B();
#endif
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(krem);
#endif
        }

#ifndef ALLOW_READ_OVERRUNS
    } else {
        for (int h = 0; h < k_align; h += 4) {
            // Load A
            for (int z = 0; z < 2; z++) {
                if (irem > z * 16) {
                    VLOAD4_A(z, (a_ptrs[z] + h));
#ifdef BOFFNONZERO
                    ADD_ROW_A(z);
#endif
                }
            }
            // Load B
            if (jrem > lid) {
                VLOAD4_B((b_ptrs + h));
#ifdef AOFFNONZERO
                ADD_COL_B();
#endif
            }
            // Compute
            FMA_I_LOOP(0);

#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(4);
#endif
        }
        // Remainder Loop
        int krem = k & 3;
        if (krem > 0) {
            ai[0] = 0;
            ai[1] = 0;
            bi = 0;
            // Load A
            for (int z = 0; z < 2; z++) {
                if (irem > z * 16) {
                    LOADA_REM(z, (a_ptrs[z] + k_align));
#ifdef BOFFNONZERO
                    ADD_ROW_A(z);
#endif
                }
            }
            // Load B
            if (jrem > lid) {
                LOADB_REM((b_ptrs + k_align));
#ifdef AOFFNONZERO
                ADD_COL_B();
#endif
            }
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(krem);
#endif
        }
    }
#endif /* ALLOW_READ_OVERHEAD */

    // Store C
    global int *c2 = c + 16;

    // Update C
    if (beta == 0)
        UPDATE_C(1);
    else
        UPDATE_C(0);
}
#endif //TN

#ifdef NN
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12lp_gemm_compute_x8x8s32(global A_TYPE *a, global B_TYPE *b, global int *c,
        int offsetA, int offsetB, int offsetC, int lda, int ldb, int ldc, int m,
        int n, int k, int beta, int ao, int bo, global int *co, int offsetCO,
        int apply_co, local A_TYPE *sa, local B_TYPE *sb, int apply_eltwise,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale) {

    // clang-format off
    A_TYPE2 ai[4];  // 32x4 block of A, 4x 32x1 block access
    B_TYPE4 bi;     // 4x16 block of B, 1x 4x16 scattered access
    int ci[2][16]; // 32x16 block of C, 16x1 x 2x16 scattered access
    // clang-format on

    int sumRowA[2] = {0, 0};
    int sumColB = 0;

    A_TYPE4 ait[2];

    int idM = get_group_id(0);
    int idN = get_group_id(1);
    int idlM = get_local_id(0);
    int idlN = get_local_id(1);
    int lid = get_sub_group_local_id();
    int lsm = 32;
    int lsn = 8;

    int i0 = (idM * lsm / 16) * 32 + (get_local_id(0) / 16) * 32;
    int j0 = idlN * 16 + (idN * lsn * 16);

    int irem = m - i0 - lid;
    int jrem = n - j0;
    int irem2 = m - i0;
    int jrem2 = n - j0;

    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;

#ifdef ALIGNED
    a += offsetA + i0;
#else
    a += offsetA + i0 + lid;
#endif

    b += offsetB + (j0 * ldb) + (lid * ldb);
    c += offsetC + (i0) + (j0 * ldc) + lid;

    int c_offset_type = 0;

#ifdef FF
    co += offsetCO;
    c_offset_type = 0;
#endif
#ifdef RR
    co += offsetCO + i0 + lid;
    c_offset_type = 1;
#endif
#ifdef CC
    co += offsetCO + (j0);
    c_offset_type = 2;
#endif

    global A_TYPE *a_ptrs[4] = {a, a + 1 * lda, a + 2 * lda, a + 3 * lda};
    global B_TYPE *b_ptrs = {b};

    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 2; z++) {
            ci[z][y] = 0;
        }
    }
    int k_align = k & ~3;

#ifndef ALLOW_READ_OVERRUNS
    if (irem2 >= 32 && jrem2 >= 16) {
#endif
        for (int h = 0; h < k_align; h += 4) {
            // Load A
            for (int hh = 0; hh < 4; hh++) {
                BLOCK_READ_A(h, hh);
#ifdef BOFFNONZERO
                ADD_ROW_AT();
#endif
            }
            // Copy A
            COPYA();
            // Load B
            VLOAD4_B((b_ptrs + h));
#ifdef AOFFNONZERO
            ADD_COL_B();
#endif
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(4);
#endif
        }
        // Remainder Loop
        int krem = k & 3;
        if (krem > 0) {
            ai[0] = 0;
            ai[1] = 0;
            ai[2] = 0;
            ai[3] = 0;
            bi = 0;
            // Load A
            for (int hh = 0; hh < krem; hh++) {
                BLOCK_READ_A(k_align, hh);
#ifdef BOFFNONZERO
                ADD_ROW_AT();
#endif
            }
            // Copy A
            COPYA();
            // Load B
            LOADB_REM((b_ptrs + k_align));
#ifdef AOFFNONZERO
            ADD_COL_B();
#endif
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(krem);
#endif
        }

#ifndef ALLOW_READ_OVERRUNS
    } else {
        for (int h = 0; h < k_align; h += 4) {
            // Load A
            for (int hh = 0; hh < 4; hh++) {
                if (irem2 > lid) {
#ifdef ALIGNED
                    ai[hh].s0 = *((a_ptrs[hh] + h * lda) + 0 + lid);
                    ai[hh].s1 = *((a_ptrs[hh] + h * lda) + 16 + lid);
#else
                    ai[hh].s0 = *((a_ptrs[hh] + h * lda) + 0);
                    ai[hh].s1 = *((a_ptrs[hh] + h * lda) + 16);
#endif
#ifdef BOFFNONZERO
                    ADD_ROW_AT();
#endif
                }
            }
            // Copy A
            COPYA();
            // Load B
            if (jrem > lid) {
                VLOAD4_B((b_ptrs + h));
#ifdef AOFFNONZERO
                ADD_COL_B();
#endif
            }
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(4);
#endif
        }
        // Remainder Loop
        int krem = k & 3;
        if (krem > 0) {
            ai[0] = 0;
            ai[1] = 0;
            ai[2] = 0;
            ai[3] = 0;
            bi = 0;
            // Load A
            for (int hh = 0; hh < krem; hh++) {
                if (irem2 > lid) {
#ifdef ALIGNED
                    ai[hh].s0 = *((a_ptrs[hh] + k_align * lda) + 0 + lid);
                    ai[hh].s1 = *((a_ptrs[hh] + k_align * lda) + 16 + lid);
#else
                    ai[hh].s0 = *((a_ptrs[hh] + k_align * lda) + 0);
                    ai[hh].s1 = *((a_ptrs[hh] + k_align * lda) + 16);
#endif
#ifdef BOFFNONZERO
                    ADD_ROW_AT();
#endif
                }
            }
            // Copy A
            COPYA();
            // Load B
            if (jrem > lid) {
                LOADB_REM((b_ptrs + k_align));
#ifdef AOFFNONZERO
                ADD_COL_B();
#endif
            }
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(krem);
#endif
        }
    }
#endif /* ALLOW_READ_OVERHEAD */

    // Store C
    global int *c2 = c + 16;

    // Update C
    if (beta == 0)
        UPDATE_C(1);
    else
        UPDATE_C(0);
}
#endif // NN

#ifdef NT
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12lp_gemm_compute_x8x8s32(global A_TYPE *a, global B_TYPE *b, global int *c,
        int offsetA, int offsetB, int offsetC, int lda, int ldb, int ldc, int m,
        int n, int k, int beta, int ao, int bo, global int *co, int offsetCO,
        int apply_co, local A_TYPE *sa, local B_TYPE *sb, int apply_eltwise,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale) {

    // clang-format off
    A_TYPE2 ai[4];   // 32x4 block of A, 4x 32x1 block access
    B_TYPE bi[4];    // 4x16 block of B, 4x 1x16 block access
    int ci[2][16];  // 32x16 block of C, 16x1 x 2x16 scattered access
    // clang-format on

    int sumRowA[2] = {0, 0};
    int sumColB = 0;

    A_TYPE4 ait[2];
    A_TYPE4 biit;

    int idM = get_group_id(0);
    int idN = get_group_id(1);
    int idlM = get_local_id(0);
    int idlN = get_local_id(1);
    int lid = get_sub_group_local_id();
    int lsm = 32;
    int lsn = 8;

    int i0 = (idM * lsm / 16) * 32 + (get_local_id(0) / 16) * 32;
    int j0 = idlN * 16 + (idN * lsn * 16);

    int irem = m - i0 - lid;
    int jrem = n - j0;
    int irem2 = m - i0;
    int jrem2 = n - j0;

    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;

#ifdef ALIGNED
    a += offsetA + i0;
#else
    a += offsetA + i0 + lid;
#endif

#ifdef ALIGNED
    b += offsetB + j0;
#else
    b += offsetB + j0 + lid;
#endif

    c += offsetC + (i0) + (j0 * ldc) + lid;

    int c_offset_type = 0;

#ifdef FF
    co += offsetCO;
    c_offset_type = 0;
#endif
#ifdef RR
    co += offsetCO + i0 + lid;
    c_offset_type = 1;
#endif
#ifdef CC
    co += offsetCO + (j0);
    c_offset_type = 2;
#endif

    global A_TYPE *a_ptrs[4] = {a, a + 1 * lda, a + 2 * lda, a + 3 * lda};
    global B_TYPE *b_ptrs[4] = {b, b + 1 * ldb, b + 2 * ldb, b + 3 * ldb};

    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 2; z++) {
            ci[z][y] = 0;
        }
    }

    int insidea1 = 5;
    int insidea2 = 5;
    int insideb = 5;

    int k_align = k & ~3;

#ifndef ALLOW_READ_OVERRUNS
    if (irem2 >= 32 && jrem2 >= 16) {
#endif
        for (int h = 0; h < k_align; h += 4) {
            // Load A
            for (int hh = 0; hh < 4; hh++) {
                BLOCK_READ_A(h, hh);
#ifdef BOFFNONZERO
                ADD_ROW_AT();
#endif
            }
            // Copy A
            COPYA();
            // Load B
            for (int hh = 0; hh < 4; hh++) {
                BLOCK_READ_B(h, hh);
#ifdef AOFFNONZERO
                ADD_COL_BT();
#endif
            }
            // Copy B
            COPYB();
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(4);
#endif
        }
        // Remainder Loop
        int krem = k & 3;
        if (krem > 0) {
            ai[0] = 0;
            ai[1] = 0;
            ai[2] = 0;
            ai[3] = 0;

            bi[0] = 0;
            bi[1] = 0;
            bi[2] = 0;
            bi[3] = 0;
            // Load A
            for (int hh = 0; hh < krem; hh++) {
                BLOCK_READ_A(k_align, hh);
#ifdef BOFFNONZERO
                ADD_ROW_AT();
#endif
            }
            // Copy A
            COPYA();
            // Load B
            for (int hh = 0; hh < krem; hh++) {
                BLOCK_READ_B(k_align, hh);
#ifdef AOFFNONZERO
                ADD_COL_BT();
#endif
            }
            // Copy B
            COPYB();
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(krem);
#endif
        }

#ifndef ALLOW_READ_OVERRUNS
    } else {
        for (int h = 0; h < k_align; h += 4) {
            // Load A
            for (int hh = 0; hh < 4; hh++) {
                if (irem2 > lid) {
#ifdef ALIGNED
                    ai[hh].s0 = *((a_ptrs[hh] + h * lda) + 0 + lid);
                    ai[hh].s1 = *((a_ptrs[hh] + h * lda) + 16 + lid);
#else
                    ai[hh].s0 = *((a_ptrs[hh] + h * lda) + 0);
                    ai[hh].s1 = *((a_ptrs[hh] + h * lda) + 16);
#endif
#ifdef BOFFNONZERO
                    ADD_ROW_AT();
#endif
                }
            }
            // Copy A
            COPYA();
            // Load B
            for (int hh = 0; hh < 4; hh++) {
                if (jrem > lid) {
#ifdef ALIGNED
                    bi[hh] = *((b_ptrs[hh] + h * ldb) + lid);
#else
                    bi[hh] = *(b_ptrs[hh] + h * ldb);
#endif
#ifdef AOFFNONZERO
                    ADD_COL_BT();
#endif
                }
            }
            // Copy B
            COPYB();
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(4);
#endif
        }
        // Remainder Loop
        int krem = k & 3;
        if (krem > 0) {
            ai[0] = 0;
            ai[1] = 0;
            ai[2] = 0;
            ai[3] = 0;

            bi[0] = 0;
            bi[1] = 0;
            bi[2] = 0;
            bi[3] = 0;
            // Load A
            for (int hh = 0; hh < krem; hh++) {
                if (irem2 > lid) {
#ifdef ALIGNED
                    ai[hh].s0 = *((a_ptrs[hh] + k_align * lda) + 0 + lid);
                    ai[hh].s1 = *((a_ptrs[hh] + k_align * lda) + 16 + lid);
#else
                    ai[hh].s0 = *((a_ptrs[hh] + k_align * lda) + 0);
                    ai[hh].s1 = *((a_ptrs[hh] + k_align * lda) + 16);
#endif
#ifdef BOFFNONZERO
                    ADD_ROW_AT();
#endif
                }
            }
            // Copy A
            COPYA();
            // Load B
            for (int hh = 0; hh < krem; hh++) {
                if (jrem > lid) {
#ifdef ALIGNED
                    bi[hh] = *((b_ptrs[hh] + k_align * ldb) + lid);
#else
                    bi[hh] = *(b_ptrs[hh] + k_align * ldb);
#endif
#ifdef AOFFNONZERO
                    ADD_COL_BT();
#endif
                }
            }
            // Copy B
            COPYB();
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(krem);
#endif
        }
    }
#endif /* ALLOW_READ_OVERHEAD */

    // Store C
    global int *c2 = c + 16;

    // Update C
    if (beta == 0)
        UPDATE_C(1);
    else
        UPDATE_C(0);
}
#endif // NT

#ifdef TT
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12lp_gemm_compute_x8x8s32(global A_TYPE *a, global B_TYPE *b, global int *c,
        int offsetA, int offsetB, int offsetC, int lda, int ldb, int ldc, int m,
        int n, int k, int beta, int ao, int bo, global int *co, int offsetCO,
        int apply_co, local A_TYPE *sa, local B_TYPE *sb, int apply_eltwise,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale) {

    // clang-format off
    A_TYPE4 ai[2];  // 32x4 block of A, 2x 16x4 scattered access
    B_TYPE bi[4];   // 4x16 block of B, 4x 1x16 block access
    int ci[2][16]; // 32x16 block of C, 16x1 x 2x16 scattered access
    // clang-format on

    int sumRowA[2] = {0, 0};
    int sumColB = 0;

    A_TYPE4 biit;

    int idM = get_group_id(0);
    int idN = get_group_id(1);
    int idlM = get_local_id(0);
    int idlN = get_local_id(1);
    int lid = get_sub_group_local_id();
    int lsm = 32;
    int lsn = 8;

    int i0 = (idM * lsm / 16) * 32 + (get_local_id(0) / 16) * 32;
    int j0 = idlN * 16 + (idN * lsn * 16);

    int irem = m - i0 - lid;
    int jrem = n - j0;

    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;
    int irem2 = m - i0;
    int jrem2 = n - j0;

    a += offsetA + (i0 * lda) + (lid * lda);

#ifdef ALIGNED
    b += offsetB + j0;
#else
    b += offsetB + j0 + lid;
#endif

    c += offsetC + (i0) + (j0 * ldc) + lid;

    int c_offset_type = 0; //0:Fixed, 1:Column, 2:Row

#ifdef FF
    co += offsetCO;
    c_offset_type = 0;
#endif
#ifdef RR
    co += offsetCO + i0 + lid;
    c_offset_type = 1;
#endif
#ifdef CC
    co += offsetCO + (j0);
    c_offset_type = 2;
#endif

    global A_TYPE *a_ptrs[2] = {a, a + 16 * lda};
    global B_TYPE *b_ptrs[4] = {b, b + 1 * ldb, b + 2 * ldb, b + 3 * ldb};

    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 2; z++) {
            ci[z][y] = 0;
        }
    }

    int k_align = k & ~3;

#ifndef ALLOW_READ_OVERRUNS
    if (irem2 >= 32 && jrem2 >= 16) {
#endif
        for (int h = 0; h < k_align; h += 4) {
            // Load A
            for (int z = 0; z < 2; z++) {
                VLOAD4_A(z, ((a_ptrs[z]) + h));
#ifdef BOFFNONZERO
                ADD_ROW_A(z);
#endif
            }
            // Load B
            for (int hh = 0; hh < 4; hh++) {
                BLOCK_READ_B(h, hh);
#ifdef AOFFNONZERO
                ADD_COL_BT();
#endif
            }
            // Copy B
            COPYB();
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(4);
#endif
        }
        // Remainder Loop
        int krem = k & 3;
        if (krem > 0) {
            ai[0] = 0;
            ai[1] = 0;

            bi[0] = 0;
            bi[1] = 0;
            bi[2] = 0;
            bi[3] = 0;

            // Load A
            for (int z = 0; z < 2; z++) {
                LOADA_REM(z, ((a_ptrs[z]) + k_align));
#ifdef BOFFNONZERO
                ADD_ROW_A(z);
#endif
            }
            // Load B
            for (int hh = 0; hh < krem; hh++) {
                BLOCK_READ_B(k_align, hh);
#ifdef AOFFNONZERO
                ADD_COL_BT();
#endif
            }
            // Copy B
            COPYB();
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(krem);
#endif
        }

#ifndef ALLOW_READ_OVERRUNS
    } else {

        for (int h = 0; h < k_align; h += 4) {
            // Load A
            for (int z = 0; z < 2; z++) {
                if (irem > z * 16) {
                    VLOAD4_A(z, (a_ptrs[z] + h));
#ifdef BOFFNONZERO
                    ADD_ROW_A(z);
#endif
                }
            }
            // Load B
            for (int hh = 0; hh < 4; hh++) {
                if (jrem > lid) {
#ifdef ALIGNED
                    bi[hh] = *((b_ptrs[hh] + h * ldb) + lid);
#else
                    bi[hh] = *(b_ptrs[hh] + h * ldb);
#endif
#ifdef AOFFNONZERO
                    ADD_COL_BT();
#endif
                }
            }
            // Copy B
            COPYB();
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(4);
#endif
        }
        // Remainder Loop
        int krem = k & 3;
        if (krem > 0) {
            ai[0] = 0;
            ai[1] = 0;

            bi[0] = 0;
            bi[1] = 0;
            bi[2] = 0;
            bi[3] = 0;

            // Load A
            for (int z = 0; z < 2; z++) {
                if (irem > z * 16) {
                    LOADA_REM(z, (a_ptrs[z] + k_align));
#ifdef BOFFNONZERO
                    ADD_ROW_A(z);
#endif
                }
            }
            // Load B
            for (int hh = 0; hh < krem; hh++) {
                if (jrem > lid) {
#ifdef ALIGNED
                    bi[hh] = *((b_ptrs[hh] + k_align * ldb) + lid);
#else
                    bi[hh] = *(b_ptrs[hh] + k_align * ldb);
#endif
#ifdef AOFFNONZERO
                    ADD_COL_BT();
#endif
                }
            }
            // Copy B
            COPYB();
            // Compute
            FMA_I_LOOP(0);
#ifdef BOFFNONZERO
            ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
            ADD_AOFF_LOOP(krem);
#endif
        }
    }
#endif /* ALLOW_READ_OVERHEAD */

    // Store C
    global int *c2 = c + 16;

    // Update C
    if (beta == 0)
        UPDATE_C(1);
    else
        UPDATE_C(0);
}
#endif // TT
