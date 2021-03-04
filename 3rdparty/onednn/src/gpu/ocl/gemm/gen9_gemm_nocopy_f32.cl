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

#if DT_F32 != 1
#error "Incorrect datatype."
#endif

#define DO_FMA_NN(hh, i_mod_16, i_div_16, i_mod_4, i_div_4) \
    do { \
        c[i_div_4].s##i_mod_4 \
                = mad(sub_group_broadcast(a[hh].s##i_div_16, i_mod_16), \
                        b.s##hh, c[i_div_4].s##i_mod_4); \
    } while (0)

#define DO_FMA_NT(hh, i_mod_16, i_div_16, i_mod_4, i_div_4) \
    do { \
        c[i_div_4].s##i_mod_4 \
                = mad(sub_group_broadcast(a[hh].s##i_div_16, i_mod_16), b[hh], \
                        c[i_div_4].s##i_mod_4); \
    } while (0)

#define DO_FMA_TN(hh, i, i_mod_4, i_div_4) \
    do { \
        c[i_div_4][0].s##i_mod_4 = mad(sub_group_broadcast(a.s##hh, i), \
                b[0].s##hh, c[i_div_4][0].s##i_mod_4); \
        c[i_div_4][1].s##i_mod_4 = mad(sub_group_broadcast(a.s##hh, i), \
                b[1].s##hh, c[i_div_4][1].s##i_mod_4); \
    } while (0)

#define DO_FMA_TT(hh, i, i_mod_4, i_div_4) \
    do { \
        c[i_div_4][0].s##i_mod_4 = mad(sub_group_broadcast(a.s##hh, i), \
                b[hh].s0, c[i_div_4][0].s##i_mod_4); \
        c[i_div_4][1].s##i_mod_4 = mad(sub_group_broadcast(a.s##hh, i), \
                b[hh].s1, c[i_div_4][1].s##i_mod_4); \
    } while (0)

#if !defined(TRANS_A)
#if !defined(TRANS_B)
#define NN
#define DO_FMA DO_FMA_NN
#else
#define NT
#define DO_FMA DO_FMA_NT
#endif
#else
#if !defined(TRANS_B)
#define TN
#define DO_FMA DO_FMA_TN
#else
#define TT
#define DO_FMA DO_FMA_TT
#endif
#endif

#if WITH_ELTWISE == 1
#define POST_OP(val) \
    do { \
        if (last_k_block && last_k_unroll) \
            val = fwd_eltwise( \
                    val, eltwise_alpha, eltwise_beta, eltwise_scale); \
    } while (0)
#else
#define POST_OP(val)
#endif

#define FMA_I_LOOP_32_ROW(hh) \
    do { \
        DO_FMA(hh, 0, 0, 0, 0); \
        DO_FMA(hh, 1, 0, 1, 0); \
        DO_FMA(hh, 2, 0, 2, 0); \
        DO_FMA(hh, 3, 0, 3, 0); \
        DO_FMA(hh, 4, 0, 0, 1); \
        DO_FMA(hh, 5, 0, 1, 1); \
        DO_FMA(hh, 6, 0, 2, 1); \
        DO_FMA(hh, 7, 0, 3, 1); \
        DO_FMA(hh, 8, 0, 0, 2); \
        DO_FMA(hh, 9, 0, 1, 2); \
        DO_FMA(hh, 10, 0, 2, 2); \
        DO_FMA(hh, 11, 0, 3, 2); \
        DO_FMA(hh, 12, 0, 0, 3); \
        DO_FMA(hh, 13, 0, 1, 3); \
        DO_FMA(hh, 14, 0, 2, 3); \
        DO_FMA(hh, 15, 0, 3, 3); \
        DO_FMA(hh, 16, 1, 0, 4); \
        DO_FMA(hh, 17, 1, 1, 4); \
        DO_FMA(hh, 18, 1, 2, 4); \
        DO_FMA(hh, 19, 1, 3, 4); \
        DO_FMA(hh, 20, 1, 0, 5); \
        DO_FMA(hh, 21, 1, 1, 5); \
        DO_FMA(hh, 22, 1, 2, 5); \
        DO_FMA(hh, 23, 1, 3, 5); \
        DO_FMA(hh, 24, 1, 0, 6); \
        DO_FMA(hh, 25, 1, 1, 6); \
        DO_FMA(hh, 26, 1, 2, 6); \
        DO_FMA(hh, 27, 1, 3, 6); \
        DO_FMA(hh, 28, 1, 0, 7); \
        DO_FMA(hh, 29, 1, 1, 7); \
        DO_FMA(hh, 30, 1, 2, 7); \
        DO_FMA(hh, 31, 1, 3, 7); \
    } while (0)

#define FMA_I_LOOP_16_ROW(hh) \
    do { \
        DO_FMA(hh, 0, 0, 0); \
        DO_FMA(hh, 1, 1, 0); \
        DO_FMA(hh, 2, 2, 0); \
        DO_FMA(hh, 3, 3, 0); \
        DO_FMA(hh, 4, 0, 1); \
        DO_FMA(hh, 5, 1, 1); \
        DO_FMA(hh, 6, 2, 1); \
        DO_FMA(hh, 7, 3, 1); \
        DO_FMA(hh, 8, 0, 2); \
        DO_FMA(hh, 9, 1, 2); \
        DO_FMA(hh, 10, 2, 2); \
        DO_FMA(hh, 11, 3, 2); \
        DO_FMA(hh, 12, 0, 3); \
        DO_FMA(hh, 13, 1, 3); \
        DO_FMA(hh, 14, 2, 3); \
        DO_FMA(hh, 15, 3, 3); \
    } while (0)

#define UPDATE_C_ROW(i, ii, betaZero) \
    do { \
        if (jrem > 0) \
            if (irem > i) { \
                float val = alpha * c[i / 4].s##ii \
                        + ((betaZero) ? 0 : beta * *C); \
                POST_OP(val); \
                *C = val; \
            } \
        C++; \
    } while (0)

#define UPDATE_C_ROW_2X(i, ii, betaZero) \
    do { \
        if (irem > i) { \
            if (jrem > 0) { \
                float val = alpha * c[i / 4][0].s##ii \
                        + ((betaZero) ? 0 : beta * *(C_ptrs[0])); \
                POST_OP(val); \
                *(C_ptrs[0]) = val; \
            } \
            if (jrem > 16) { \
                float val = alpha * c[i / 4][1].s##ii \
                        + ((betaZero) ? 0 : beta * *(C_ptrs[1])); \
                POST_OP(val); \
                *(C_ptrs[1]) = val; \
            } \
        } \
        C_ptrs[0]++; \
        C_ptrs[1]++; \
    } while (0)

#define UPDATE_C_32_ROW(betaZero) \
    do { \
        UPDATE_C_ROW(0, 0, betaZero); \
        UPDATE_C_ROW(1, 1, betaZero); \
        UPDATE_C_ROW(2, 2, betaZero); \
        UPDATE_C_ROW(3, 3, betaZero); \
        UPDATE_C_ROW(4, 0, betaZero); \
        UPDATE_C_ROW(5, 1, betaZero); \
        UPDATE_C_ROW(6, 2, betaZero); \
        UPDATE_C_ROW(7, 3, betaZero); \
        UPDATE_C_ROW(8, 0, betaZero); \
        UPDATE_C_ROW(9, 1, betaZero); \
        UPDATE_C_ROW(10, 2, betaZero); \
        UPDATE_C_ROW(11, 3, betaZero); \
        UPDATE_C_ROW(12, 0, betaZero); \
        UPDATE_C_ROW(13, 1, betaZero); \
        UPDATE_C_ROW(14, 2, betaZero); \
        UPDATE_C_ROW(15, 3, betaZero); \
        UPDATE_C_ROW(16, 0, betaZero); \
        UPDATE_C_ROW(17, 1, betaZero); \
        UPDATE_C_ROW(18, 2, betaZero); \
        UPDATE_C_ROW(19, 3, betaZero); \
        UPDATE_C_ROW(20, 0, betaZero); \
        UPDATE_C_ROW(21, 1, betaZero); \
        UPDATE_C_ROW(22, 2, betaZero); \
        UPDATE_C_ROW(23, 3, betaZero); \
        UPDATE_C_ROW(24, 0, betaZero); \
        UPDATE_C_ROW(25, 1, betaZero); \
        UPDATE_C_ROW(26, 2, betaZero); \
        UPDATE_C_ROW(27, 3, betaZero); \
        UPDATE_C_ROW(28, 0, betaZero); \
        UPDATE_C_ROW(29, 1, betaZero); \
        UPDATE_C_ROW(30, 2, betaZero); \
        UPDATE_C_ROW(31, 3, betaZero); \
    } while (0)

#define UPDATE_C_16_ROW(betaZero) \
    do { \
        UPDATE_C_ROW_2X(0, 0, betaZero); \
        UPDATE_C_ROW_2X(1, 1, betaZero); \
        UPDATE_C_ROW_2X(2, 2, betaZero); \
        UPDATE_C_ROW_2X(3, 3, betaZero); \
        UPDATE_C_ROW_2X(4, 0, betaZero); \
        UPDATE_C_ROW_2X(5, 1, betaZero); \
        UPDATE_C_ROW_2X(6, 2, betaZero); \
        UPDATE_C_ROW_2X(7, 3, betaZero); \
        UPDATE_C_ROW_2X(8, 0, betaZero); \
        UPDATE_C_ROW_2X(9, 1, betaZero); \
        UPDATE_C_ROW_2X(10, 2, betaZero); \
        UPDATE_C_ROW_2X(11, 3, betaZero); \
        UPDATE_C_ROW_2X(12, 0, betaZero); \
        UPDATE_C_ROW_2X(13, 1, betaZero); \
        UPDATE_C_ROW_2X(14, 2, betaZero); \
        UPDATE_C_ROW_2X(15, 3, betaZero); \
    } while (0)

#ifdef NN
__attribute__((intel_reqd_sub_group_size(16))) // attr:no-format
kernel void
gen9_gemm_nocopy_f32(global float *A, global float *B, global float *C,
        long offset_a, long offset_b, long offset_c, int lda, int ldb, int ldc,
        int m, int n, int k, float alpha, float beta, int last_k_block,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale
#ifdef WITH_K_UNROLL
        ,
        volatile global int *flag, long offset_f) {
#else
) {
#endif

    // clang-format off
    float2 a[4]; // 32 x 4  block of A, 4x 32x1 block accesses   [col major]
    float4 b;    // 4  x 16 block of B, 1x 4x16 scattered access [row major]
    float4 c[8]; // 32 x 16 block of C, 8x 4x16 scattered access [row major]
    // clang-format on
    int idM = get_global_id(1);
    int idN = get_global_id(0);
    int lid = get_sub_group_local_id();
    int idK = get_global_id(2);
    int nku = get_global_size(2);

    int i0 = idM * 32;
    int j0 = idN;
    int irem = m - i0;
    int jrem = n - j0;
    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;

    int last_k_unroll = (idK == nku - 1);

#ifdef WITH_K_UNROLL
    int k0 = idK * UNROLL_K;
    int kt = k - k0;
    if (kt < 0) kt = 0;
    if (kt > UNROLL_K) kt = UNROLL_K;
    A += offset_a + i0 + k0 * lda;
    B += offset_b + j0 * ldb + k0;
    k = kt;
#else
    A += offset_a + i0;
    B += offset_b + j0 * ldb;
#endif
    C += offset_c + i0 + j0 * ldc;

    global float *A_cols[4] = {A, A + lda, A + 2 * lda, A + 3 * lda};

    int ldax4 = lda << 2;
    int ldbx4 = ldb << 2;

    for (int z = 0; z < 8; z++)
        c[z] = 0.f;

#ifdef WITH_K_UNROLL
    flag += offset_f + idM;
    if (idK == 0 && lid == 0) *flag = 0;
#endif

#ifndef ALLOW_READ_OVERRUNS
    if (irem >= 32 && sub_group_broadcast(jrem, 0) >= 16) {
#endif
        // Non-remainder kernel.
        for (int h = 0; h < (k >> 2); h++) {
            // Load A
            for (int hh = 0; hh < 4; hh++) {
                a[hh] = as_float2(
                        intel_sub_group_block_read2((global uint *)A_cols[hh]));
                A_cols[hh] += ldax4;
            }

            // Load B
            b = vload4(0, B);
            B += 4;

            // FMAs
            FMA_I_LOOP_32_ROW(0);
            FMA_I_LOOP_32_ROW(1);
            FMA_I_LOOP_32_ROW(2);
            FMA_I_LOOP_32_ROW(3);
        }
        int krem = k & 3;
        for (int h = 0; h < krem; h++) {
            a[0] = as_float2(
                    intel_sub_group_block_read2((global uint *)A_cols[0]));
            A_cols[0] += lda;

            b = *B++;

            FMA_I_LOOP_32_ROW(0);
        }
#ifndef ALLOW_READ_OVERRUNS
    } else {
        // Remainder kernel: use masked loads.
        for (int h = 0; h < (k >> 1); h++) {
            for (int hh = 0; hh < 2; hh++) {
                if (irem > lid) a[hh].s0 = A_cols[hh][lid];
                if (irem > (lid + 16)) a[hh].s1 = A_cols[hh][lid + 16];
                A_cols[hh] += (lda << 1);
            }

            if (jrem > 0) b.s01 = vload2(0, B);
            B += 2;

            FMA_I_LOOP_32_ROW(0);
            FMA_I_LOOP_32_ROW(1);
        }

        if (k & 1) {
            if (irem > lid) a[0].s0 = A_cols[0][lid];
            if (irem > (lid + 16)) a[0].s1 = A_cols[0][lid + 16];

            if (jrem > 0) b = *B;

            FMA_I_LOOP_32_ROW(0);
        }
    }
#endif // ALLOW_READ_OVERRUNS

    // Update C.
#ifdef WITH_K_UNROLL
    do {
        read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    } while (*flag != idK);

    if (idK == 0) {
        if (beta == 0)
            UPDATE_C_32_ROW(1);
        else
            UPDATE_C_32_ROW(0);
    } else {
        beta = 1.0;
        UPDATE_C_32_ROW(0);
    }
    if (lid == 0) *flag = idK + 1;
#else
    if (beta == 0)
        UPDATE_C_32_ROW(1);
    else
        UPDATE_C_32_ROW(0);
#endif
}
#endif

#ifdef NT
__attribute__((intel_reqd_sub_group_size(16))) // attr:no-format
kernel void
gen9_gemm_nocopy_f32(global float *A, global float *B, global float *C,
        long offset_a, long offset_b, long offset_c, int lda, int ldb, int ldc,
        int m, int n, int k, float alpha, float beta, int last_k_block,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale
#ifdef WITH_K_UNROLL
        ,
        volatile global int *flag, long offset_f) {
#else
) {
#endif

    // clang-format off
    float2 a[2]; // 32 x 2  block of A, 2x 32x1 block accesses   [col major]
    float b[2];  // 2  x 16 block of B, 2x 1x16 block accesses   [row major]
    float4 c[8]; // 32 x 16 block of C, 8x 4x16 scattered access [row major]
    // clang-format on
    int idM = get_global_id(1);
    int idN = get_global_id(0);
    int lid = get_sub_group_local_id();
    int idK = get_global_id(2);
    int nku = get_global_size(2);

    int i0 = idM * 32;
    int j0 = idN;
    int irem = m - i0;
    int jrem = n - j0;
    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;

    int last_k_unroll = (idK == nku - 1);

#ifdef WITH_K_UNROLL
    int k0 = idK * UNROLL_K;
    int kt = k - k0;
    if (kt < 0) kt = 0;
    if (kt > UNROLL_K) kt = UNROLL_K;
    A += offset_a + i0 + k0 * lda;
    B += offset_b + sub_group_broadcast(j0, 0) + k0 * ldb;
    k = kt;
#else
    A += offset_a + i0;
    B += offset_b + sub_group_broadcast(j0, 0);
#endif
    C += offset_c + i0 + j0 * ldc;

    global float *A_cols[2] = {A, A + lda};
    global float *B_rows[2] = {B, B + ldb};

    int ldax2 = lda << 1;
    int ldbx2 = ldb << 1;

    for (int z = 0; z < 8; z++)
        c[z] = 0.f;

#ifdef WITH_K_UNROLL
    flag += offset_f + idM;
    if (idK == 0 && lid == 0) *flag = 0;
#endif

#ifndef ALLOW_READ_OVERRUNS
    if (irem >= 32 && sub_group_broadcast(jrem, 0) >= 16) {
#endif
        for (int h = 0; h < (k >> 1); h++) {
            // Load A
            for (int hh = 0; hh < 2; hh++) {
                a[hh] = as_float2(
                        intel_sub_group_block_read2((global uint *)A_cols[hh]));
                A_cols[hh] += ldax2;
            }

            // Load B
            for (int hh = 0; hh < 2; hh++) {
                b[hh] = as_float(
                        intel_sub_group_block_read((global uint *)B_rows[hh]));
                B_rows[hh] += ldbx2;
            }

            // FMAs
            FMA_I_LOOP_32_ROW(0);
            FMA_I_LOOP_32_ROW(1);
        }

        int krem = k & 1;
        if (krem > 0) {
            a[0] = as_float2(
                    intel_sub_group_block_read2((global uint *)A_cols[0]));
            b[0] = as_float(
                    intel_sub_group_block_read((global uint *)B_rows[0]));

            FMA_I_LOOP_32_ROW(0);
        }
#ifndef ALLOW_READ_OVERRUNS
    } else {
        // Remainder kernel
        for (int h = 0; h < (k >> 1); h++) {
            // Load A
            for (int hh = 0; hh < 2; hh++) {
                if (irem > lid) a[hh].s0 = A_cols[hh][lid];
                if (irem > (lid + 16)) a[hh].s1 = A_cols[hh][lid + 16];
                A_cols[hh] += ldax2;
            }

            // Load B
            for (int hh = 0; hh < 2; hh++) {
                if (jrem > 0) b[hh] = B_rows[hh][lid];
                B_rows[hh] += ldbx2;
            }

            // FMAs
            FMA_I_LOOP_32_ROW(0);
            FMA_I_LOOP_32_ROW(1);
        }

        int krem = k & 1;
        if (krem > 0) {
            if (irem > lid) a[0].s0 = A_cols[0][lid];
            if (irem > (lid + 16)) a[0].s1 = A_cols[0][lid + 16];

            if (jrem > 0) b[0] = B_rows[0][lid];

            FMA_I_LOOP_32_ROW(0);
        }
    }
#endif // ALLOW_READ_OVERRUNS

    // Update C.
#ifdef WITH_K_UNROLL
    do {
        read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    } while (*flag != idK);

    if (idK == 0) {
        if (beta == 0)
            UPDATE_C_32_ROW(1);
        else
            UPDATE_C_32_ROW(0);
    } else {
        beta = 1.0;
        UPDATE_C_32_ROW(0);
    }
    if (lid == 0) *flag = idK + 1;
#else
    if (beta == 0)
        UPDATE_C_32_ROW(1);
    else
        UPDATE_C_32_ROW(0);
#endif
}
#endif

#ifdef TN
__attribute__((intel_reqd_sub_group_size(16))) // attr:no-format
kernel void
gen9_gemm_nocopy_f32(global float *A, global float *B, global float *C,
        long offset_a, long offset_b, long offset_c, int lda, int ldb, int ldc,
        int m, int n, int k, float alpha, float beta, int last_k_block,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale
#ifdef WITH_K_UNROLL
        ,
        volatile global int *flag, long offset_f) {
#else
) {
#endif

    // clang-format off
    float4 a;       // 16 x 4  block of A, 1x     16x4 scattered [col major]
    float4 b[2];    // 4  x 32 block of B, 2x     4x16 scattered [row major]
    float4 c[4][2]; // 16 x 32 block of C, (4x2)x 4x16 scattered [row major]
    // clang-format on
    int idM = get_global_id(1);
    int idN = get_global_id(0);
    int lid = get_sub_group_local_id();
    int idK = get_global_id(2);
    int nku = get_global_size(2);

    int i0 = idM * 16;
    int j0 = sub_group_broadcast(idN, 0) * 2 + lid;
    int irem = m - i0;
    int jrem = n - j0;
    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;

    int last_k_unroll = (idK == nku - 1);

#ifdef WITH_K_UNROLL
    int k0 = idK * UNROLL_K;
    int kt = k - k0;
    if (kt < 0) kt = 0;
    if (kt > UNROLL_K) kt = UNROLL_K;
    A += offset_a + (i0 + lid) * lda + k0;
    B += offset_b + j0 * ldb + k0;
    k = kt;
#else
    A += offset_a + (i0 + lid) * lda;
    B += offset_b + j0 * ldb;
#endif
    C += offset_c + i0 + j0 * ldc;

    global float *B_ptrs[2] = {B, B + 16 * ldb};

    for (int ii = 0; ii < 4; ii++)
        for (int jj = 0; jj < 2; jj++)
            c[ii][jj] = 0.f;

#ifdef WITH_K_UNROLL
    flag += offset_f + idM;
    if (idK == 0 && lid == 0) *flag = 0;
#endif

    for (int h = 0; h < (k >> 2); h++) {
        // Load A
        if (irem > lid) a = vload4(0, A);
        A += 4;

        // Load B
        for (int hh = 0; hh < 2; hh++) {
            if (jrem > hh * 16) b[hh] = vload4(0, B_ptrs[hh]);
            B_ptrs[hh] += 4;
        }

        // FMAs
        FMA_I_LOOP_16_ROW(0);
        FMA_I_LOOP_16_ROW(1);
        FMA_I_LOOP_16_ROW(2);
        FMA_I_LOOP_16_ROW(3);
    }

    int krem = k & 3;
    for (int h = 0; h < krem; h++) {
        if (irem > lid) a = *A++;

        for (int hh = 0; hh < 2; hh++) {
            if (jrem > hh * 16) b[hh] = *B_ptrs[hh];
            B_ptrs[hh]++;
        }

        FMA_I_LOOP_16_ROW(0);
    }

    // Update C.
    global float *C_ptrs[2] = {C, C + 16 * ldc};

#ifdef WITH_K_UNROLL
    do {
        read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    } while (*flag != idK);

    if (idK == 0) {
        if (beta == 0)
            UPDATE_C_16_ROW(1);
        else
            UPDATE_C_16_ROW(0);
    } else {
        beta = 1.0;
        UPDATE_C_16_ROW(0);
    }
    if (lid == 0) *flag = idK + 1;
#else
    if (beta == 0)
        UPDATE_C_16_ROW(1);
    else
        UPDATE_C_16_ROW(0);
#endif
}
#endif

#ifdef TT
__attribute__((intel_reqd_sub_group_size(16))) // attr:no-format
kernel void
gen9_gemm_nocopy_f32(global float *A, global float *B, global float *C,
        long offset_a, long offset_b, long offset_c, int lda, int ldb, int ldc,
        int m, int n, int k, float alpha, float beta, int last_k_block,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale
#ifdef WITH_K_UNROLL
        ,
        volatile global int *flag, long offset_f) {
#else
) {
#endif

    // clang-format off
    float4 a;       // 16 x 4  block of A, 1x     16x4 scattered [col major]
    float2 b[4];    // 4  x 32 block of B, 4x     1x32 block     [row major]
    float4 c[4][2]; // 16 x 32 block of C, (4x2)x 4x16 scattered [row major]
    // clang-format on

    int idM = get_global_id(1);
    int idN = get_global_id(0);
    int lid = get_sub_group_local_id();
    int idK = get_global_id(2);
    int nku = get_global_size(2);

    int i0 = idM * 16;
    int j0 = sub_group_broadcast(idN, 0) * 2 + lid;
    int irem = m - i0;
    int jrem = n - j0;
    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;

    int last_k_unroll = (idK == nku - 1);

#ifdef WITH_K_UNROLL
    int k0 = idK * UNROLL_K;
    int kt = k - k0;
    if (kt < 0) kt = 0;
    if (kt > UNROLL_K) kt = UNROLL_K;
    A += offset_a + (i0 + lid) * lda + k0;
    B += offset_b + sub_group_broadcast(j0, 0) + k0 * ldb;
    k = kt;
#else
    A += offset_a + (i0 + lid) * lda;
    B += offset_b + sub_group_broadcast(j0, 0);
#endif
    C += offset_c + i0 + j0 * ldc;

    global float *B_rows[4] = {B, B + ldb, B + 2 * ldb, B + 3 * ldb};

    int ldbx4 = ldb << 2;

    for (int ii = 0; ii < 4; ii++)
        for (int jj = 0; jj < 2; jj++)
            c[ii][jj] = 0.f;

#ifdef WITH_K_UNROLL
    flag += offset_f + idM;
    if (idK == 0 && lid == 0) *flag = 0;
#endif

#ifndef ALLOW_READ_OVERRUNS
    if (irem >= 16 && sub_group_broadcast(jrem, 0) >= 32) {
#endif
        for (int h = 0; h < (k >> 2); h++) {
            // Load A
            a = vload4(0, A);
            A += 4;

            // Load B
            for (int hh = 0; hh < 4; hh++) {
                b[hh] = as_float2(
                        intel_sub_group_block_read2((global uint *)B_rows[hh]));
                B_rows[hh] += ldbx4;
            }

            // FMAs
            FMA_I_LOOP_16_ROW(0);
            FMA_I_LOOP_16_ROW(1);
            FMA_I_LOOP_16_ROW(2);
            FMA_I_LOOP_16_ROW(3);
        }

        int krem = k & 3;
        for (int h = 0; h < krem; h++) {
            // Load A
            a = *A++;

            // Load B
            b[0] = as_float2(
                    intel_sub_group_block_read2((global uint *)B_rows[0]));
            B_rows[0] += ldb;

            FMA_I_LOOP_16_ROW(0);
        }
#ifndef ALLOW_READ_OVERRUNS
    } else {
        for (int h = 0; h < (k >> 2); h++) {
            // Load A
            if (irem > lid) a = vload4(0, A);
            A += 4;

            // Load B
            for (int hh = 0; hh < 4; hh++) {
                if (jrem > 0) b[hh].s0 = B_rows[hh][lid];
                if (jrem > 16) b[hh].s1 = B_rows[hh][lid + 16];
                B_rows[hh] += ldbx4;
            }

            // FMAs
            FMA_I_LOOP_16_ROW(0);
            FMA_I_LOOP_16_ROW(1);
            FMA_I_LOOP_16_ROW(2);
            FMA_I_LOOP_16_ROW(3);
        }

        int krem = k & 3;
        for (int h = 0; h < krem; h++) {
            if (irem > lid) a = *A++;

            if (jrem > 0) b[0].s0 = B_rows[0][lid];
            if (jrem > 16) b[0].s1 = B_rows[0][lid + 16];
            B_rows[0] += ldb;

            FMA_I_LOOP_16_ROW(0);
        }
    }
#endif // ALLOW_READ_OVERRUNS

    // Update C.
    global float *C_ptrs[2] = {C, C + 16 * ldc};

#ifdef WITH_K_UNROLL
    do {
        read_mem_fence(CLK_GLOBAL_MEM_FENCE);
    } while (*flag != idK);

    if (idK == 0) {
        if (beta == 0)
            UPDATE_C_16_ROW(1);
        else
            UPDATE_C_16_ROW(0);
    } else {
        beta = 1.0;
        UPDATE_C_16_ROW(0);
    }
    if (lid == 0) *flag = idK + 1;
#else
    if (beta == 0)
        UPDATE_C_16_ROW(1);
    else
        UPDATE_C_16_ROW(0);
#endif
}
#endif
