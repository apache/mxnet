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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#define BLOCK_SIZE OC_BLOCK
#define BLOCKED_DATA_T CONCAT2(DATA_T, BLOCK_SIZE)
#define BLOCKED_READ(ptr) vload16(0, ptr)

// Using for loop instead of vstore16 due incorrect results
#define BLOCKED_WRITE(data, ptr) \
    do { \
        BLOCKED_DATA_T result = data; \
        unroll_for(int _i = 0; _i < BLOCK_SIZE; _i++) { \
            (ptr)[_i] = result[_i]; \
        } \
    } while (0)

#define VECT_SIZE 4
#define VECT_DATA_T CONCAT2(DATA_T, VECT_SIZE)
#define AS_VECT_DATA_T AS_DATA4_T
#define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA4_T

#define OC_OUTER_BLOCK OC_BLOCK
#define IC_OUTER_BLOCK IC_BLOCK

#define WINO_D (WINO_M + WINO_R - 1)

#define VW WINO_IW
#define VH WINO_IH
#define VC WINO_IC

#define MW WINO_OW
#define MH WINO_OH
#define MC WINO_OC

static inline int off_nCdhw16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += n * (C / 16) * D * H * W * 16;
    off += (c / 16) * D * H * W * 16;
    off += d * H * W * 16;
    off += h * W * 16;
    off += w * 16;
    off += c % 16;
    return off;
}

static inline int off_NCdhw16n16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += (n / 16) * (C / 16) * D * H * W * 16 * 16;
    off += (c / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (n % 16) * 16;
    off += (c % 16);
    return off;
}

static inline int off_gIOdhw16i16o(int g, int o, int i, int d, int h, int w,
        int O, int I, int D, int H, int W) {
    int off = 0;
    off += g * (I / 16) * (O / 16) * D * H * W * 16 * 16;
    off += (i / 16) * (O / 16) * D * H * W * 16 * 16;
    off += (o / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (i % 16) * 16;
    off += (o % 16);
    return off;
}

static inline int src_off(int n, int c, int d, int h, int w) {
    if (SRC_W16C) return off_nCdhw16c(n, c, d, h, w, G * IC, 1, IH, IW);
    if (SRC_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * IC, 1, IH, IW);
    return 0;
}

static inline int wei_off(int g, int o, int i, int d, int h, int w) {
    return off_gIOdhw16i16o(g, o, i, d, h, w, OC, IC, 1, KH, KW);
}

static inline int U_off(int o, int i, int h, int z) {

    // Needs to be blocked so that ic overflows to the next row.
    int off = z * KH * WINO_IC * WINO_OC;
    off += h * WINO_IC * WINO_OC;
    off += i * WINO_OC;
    off += o;
    return off;
}

static inline int V_off(int n, int i, int h, int w, int z) {

    // Needs to be blocked so that ic overflows to the next row.
    int off = n * WINO_D * VW * VH * VC;
    off += z * VW * VH * VC;
    off += w * VH * VC;
    off += (h + PH) * VC;
    off += i;
    return off;
}

static inline int M_off(int n, int o, int h, int w, int z) {

    // Needs to be blocked so that ic overflows to the next row.
    int off = n * WINO_D * MW * MH * MC;
    off += z * MW * MH * MC;
    off += w * MH * MC;
    off += h * MC;
    off += o;
    return off;
}

static inline int dst_off(int n, int c, int d, int h, int w) {
    if (DST_W16C) return off_nCdhw16c(n, c, d, h, w, G * OC, 1, OH, OW);
    if (DST_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * OC, 1, OH, OW);
    return 0;
}

__kernel void gen9_wino_wei_transform(
        __global DATA_T *U, const __global DATA_T *weights) {
    const uint weights_tile_width = WINO_M;
    const uint weights_tile_height = 1;
    const uint in_kw = get_global_id(0) * weights_tile_width;
    const uint in_kh = get_global_id(1) * weights_tile_height;

    const uint U_tile_width = WINO_D;
    const uint U_tile_height = 1;

    const uint out_kw = get_global_id(0) * U_tile_width;
    const uint out_kh = get_global_id(1) * U_tile_height;
    const uint ic = get_global_id(2) % WINO_IC;
    const uint oc = get_global_id(2) / WINO_IC;

    uint in_idx = wei_off(0, oc, ic, 0, in_kh, in_kw);
    bool is_valid = ic < IC || oc < OC;

    VECT_DATA_T tile;
    tile.x = is_valid ? weights[in_idx] : 0;
    in_idx += wei_off(0, 0, 0, 0, 0, 1);
    tile.y = is_valid ? weights[in_idx] : 0;
    in_idx += wei_off(0, 0, 0, 0, 0, 1);
    tile.z = is_valid ? weights[in_idx] : 0;

    uint out_idx = U_off(oc, ic, out_kh, out_kw);

    U[out_idx] = tile.x;
    out_idx += U_off(0, 0, 0, 1);
    U[out_idx] = (tile.x + tile.y + tile.z) / 2;
    out_idx += U_off(0, 0, 0, 1);
    U[out_idx] = (tile.x - tile.y + tile.z) / 2;
    out_idx += U_off(0, 0, 0, 1);
    U[out_idx] = tile.z;
}

__kernel void gen9_wino_src_transform(
        __global DATA_T *V, const __global DATA_T *src) {
    const uint tile_id_x = get_global_id(0);
    const uint tile_id_y = get_global_id(1);
    const uint stride_x = WINO_M;
    const uint stride_y = 1;
    const uint iw = tile_id_x * stride_x - PW;
    const uint ih = tile_id_y * stride_y - PH;
    const uint ic = (get_global_id(2) % (WINO_IC / IC_BLOCK)) * IC_BLOCK;
    const uint n = get_global_id(2) / (WINO_IC / IC_BLOCK);

    const bool w0 = iw < 0 || iw >= IW;
    const bool w1 = iw + 1 < 0 || iw + 1 >= IW;
    const bool w2 = iw + 2 < 0 || iw + 2 >= IW;
    const bool w3 = iw + 3 < 0 || iw + 3 >= IW;
    const bool h0 = ih < 0 || ih >= IH || ic > IC;

    BLOCKED_DATA_T d0, d1, d2, d3;
    int in_idx = src_off(n, ic, 0, ih, iw);
    d0 = (h0 || w0) ? 0 : BLOCKED_READ(&src[in_idx]);
    in_idx += src_off(0, 0, 0, 0, 1);
    d1 = (h0 || w1) ? 0 : BLOCKED_READ(&src[in_idx]);
    in_idx += src_off(0, 0, 0, 0, 1);
    d2 = (h0 || w2) ? 0 : BLOCKED_READ(&src[in_idx]);
    in_idx += src_off(0, 0, 0, 0, 1);
    d3 = (h0 || w3) ? 0 : BLOCKED_READ(&src[in_idx]);

    int out_idx = V_off(n, ic, ih, tile_id_x, 0);
    BLOCKED_WRITE(d0 - d2, &V[out_idx]);
    out_idx += V_off(0, 0, -PH, 0, 1);
    BLOCKED_WRITE(d1 + d2, &V[out_idx]);
    out_idx += V_off(0, 0, -PH, 0, 1);
    BLOCKED_WRITE(-d1 + d2, &V[out_idx]);
    out_idx += V_off(0, 0, -PH, 0, 1);
    BLOCKED_WRITE(d1 - d3, &V[out_idx]);
}

__kernel void gen9_wino_dst_transform(__global DATA_T *dst,
        const __global DATA_T *M, const __global DATA_T *bias POST_OP_ARGS) {

    const uint tile_id_x = get_global_id(0);
    const uint tile_id_y = get_global_id(1);
    const uint dst_tile_width_x = WINO_M;
    const uint dst_tile_width_y = 1;
    const uint ow = tile_id_x * dst_tile_width_x;
    const uint oh = tile_id_y * dst_tile_width_y;
    const uint oc = (get_global_id(2) % (OC / OC_BLOCK)) * OC_BLOCK;
    const uint n = get_global_id(2) / (OC / OC_BLOCK);

    BLOCKED_DATA_T m0, m1, m2, m3;
    int M_idx = M_off(n, oc, tile_id_y, tile_id_x, 0);

    m0 = BLOCKED_READ(&M[M_idx]);
    M_idx += M_off(0, 0, 0, 0, 1);
    m1 = BLOCKED_READ(&M[M_idx]);
    M_idx += M_off(0, 0, 0, 0, 1);
    m2 = BLOCKED_READ(&M[M_idx]);
    M_idx += M_off(0, 0, 0, 0, 1);
    m3 = BLOCKED_READ(&M[M_idx]);

    BLOCKED_DATA_T C1 = m0 + m1 + m2;
    BLOCKED_DATA_T C2 = m1 - m2 - m3;

    if (WITH_BIAS || WITH_POST_OP) {
        const int c_size = WINO_M * OC_BLOCK;
        DATA_T C[c_size];
        BLOCKED_WRITE(C1, &C[0]);
        BLOCKED_WRITE(C2, &C[OC_BLOCK]);
        if (WITH_BIAS) {
            for (int oc_outer = 0; oc_outer < OC_BLOCK; oc_outer++) {
                for (int ow_block = 0; ow_block < WINO_M; ow_block++) {
                    const int c_off = ow_block * OC_BLOCK + oc_outer;
                    const int bc_off = oc + oc_outer;
                    C[c_off] += (OC_WO_PADDING % OC_BLOCK == 0
                                        || bc_off < OC_WO_PADDING)
                            ? bias[bc_off]
                            : DATA_ZERO;
                }
            }
        }

        DATA_T S[c_size];
        if (WITH_SUM) {
            BLOCKED_DATA_T S1, S2;
            int dst_idx = dst_off(n, oc, 0, oh, ow);
            S1 = BLOCKED_READ(&dst[dst_idx]);
            if (OW % WINO_M == 0 || ow < OW - 1) {
                dst_idx += dst_off(0, 0, 0, 0, 1);
                S2 = BLOCKED_READ(&dst[dst_idx]);
            } else {
                S2 = 0;
            }
            BLOCKED_WRITE(S1, &S[0]);
            BLOCKED_WRITE(S2, &S[OC_BLOCK]);
        }
        for (int didx = 0; didx < c_size; ++didx) {
            float accum = CONVERT_FLOAT_T(C[didx]);
            float sum = CONVERT_FLOAT_T(S[didx]);
            int po_oc = oc + c_size % OC_BLOCK;
            APPLY_POST_OPS(C, DATA_T, S, DATA_T, n, 1, po_oc, 1, 0, 1, 0, 1, 0,
                    1, 0, 1);
            C[didx] = TO_DATA_T(accum);
        }

        C1 = BLOCKED_READ(&C[0]);
        C2 = BLOCKED_READ(&C[OC_BLOCK]);
    }

    int dst_idx = dst_off(n, oc, 0, oh, ow);
    BLOCKED_WRITE(C1, &dst[dst_idx]);
    if (OW % WINO_M == 0 || ow < OW - 1) {
        dst_idx += dst_off(0, 0, 0, 0, 1);
        BLOCKED_WRITE(C2, &dst[dst_idx]);
    }
}

__attribute__((reqd_work_group_size(8, 1, 1))) __kernel void gen9_wino_conv_fwd(
        __global DATA_T *M, const __global DATA_T *V,
        const __global DATA_T *U_param) {
    const int VH_SIZE_VECT = V_off(0, 0, 1 - PH, 0, 0) / VECT_SIZE;
    const int MH_SIZE_VECT = M_off(0, 0, 1, 0, 0) / VECT_SIZE;
    const int U_IC_SIZE_VECT = U_off(0, 1, 0, 0) / VECT_SIZE;

    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int group_z = get_group_id(2);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int local_z = get_local_id(2);

    const int no_of_tiles_x = MW;
    const int no_of_tiles_y = MH;

    const int ow = (group_y * OH_BLOCK) / no_of_tiles_y;
    const int oh = (group_y * OH_BLOCK) % no_of_tiles_y;
    const int oc = group_x * WINO_OC_BLOCK
            + local_x * VECT_SIZE; // Divide oc across local work group
    const int n = group_z / WINO_D;
    const int tile_w_offset = group_z % WINO_D;

    const int ih = oh - PH;
    const int iw = ow;
    const int ic = local_x * VECT_SIZE; // Divide ic across local work group

    // Result tile is OH_BLOCK output height x WINO_OC_BLOCK output channels
    // OH_BLOCK = 8, we have 1 row of work-items, so we need 8/1 = 8 results down
    // WINO_OC_BLOCK = 32, output channels is spread across local id, so we need
    // to compute 32/8 = 4 output channels in each thread;

    // Initialize results tile
    VECT_DATA_T M0 = (VECT_DATA_T)(0.f);
    VECT_DATA_T M1 = (VECT_DATA_T)(0.f);
    VECT_DATA_T M2 = (VECT_DATA_T)(0.f);
    VECT_DATA_T M3 = (VECT_DATA_T)(0.f);
    VECT_DATA_T M4 = (VECT_DATA_T)(0.f);
    VECT_DATA_T M5 = (VECT_DATA_T)(0.f);
    VECT_DATA_T M6 = (VECT_DATA_T)(0.f);
    VECT_DATA_T M7 = (VECT_DATA_T)(0.f);

    const int M_idx = M_off(n, oc, oh, ow, tile_w_offset);
    __global VECT_DATA_T *dst = (__global VECT_DATA_T *)(M + M_idx);

    const int V_idx = V_off(n, ic, ih, iw, tile_w_offset);
    const __global VECT_DATA_T *V_tile = (__global VECT_DATA_T *)(V + V_idx);

    const int U_idx = U_off(oc, 0, 0, tile_w_offset);
    const __global VECT_DATA_T *U_tile
            = (__global VECT_DATA_T *)(U_param + U_idx);

    VECT_DATA_T a;

    // Implementation relies on IC overflowing to next row in U, V and M
    for_(int kh = 0; kh < KH; kh++)
    for (int ic_idx = 0; ic_idx < WINO_IC; ic_idx += WINO_IC_BLOCK) {

        // V_tile is OH_BLOCK input height x WINO_IC_BLOCK input channels
        // OH_BLOCK = 8, so we need 8/1 = 8 results
        // WINO_IC_BLOCK = 32, input channels is spread across local id, so we
        // need to load 32/8 = 4 input channels in each thread;

        // Load V tile
        const VECT_DATA_T V0 = V_tile[0 * VH_SIZE_VECT];
        const VECT_DATA_T V1 = V_tile[1 * VH_SIZE_VECT];
        const VECT_DATA_T V2 = V_tile[2 * VH_SIZE_VECT];
        const VECT_DATA_T V3 = V_tile[3 * VH_SIZE_VECT];
        const VECT_DATA_T V4 = V_tile[4 * VH_SIZE_VECT];
        const VECT_DATA_T V5 = V_tile[5 * VH_SIZE_VECT];
        const VECT_DATA_T V6 = V_tile[6 * VH_SIZE_VECT];
        const VECT_DATA_T V7 = V_tile[7 * VH_SIZE_VECT];

#define DOT_PRODUCT(_i, _j) \
    do { \
        a = AS_VECT_DATA_T( \
                intel_sub_group_shuffle(AS_VECT_BLOCK_DATA_T(V##_i), _j)); \
        M##_i = mad(a.x, U0, mad(a.y, U1, mad(a.z, U2, mad(a.w, U3, M##_i)))); \
    } while (0)

        // We need WINO_IC_BLOCK/VECT_SIZE iterations.
        // WINO_IC_BLOCK = 32, VECT_SIZE = 4, so 32/4 = 8 iterations.
        unroll_for(int j = 0; j < WINO_IC_BLOCK / VECT_SIZE; j++) {
            const VECT_DATA_T U0 = U_tile[0];
            U_tile += U_IC_SIZE_VECT;
            const VECT_DATA_T U1 = U_tile[0];
            U_tile += U_IC_SIZE_VECT;
            const VECT_DATA_T U2 = U_tile[0];
            U_tile += U_IC_SIZE_VECT;
            const VECT_DATA_T U3 = U_tile[0];
            U_tile += U_IC_SIZE_VECT;
            DOT_PRODUCT(0, j);
            DOT_PRODUCT(1, j);
            DOT_PRODUCT(2, j);
            DOT_PRODUCT(3, j);
            DOT_PRODUCT(4, j);
            DOT_PRODUCT(5, j);
            DOT_PRODUCT(6, j);
            DOT_PRODUCT(7, j);
        }

#undef DOT_PRODUCT

        V_tile += WINO_IC_BLOCK / VECT_SIZE;
    }

    dst[0] = M0;
    dst += MH_SIZE_VECT;
    dst[0] = M1;
    dst += MH_SIZE_VECT;
    dst[0] = M2;
    dst += MH_SIZE_VECT;
    dst[0] = M3;
    dst += MH_SIZE_VECT;
    dst[0] = M4;
    dst += MH_SIZE_VECT;
    dst[0] = M5;
    dst += MH_SIZE_VECT;
    dst[0] = M6;
    dst += MH_SIZE_VECT;
    dst[0] = M7;
}
