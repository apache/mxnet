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

#define _BLOCK_READ8(ptr) \
    AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ4(ptr) \
    AS_DATA4_T(BLOCK_READ4((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ2(ptr) \
    AS_DATA2_T(BLOCK_READ2((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ(ptr) \
    AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)(ptr)))

#define _BLOCK_WRITE8(ptr, v) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)(ptr), AS_BLOCK_DATA8_T(v))
#define _BLOCK_WRITE4(ptr, v) \
    BLOCK_WRITE4((__global BLOCK_DATA_T *)(ptr), AS_BLOCK_DATA4_T(v))
#define _BLOCK_WRITE2(ptr, v) \
    BLOCK_WRITE2((__global BLOCK_DATA_T *)(ptr), AS_BLOCK_DATA2_T(v))
#define _BLOCK_WRITE(ptr, v) \
    BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), AS_BLOCK_DATA_T(v))

#define IS_3D (OD > 1)
#define IS_1STCONV (IC == 3)

#define HAS_PAD_D (PD > 0 || (OD - 1) * SD - PD + (KD - 1) * (1 + DD) >= ID)
#define HAS_PAD_H (PH > 0 || (OH - 1) * SH - PH + (KH - 1) * (1 + DH) >= IH)
#define HAS_PAD_W (PW > 0 || (OW - 1) * SW - PW + (KW - 1) * (1 + DW) >= IW)

#define ENABLE_SRC_BUF (MB_BLOCK == 1 && KW >= 3)

#define W_VEC (IS_1STCONV && ENABLE_SRC_BUF && KW >= 5 && SW <= 3)
#define C_VEC (!W_VEC)

#define IC_INNER (C_VEC ? (IC < 16 ? IC : 16) : 1)

#define IC_OUTER (IC_BLOCK / IC_INNER)
#define OC_OUTER (OC_BLOCK / 16)
#define IW_BLOCK (SW * (OW_BLOCK - 1) + (KW - 1) * (1 + DW) + 1)

#define OW_INNER (C_VEC ? 1 : 16)
#define IW_INNER OW_INNER

#define OW_OUTER ((OW_BLOCK + OW_INNER - 1) / OW_INNER)
#define IW_OUTER ((IW_BLOCK + IW_INNER - 1) / IW_INNER)

#define C_SIZE (MB_BLOCK * OC_OUTER * OW_BLOCK)

#if OW_BLOCK >= 32
#error "Block is too big for unrolled_read and unrolled_write."
#endif

int off_ncdhw(int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += n * C * D * H * W;
    off += c * D * H * W;
    off += d * H * W;
    off += h * W;
    off += w;
    return off;
}
int off_ndhwc(int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += n * D * H * W * C;
    off += d * H * W * C;
    off += h * W * C;
    off += w * C;
    off += c;
    return off;
}

int off_nCdhw16c(
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

int off_NCdhw16n16c(
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

int off_gOdhwi16o(int g, int o, int i, int d, int h, int w, int O, int I, int D,
        int H, int W) {
    int off = 0;
    off += g * (O / 16) * D * H * W * I * 16;
    off += (o / 16) * D * H * W * I * 16;
    off += d * H * W * I * 16;
    off += h * W * I * 16;
    off += w * I * 16;
    off += i * 16;
    off += (o % 16);
    return off;
}

int off_gOIdhw16i16o(int g, int o, int i, int d, int h, int w, int O, int I,
        int D, int H, int W) {
    int off = 0;
    off += g * (O / 16) * (I / 16) * D * H * W * 16 * 16;
    off += (o / 16) * (I / 16) * D * H * W * 16 * 16;
    off += (i / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (i % 16) * 16;
    off += (o % 16);
    return off;
}

int off_gIOdhw16i16o(int g, int o, int i, int d, int h, int w, int O, int I,
        int D, int H, int W) {
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

int src_off(int n, int c, int d, int h, int w) {
    if (SRC_NCHW) return off_ncdhw(n, c, d, h, w, G * IC, ID, IH, IW);
    if (SRC_NHWC) return off_ndhwc(n, c, d, h, w, G * IC, ID, IH, IW);
    if (SRC_W16C) return off_nCdhw16c(n, c, d, h, w, G * IC, ID, IH, IW);
    if (SRC_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * IC, ID, IH, IW);
    return 0;
}

int wei_off(int g, int o, int i, int d, int h, int w) {
    if (WEI_I16O) return off_gOdhwi16o(g, o, i, d, h, w, OC, IC, KD, KH, KW);
    if (WEI_16I16O)
        return off_gOIdhw16i16o(g, o, i, d, h, w, OC, IC, KD, KH, KW);
    if (WEI_16I16O_FLIPPED)
        return off_gIOdhw16i16o(g, o, i, d, h, w, OC, IC, KD, KH, KW);
    return 0;
}

int dst_off(int n, int c, int d, int h, int w) {
    if (DST_W16C) return off_nCdhw16c(n, c, d, h, w, G * OC, OD, OH, OW);
    if (DST_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * OC, OD, OH, OW);
    return 0;
}

// Layout is:
// - cwn[16c] for NCdhw16n16c (16c is mapped to sub-group)
// - ncw[16c] for others (16c is mapped to sub-group)
int src_idx_c_vec(int mb_block, int ic_outer, int ow_outer) {
    if (SRC_16N16C)
        return ic_outer * OW_BLOCK * MB_BLOCK + ow_outer * MB_BLOCK + mb_block;
    return mb_block * IC_OUTER * OW_BLOCK + ic_outer * OW_BLOCK + ow_outer;
}

// Layout is ncW[16w] (16w is mapped to sub-group).
int src_idx_w_vec(int mb_block, int ic_outer, int ow_outer) {
    return mb_block * IC_OUTER * OW_OUTER + ic_outer * OW_OUTER + ow_outer;
}

int src_idx(int mb_block, int ic_outer, int ow_outer) {
    if (C_VEC) return src_idx_c_vec(mb_block, ic_outer, ow_outer);
    return src_idx_w_vec(mb_block, ic_outer, ow_outer);
}

// Same as src_idx_c_vec but contains contiguous input spatial (even unused).
int src_buf_idx_c_vec(int mb_block, int ic_outer, int iw_outer) {
    if (SRC_16N16C)
        return mb_block * IC_OUTER * IW_OUTER + ic_outer * IW_OUTER + iw_outer;
    return ic_outer * IW_OUTER * MB_BLOCK + iw_outer * MB_BLOCK + mb_block;
}

// Same as src_idx_w_vec but contains contiguous input spatial (even unused).
int src_buf_idx_w_vec(int mb_block, int ic_outer, int iw_outer) {
    return mb_block * IC_OUTER * IW_OUTER + ic_outer * IW_OUTER + iw_outer;
}

int src_buf_idx(int mb_block, int ic_outer, int iw_outer) {
    if (C_VEC) return src_buf_idx_c_vec(mb_block, ic_outer, iw_outer);
    return src_buf_idx_w_vec(mb_block, ic_outer, iw_outer);
}

// Layout is IO16i[16o] (16o is mapped to sub-group).
int wei_idx_c_vec(int oc_outer, int ic_outer) {
    return ic_outer * OC_OUTER * IC_INNER + oc_outer * IC_INNER;
}

// Layout is Oi[16o] (16o is mapped to sub-group).
int wei_idx_w_vec(int oc_outer, int ic_outer) {
    return oc_outer * IC_OUTER + ic_outer;
}

int wei_idx(int oc_outer, int ic_outer) {
    if (C_VEC) return wei_idx_c_vec(oc_outer, ic_outer);
    return wei_idx_w_vec(oc_outer, ic_outer);
}

// Layout is:
// - cwn[16c] for NCdhw16n16c (16c is mapped to sub-group)
// - ncw[16c] for nCdhw16c (16c is mapped to sub-group)
int dst_idx(int mb_block, int oc_outer, int ow_block) {
    if (DST_16N16C)
        return oc_outer * OW_BLOCK * MB_BLOCK + ow_block * MB_BLOCK + mb_block;
    return mb_block * OC_OUTER * OW_BLOCK + oc_outer * OW_BLOCK + ow_block;
}

#define copy(dst, src, n) \
    do { \
        for (int i = 0; i < (n); i++) \
            (dst)[i] = (src)[i]; \
    } while (0)

// block[i].j = ptr[i * 16 + j], 0 <= i < n, 0 <= j < 16
#define unrolled_read(n, block, ptr) \
    do { \
        if ((n)&16) { \
            *((DATA8_T *)(block)) = _BLOCK_READ8((ptr)); \
            *((DATA8_T *)((block) + 8)) = _BLOCK_READ8((ptr) + 8 * 16); \
        } \
        if ((n)&8) \
            *((DATA8_T *)((block) + ((n) & ~15))) \
                    = _BLOCK_READ8((ptr) + ((n) & ~15) * 16); \
        if ((n)&4) \
            *((DATA4_T *)((block) + ((n) & ~7))) \
                    = _BLOCK_READ4((ptr) + ((n) & ~7) * 16); \
        if ((n)&2) \
            *((DATA2_T *)((block) + ((n) & ~3))) \
                    = _BLOCK_READ2((ptr) + ((n) & ~3) * 16); \
        if ((n)&1) \
            *((block) + ((n) & ~1)) = _BLOCK_READ((ptr) + ((n) & ~1) * 16); \
    } while (0)

// block[0].i = ptr[i * stride], 0 <= i < n
// block[0].i = 0, n <= i < 16
#define strided_read(block, ptr, n, stride) \
    do { \
        if ((n) == 16 && (stride) == 1) { \
            (block)[0] = _BLOCK_READ((ptr)); \
        } else { \
            int local_id = get_local_id(0); \
            (block)[0] = (local_id < (n)) ? (ptr)[local_id * (stride)] : 0; \
        } \
    } while (0)

// ptr[i * 16 + j] = block[i].j, 0 <= i < n, 0 <= j < 16
#define unrolled_write(n, block, ptr) \
    do { \
        if ((n)&16) { \
            _BLOCK_WRITE8((ptr), *(DATA8_T *)((block))); \
            _BLOCK_WRITE8((ptr) + 8 * 16, *(DATA8_T *)((block) + 8)); \
        } \
        if ((n)&8) \
            _BLOCK_WRITE8((ptr) + ((n) & ~15) * 16, \
                    *(DATA8_T *)((block) + ((n) & ~15))); \
        if ((n)&4) \
            _BLOCK_WRITE4((ptr) + ((n) & ~7) * 16, \
                    *(DATA4_T *)((block) + ((n) & ~7))); \
        if ((n)&2) \
            _BLOCK_WRITE2((ptr) + ((n) & ~3) * 16, \
                    *(DATA2_T *)((block) + ((n) & ~3))); \
        if ((n)&1) \
            _BLOCK_WRITE((ptr) + ((n) & ~1) * 16, *((block) + ((n) & ~1))); \
    } while (0)

// Supports C vectorization only.
#define multiply_row(C, A, B, mb_block, oc_outer, ic_outer, ow_outer) \
    do { \
        int b_off = wei_idx((oc_outer), (ic_outer)); \
        int c_off = dst_idx((mb_block), (oc_outer), (ow_outer)); \
        for (int ic_inner = 0; ic_inner < min(IC_WO_PADDING, IC_INNER); \
                ic_inner++) { \
            (C)[c_off] = fma(intel_sub_group_shuffle((A), ic_inner), \
                    (B)[b_off + ic_inner], (C)[c_off]); \
        } \
    } while (0)

// Supports C vectorization only.
#define read_src_and_multiply_w16c_dense(ptr, iw, kw, C, B) \
    do { \
        for (int mb_block = 0; mb_block < MB_BLOCK; mb_block++) \
            for (int ic_outer = 0; ic_outer < IC_OUTER; ic_outer++) { \
                int idx = src_idx(mb_block, ic_outer, 0); \
                for (int ow_block = 0; ow_block < OW_BLOCK; ow_block += 8) { \
                    int ow_bound = min(8, OW_BLOCK - ow_block); \
                    DATA_T A[8]; \
                    int off = src_off(mb_block, ic_outer * IC_INNER, 0, 0, \
                            ow_block + (kw) * (1 + DW)); \
                    unrolled_read(ow_bound, A, &(ptr)[off]); \
                    for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) { \
                        for (int i = 0; i < ow_bound; i++) { \
                            multiply_row((C), A[i], (B), mb_block, oc_outer, \
                                    ic_outer, ow_block + i); \
                        } \
                    } \
                } \
            } \
    } while (0)

// Supports C vectorization only.
#define read_src_and_multiply_16n16c(ptr, iw, kw, do_w_check, C, B) \
    do { \
        for (int ow_block = 0; ow_block < OW_BLOCK; ow_block++) { \
            int iw_off = ow_block * SW + (kw) * (1 + DW); \
            if ((do_w_check) && HAS_PAD_W \
                    && ((iw) + iw_off < 0 || (iw) + iw_off >= IW)) \
                continue; \
            for (int ic_outer = 0; ic_outer < IC_OUTER; ic_outer++) \
                __attribute__((opencl_unroll_hint)) /*  attr:no-format */ \
                        for (int mb_block = 0; mb_block < MB_BLOCK; \
                                mb_block += 8) { \
                    int mb_bound = min(8, MB_BLOCK - mb_block); \
                    DATA_T A[8]; \
                    int off = src_off( \
                            mb_block, ic_outer * IC_INNER, 0, 0, iw_off); \
                    unrolled_read(mb_bound, A, &(ptr)[off]); \
                    for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) { \
                        for (int i = 0; i < mb_bound; i++) { \
                            multiply_row((C), A[i], (B), mb_block + i, \
                                    oc_outer, ic_outer, ow_block); \
                        } \
                    } \
                } \
        } \
    } while (0)

// Supports C vectorization only.
#define read_src_and_multiply_common(ptr, iw, kw, do_w_check, C, B) \
    do { \
        for (int i = 0; i < OW_BLOCK; i++) { \
            int iw_off = i * SW + (kw) * (1 + DW); \
            if ((do_w_check) && HAS_PAD_W \
                    && ((iw) + iw_off < 0 || (iw) + iw_off >= IW)) \
                continue; \
            for (int ic_outer = 0; ic_outer < IC_OUTER; ic_outer++) { \
                for (int mb_block = 0; mb_block < MB_BLOCK; mb_block++) { \
                    int off = src_off( \
                            mb_block, ic_outer * IC_INNER, 0, 0, iw_off); \
                    DATA_T A; \
                    strided_read(&A, &(ptr)[off], IC_INNER, \
                            src_off(0, 1, 0, 0, 0)); \
                    for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) { \
                        int b_off = wei_idx(oc_outer, ic_outer); \
                        int c_off = dst_idx(mb_block, oc_outer, i); \
                        for (int ic_inner = 0; \
                                ic_inner < min(IC_WO_PADDING, IC_INNER); \
                                ic_inner++) { \
                            (C)[c_off] = fma( \
                                    intel_sub_group_shuffle(A, ic_inner), \
                                    (B)[b_off + ic_inner], (C)[c_off]); \
                        } \
                    } \
                } \
            } \
        } \
    } while (0)

// Read MB_BLOCK x IC_BLOCK x OW_BLOCK block of src.
// Supports C vectorization only.
#define read_src_and_multiply(ptr, iw, kw, do_w_check, C, B) \
    do { \
        if (SRC_W16C && (!(do_w_check) || (!HAS_PAD_W && SW == 1))) { \
            read_src_and_multiply_w16c_dense((ptr), (iw), (kw), (C), (B)); \
        } else if (SRC_16N16C) { \
            read_src_and_multiply_16n16c( \
                    (ptr), (iw), (kw), (do_w_check), (C), (B)); \
        } else { \
            read_src_and_multiply_common( \
                    (ptr), (iw), (kw), (do_w_check), (C), (B)); \
        } \
    } while (0)

#define read_src_buf(buf, ptr, iw) \
    do { \
        for (int iw_outer = 0; iw_outer < IW_OUTER; iw_outer++) { \
            int iw_inner = (C_VEC ? 0 : get_local_id(0)); \
            int iw_block = iw_outer * IW_INNER + iw_inner; \
            if (HAS_PAD_W && ((iw) + iw_block < 0 || (iw) + iw_block >= IW)) \
                continue; \
            for (int ic_outer = 0; ic_outer < IC_OUTER; ic_outer++) { \
                for (int mb_block = 0; mb_block < MB_BLOCK; mb_block++) { \
                    int off = src_off( \
                            mb_block, ic_outer * IC_INNER, 0, 0, iw_block); \
                    int idx = src_buf_idx(mb_block, ic_outer, iw_outer); \
                    if (C_VEC) { \
                        strided_read(&(buf)[idx], &(ptr)[off], IC_INNER, \
                                src_off(0, 1, 0, 0, 0)); \
                    } else { \
                        (buf)[idx] = (ptr)[off]; \
                    } \
                } \
            } \
        } \
    } while (0)

// Read IC_BLOCK x OC_BLOCK block of weights.
#define read_wei_block(block, ptr) \
    do { \
        for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) { \
            int ic_bound = min(IC_WO_PADDING, IC_BLOCK); \
            for (int ic_block = 0; ic_block < ic_bound; ic_block += 16) { \
                int off = wei_off(0, oc_outer * 16, ic_block, 0, 0, 0); \
                int idx = wei_idx(oc_outer, ic_block); \
                unrolled_read(min(16, ic_bound - ic_block), &(block)[idx], \
                        &(ptr)[off]); \
            } \
        } \
    } while (0)

#define read_wei_and_multiply_c_vec(wei, kw, C, A) \
    do { \
        for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) { \
            int ic_bound1 = min(IC_WO_PADDING, IC_BLOCK); \
            for (int ic_block = 0; ic_block < ic_bound1; ic_block += 8) { \
                int ic_bound2 = min(8, ic_bound1 - ic_block); \
                int off = wei_off(0, oc_outer * 16, ic_block, 0, 0, 0); \
                DATA_T B[8]; \
                unrolled_read(ic_bound2, B, &(wei)[off]); \
                for (int mb_block = 0; mb_block < MB_BLOCK; mb_block++) { \
                    for (int ow_block = 0; ow_block < OW_BLOCK; ow_block++) { \
                        int iw_off = ow_block * SW + (kw) * (1 + DW); \
                        int buf_idx = src_buf_idx(mb_block, 0, iw_off); \
                        int c_off = dst_idx(mb_block, oc_outer, ow_block); \
                        for (int i = 0; i < ic_bound2; i++) { \
                            (C)[c_off] \
                                    = fma(intel_sub_group_shuffle( \
                                                  (A)[buf_idx], ic_block + i), \
                                            B[i], (C)[c_off]); \
                        } \
                    } \
                } \
            } \
        } \
    } while (0)

DATA_T shuffle_a_value(int mb_block, int ic_block, int ow_outer, int ow_inner,
        int kw, const DATA_T *A) {
    int iw_off0 = ow_outer * OW_INNER * SW + kw * (1 + DW);
    int iw_outer0 = iw_off0 / IW_INNER;
    int buf_idx0 = src_buf_idx(mb_block, ic_block, iw_outer0);
    int iw_off = iw_off0 + ow_inner * SW;
    int iw_outer = iw_off / IW_INNER;
    DATA4_T A_shuf = 0;
    for (int i = 0; i < SW + 1; i++) {
        A_shuf[i] = (iw_outer0 + i < IW_OUTER) ? A[buf_idx0 + i] : 0;
    }
    A_shuf = AS_DATA4_T(intel_sub_group_shuffle(
            AS_BLOCK_DATA4_T(A_shuf), iw_off % IW_INNER));
    return A_shuf[iw_outer - iw_outer0];
}

#define read_wei_and_multiply_w_vec(wei, kw, C, A) \
    do { \
        for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) { \
            int ic_bound1 = min(IC_WO_PADDING, IC_BLOCK); \
            for (int ic_block = 0; ic_block < ic_bound1; ic_block += 8) { \
                int ic_bound2 = min(8, ic_bound1 - ic_block); \
                int off = wei_off(0, oc_outer * 16, ic_block, 0, 0, 0); \
                DATA_T B[8]; \
                unrolled_read(ic_bound2, B, &(wei)[off]); \
                for (int mb_block = 0; mb_block < MB_BLOCK; mb_block++) \
                    /*  IC_INNER is 1 with W vectorization. */ \
                    for (int ic_inner = 0; ic_inner < ic_bound2; ic_inner++) \
                        for (int ow_outer = 0; ow_outer < OW_OUTER; \
                                ow_outer++) { \
                            int ow_bound = min( \
                                    OW_INNER, OW_BLOCK - ow_outer * OW_INNER); \
                            for (int i = 0; i < ow_bound; i++) { \
                                DATA_T A_val = shuffle_a_value(mb_block, \
                                        ic_block + ic_inner, ow_outer, i, \
                                        (kw), (A)); \
                                int c_off = dst_idx(mb_block, oc_outer, \
                                        ow_outer * OW_INNER + i); \
                                (C)[c_off] = fma(A_val, \
                                        B[ic_block + ic_inner], (C)[c_off]); \
                            } \
                        } \
            } \
        } \
    } while (0)

#define read_wei_and_multiply(wei, kw, C, A) \
    do { \
        if (W_VEC) \
            read_wei_and_multiply_w_vec((wei), (kw), (C), (A)); \
        else \
            read_wei_and_multiply_c_vec((wei), (kw), (C), (A)); \
    } while (0)

// Read MB_BLOCK x OC_BLOCK x OW_BLOCK block of dst.
#define read_dst_block(block, ptr, ow) \
    do { \
        int ow_bound \
                = (OW % OW_BLOCK == 0) ? OW_BLOCK : min(OW_BLOCK, OW - (ow)); \
        for (int mb_block = 0; mb_block < MB_BLOCK; mb_block++) \
            for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) \
                for (int ow_block = 0; ow_block < ow_bound; ow_block++) { \
                    int off = dst_off( \
                            mb_block, oc_outer * 16, 0, 0, ow_block); \
                    int idx = dst_idx(mb_block, oc_outer, ow_block); \
                    (block)[idx] = _BLOCK_READ(&(ptr)[off]); \
                } \
    } while (0)

#define write_dst_block(block, ptr, ow) \
    do { \
        if (DST_W16C) { \
            for (int mb_block = 0; mb_block < MB_BLOCK; mb_block++) \
                for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) { \
                    int off = dst_off(mb_block, oc_outer * 16, 0, 0, 0); \
                    int idx = dst_idx(mb_block, oc_outer, 0); \
                    if (OW % OW_BLOCK == 0 || (ow) + OW_BLOCK <= OW) { \
                        unrolled_write(OW_BLOCK, &(block)[idx], &(ptr)[off]); \
                    } else { \
                        unrolled_write( \
                                OW % OW_BLOCK, &(block)[idx], &(ptr)[off]); \
                    } \
                } \
        } else if (DST_16N16C) { \
            int ow_bound = (OW % OW_BLOCK == 0) ? OW_BLOCK \
                                                : min(OW_BLOCK, OW - (ow)); \
            for (int ow_block = 0; ow_block < ow_bound; ow_block++) \
                for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) \
                    for (int mb_block = 0; mb_block < MB_BLOCK; \
                            mb_block += 16) { \
                        int off = dst_off( \
                                mb_block, oc_outer * 16, 0, 0, ow_block); \
                        int idx = dst_idx(mb_block, oc_outer, ow_block); \
                        unrolled_write(min(16, MB_BLOCK), &(block)[idx], \
                                &(ptr)[off]); \
                    } \
        } \
    } while (0)

#define loop_ic_outermost(src, wei, C, id, ih, iw) \
    do { \
        __attribute__((opencl_unroll_hint(1))) /*  attr:no-format */ \
                for (int ic = 0; ic < IC; ic += IC_BLOCK) { \
            __attribute__((opencl_unroll_hint(1))) /*  attr:no-format */ \
                    for (int kd = 0; kd < KD; kd++) { \
                if (HAS_PAD_D \
                        && ((id) + kd * (1 + DD) < 0 \
                                || (id) + kd * (1 + DD) >= ID)) \
                    continue; \
                __attribute__((opencl_unroll_hint(1))) /*  attr:no-format */ \
                        for (int kh = 0; kh < KH; kh++) { \
                    if (HAS_PAD_H \
                            && ((ih) + kh * (1 + DH) < 0 \
                                    || (ih) + kh * (1 + DH) >= IH)) \
                        continue; \
                    const __global DATA_T *src1 = (src) \
                            + src_off(0, 0, kd * (1 + DD), kh * (1 + DH), 0); \
                    DATA_T A_buf[MB_BLOCK * IC_OUTER * IW_OUTER] = {0}; \
                    if (ENABLE_SRC_BUF) read_src_buf(A_buf, src1, (iw)); \
                    __attribute__( \
                            (opencl_unroll_hint(KW))) /*  attr:no-format */ \
                            for (int kw = 0; kw < KW; kw++) { \
                        const __global DATA_T *wei1 \
                                = (wei) + wei_off(0, 0, 0, kd, kh, kw); \
                        DATA_T B[IC_OUTER * OC_OUTER * IC_INNER]; \
                        if (ENABLE_SRC_BUF) { \
                            read_wei_and_multiply(wei1, kw, (C), A_buf); \
                        } else { \
                            read_wei_block(B, wei1); \
                            read_src_and_multiply(src1, (iw), kw, 1, (C), B); \
                        } \
                    } \
                } \
            } \
            (src) += src_off(0, IC_BLOCK, 0, 0, 0); \
            (wei) += wei_off(0, 0, IC_BLOCK, 0, 0, 0); \
        } \
    } while (0)

#define loop_kdhw_outermost(src, wei, C, id, ih, iw) \
    do { \
        for (int kd = 0; kd < KD; kd++) { \
            if (HAS_PAD_D \
                    && ((id) + kd * (1 + DD) < 0 \
                            || (id) + kd * (1 + DD) >= ID)) \
                continue; \
            for (int kh = 0; kh < KH; kh++) { \
                if (HAS_PAD_H \
                        && ((ih) + kh * (1 + DH) < 0 \
                                || (ih) + kh * (1 + DH) >= IH)) \
                    continue; \
                for (int kw = 0; kw < KW; kw++) { \
                    if (HAS_PAD_W \
                            && ((iw) + kw * (1 + DW) < 0 \
                                    || (iw) + kw * (1 + DW) >= IW)) \
                        continue; \
                    /*  XXX: kw offset is applied in read_src_and_multiply(). */ \
                    const __global DATA_T *src1 = (src) \
                            + src_off(0, 0, kd * (1 + DD), kh * (1 + DH), 0); \
                    const __global DATA_T *wei1 \
                            = (wei) + wei_off(0, 0, 0, kd, kh, kw); \
                    __attribute__((opencl_unroll_hint)) /*  attr:no-format */ \
                            for (int ic = 0; ic < IC; ic += IC_BLOCK) { \
                        DATA_T B[IC_OUTER * OC_OUTER * IC_INNER]; \
                        read_wei_block(B, wei1); \
                        read_src_and_multiply(src1, (iw), kw, 0, (C), B); \
                        src1 += src_off(0, IC_BLOCK, 0, 0, 0); \
                        wei1 += wei_off(0, 0, IC_BLOCK, 0, 0, 0); \
                    } \
                } \
            } \
        } \
    } while (0)

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_conv_fwd(const __global DATA_T *src, const __global DATA_T *wei,
        const __global DATA_T *bia, __global DATA_T *dst POST_OP_ARGS) {

    int local_id = get_local_id(0);
    int g_ocb = get_group_id(0);
    int g = g_ocb / (OCB / OC_BLOCK);
    int ocb = g_ocb % (OCB / OC_BLOCK) * OC_BLOCK;

    int odhw = get_group_id(1) / (OMB / MB_BLOCK);
    int omb = get_group_id(1) % (OMB / MB_BLOCK) * MB_BLOCK;

    int ohw = IS_3D ? odhw % (OWB * OHB) : odhw;
    int od = IS_3D ? odhw / (OWB * OHB) : 0;
    int oh = (ohw / OWB) * OH_BLOCK;
    int ow = (ohw % OWB) * OW_BLOCK;

    int ocb_idx_omb_idx = get_group_id(2);
    int ocb_idx = ocb_idx_omb_idx / (MB / OMB);
    int omb_idx = ocb_idx_omb_idx % (MB / OMB);
    int oc = ocb_idx * OCB + ocb;
    int mb = omb_idx * OMB + omb;

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;
    int id = od * SD - PD;

    // Vector type variables have less chance of being spilled for half data
    // type.
#if DT_F16 && C_SIZE == 8
    DATA8_T C = 0;
#elif DT_F16 && C_SIZE == 16
    DATA16_T C = 0;
#else
    DATA_T C[C_SIZE] = {0};
#endif

    if (WITH_BIAS) {
        for (int mb_block = 0; mb_block < MB_BLOCK; mb_block++) {
            for (int oc_outer = 0; oc_outer < OC_OUTER; oc_outer++) {
                const int bg_off = g * OC;
                const int bc_off = oc + oc_outer * 16 + local_id;
                if (OC_WO_PADDING % OC_BLOCK == 0 || bc_off < OC_WO_PADDING) {
                    for (int ow_block = 0; ow_block < OW_BLOCK; ow_block++) {
                        const int c_off = dst_idx(mb_block, oc_outer, ow_block);
                        C[c_off] = bia[bg_off + bc_off];
                    } // ow_block
                } // copy-bias
            } // oc_outer
        } // mb_block
    } // if-bias

    src += src_off(mb, g * IC, id, ih, iw);
    wei += wei_off(g, oc, 0, 0, 0, 0);
    dst += dst_off(mb, g * OC + oc, od, oh, ow);

    if (OW_BLOCK == 1) {
        loop_kdhw_outermost(src, wei, C, id, ih, iw);
    } else {
        loop_ic_outermost(src, wei, C, id, ih, iw);
    }

    DATA_T S[MB_BLOCK * OC_OUTER * OW_BLOCK];

    if (WITH_SUM) { read_dst_block(S, dst, ow); }

    for (int didx = 0; didx < MB_BLOCK * OC_OUTER * OW_BLOCK; ++didx) {
        float accum = CONVERT_FLOAT_T(C[didx]);
        float sum = CONVERT_FLOAT_T(S[didx]);
        const int po_mb = (mb + didx / (OC_OUTER * OW_BLOCK)) % MB;
        const int po_oc = (g * OC + oc + local_id
                                  + (((didx / OW_BLOCK) % OC_OUTER) * 16))
                % (OC * G);
        APPLY_POST_OPS(accum, float, sum, float, po_mb, 1, po_oc, 1, od, 1, oh,
                1, ow, 1, 0, 1);
        C[didx] = TO_DATA_T(accum);
    }

    write_dst_block((DATA_T *)(&C), dst, ow);
}
