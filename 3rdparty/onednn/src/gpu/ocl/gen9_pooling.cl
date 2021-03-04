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

// Read functions.
inline VECT_DATA_T read_vect_c_block(int idx, const __global DATA_T *ptr, int c,
        int blocks_stride, int chunks_per_block);
inline VECT_INT_T read_vect_c_block_int(int idx, const __global int *ptr, int c,
        int blocks_stride, int chunks_per_block);

// Write functions.
inline void write_vect_c_block(int idx, __global DATA_T *ptr, int c,
        int blocks_stride, int chunks_per_block, VECT_DATA_T block);
inline void write_vect_c_block_int(int idx, __global int *ptr, int c,
        int blocks_stride, int chunks_per_block, VECT_INT_T block);

#if IS_FWD
KERNEL_ATTR
__kernel void gen9_pooling_fwd(__global DATA_T *src, __global int *ws,
        __global DATA_T *dst POST_OP_ARGS) {
    const int mb = GWS_GET_MB();
    const int c = GWS_GET_C();
    const int od = GWS_GET_OD();
    const int oh = GWS_GET_OH();

    // Calculate number of subgroup chunks inside C block
    // and stride between consecutive MB/C blocks
#if USE_MB_C_BLOCK
    const int src_stride = (SRC_SB0 > 1) ? SRC_SB0 : SRC_S0;
    const int dst_stride = (DST_SB0 > 1) ? DST_SB0 : DST_S0;
    const int src_chunks_per_c_block = CHUNKS_PER_C_BLOCK;
    const int dst_chunks_per_c_block = CHUNKS_PER_C_BLOCK;
#elif USE_ONLY_C_BLOCK
    const int src_stride = (SRC_B1 > 1) ? SRC_S1 : SUB_GROUP_SIZE;
    const int dst_stride = (DST_B1 > 1) ? DST_S1 : SUB_GROUP_SIZE;
    const int src_chunks_per_c_block
            = (SRC_B1 > 1) ? (SRC_B1 / SUB_GROUP_SIZE) : 1;
    const int dst_chunks_per_c_block
            = (DST_B1 > 1) ? (DST_B1 / SUB_GROUP_SIZE) : 1;
#endif
    const int ws_stride = dst_stride;
    const int ws_chunks_per_c_block = dst_chunks_per_c_block;

    for (int ow = 0; ow < OW; ++ow) {
        const int id = od * SD - PD;
        const int ih = oh * SH - PH;
        const int iw = ow * SW - PW;

        VECT_FLOAT_T D0 = ALG_MAX ? DATA_MIN : DATA_ZERO;
        VECT_FLOAT_T D1 = ALG_MAX ? DATA_MIN : DATA_ZERO;
        VECT_INT_T WS0 = 0, WS1 = 0;

        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    if (id + kd < 0 || id + kd >= ID) continue;
                    if (ih + kh < 0 || ih + kh >= IH) continue;
                    if (iw + kw < 0 || iw + kw >= IW) continue;

                    int src_off = SRC_OFF(mb, c, id + kd, ih + kh, iw + kw);

                    VECT_FLOAT_T S0 = CONVERT_VECT_FLOAT_T(
                            read_vect_c_block(0, &src[src_off], c, src_stride,
                                    src_chunks_per_c_block));
                    VECT_FLOAT_T S1 = CONVERT_VECT_FLOAT_T(
                            read_vect_c_block(1, &src[src_off], c, src_stride,
                                    src_chunks_per_c_block));

#if ALG_MAX
#if IS_TRAINING
                    VECT_INT_T CMP0 = isless(D0, S0);
                    WS0 = select(WS0, kd * KH * KW + kh * KW + kw, CMP0);
                    D0 = select(D0, S0, CMP0);

                    VECT_INT_T CMP1 = isless(D1, S1);
                    WS1 = select(WS1, kd * KH * KW + kh * KW + kw, CMP1);
                    D1 = select(D1, S1, CMP1);

#else // TRAINING
                    D0 = max(D0, S0);
                    D1 = max(D1, S1);
#endif // TRAINING
#else // ALG_MAX
                    D0 += S0;
                    D1 += S1;
#endif // ALG_MAX
                }
            }

#if ALG_AVG_P
        D0 = D0 / (KD * KH * KW);
        D1 = D1 / (KD * KH * KW);

#endif // ALG_AVG_P

#if ALG_AVG_NP
        const int id_start = max(od * SD - PD, 0);
        const int ih_start = max(oh * SH - PH, 0);
        const int iw_start = max(ow * SW - PW, 0);
        const int id_end = min(od * SD - PD + KD, ID);
        const int ih_end = min(oh * SH - PH + KH, IH);
        const int iw_end = min(ow * SW - PW + KW, IW);
        const DATA_T num_summands = (ih_end - ih_start) * (iw_end - iw_start)
                * (id_end - id_start);
        D0 = D0 / num_summands;
        D1 = D1 / num_summands;
#endif // ALG_AVG_NP

        int dst_off = DST_OFF(mb, c, od, oh, ow);
        VECT_DATA_T sum0;
        VECT_DATA_T sum1;
#if WITH_SUM
        sum0 = read_vect_c_block(
                0, &dst[dst_off], c, dst_stride, dst_chunks_per_c_block);
        sum1 = read_vect_c_block(
                1, &dst[dst_off], c, dst_stride, dst_chunks_per_c_block);
#endif

        const int local_id = get_sub_group_local_id();

#if VECT_DT_N == 1
        const int po_mb = mb;
        const int po_oc = c + local_id;
        POST_OP_DATA_T po_sum0 = DATA_TO_REF(sum0);
        APPLY_POST_OPS(D0, float, po_sum0, POST_OP_DATA_T, po_mb, 1, po_oc, 1,
                0, 1, 0, 1, 0, 1, 0, 1);

        POST_OP_DATA_T po_sum1 = DATA_TO_REF(sum1);
        APPLY_POST_OPS(D1, POST_OP_DATA_T, po_sum1, POST_OP_DATA_T, po_mb, 1,
                po_oc, 1, 0, 1, 0, 1, 0, 1, 0, 1);
#else
        for (int idx = 0; idx < VECT_DT_N; ++idx) {
#if USE_MB_C_BLOCK
            int c_sub_block_id = idx % CHUNKS_PER_C_BLOCK;
            int mb_sub_block_id = idx / CHUNKS_PER_C_BLOCK;
            const int po_oc = c + c_sub_block_id * SUB_GROUP_SIZE + local_id;
            int po_mb = (mb + mb_sub_block_id) % MB;
#else // USE_MB_C_BLOCK
            const int po_oc = c + idx * SUB_GROUP_SIZE + local_id;
            int po_mb = mb;
#endif // USE_MB_C_BLOCK

            POST_OP_DATA_T d0_i = D0[idx];
            POST_OP_DATA_T sum0_i = DATA_TO_REF(sum0[idx]);
            APPLY_POST_OPS(d0_i, POST_OP_DATA_T, sum0_i, POST_OP_DATA_T, po_mb,
                    1, po_oc, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            D0[idx] = d0_i;

            POST_OP_DATA_T d1_i = D1[idx];
            POST_OP_DATA_T sum1_i = DATA_TO_REF(sum1[idx]);
            po_mb += VECT_DT_N;
            APPLY_POST_OPS(d1_i, POST_OP_DATA_T, sum1_i, POST_OP_DATA_T, po_mb,
                    1, po_oc, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            D1[idx] = d1_i;
        }
#endif // #if VECT_DT_N == 1
        write_vect_c_block(0, &dst[dst_off], c, dst_stride,
                dst_chunks_per_c_block, CONVERT_VECTOR_DATA_T(D0));
        write_vect_c_block(1, &dst[dst_off], c, dst_stride,
                dst_chunks_per_c_block, CONVERT_VECTOR_DATA_T(D1));

#if ALG_MAX && IS_TRAINING
        int ws_off = dst_off;
        write_vect_c_block_int(
                0, &ws[ws_off], c, ws_stride, ws_chunks_per_c_block, WS0);
        write_vect_c_block_int(
                1, &ws[ws_off], c, ws_stride, ws_chunks_per_c_block, WS1);
#endif // ALG_MAX && IS_TRAINING
    }
}
#endif

#if IS_BWD
KERNEL_ATTR
__kernel void gen9_pooling_bwd(__global DATA_T *diff_src, __global int *ws,
        __global DATA_T *diff_dst) {

    const int mb = GWS_GET_MB();
    const int c = GWS_GET_C();
    const int id = GWS_GET_ID();
    const int ih = GWS_GET_IH();
    const int iw = GWS_GET_IW();

    // Calculate number of subgroup chunks inside C block
    // and stride between consecutive MB/C blocks
#if USE_MB_C_BLOCK
    const int src_stride = (SRC_SB0 > 1) ? SRC_SB0 : SRC_S0;
    const int dst_stride = (DST_SB0 > 1) ? DST_SB0 : DST_S0;
    const int src_chunks_per_c_block = CHUNKS_PER_C_BLOCK;
    const int dst_chunks_per_c_block = CHUNKS_PER_C_BLOCK;
#elif USE_ONLY_C_BLOCK
    const int src_stride = (SRC_B1 > 1) ? SRC_S1 : SUB_GROUP_SIZE;
    const int dst_stride = (DST_B1 > 1) ? DST_S1 : SUB_GROUP_SIZE;
    const int src_chunks_per_c_block
            = (SRC_B1 > 1) ? (SRC_B1 / SUB_GROUP_SIZE) : 1;
    const int dst_chunks_per_c_block
            = (DST_B1 > 1) ? (DST_B1 / SUB_GROUP_SIZE) : 1;
#endif
    const int ws_stride = dst_stride;
    const int ws_chunks_per_c_block = dst_chunks_per_c_block;

    VECT_DATA_T S0 = 0, S1 = 0;
    for (int kd = 0; kd < KD; kd++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int od = (id + PD - kd);
                int oh = (ih + PH - kh);
                int ow = (iw + PW - kw);
                if (od % SD != 0 || oh % SH != 0 || ow % SW != 0) continue;
                od /= SD;
                oh /= SH;
                ow /= SW;
                if (od < 0 || od >= OD) continue;
                if (oh < 0 || oh >= OH) continue;
                if (ow < 0 || ow >= OW) continue;

                const int dst_off = DST_OFF(mb, c, od, oh, ow);
                VECT_DATA_T D0 = read_vect_c_block(0, &diff_dst[dst_off], c,
                        dst_stride, dst_chunks_per_c_block);
                VECT_DATA_T D1 = read_vect_c_block(1, &diff_dst[dst_off], c,
                        dst_stride, dst_chunks_per_c_block);

#if ALG_MAX
                VECT_INT_T WS0 = read_vect_c_block_int(
                        0, &ws[dst_off], c, ws_stride, ws_chunks_per_c_block);
                VECT_INT_T WS1 = read_vect_c_block_int(
                        1, &ws[dst_off], c, ws_stride, ws_chunks_per_c_block);

                VECT_INT_T CMP0 = isnotequal(
                        AS_VECT_DATA_T(WS0 - kd * KH * KW - kh * KW - kw),
                        (VECT_DATA_T)0);
                D0 = select(D0, (VECT_DATA_T)0, CMP0);

                VECT_INT_T CMP1 = isnotequal(
                        AS_VECT_DATA_T(WS1 - kd * KH * KW - kh * KW - kw),
                        (VECT_DATA_T)0);
                D1 = select(D1, (VECT_DATA_T)0, CMP1);
#endif
#if ALG_AVG_NP
                const int id_start = max(id - kd, 0);
                const int ih_start = max(ih - kh, 0);
                const int iw_start = max(iw - kw, 0);
                const int id_end = min(id - kd + KD, ID);
                const int ih_end = min(ih - kh + KH, IH);
                const int iw_end = min(iw - kw + KW, IW);
                const DATA_T num_summands = (ih_end - ih_start)
                        * (iw_end - iw_start) * (id_end - id_start);
                D0 /= num_summands;
                D1 /= num_summands;
#endif
                S0 += D0;
                S1 += D1;
            }
        }
    }
#if ALG_AVG_P
    S0 /= KD * KH * KW;
    S1 /= KD * KH * KW;
#endif

    int src_off = SRC_OFF(mb, c, id, ih, iw);
    write_vect_c_block(
            0, &diff_src[src_off], c, src_stride, src_chunks_per_c_block, S0);
    write_vect_c_block(
            1, &diff_src[src_off], c, src_stride, src_chunks_per_c_block, S1);
}
#endif

inline DATA_T read_c_block(const __global DATA_T *ptr, int c) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    int tail = C_WO_PADDING - c;
    return (local_id < tail) ? ptr[local_id] : 0;
#else
    if (c >= C_WO_PADDING) { return 0; }
    return AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)ptr));
#endif
}

inline VECT_DATA_T read_vect_c_block(int idx, const __global DATA_T *ptr, int c,
        int blocks_stride, int chunks_per_block) {
    if (idx >= NVECT) return 0;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        return AS_VECT_DATA_T(VECT_BLOCK_READ((const __global BLOCK_DATA_T *)ptr
                + idx * VECT_DT_N * SUB_GROUP_SIZE));
    } else {
        VECT_DATA_T ret;
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const int ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off
                    = (USE_ONLY_C_BLOCK ? offset_index * SUB_GROUP_SIZE
                                        : local_c_block_index * SUB_GROUP_SIZE);
#if VECT_DT_N == 1
            ret = read_c_block(ptr + ptr_offset, c + c_off);
#else
            ret[i] = read_c_block(ptr + ptr_offset, c + c_off);
#endif
        }
        return ret;
    }
}

inline int read_c_block_int(const __global int *ptr, int c) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    int tail = C_WO_PADDING - c;
    return (local_id < tail) ? ptr[local_id] : 0;
#else
    if (c >= C_WO_PADDING) { return 0; }
    return as_int(intel_sub_group_block_read((const __global uint *)ptr));
#endif
}

inline VECT_INT_T read_vect_c_block_int(int idx, const __global int *ptr, int c,
        int blocks_stride, int chunks_per_block) {
    if (idx >= NVECT) return 0;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        return AS_VECT_INT_T(VECT_UINT_READ(
                (const __global uint *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE));
    } else {
        VECT_INT_T ret;
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const int ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off
                    = (USE_ONLY_C_BLOCK ? offset_index * SUB_GROUP_SIZE
                                        : local_c_block_index * SUB_GROUP_SIZE);
#if VECT_DT_N == 1
            ret = read_c_block_int(ptr + ptr_offset, c + c_off);
#else
            ret[i] = read_c_block_int(ptr + ptr_offset, c + c_off);
#endif
        }
        return ret;
    }
}

inline void write_c_block(__global DATA_T *ptr, int c, DATA_T value) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    int tail = C_WO_PADDING - c;
    if (local_id < tail) ptr[local_id] = value;
#else
    if (c >= C_WO_PADDING) { return; }
    BLOCK_WRITE((__global BLOCK_DATA_T *)ptr, AS_BLOCK_DATA_T(value));
#endif
}

inline void write_vect_c_block(int idx, __global DATA_T *ptr, int c,
        int blocks_stride, int chunks_per_block, VECT_DATA_T block) {
    if (idx >= NVECT) return;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        VECT_BLOCK_WRITE(
                (__global BLOCK_DATA_T *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE,
                AS_VECT_BLOCK_DATA_T(block));
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const int ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off
                    = (USE_ONLY_C_BLOCK ? offset_index * SUB_GROUP_SIZE
                                        : local_c_block_index * SUB_GROUP_SIZE);
#if VECT_DT_N == 1
            write_c_block(ptr + ptr_offset, c + c_off, block);
#else
            write_c_block(ptr + ptr_offset, c + c_off, block[i]);
#endif
        }
    }
}

inline void write_c_block_int(__global int *ptr, int c, int value) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    int tail = C_WO_PADDING - c;
    if (local_id < tail) ptr[local_id] = value;
#else
    if (c >= C_WO_PADDING) { return; }
    intel_sub_group_block_write((__global uint *)ptr, as_uint(value));
#endif
}

inline void write_vect_c_block_int(int idx, __global int *ptr, int c,
        int blocks_stride, int chunks_per_block, VECT_INT_T block) {
    if (idx >= NVECT) return;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        VECT_UINT_WRITE((__global uint *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE,
                AS_VECT_UINT_T(block));
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const int ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off
                    = (USE_ONLY_C_BLOCK ? offset_index * SUB_GROUP_SIZE
                                        : local_c_block_index * SUB_GROUP_SIZE);
#if VECT_DT_N == 1
            write_c_block_int(ptr + ptr_offset, c + c_off, block);
#else
            write_c_block_int(ptr + ptr_offset, c + c_off, block[i]);
#endif
        }
    }
}
