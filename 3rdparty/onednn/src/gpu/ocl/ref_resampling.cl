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

#if IS_FWD == 1
KERNEL_ATTR
__kernel void ref_resampling_fwd(
        __global const DATA_T *src, __global DATA_T *dst) {
    const uint mb = GWS_GET_MB();
    const uint ch = GWS_GET_C();
    const uint od = GWS_GET_OD();
    const uint oh = GWS_GET_OH();
    const uint ow = GWS_GET_OW();
    const float id = (od + .5f) * ID / OD;
    const float ih = (oh + .5f) * IH / OH;
    const float iw = (ow + .5f) * IW / OW;
#if NEAREST
    for (int c = ch; c < min((uint)C, ch + (uint)C_BLOCK); c++) {
        const uint src_index = SRC_OFF(mb, c, (uint)id, (uint)ih, (uint)iw);
        const uint dst_index = DST_OFF(mb, c, od, oh, ow);
        dst[dst_index] = src[src_index];
    }
#else
    const uint id0 = max((uint)floor(id - .5f), (uint)0);
    const uint id1 = min((uint)ceil(id - .5f), (uint)ID - 1);
    const uint ih0 = max((uint)floor(ih - .5f), (uint)0);
    const uint ih1 = min((uint)ceil(ih - .5f), (uint)IH - 1);
    const uint iw0 = max((uint)floor(iw - .5f), (uint)0);
    const uint iw1 = min((uint)ceil(iw - .5f), (uint)IW - 1);
    float Wid = 1.0f - fabs(id - .5f - id0);
    float Wih = 1.0f - fabs(ih - .5f - ih0);
    float Wiw = 1.0f - fabs(iw - .5f - iw0);
    for (int c = ch; c < min((uint)C, ch + (uint)C_BLOCK); c++) {
        dst[DST_OFF(mb, c, od, oh, ow)] = CONVERT_DATA_T(
                ((CONVERT_FLOAT_T(src[SRC_OFF(mb, c, id0, ih0, iw0)]) * Wih
                         * Wiw)
                        + (CONVERT_FLOAT_T(src[SRC_OFF(mb, c, id0, ih1, iw0)])
                                * (1.f - Wih) * Wiw)
                        + (CONVERT_FLOAT_T(src[SRC_OFF(mb, c, id0, ih0, iw1)])
                                * Wih * (1 - Wiw))
                        + (CONVERT_FLOAT_T(src[SRC_OFF(mb, c, id0, ih1, iw1)])
                                * (1.f - Wih) * (1.f - Wiw)))
                        * Wid
                + ((CONVERT_FLOAT_T(src[SRC_OFF(mb, c, id1, ih0, iw0)]) * Wih
                           * Wiw)
                          + (CONVERT_FLOAT_T(src[SRC_OFF(mb, c, id1, ih1, iw0)])
                                  * (1.f - Wih) * Wiw)
                          + (CONVERT_FLOAT_T(src[SRC_OFF(mb, c, id1, ih0, iw1)])
                                  * Wih * (1 - Wiw))
                          + (CONVERT_FLOAT_T(src[SRC_OFF(mb, c, id1, ih1, iw1)])
                                  * (1.f - Wih) * (1.f - Wiw)))
                        * (1.f - Wid));
    }
#endif
}
#endif
#if IS_BWD == 1
float linear(uint x, int fo, int fi) {
    return ((x + .5f) * fo / fi) - .5f;
}
KERNEL_ATTR
__kernel void ref_resampling_bwd(
        __global DATA_T *diff_src, __global const DATA_T *diff_dst) {
#define CEIL(x) max((uint)ceil(x), (uint)0)
#define L(x, fo, fi) linear(x, fo, fi)
#define LS(x, fo, fi) CEIL(L(x, fo, fi))
#define RS(x, fo, fi) L(x - 1, fo, fi) < 0 ? 0 : (uint)(L(x - 1, fo, fi)) + 1
#define LE(x, fo, fi, lim) min(CEIL(L(x + 1, fo, fi)), (uint)lim)
#define RE(x, fo, fi, lim) \
    min((L(x, fo, fi) < 0 ? 0 : (uint)(L(x, fo, fi)) + 1), (uint)lim)
    const uint mb = GWS_GET_MB();
    const uint c = GWS_GET_C();
    const uint id = GWS_GET_ID();
    const uint ih = GWS_GET_IH();
    const uint iw = GWS_GET_IW();
#if NEAREST
    uint od_start = CEIL(id * FD - .5f);
    uint oh_start = CEIL(ih * FH - .5f);
    uint ow_start = CEIL(iw * FW - .5f);
    uint od_end = CEIL((id + 1.f) * FD - .5f);
    uint oh_end = CEIL((ih + 1.f) * FH - .5f);
    uint ow_end = CEIL((iw + 1.f) * FW - .5f);
    uint src_index = SRC_OFF(mb, c, id, ih, iw);
    DEF_ACC_DATA_T src_val = 0;
    for (int i = od_start; i < od_end; i++) {
        for (int j = oh_start; j < oh_end; j++) {
            for (int k = ow_start; k < ow_end; k++) {
                const int dst_index = DST_OFF(mb, c, i, j, k);
                src_val += TO_DEF_ACC_DATA_T(diff_dst[dst_index]);
            }
        }
    }
    diff_src[src_index] = TO_DATA_T(src_val);
#else
    uint left_sd = id == 0 ? 0 : LS(id, OD, ID);
    uint left_sh = ih == 0 ? 0 : LS(ih, OH, IH);
    uint left_sw = iw == 0 ? 0 : LS(iw, OW, IW);
    uint right_sd = RS(id, OD, ID);
    uint right_sh = RS(ih, OH, IH);
    uint right_sw = RS(iw, OW, IW);
    uint left_ed = LE(id, OD, ID, OD);
    uint left_eh = LE(ih, OH, IH, OH);
    uint left_ew = LE(iw, OW, IW, OW);
    uint right_ed = id == (ID - 1) ? OD : RE(id, OD, ID, OD);
    uint right_eh = ih == (IH - 1) ? OH : RE(ih, OH, IH, OH);
    uint right_ew = iw == (IW - 1) ? OW : RE(iw, OW, IW, OW);
    uint od_start[2] = {left_sd, right_sd};
    uint oh_start[2] = {left_sh, right_sh};
    uint ow_start[2] = {left_sw, right_sw};
    uint od_end[2] = {left_ed, right_ed};
    uint oh_end[2] = {left_eh, right_eh};
    uint ow_end[2] = {left_ew, right_ew};
    uint src_index = SRC_OFF(mb, c, id, ih, iw);
    DEF_ACC_DATA_T src_val = 0.0;
    for (int c1 = 0; c1 < 2; c1++) {
        for (int c2 = 0; c2 < 2; c2++) {
            for (int c3 = 0; c3 < 2; c3++) {
                for (int i = od_start[c1]; i < od_end[c1]; i++) {
                    for (int j = oh_start[c2]; j < oh_end[c2]; j++) {
                        for (int k = ow_start[c3]; k < ow_end[c3]; k++) {
                            DEF_ACC_DATA_T dst_val = TO_DEF_ACC_DATA_T(
                                    diff_dst[DST_OFF(mb, c, i, j, k)]);
                            float d = L(i, ID, OD);
                            float h = L(j, IH, OH);
                            float w = L(k, IW, OW);
                            float Wid = c1 == 0 ? 1.f - fabs(d - (int)d)
                                                : fabs(d - (int)d);
                            float Wih = c2 == 0 ? 1.f - fabs(h - (int)h)
                                                : fabs(h - (int)h);
                            float Wiw = c3 == 0 ? 1.f - fabs(w - (int)w)
                                                : fabs(w - (int)w);
                            src_val += dst_val * Wid * Wih * Wiw;
                        }
                    }
                }
            }
        }
    }
    diff_src[src_index] = TO_DATA_T(src_val);
#endif
}
#endif
