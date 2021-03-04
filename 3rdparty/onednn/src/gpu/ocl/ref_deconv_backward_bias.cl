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

__kernel void ref_deconv_backward_bias(
        __global DST_DATA_T *diff_dst, __global BIA_DATA_T *diff_bias) {
    const int g = get_global_id(0) / OC;
    const int oc = get_global_id(0) % OC;
    ACC_DATA_T db = 0;
    for (int mb = 0; mb < MB; ++mb)
        for (int od = 0; od < OD; ++od)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    uint diff_dst_off = DST_OFF(mb, g * OC + oc, od, oh, ow);
                    db += DST_TO_REF(diff_dst[diff_dst_off]);
                }

    diff_bias[g * OC + oc] = TO_BIA(db);
}
