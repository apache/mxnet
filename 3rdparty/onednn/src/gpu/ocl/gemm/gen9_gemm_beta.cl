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

__kernel void gen9_gemm_beta(long m, long n, DATA_T alpha,
        __global DST_DATA_T *a, long offset, long lda) {
    int idy = get_group_id(1);
    long count = m;

    offset += idy * lda;

    if (idy < n) {
        while (count > 0) {
            a[offset] = REF_TO_DST(DST_TO_REF(a[offset]) * alpha);
            offset++;
            count--;
        }
    }
}
