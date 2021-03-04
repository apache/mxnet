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

#define MAX_SUPPORTED_NDIMS 6

#define OFF1D(x, dim) \
    ((x) % SRC_B##dim) * SRC_SB##dim + ((x) / SRC_B##dim) * SRC_S##dim

#if NDIMS == 1
#define OFF(x0, x1, x2, x3, x4, x5) OFF1D(x0, 0)
#elif NDIMS == 2
#define OFF(x0, x1, x2, x3, x4, x5) OFF1D(x0, 0) + OFF1D(x1, 1)
#elif NDIMS == 3
#define OFF(x0, x1, x2, x3, x4, x5) OFF1D(x0, 0) + OFF1D(x1, 1) + OFF1D(x2, 2)
#elif NDIMS == 4
#define OFF(x0, x1, x2, x3, x4, x5) \
    OFF1D(x0, 0) + OFF1D(x1, 1) + OFF1D(x2, 2) + OFF1D(x3, 3)
#elif NDIMS == 5
#define OFF(x0, x1, x2, x3, x4, x5) \
    OFF1D(x0, 0) + OFF1D(x1, 1) + OFF1D(x2, 2) + OFF1D(x3, 3) + OFF1D(x4, 4)
#elif NDIMS == 6
#define OFF(x0, x1, x2, x3, x4, x5) \
    OFF1D(x0, 0) + OFF1D(x1, 1) + OFF1D(x2, 2) + OFF1D(x3, 3) + OFF1D(x4, 4) \
            + OFF1D(x5, 5)
#else
#error Unsupported number of dimensions
#endif

#define GETDIM(d) SRC_D##d

int rev_transposed(int a) {
    return ((a % TRANSPOSE_COL) * TRANSPOSE_ROW + a / TRANSPOSE_COL);
}

__kernel void ref_shuffle(__global DATA_T *src, __global DATA_T *dst) {

    const int inner_dims_idx = get_global_id(0);
    const int axis_idx = get_global_id(1);
    const int outer_dims_idx = get_global_id(2);

    int d[MAX_SUPPORTED_NDIMS] = {};
    int s[MAX_SUPPORTED_NDIMS] = {};
    int D[MAX_SUPPORTED_NDIMS] = {
            GETDIM(0), GETDIM(1), GETDIM(2), GETDIM(3), GETDIM(4), GETDIM(5)};

    const ulong offdst = outer_dims_idx * AXIS_SIZE * INNER_SIZE
            + inner_dims_idx + axis_idx * INNER_SIZE;

    ulong dimprod = 1;
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = NDIMS - 1; i >= 0; i--) {
        s[i] = d[i] = (offdst / dimprod) % D[i];
        dimprod *= D[i];
    }
    d[AXIS] = axis_idx;
    s[AXIS] = rev_transposed(axis_idx);

    dst[OFF(d[0], d[1], d[2], d[3], d[4], d[5])]
            = src[OFF(s[0], s[1], s[2], s[3], s[4], s[5])];
}
