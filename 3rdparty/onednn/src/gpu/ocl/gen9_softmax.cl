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

#include "gpu/ocl/ocl_types.h"

#define OFF(dim, idx) \
    (dim % CONCAT2(DATA_B, idx)) * CONCAT2(DATA_SB, idx) \
            + (dim / CONCAT2(DATA_B, idx)) * CONCAT2(DATA_S, idx)

#if SOFTMAX_AXIS_IDX == 0
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(softmax_dim, 0) + OFF(dim0, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 1
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(softmax_dim, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 2
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(softmax_dim, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 3
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(softmax_dim, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 4
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) \
            + OFF(softmax_dim, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 5
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) + OFF(dim4, 4) \
            + OFF(softmax_dim, 5)
#else
#error unsupported softmax dimension
#endif

#define LOAD_DATA_8x16(ptr) \
    CONVERT_FLOAT8_T( \
            AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr))))

#define STORE_DATA_8x16(ptr, val) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)ptr, \
            AS_BLOCK_DATA8_T(CONVERT_DATA8_T(val)))

#define VECT_SIZE 8
#define NUM_BUF (SOFTMAX_AXIS_SIZE / SUB_GROUP_SIZE / VECT_SIZE)

#if IS_FWD

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_softmax_fwd(__global DATA_T *src, __global DATA_T *dst) {

    const int dim[] = {
            (get_global_id(0) / GROUP_SIZE) % BLOCK_0,
            get_global_id(1) % BLOCK_1,
            get_global_id(2) % BLOCK_2,
            (get_global_id(0) / GROUP_SIZE) / BLOCK_0,
            get_global_id(1) / BLOCK_1,
            get_global_id(2) / BLOCK_2,
    };

    float8 d[NUM_BUF];

    int local_id = get_sub_group_local_id();
    int begin = local_id * (SOFTMAX_AXIS_SIZE / VECT_SIZE);

    float max_ = -FLT_MAX;
    float denom_ = 0.f;

    size_t data_off = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], begin);
    src += data_off;

    for (int k = 0; k < NUM_BUF; ++k) {
        d[k] = LOAD_DATA_8x16(&src[k * VECT_SIZE * SUB_GROUP_SIZE]);
        for (int i = 0; i < VECT_SIZE; ++i) {
            max_ = max(d[k][i], max_);
        }
    }

    max_ = sub_group_reduce_max(max_);

    for (int k = 0; k < NUM_BUF; ++k) {
#if LOGSOFTMAX
        for (int i = 0; i < VECT_SIZE; ++i)
            denom_ += exp(d[k][i] - max_);
#else
        d[k] = exp(d[k] - max_);
        for (int i = 0; i < VECT_SIZE; ++i)
            denom_ += d[k][i];
#endif
    }

    denom_ = sub_group_reduce_add(denom_);

#if LOGSOFTMAX
    denom_ = log(denom_);
#else
    denom_ = 1.0 / denom_;
#endif

    dst += data_off;
    for (int k = 0; k < NUM_BUF; ++k) {
#if LOGSOFTMAX
        d[k] = d[k] - max_ - denom_;
#else
        d[k] = d[k] * denom_;
#endif
        STORE_DATA_8x16(&dst[k * VECT_SIZE * SUB_GROUP_SIZE], d[k]);
    }
}

#endif
