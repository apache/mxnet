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

#ifndef GPU_ZERO_PAD_ZERO_PAD_STRUCT_H
#define GPU_ZERO_PAD_ZERO_PAD_STRUCT_H

#define ZERO_PAD_MAX_STEP_SIZE 1024

#ifdef IS_OCL_KERNEL
#define ZERO_PAD_MASK_DATA_TYPE uchar
#else
#define ZERO_PAD_MASK_DATA_TYPE unsigned char
#endif

#define ZERO_PAD_MASK_SIZE \
    (ZERO_PAD_MAX_STEP_SIZE / (8 * sizeof(ZERO_PAD_MASK_DATA_TYPE)))
typedef struct {
    ZERO_PAD_MASK_DATA_TYPE mask[ZERO_PAD_MASK_SIZE];
} zero_pad_mask_t;

#endif
