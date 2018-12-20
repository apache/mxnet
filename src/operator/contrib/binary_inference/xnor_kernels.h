/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * Copyright (c) 2018 by Contributors
 * \file binary_inference_convolution-inl.h
 * \brief
 * \ref: https://arxiv.org/abs/1705.09864
 * \author HPI-DeepLearning
*/

#ifndef MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_XNOR_KERNELS_H
#define MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_XNOR_KERNELS_H

typedef unsigned int BINARY_WORD;
const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);
#define BLOCK_SIZE_XNOR 1

__device__ unsigned int concatenate(float* array);
__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size);
__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int m, int n);
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k);

#endif /* MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_XNOR_KERNELS_H */
