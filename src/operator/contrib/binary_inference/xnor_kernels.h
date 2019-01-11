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

namespace xnor_cuda {
// typedef unsigned int BINARY_WORD;
//uint32_t, uint64_t
#if BINARY_WORD_32 == 1
typedef unsigned int BINARY_WORD;
#endif
#if BINARY_WORD_64 == 1
typedef unsigned long long int BINARY_WORD;
#endif

const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);

__device__ unsigned int concatenate(float* array);
__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size);
__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int m, int n);
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k, int block_size);

inline int get_next_block_dim(int c){
	if (c%8 == 0)
		return 8;
	if (c%4 == 0)
		return 4;
	if (c%2 == 0)
		return 2;
	return 1;											
}

inline int get_next_block_dim(int m, int n, int k){	
	if (m%128 == 0 && n%128 == 0 && k%128 == 0)
		return 128;
	if (m%64 == 0 && n%64 == 0 && k%64 == 0)
		return 64;
	if (m%32 == 0 && n%32 == 0 && k%32 == 0)
		return 32;			
	if (m%16 == 0 && n%16 == 0 && k%16 == 0)
		return 16;
	if (m%8 == 0 && n%8 == 0 && k%8 == 0)
		return 8;
	if (m%4 == 0 && n%4 == 0 && k%4 == 0)
		return 4;
	if (m%2 == 0 && n%2 == 0 && k%2 == 0)
		return 2;
	return 1;											
}
} // namespace xnor_cuda

#endif /* MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_XNOR_KERNELS_H */
