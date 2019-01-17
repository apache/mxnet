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
#include <vector>
#include <iostream>
#include <mshadow/tensor.h>
#include <algorithm>
namespace xnor_cuda {
#if BINARY_WORD_32 == 1
	typedef unsigned int BINARY_WORD;
#endif
#if BINARY_WORD_64 == 1
	typedef unsigned long long int BINARY_WORD;
#endif

const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);

__device__ BINARY_WORD concatenate(float* array);
__global__ void concatenate_rows_kernel(float *a, BINARY_WORD *b, int size);
__global__ void concatenate_cols_kernel(float *a, BINARY_WORD *b, int m, int n);
__global__ void xnor_gemm(BINARY_WORD* A, BINARY_WORD* B, float* C, int m, int n, int k, int block_size);

inline std::vector<int> get_divisors(int num){
	std::vector<int> dv;
	int square_root = (int) sqrt(num) + 1;
	for (int i = 1; i < square_root; i++) { 
		if (num % i == 0 && i*i==num){
			dv.push_back(i);
		}else if (num % i == 0 && i*i!=num){
	    	dv.push_back(i);
	    	dv.push_back(num/i);
		}
	}
	return dv;
}

inline int get_next_block_dim(int c){
	std::vector<int> divs = get_divisors(c);
	if (!divs.empty()){
		int dim = divs.at(divs.size()-1) < mshadow::cuda::kMaxThreadsPerBlock ? divs.at(divs.size()-1) : mshadow::cuda::kMaxThreadsPerBlock;
		return dim;
	}
	return 1;											
}

/*
 * m: number of output channels (num_filter) per group
 * n: number of input channels per group * kernel size(e.g., 3x3=9) / BITS_PER_BINARY_WORD
 * k: number of pixels of output images per channel (output dimension)
 */
inline int get_next_block_dim(int m, int n, int k){	
	std::vector<int> divs = get_divisors(n);
	if (!divs.empty()){
		std::sort(divs.begin(), divs.end());
		for (int i = divs.size()-1; i > -1 ; --i){
			int val = divs[i];			
			int sel_mid =  val < mshadow::cuda::kMaxThreadsPerBlock ? val : mshadow::cuda::kMaxThreadsPerBlock;
			if (sel_mid < m/2 && sel_mid < k/2){
				return sel_mid;
			}
		}					
	}
	return 1;											
}
} // namespace xnor_cuda

#endif /* MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_XNOR_KERNELS_H */
