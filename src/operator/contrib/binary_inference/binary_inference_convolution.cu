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

#include "./binary_inference_convolution-inl.h"
#include <mshadow/tensor.h>
#include "./xnor_kernels.h"

namespace mshadow {
namespace cuda {


/*
 * m: number of output channels (num_filter) per group
 * n: number of pixels of output images per channel (output dimension)
 * k: number of input channels per group * kernel size
 */
inline void _BinaryConvolutionForward(int m, int n, int k,
										mxnet::op::xnor::BINARY_WORD* wmat_binarized,
										Tensor<gpu, 1, float> &workspace,
										const Tensor<gpu, 2, float> &in_col,
										Tensor<gpu, 2, float> &temp_dst) {

	CHECK_EQ(workspace.shape_.Size() * sizeof(workspace[0]) * CHAR_BIT, n * k);

	// check matrix dims:
	// temp_dst should have dims (m x n)
	CHECK_EQ(k, (int)in_col.size(0));
	CHECK_EQ((int)temp_dst.size(0), m);
	CHECK_EQ(n, (int)in_col.size(1));
	CHECK_EQ((int)temp_dst.size(1), n);	
	
	cudaStream_t stream = Stream<gpu>::GetStream(temp_dst.stream_);
	
	//set memory
	float *fB = in_col.dptr_;
	float *fC = temp_dst.dptr_;					
	BINARY_WORD* binary_col = (BINARY_WORD*) workspace.dptr_;	

	//concatinates matrix (n x k) -> (n/32 x k)
	int threads_per_block = BLOCK_SIZE_XNOR;
	dim3 conc_block(threads_per_block,1,1);
  	dim3 conc_grid(n/threads_per_block+1,1);
	concatenate_cols_kernel<<<conc_grid, conc_block, 0, stream>>>(fB, binary_col, k, n);
	cudaDeviceSynchronize();
	
	//perform xnor gemm
	threads_per_block = BLOCK_SIZE_XNOR;
	dim3 block(threads_per_block, threads_per_block, 1);
	dim3 grid(n/threads_per_block + 1, m/threads_per_block + 1);
	xnor_gemm<<<grid, block, 0, stream>>>(wmat_binarized, binary_col, fC, m, k/BITS_PER_BINARY_WORD, n);		
	cudaDeviceSynchronize();	

	// NOTE: gemm not correct for conv layer!!!!
    // float* bcol_host = (float*)malloc(n*sizeof(float));
	// cudaMemcpy(bcol_host, fC, n*sizeof(float), cudaMemcpyDeviceToHost);
	// //print
	// for (int i=0; i<n; i++) {		
	// 	std::cout << bcol_host[i] << ' ';
	// }
	// std::cout << std::endl;
	// free(bcol_host);
}
}  // namespace cuda

	inline void BinaryConvolutionForward(int m, int n, int k,
									mxnet::op::xnor::BINARY_WORD* wmat_binarized,
									Tensor<gpu, 1, float> &workspace,
									const Tensor<gpu, 2, float> &in_col,
									Tensor<gpu, 2, float> &temp_dst) {

		cuda::_BinaryConvolutionForward(m, n, k, wmat_binarized, workspace, in_col, temp_dst);
	}

	inline void BinaryConvolutionForward(int m, int n, int k,
									const Tensor<gpu, 2, float> &wmat,
									Tensor<gpu, 1, float> &workspace,
									const Tensor<gpu, 2, float> &in_col,
									Tensor<gpu, 2, float> &temp_dst) {
    	CHECK(false) << "cuda for non-concatenated weights not implemented";
	}

	template<typename DType>
	inline void BinaryConvolutionForward(int m, int n, int k,
									const Tensor<gpu, 2, DType> &wmat,
									Tensor<gpu, 1, DType> &workspace,
									const Tensor<gpu, 2, DType> &in_col,
									Tensor<gpu, 2, DType> &temp_dst) {
		CHECK(false) << "only float supported";
	}

	template<typename DType>
	inline void BinaryConvolutionForward(int m, int n, int k,
									mxnet::op::xnor::BINARY_WORD* wmat_binarized,
									Tensor<gpu, 1, DType> &workspace,
									const Tensor<gpu, 2, DType> &in_col,
									Tensor<gpu, 2, DType> &temp_dst) {
		CHECK(false) << "only float supported";
	}
} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(BinaryInferenceConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BinaryInferenceConvolutionOp<gpu, DType>(param);
  })
  return op;

}

}  // namespace op
}  // namespace mxnet

