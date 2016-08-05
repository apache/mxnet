/*!
 * Copyright (c) 2016 by Contributors
 * \file moon_output.cu
 * \brief
 * \author Wei Wu
*/
#include <vector>
#include <mshadow/tensor.h>
#include <mshadow/cuda/tensor_gpu-inl.cuh>
#include "./moon_output-inl.h"

#define CU2DBLOCK_X 32
#define CU2DBLOCK_Y 32

namespace mshadow {
namespace cuda{
template<typename DType>
__global__ void MoonBackwardKernel(DType *grad, const DType *data, const DType *label, const float *src_dist,
	const int cols, const int rows, const int stride) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int num_threads_x = blockDim.x * gridDim.x;
	int num_threads_y = blockDim.y * gridDim.y;
	DType weight = 0.0;
	for (int index = 0; i < cols && j < rows; i += num_threads_x, j += num_threads_y) {
		index = i * stride + j;
		if (1 == int(label[index]) && src_dist[i] > 0.5) {
			weight = (1 - src_dist[i]) / src_dist[i];
		}
		else if (-1 == int(label[index]) && src_dist[i] < 0.5) {
			weight = src_dist[i] / (1 - src_dist[i]);
		}
		else {
			weight = 1.0;
		}
		grad[index] = 2.0 * (data[index] - label[index]) * weight;
	}
}

template<typename DType>
inline void MoonBackward(const Tensor<gpu, 2, DType> &grad_data,
							const Tensor<gpu, 2, DType> &out_data,
							const Tensor<gpu, 2, DType> &input_label,
							const std::vector<float> &src_dist) {
	const DType *data = out_data.dptr_;
	const DType *label = input_label.dptr_;
	DType *grad = grad_data.dptr_;
	dim3 threads_per_block(CU2DBLOCK_X, CU2DBLOCK_Y);
	dim3 num_blocks((out_data.size(1) + threads_per_block.x - 1) / threads_per_block.x, 
					(out_data.size(0) + threads_per_block.y - 1) / threads_per_block.y);
	CheckLaunchParam(num_blocks, threads_per_block, "Moon Backward");
	cudaStream_t stream = Stream<gpu>::GetStream(grad_data.stream_);
	// maybe these is a better solutive to construct a Tensor<gpu> with a std::vector
	float *dist;
	cudaMalloc((void**)&dist, src_dist.size()*sizeof(float));
	cudaMemcpyAsync(dist, &src_dist[0], src_dist.size()*sizeof(float), cudaMemcpyHostToDevice, stream);
	MoonBackwardKernel<DType> << <num_blocks, threads_per_block, 0, stream >> >(grad, data, label, dist, 
		out_data.size(1), out_data.size(0), out_data.size(0));
}
} // namespace cuda

template<typename DType>
inline void MoonBackward(const Tensor<gpu, 2, DType> &grad_data,
	const Tensor<gpu, 2, DType> &out_data,
	const Tensor<gpu, 2, DType> &input_label,
	const std::vector<float> &src_dist) {
	cuda::MoonBackward(grad_data, out_data, input_label, src_dist);
}
} // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MoonOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MoonOutputOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

