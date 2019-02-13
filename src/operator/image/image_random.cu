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
* \file image_random.cu
* \brief GPU Implementation of image transformation operators
*/
#include <cuda_runtime_api.h>
#include "./image_random-inl.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
namespace image {

using namespace mshadow;

// ToTensor Kernel for 3D input
/*
 * In order to not generate the code that uses too many
 * registers (resulting in too many resources requested
 * error) we need to tell the compiler that we will be
 * launching this kernel with cuda::kMaxThreadsPerBlock
 * threads per block. Setting __launch_bounds__ ensures
 * that such configuration can always be launched.
 */
template<typename xpu, typename Dtype>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
ToTensorCudaKernel(const Tensor<xpu, 3, Dtype> input,
                   const Tensor<xpu, 3, float> output,
                   const int req,
                   const int N,
                   const int H,
                   const int W,
                   const int C,
                   const float normalize_factor) {
    // We process one image per thread block.
    // In 3D case, we have only 1 block i.e., blockIdx.x
    // We do not use it.
    for (int c = 0; c < C; ++c) {
        for (int h = threadIdx.y; h < H; h += blockDim.y) {
            for (int w = threadIdx.x; w < W; w += blockDim.x) {
                KERNEL_ASSIGN(output[c][h][w], req,
                              input[h][w][c] / normalize_factor);
            }
        }
    }
}

// ToTensor Kernel for 4D input
template<typename xpu, typename Dtype>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
ToTensorCudaKernel(const Tensor<xpu, 4, Dtype> input,
                   const Tensor<xpu, 4, float> output,
                   const int req,
                   const int N,
                   const int H,
                   const int W,
                   const int C,
                   const float normalize_factor) {
    // We process one image per thread block.
    const int n = blockIdx.x;

    for (int c = 0; c < C; ++c) {
        for (int h = threadIdx.y; h < H; h += blockDim.y) {
            for (int w = threadIdx.x; w < W; w += blockDim.x) {
                KERNEL_ASSIGN(output[n][c][h][w], req,
                              input[n][h][w][c] / normalize_factor);
            }
        }
    }
}

template<typename DType, typename T1, typename T2>
void ToTensorImplCUDA(mshadow::Stream<gpu> *s,
                      const T1 input,
                      const T2 output,
                      const int req,
                      const float normalize_factor) {
    int blocks, H, W, C, N;
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    if (std::is_same<T1, Tensor<gpu, 3, DType>>::value) {
        // 3D Input - (H, W, C)
        N = 0;
        H = input.size(0);
        W = input.size(1);
        C = input.size(2);
        blocks = 1;
    } else {
        // 4D Input - (N, H, W, C)
        N = input.size(0);
        H = input.size(1);
        W = input.size(2);
        C = input.size(3);
        blocks = N > 0 ? N : 1;
    }

    ToTensorCudaKernel<gpu, DType>
            <<<blocks, dim3(H, cuda::kMaxThreadsPerBlock / H), 0, stream>>>(input, output,
                req, N, H, W, C, normalize_factor);
        MSHADOW_CUDA_POST_KERNEL_CHECK(ToTensorCudaKernel);
}

// Normalize Kernel for 3D input
template<typename xpu, typename DType>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
NormalizeCudaKernel(const Tensor<xpu, 3, DType> input,
                    const Tensor<xpu, 3, DType> output,
                    const int req,
                    const int N,
                    const int H,
                    const int W,
                    const int C,
                    const float mean_d0,
                    const float mean_d1,
                    const float mean_d2,
                    const float std_d0,
                    const float std_d1,
                    const float std_d2) {
    // We process one image per thread block.
    // In 3D case, we have only 1 block i.e., blockIdx.x
    // We do not use it.

    float mean = mean_d0;
    float std = std_d0;
    for (int c = 0; c < C; ++c) {
        switch (c) {
            case 0 : mean = mean_d0;
                     std = std_d0;
                     break;
            case 1 : mean = mean_d1;
                     std = std_d1;
                     break;
            case 2 : mean = mean_d2;
                     std = std_d2;
                     break;
        }
        for (int h = threadIdx.y; h < H; h += blockDim.y) {
            for (int w = threadIdx.x; w < W; w += blockDim.x) {
                KERNEL_ASSIGN(output[c][h][w], req,
                              (input[c][h][w] - mean) / std);
            }
        }
    }
}

// Normalize Kernel for 4D input
template<typename xpu, typename DType>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
NormalizeCudaKernel(const Tensor<xpu, 4, DType> input,
                    const Tensor<xpu, 4, DType> output,
                    const int req,
                    const int N,
                    const int H,
                    const int W,
                    const int C,
                    const float mean_d0,
                    const float mean_d1,
                    const float mean_d2,
                    const float std_d0,
                    const float std_d1,
                    const float std_d2) {
    // We process one image per thread block.
    const int n = blockIdx.x;

    float mean = mean_d0;
    float std = std_d0;
    for (int c = 0; c < C; ++c) {
        switch (c) {
            case 0 : mean = mean_d0;
                     std = std_d0;
                     break;
            case 1 : mean = mean_d1;
                     std = std_d1;
                     break;
            case 2 : mean = mean_d2;
                     std = std_d2;
                     break;
        }
        for (int h = threadIdx.y; h < H; h += blockDim.y) {
            for (int w = threadIdx.x; w < W; w += blockDim.x) {
                KERNEL_ASSIGN(output[n][c][h][w], req,
                              (input[n][c][h][w] -  mean) / std);
            }
        }
    }
}

template<typename DType, typename T>
void NormalizeImplCUDA(mshadow::Stream<gpu> *s,
                       const T input,
                       const T output,
                       const int req,
                       const float mean_d0,
                       const float mean_d1,
                       const float mean_d2,
                       const float std_d0,
                       const float std_d1,
                       const float std_d2) {
    int blocks, H, W, C, N;
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    if (std::is_same<T, Tensor<gpu, 3, DType>>::value) {
        // 3D Input - (C, H, W)
        N = 0;
        C = input.size(0);
        H = input.size(1);
        W = input.size(2);
        blocks = 1;
    } else {
        // 4D Input - (N, C, H, W)
        N = input.size(0);
        C = input.size(1);
        H = input.size(2);
        W = input.size(3);
        blocks = N > 0 ? N : 1;
    }
    // One block per image.
    NormalizeCudaKernel<gpu, DType>
        <<<blocks, dim3(H, cuda::kMaxThreadsPerBlock / H), 0, stream>>>(input, output,
            req, N, H, W, C, mean_d0, mean_d1, mean_d2,
            std_d0, std_d1, std_d2);
    MSHADOW_CUDA_POST_KERNEL_CHECK(NormalizeCudaKernel);
}

// Normalize Backward Kernel for 3D input
template<typename xpu, typename DType>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
NormalizeBackwardCudaKernel(const Tensor<xpu, 3, DType> out_grad,
                            const Tensor<xpu, 3, DType> in_grad,
                            const int req,
                            const int N,
                            const int H,
                            const int W,
                            const int C,
                            const float std_d0,
                            const float std_d1,
                            const float std_d2) {
    // We process one image per thread block.
    // In 3D case, we have only 1 block i.e., blockIdx.x
    // We do not use it.

    float std = std_d0;
    for (int c = 0; c < C; ++c) {
        switch (c) {
            case 0 : std = std_d0;
                     break;
            case 1 : std = std_d1;
                     break;
            case 2 : std = std_d2;
                     break;
        }
        for (int h = threadIdx.y; h < H; h += blockDim.y) {
            for (int w = threadIdx.x; w < W; w += blockDim.x) {
                KERNEL_ASSIGN(in_grad[c][h][w], req,
                              out_grad[c][h][w] * (1 / std));
            }
        }
    }
}

// Normalize Backward Kernel for 3D input
template<typename xpu, typename DType>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
NormalizeBackwardCudaKernel(const Tensor<xpu, 4, DType> out_grad,
                            const Tensor<xpu, 4, DType> in_grad,
                            const int req,
                            const int N,
                            const int H,
                            const int W,
                            const int C,
                            const float std_d0,
                            const float std_d1,
                            const float std_d2) {
    // We process one image per thread block.
    const int n = blockIdx.x;

    float std = std_d0;
    for (int c = 0; c < C; ++c) {
        switch (c) {
            case 0 : std = std_d0;
                     break;
            case 1 : std = std_d1;
                     break;
            case 2 : std = std_d2;
                     break;
        }
        for (int h = threadIdx.y; h < H; h += blockDim.y) {
            for (int w = threadIdx.x; w < W; w += blockDim.x) {
                KERNEL_ASSIGN(in_grad[n][c][h][w], req,
                              out_grad[n][c][h][w] * (1 / std));
            }
        }
    }
}

template<typename DType, typename T>
void NormalizeBackwardImplCUDA(mshadow::Stream<gpu> *s,
                               const T out_grad,
                               const T in_grad,
                               const int req,
                               const float std_d0,
                               const float std_d1,
                               const float std_d2) {
    int blocks, H, W, C, N;
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    if (std::is_same<T, Tensor<gpu, 3, DType>>::value) {
        // 3D Input - (C, H, W)
        N = 0;
        C = out_grad.size(0);
        H = out_grad.size(1);
        W = out_grad.size(2);
        blocks = 1;
    } else {
        // 4D Input - (N, C, H, W)
        N = out_grad.size(0);
        C = out_grad.size(1);
        H = out_grad.size(2);
        W = out_grad.size(3);
        blocks = N > 0 ? N : 1;
    }
    // One block per image.
    NormalizeBackwardCudaKernel<gpu, DType>
        <<<blocks, dim3(H, cuda::kMaxThreadsPerBlock / H), 0, stream>>>(out_grad, in_grad,
            req, N, H, W, C, std_d0, std_d1, std_d2);
    MSHADOW_CUDA_POST_KERNEL_CHECK(NormalizeBackwardCudaKernel);
}

NNVM_REGISTER_OP(_image_to_tensor)
.set_attr<FCompute>("FCompute<gpu>", ToTensorOpForward<gpu>);

NNVM_REGISTER_OP(_image_normalize)
.set_attr<FCompute>("FCompute<gpu>", NormalizeOpForward<gpu>);

NNVM_REGISTER_OP(_backward_image_normalize)
.set_attr<FCompute>("FCompute<gpu>", NormalizeOpBackward<gpu>);

}  // namespace image
}  // namespace op
}  // namespace mxnet
