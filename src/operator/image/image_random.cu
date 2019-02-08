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

template<typename xpu, typename Dtype>
__global__ void ToTensorCudaKernel(const Tensor<xpu, 3, Dtype> input,
                                   const Tensor<xpu, 3, float> output,
                                   const int req,
                                   int N, int H, int W, int C,
                                   const float normalize_factor = 255.0f) {
    // We process one image per thread block.
    // In 3D case, we have only 1 block i.e., blockIdx.x
    // We do not use it.
    /*
    const int n = blockIdx.x;
    const int stride = H*W*C;

    // Get pointer to my blocks image
    int step = 0;
    if (N > 0) {
        step = n * stride;
    }
    */
    for (int c = 0; c < C; ++c) {
        for (int h = threadIdx.y; h < H; h += blockDim.y) {
            for (int w = threadIdx.x; w < W; w += blockDim.x) {
                KERNEL_ASSIGN(output[c][h][w], req,
                              input[h][w][c] / normalize_factor);
            }
        }
    }
}

template<typename DType, typename T1, typename T2>
void ToTensorImplCUDA(mshadow::Stream<gpu> *s,
                      const T1 input,
                      const T2 output,
                      const int req,
                      const float normalize_factor = 255.0f) {
    int blocks, H, W, C, N;
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    if (std::is_same<T1, Tensor<gpu, 3, DType>>::value) {
        // 3D Input - (H, W, C)
        N = 0;
        H = input.size(0);
        W = input.size(1);
        C = input.size(2);
        blocks = 1;
    } /*else {
        // 4D Input - (N, H, W, C)
        N = input.size()[0];
        H = input.size()[1];
        W = input.size()[2];
        C = input.size()[3];
        // blocks = N > 0 ? N : 1;
        blocks = N;
    }*/
    // One block per image.
    // Number of threads = (32, 32) is optimal, because,
    // computation is minimal and overhead of CUDA preparing
    // all threads is minimal.
    ToTensorCudaKernel<<<blocks, dim3(32, 32), 0, stream>>>(input,
        output, req, N, H, W, C, normalize_factor);
    MSHADOW_CUDA_POST_KERNEL_CHECK(ToTensorCudaKernel);
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
