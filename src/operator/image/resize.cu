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
 * Copyright (c) 2019 by Contributors
 * \file bilinear_resize.cu
 * \brief bilinear resize operator
 * \author Hang Zhang, Jake Lee
*/
#include <algorithm>
#include "./resize-inl.h"
#include "../contrib/bilinear_resize-inl.cuh"

namespace mxnet {
namespace op {
namespace image {

using namespace mshadow;

template<typename DType, typename T, typename AccReal>
void ResizeImplCUDA(mshadow::Stream<gpu> *s,
                      const T input,
                      const T output) {
  int outputHeight;
  int outputWidth;
  int inputHeight;
  int inputWidth;
  mxnet::op::ImageLayout layout;
  if (std::is_same<T, Tensor<gpu, 3, DType>>::value) {
    layout = HWC;
    outputHeight = output.size(0);
    outputWidth = output.size(1);
    inputHeight = input.size(0);
    inputWidth = input.size(1);
  } else {
    layout = NHWC;
    outputHeight = output.size(1);
    outputWidth = output.size(2);
    inputHeight = input.size(1);
    inputWidth = input.size(2);
  }
  const AccReal rheight = (outputHeight > 1) ? (AccReal)(inputHeight - 1)/
                         (outputHeight - 1) : AccReal(0);
  const AccReal rwidth = (outputWidth > 1) ? (AccReal)(inputWidth - 1)/
                         (outputWidth - 1) : AccReal(0);
  const int num_kernels = outputHeight * outputWidth;
  const int num_threads = getNumThreads(inputHeight * inputWidth, false);
  dim3 blocks(static_cast<int>(num_kernels / num_threads) + 1);
  dim3 threads(num_threads);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  caffe_gpu_interp2_kernel<gpu, DType, AccReal>
  <<<blocks, threads , 0, stream>>>(
    num_kernels, rheight, rwidth, input, output, layout);
  MSHADOW_CUDA_POST_KERNEL_CHECK(caffe_gpu_interp2_kernel);
}

NNVM_REGISTER_OP(_image_resize)
.set_attr<FCompute>("FCompute<gpu>", Resize<gpu>);

}  // namespace image
}  // namespace op
}  // namespace mxnet
