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
 * Copyright (c) 2015 by Contributors
 * \file pad.cu
 * \brief
 * \author Sebastian Bodenstein
*/
#include <algorithm>
#include "./pad-inl.h"
#include "../common/cuda_utils.h"

namespace mshadow {
namespace cuda {

////////////////////////////////////////////////////////////////////////////////
// Special Case: 2d image (so only pad width + height)

// Case 1: Replication Padding
// single_image_2d_edge adapted from Torch
// https://github.com/torch/cunn/blob/master/lib/THCUNN/SpatialReplicationPadding.cu

template <int n_bits, typename DType>
__global__ void image_2d_pad_edge_kernel(Tensor<gpu, 4, DType> dst,
                                         const Tensor<gpu, 4, DType> src,
                                         const int padT, const int padL) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= dst.size(2) * dst.size(3)) {
    return;
  }
  int outputPointX = outputPointId % dst.size(3);
  int outputPointY = outputPointId / dst.size(3);

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);

  int inputPointX =
      min(max(padL, outputPointX), src.size(3) + padL - 1) - oStartX + iStartX;
  int inputPointY =
      min(max(padT, outputPointY), src.size(2) + padT - 1) - oStartY + iStartY;

  DType valueToCopy = src[batch][plane][inputPointY][inputPointX];
  dst[batch][plane][outputPointY][outputPointX] = valueToCopy;
}

template <typename DType>
inline void image_pad_edge(Tensor<gpu, 4, DType> dst,
                           const Tensor<gpu, 4, DType> &src,
                           const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_2d_pad_edge_kernel<kBaseThreadBits,
                           DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src,
                                                                    padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_2d_pad_edge_kernel);
}

template <int n_bits, typename DType>
__global__ void image_2d_pad_edge_grad_kernel(
    Tensor<gpu, 4, DType> grad_in, const Tensor<gpu, 4, DType> grad_out,
    const int padT, const int padL) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= grad_out.size(2) * grad_out.size(3)) {
    return;
  }
  int outputPointX = outputPointId % grad_out.size(3);
  int outputPointY = outputPointId / grad_out.size(3);

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);
  int inputPointX = min(max(padL, outputPointX), grad_in.size(3) + padL - 1) -
                    oStartX + iStartX;
  int inputPointY = min(max(padT, outputPointY), grad_in.size(2) + padT - 1) -
                    oStartY + iStartY;
  DType valueToCopy = grad_out[batch][plane][outputPointY][outputPointX];
  atomicAdd(&grad_in[batch][plane][inputPointY][inputPointX], valueToCopy);
}

template <typename DType>
inline void image_pad_edge_grad(Tensor<gpu, 4, DType> grad_in,
                                const Tensor<gpu, 4, DType> &grad_out,
                                const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (grad_out.size(2) * grad_out.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_out.size(1), grad_out.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_out.stream_);
  image_2d_pad_edge_grad_kernel<kBaseThreadBits,
                                DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_2d_pad_edge_grad_kernel);
}

// Case 2: Constant Padding
template <int n_bits, typename DType>
__global__ void image_2d_pad_constant_kernel(Tensor<gpu, 4, DType> dst,
                                             const Tensor<gpu, 4, DType> src,
                                             const int padT, const int padL,
                                             const DType constant) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  if (outputPointId >= dst.size(2) * dst.size(3)) {
    return;
  }
  // cast sizes to int to use in min/max
  int Ny = src.size(2);
  int Nx = src.size(3);

  int plane = blockIdx.y;
  int batch = blockIdx.z;
  int outputPointX = outputPointId % dst.size(3);
  int outputPointY = outputPointId / dst.size(3);
  int checkT = max(0, outputPointY - padT + 1);
  int checkB = max(0, padT + Ny - outputPointY);
  int checkL = max(0, outputPointX - padL + 1);
  int checkR = max(0, padL + Nx - outputPointX);
  int inputPointX = min(max(outputPointX - padL, 0), Nx - 1);
  int inputPointY = min(max(outputPointY - padT, 0), Ny - 1);
  // 1 if need padding, 0 if not
  int need_pad = !(checkT * checkB * checkL * checkR);
  DType valueToCopy = src[batch][plane][inputPointY][inputPointX];
  dst[batch][plane][outputPointY][outputPointX] =
      valueToCopy * (!need_pad) + need_pad * constant;
}

template <typename DType>
inline void image_pad_constant(Tensor<gpu, 4, DType> dst,
                               const Tensor<gpu, 4, DType> &src,
                               const mxnet::TShape &pad, const DType constant) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_2d_pad_constant_kernel<kBaseThreadBits,
                               DType><<<dimGrid, dimBlock, 0, stream>>>(
      dst, src, padT, padL, constant);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_2d_pad_constant_kernel);
}

template <int n_bits, typename DType>
__global__ void image_2d_pad_constant_grad_kernel(
    Tensor<gpu, 4, DType> grad_in, const Tensor<gpu, 4, DType> grad_out,
    const int padT, const int padL) {
  int inPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  int pixel_num = grad_in.size(2) * grad_in.size(3);
  if (inPointId >= pixel_num) {
    return;
  }
  int inPointX = inPointId % grad_in.size(3);
  int inPointY = inPointId / grad_in.size(3);
  int outPointX = inPointX + padL;
  int outPointY = inPointY + padT;

  grad_in[batch][plane][inPointY][inPointX] =
      grad_out[batch][plane][outPointY][outPointX];
}

template <typename DType>
inline void image_pad_constant_grad(Tensor<gpu, 4, DType> grad_in,
                                    const Tensor<gpu, 4, DType> &grad_out,
                                    const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (grad_in.size(2) * grad_in.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_in.size(1), grad_in.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_in.stream_);
  image_2d_pad_constant_grad_kernel<kBaseThreadBits,
                                    DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_2d_pad_constant_grad_kernel);
}


// Case 3: Reflection Padding
// adapted from Torch
// https://github.com/torch/cunn/blob/master/lib/THCUNN/SpatialReflectionPadding.cu

template <int n_bits, typename DType>
__global__ void image_2d_pad_reflect_kernel(Tensor<gpu, 4, DType> dst,
                                         const Tensor<gpu, 4, DType> src,
                                         const int padT, const int padL) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= dst.size(2) * dst.size(3)) {
    return;
  }
  int outputPointX = outputPointId % dst.size(3);
  int outputPointY = outputPointId / dst.size(3);

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);

  int inputPointX = __sad(outputPointX, padL, 0)
                  - __sad(outputPointX, src.size(3) + padL - 1, 0)
                  - outputPointX
                  + 2 * padL + src.size(3) - 1
                  - oStartX + iStartX;

  int inputPointY = __sad(outputPointY, padT, 0)
                  - __sad(outputPointY, src.size(2) + padT - 1, 0)
                  - outputPointY
                  + 2 * padT + src.size(2) - 1
                  - oStartY + iStartY;

  DType valueToCopy = src[batch][plane][inputPointY][inputPointX];
  dst[batch][plane][outputPointY][outputPointX] = valueToCopy;
}

template <typename DType>
inline void image_pad_reflect(Tensor<gpu, 4, DType> dst,
                           const Tensor<gpu, 4, DType> &src,
                           const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_2d_pad_reflect_kernel<kBaseThreadBits,
                           DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src,
                                                                    padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_2d_pad_reflect_kernel);
}

template <int n_bits, typename DType>
__global__ void image_2d_pad_reflect_grad_kernel(
    Tensor<gpu, 4, DType> grad_in, const Tensor<gpu, 4, DType> grad_out,
    const int padT, const int padL) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= grad_out.size(2) * grad_out.size(3)) {
    return;
  }
  int outputPointX = outputPointId % grad_out.size(3);
  int outputPointY = outputPointId / grad_out.size(3);

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);

  int inputPointX = __sad(outputPointX, padL, 0)
                  - __sad(outputPointX, grad_in.size(3) + padL - 1, 0)
                  - outputPointX
                  + 2 * padL + grad_in.size(3) - 1
                  - oStartX + iStartX;

  int inputPointY = __sad(outputPointY, padT, 0)
                  - __sad(outputPointY, grad_in.size(2) + padT - 1, 0)
                  - outputPointY
                  + 2 * padT + grad_in.size(2) - 1
                  - oStartY + iStartY;

  DType valueToCopy = grad_out[batch][plane][outputPointY][outputPointX];
  atomicAdd(&grad_in[batch][plane][inputPointY][inputPointX], valueToCopy);
}

template <typename DType>
inline void image_pad_reflect_grad(Tensor<gpu, 4, DType> grad_in,
                                const Tensor<gpu, 4, DType> &grad_out,
                                const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (grad_out.size(2) * grad_out.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_out.size(1), grad_out.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_out.stream_);
  image_2d_pad_reflect_grad_kernel<kBaseThreadBits,
                                DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_2d_pad_reflect_grad_kernel);
}


////////////////////////////////////////////////////////////////////////////////
// Special Case: 3d image (pad depth + width + height)

// Case 1: Replication Padding
// single_image_3_edge adapted from Torch
// https://github.com/torch/cunn/blob/master/lib/THCUNN/VolumetricReplicationPadding.cu

template <int n_bits, typename DType>
__global__ void image_3d_pad_edge_kernel(Tensor<gpu, 5, DType> dst,
                                         const Tensor<gpu, 5, DType> src,
                                         const int padF, const int padT,
                                         const int padL) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= dst.size(2) * dst.size(3) * dst.size(4)) {
    return;
  }
  int outputPointX = outputPointId % dst.size(4);
  int outputPointY = (outputPointId / dst.size(4)) % dst.size(3);
  int outputPointZ = outputPointId / (dst.size(3) * dst.size(4));

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int iStartZ = max(0, -padF);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);
  int oStartZ = max(0, padF);

  int inputPointX =
      min(max(padL, outputPointX), src.size(4) + padL - 1) - oStartX + iStartX;
  int inputPointY =
      min(max(padT, outputPointY), src.size(3) + padT - 1) - oStartY + iStartY;
  int inputPointZ =
      min(max(padF, outputPointZ), src.size(2) + padF - 1) - oStartZ + iStartZ;

  DType valueToCopy = src[batch][plane][inputPointZ][inputPointY][inputPointX];
  dst[batch][plane][outputPointZ][outputPointY][outputPointX] = valueToCopy;
}

template <typename DType>
inline void image_pad_edge(Tensor<gpu, 5, DType> dst,
                           const Tensor<gpu, 5, DType> &src,
                           const mxnet::TShape &pad) {
  const int padF = pad[4];
  const int padT = pad[6];
  const int padL = pad[8];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) * dst.size(4) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_3d_pad_edge_kernel<kBaseThreadBits,
                           DType><<<dimGrid, dimBlock, 0, stream>>>(
      dst, src, padF, padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_3d_pad_edge_kernel);
}

template <int n_bits, typename DType>
__global__ void image_3d_pad_edge_grad_kernel(
    Tensor<gpu, 5, DType> grad_in, const Tensor<gpu, 5, DType> grad_out,
    const int padF, const int padT, const int padL) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= grad_out.size(2) * grad_out.size(3) * grad_out.size(4)) {
    return;
  }
  int outputPointX = outputPointId % grad_out.size(4);
  int outputPointY = (outputPointId / grad_out.size(4)) % grad_out.size(3);
  int outputPointZ = outputPointId / (grad_out.size(3) * grad_out.size(4));

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int iStartZ = max(0, -padF);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);
  int oStartZ = max(0, padF);

  int inputPointX = min(max(padL, outputPointX), grad_in.size(4) + padL - 1) -
                    oStartX + iStartX;
  int inputPointY = min(max(padT, outputPointY), grad_in.size(3) + padT - 1) -
                    oStartY + iStartY;
  int inputPointZ = min(max(padF, outputPointZ), grad_in.size(2) + padF - 1) -
                    oStartZ + iStartZ;
  DType valueToCopy =
      grad_out[batch][plane][outputPointZ][outputPointY][outputPointX];
  atomicAdd(&grad_in[batch][plane][inputPointZ][inputPointY][inputPointX],
            valueToCopy);
}

template <typename DType>
inline void image_pad_edge_grad(Tensor<gpu, 5, DType> grad_in,
                                const Tensor<gpu, 5, DType> &grad_out,
                                const mxnet::TShape &pad) {
  const int padF = pad[4];
  const int padT = pad[6];
  const int padL = pad[8];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize =
      (grad_out.size(2) * grad_out.size(3) * grad_out.size(4) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_out.size(1), grad_out.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_out.stream_);
  image_3d_pad_edge_grad_kernel<kBaseThreadBits,
                                DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padF, padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_3d_pad_edge_grad_kernel);
}

// Case 2: Constant Padding
template <int n_bits, typename DType>
__global__ void image_3d_pad_constant_kernel(Tensor<gpu, 5, DType> dst,
                                             const Tensor<gpu, 5, DType> src,
                                             const int padF, const int padT,
                                             const int padL,
                                             const DType constant) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  if (outputPointId >= dst.size(2) * dst.size(3) * dst.size(4)) {
    return;
  }
  // cast sizes to int to use in min/max
  int Nz = src.size(2);
  int Ny = src.size(3);
  int Nx = src.size(4);

  int plane = blockIdx.y;
  int batch = blockIdx.z;
  int outputPointX = outputPointId % dst.size(4);
  int outputPointY = (outputPointId / dst.size(4)) % dst.size(3);
  int outputPointZ = outputPointId / (dst.size(3) * dst.size(4));

  int checkFront = max(0, outputPointZ - padF + 1);
  int checkBack = max(0, padF + Nz - outputPointZ);
  int checkTop = max(0, outputPointY - padT + 1);
  int checkBottom = max(0, padT + Ny - outputPointY);
  int checkLeft = max(0, outputPointX - padL + 1);
  int checkRight = max(0, padL + Nx - outputPointX);

  int inputPointZ = min(max(outputPointZ - padF, 0), Nz - 1);
  int inputPointX = min(max(outputPointX - padL, 0), Nx - 1);
  int inputPointY = min(max(outputPointY - padT, 0), Ny - 1);
  // 1 if need padding, 0 if not
  int need_pad = !(checkFront * checkBack * checkTop * checkBottom * checkLeft *
                   checkRight);
  DType valueToCopy = src[batch][plane][inputPointZ][inputPointY][inputPointX];
  dst[batch][plane][outputPointZ][outputPointY][outputPointX] =
      valueToCopy * (!need_pad) + need_pad * constant;
}

template <typename DType>
inline void image_pad_constant(Tensor<gpu, 5, DType> dst,
                               const Tensor<gpu, 5, DType> &src,
                               const mxnet::TShape &pad, const DType constant) {
  const int padF = pad[4];
  const int padT = pad[6];
  const int padL = pad[8];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) * dst.size(4) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_3d_pad_constant_kernel<kBaseThreadBits,
                               DType><<<dimGrid, dimBlock, 0, stream>>>(
      dst, src, padF, padT, padL, constant);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_3d_pad_constant_kernel);
}

template <int n_bits, typename DType>
__global__ void image_3d_pad_constant_grad_kernel(
    Tensor<gpu, 5, DType> grad_in, const Tensor<gpu, 5, DType> grad_out,
    const int padF, const int padT, const int padL) {
  int inPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  int pixel_num = grad_in.size(2) * grad_in.size(3) * grad_in.size(4);
  if (inPointId >= pixel_num) {
    return;
  }

  int inPointX = inPointId % grad_in.size(4);
  int inPointY = (inPointId / grad_in.size(4)) % grad_in.size(3);
  int inPointZ = inPointId / (grad_in.size(3) * grad_in.size(4));

  int outPointZ = inPointZ + padF;
  int outPointX = inPointX + padL;
  int outPointY = inPointY + padT;

  grad_in[batch][plane][inPointZ][inPointY][inPointX] =
      grad_out[batch][plane][outPointZ][outPointY][outPointX];
}

template <typename DType>
inline void image_pad_constant_grad(Tensor<gpu, 5, DType> grad_in,
                                    const Tensor<gpu, 5, DType> &grad_out,
                                    const mxnet::TShape &pad) {
  const int padF = pad[4];
  const int padT = pad[6];
  const int padL = pad[8];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize =
      (grad_in.size(2) * grad_in.size(3) * grad_in.size(4) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_in.size(1), grad_in.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_in.stream_);
  image_3d_pad_constant_grad_kernel<kBaseThreadBits,
                                    DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padF, padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_3d_pad_constant_grad_kernel);
}

// Case 3: Reflection Padding

template <int n_bits, typename DType>
__global__ void image_3d_pad_reflect_kernel(Tensor<gpu, 5, DType> dst,
                                         const Tensor<gpu, 5, DType> src,
                                         const int padF, const int padT,
                                         const int padL) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= dst.size(2) * dst.size(3) * dst.size(4)) {
    return;
  }
  int outputPointX = outputPointId % dst.size(4);
  int outputPointY = (outputPointId / dst.size(4)) % dst.size(3);
  int outputPointZ = outputPointId / (dst.size(3) * dst.size(4));

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int iStartZ = max(0, -padF);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);
  int oStartZ = max(0, padF);

  int inputPointX = __sad(outputPointX, padL, 0)
                  - __sad(outputPointX, src.size(4) + padL - 1, 0)
                  - outputPointX
                  + 2 * padL + src.size(4) - 1
                  - oStartX + iStartX;

  int inputPointY = __sad(outputPointY, padT, 0)
                  - __sad(outputPointY, src.size(3) + padT - 1, 0)
                  - outputPointY
                  + 2 * padT + src.size(3) - 1
                  - oStartY + iStartY;

  int inputPointZ = __sad(outputPointZ, padF, 0)
                  - __sad(outputPointZ, src.size(2) + padF - 1, 0)
                  - outputPointZ
                  + 2 * padF + src.size(2) - 1
                  - oStartZ + iStartZ;

  DType valueToCopy = src[batch][plane][inputPointZ][inputPointY][inputPointX];
  dst[batch][plane][outputPointZ][outputPointY][outputPointX] = valueToCopy;
}

template <typename DType>
inline void image_pad_reflect(Tensor<gpu, 5, DType> dst,
                           const Tensor<gpu, 5, DType> &src,
                           const mxnet::TShape &pad) {
  const int padF = pad[4];
  const int padT = pad[6];
  const int padL = pad[8];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) * dst.size(4) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_3d_pad_reflect_kernel<kBaseThreadBits,
                           DType><<<dimGrid, dimBlock, 0, stream>>>(
      dst, src, padF, padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_3d_pad_reflect_kernel);
}

template <int n_bits, typename DType>
__global__ void image_3d_pad_reflect_grad_kernel(
    Tensor<gpu, 5, DType> grad_in, const Tensor<gpu, 5, DType> grad_out,
    const int padF, const int padT, const int padL) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= grad_out.size(2) * grad_out.size(3) * grad_out.size(4)) {
    return;
  }
  int outputPointX = outputPointId % grad_out.size(4);
  int outputPointY = (outputPointId / grad_out.size(4)) % grad_out.size(3);
  int outputPointZ = outputPointId / (grad_out.size(3) * grad_out.size(4));

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int iStartZ = max(0, -padF);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);
  int oStartZ = max(0, padF);

  int inputPointX = __sad(outputPointX, padL, 0)
                  - __sad(outputPointX, grad_in.size(4) + padL - 1, 0)
                  - outputPointX
                  + 2 * padL + grad_in.size(4) - 1
                  - oStartX + iStartX;

  int inputPointY = __sad(outputPointY, padT, 0)
                  - __sad(outputPointY, grad_in.size(3) + padT - 1, 0)
                  - outputPointY
                  + 2 * padT + grad_in.size(3) - 1
                  - oStartY + iStartY;

  int inputPointZ = __sad(outputPointZ, padF, 0)
                  - __sad(outputPointZ, grad_in.size(2) + padF - 1, 0)
                  - outputPointZ
                  + 2 * padF + grad_in.size(2) - 1
                  - oStartZ + iStartZ;

  DType valueToCopy =
      grad_out[batch][plane][outputPointZ][outputPointY][outputPointX];
  atomicAdd(&grad_in[batch][plane][inputPointZ][inputPointY][inputPointX],
            valueToCopy);
}

/*  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= grad_out.size(2) * grad_out.size(3)) {
    return;
  }
  int outputPointX = outputPointId % grad_out.size(3);
  int outputPointY = outputPointId / grad_out.size(3);

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);

  int inputPointX = __sad(outputPointX, padL, 0)
                  - __sad(outputPointX, grad_in.size(3) + padL - 1, 0)
                  - outputPointX
                  + 2 * padL + grad_in.size(3) - 1
                  - oStartX + iStartX;

  int inputPointY = __sad(outputPointY, padT, 0)
                  - __sad(outputPointY, grad_in.size(2) + padT - 1, 0)
                  - outputPointY
                  + 2 * padT + grad_in.size(2) - 1
                  - oStartY + iStartY;

  DType valueToCopy = grad_out[batch][plane][outputPointY][outputPointX];
  atomicAdd(&grad_in[batch][plane][inputPointY][inputPointX], valueToCopy);*/

template <typename DType>
inline void image_pad_reflect_grad(Tensor<gpu, 5, DType> grad_in,
                                const Tensor<gpu, 5, DType> &grad_out,
                                const mxnet::TShape &pad) {
  const int padF = pad[4];
  const int padT = pad[6];
  const int padL = pad[8];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize =
      (grad_out.size(2) * grad_out.size(3) * grad_out.size(4) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_out.size(1), grad_out.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_out.stream_);
  image_3d_pad_reflect_grad_kernel<kBaseThreadBits,
                                DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padF, padT, padL);
  MSHADOW_CUDA_POST_KERNEL_CHECK(image_3d_pad_reflect_grad_kernel);
}

////////////////////////////////////////////////////////////////////////////////
}  // namespace cuda

template <int dim, typename DType>
void pad_image(Tensor<gpu, dim, DType> dst, const Tensor<gpu, dim, DType> src,
               const mxnet::TShape pad, int mode, const DType constant_value) {
  switch (mode) {
    case mxnet::op::pad_enum::kEdge:
      cuda::image_pad_edge(dst, src, pad);
      break;
    case mxnet::op::pad_enum::kConstant:
      cuda::image_pad_constant(dst, src, pad, constant_value);
      break;
    case mxnet::op::pad_enum::kReflect:
      cuda::image_pad_reflect(dst, src, pad);
      break;
  }
}

template <int dim, typename DType>
void pad_image_grad(Tensor<gpu, dim, DType> grad_in,
                    const Tensor<gpu, dim, DType> grad_out,
                    const mxnet::TShape pad, int mode) {
  switch (mode) {
    case mxnet::op::pad_enum::kEdge:
      cuda::image_pad_edge_grad(grad_in, grad_out, pad);
      break;
    case mxnet::op::pad_enum::kConstant:
      cuda::image_pad_constant_grad(grad_in, grad_out, pad);
      break;
    case mxnet::op::pad_enum::kReflect:
      cuda::image_pad_reflect_grad(grad_in, grad_out, pad);
      break;
  }
}

}  // namespace mshadow

////////////////////////////////////////////////////////////////////////////////

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<gpu>(PadParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new PadOp<gpu, DType>(param); })
  return op;
}

}  // namespace op
}  // namespace mxnet
