/*!
 * Copyright (c) 2015 by Contributors
 * \file pad.cu
 * \brief
 * \author Sebastian Bodenstein
*/
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
__global__ void image_2d_edge_kernel(Tensor<gpu, 4, DType> dst,
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
inline void image_2d_edge(Tensor<gpu, 4, DType> dst,
                          const Tensor<gpu, 4, DType> &src,
                          const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_2d_edge_kernel<kBaseThreadBits,
                       DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src, padT,
                                                                padL);
}

template <int n_bits, typename DType>
__global__ void image_2d_edge_grad_kernel(Tensor<gpu, 4, DType> grad_in,
                                          const Tensor<gpu, 4, DType> grad_out,
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
inline void image_2d_edge_grad(Tensor<gpu, 4, DType> grad_in,
                               const Tensor<gpu, 4, DType> &grad_out,
                               const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (grad_out.size(2) * grad_out.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_out.size(1), grad_out.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_out.stream_);
  image_2d_edge_grad_kernel<kBaseThreadBits,
                            DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padT, padL);
}

// Case 2: Constant Padding
template <int n_bits, typename DType>
__global__ void image_2d_constant_kernel(Tensor<gpu, 4, DType> dst,
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
inline void image_2d_constant(Tensor<gpu, 4, DType> dst,
                              const Tensor<gpu, 4, DType> &src,
                              const mxnet::TShape &pad, const DType constant) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_2d_constant_kernel<kBaseThreadBits,
                           DType><<<dimGrid, dimBlock, 0, stream>>>(
      dst, src, padT, padL, constant);
}

template <int n_bits, typename DType>
__global__ void image_2d_constant_grad_kernel(
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
inline void image_2d_constant_grad(Tensor<gpu, 4, DType> grad_in,
                                   const Tensor<gpu, 4, DType> &grad_out,
                                   const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (grad_in.size(2) * grad_in.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_in.size(1), grad_in.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_in.stream_);
  image_2d_constant_grad_kernel<kBaseThreadBits,
                                DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padT, padL);
}

////////////////////////////////////////////////////////////////////////////////
}  // namespace cuda

template <typename DType>
void pad_image_2d(Tensor<gpu, 4, DType> dst, const Tensor<gpu, 4, DType> src,
                  const mxnet::TShape pad, int mode,
                  const DType constant_value) {
  switch (mode) {
    case mxnet::op::pad_enum::kEdge:
      cuda::image_2d_edge(dst, src, pad);
      break;
    case mxnet::op::pad_enum::kConstant:
      cuda::image_2d_constant(dst, src, pad, constant_value);
      break;
  }
}

template <typename DType>
void pad_image_2d_grad(Tensor<gpu, 4, DType> grad_in,
                       const Tensor<gpu, 4, DType> grad_out,
                       const mxnet::TShape pad, int mode) {
  switch (mode) {
    case mxnet::op::pad_enum::kEdge:
      cuda::image_2d_edge_grad(grad_in, grad_out, pad);
      break;
    case mxnet::op::pad_enum::kConstant:
      cuda::image_2d_constant_grad(grad_in, grad_out, pad);
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
