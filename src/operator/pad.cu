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
// single_image_2d_replicate adapted from Torch
// https://github.com/torch/cunn/blob/master/lib/THCUNN/SpatialReplicationPadding.cu

template <int n_bits, typename DType>
__global__ void image_2d_replicate_kernel(Tensor<gpu, 4, DType> dst,
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
inline void image_2d_replicate(Tensor<gpu, 4, DType> &dst,
                               const Tensor<gpu, 4, DType> &src,
                               const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_2d_replicate_kernel<kBaseThreadBits,
                            DType><<<dimGrid, dimBlock, 0, stream>>>(
      dst, src, padT, padL);
}

template <int n_bits, typename DType>
__global__ void image_2d_replicate_grad_kernel(
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
inline void image_2d_replicate_grad(Tensor<gpu, 4, DType> &grad_in,
                                    const Tensor<gpu, 4, DType> &grad_out,
                                    const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (grad_out.size(2) * grad_out.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_out.size(1), grad_out.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_out.stream_);
  image_2d_replicate_grad_kernel<kBaseThreadBits,
                                 DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padT, padL);
}

// Case 2: Constant Padding
template <int n_bits, typename DType>
__global__ void image_2d_constant_kernel(Tensor<gpu, 4, DType> dst,
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
inline void image_2d_constant(Tensor<gpu, 4, DType> &dst,
                              const Tensor<gpu, 4, DType> &src,
                              const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (dst.size(2) * dst.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, dst.size(1), dst.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  image_2d_constant_kernel<kBaseThreadBits,
                           DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src,
                                                                    padT, padL);
}

template <int n_bits, typename DType>
__global__ void image_2d_constant_grad_kernel(
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
inline void image_2d_constantgrad(Tensor<gpu, 4, DType> &grad_in,
                                  const Tensor<gpu, 4, DType> &grad_out,
                                  const mxnet::TShape &pad) {
  const int padT = pad[4];
  const int padL = pad[6];
  dim3 dimBlock(kBaseThreadNum);
  int xGridSize = (grad_out.size(2) * grad_out.size(3) + 256 - 1) / 256;
  dim3 dimGrid(xGridSize, grad_out.size(1), grad_out.size(0));
  CheckLaunchParam(dimGrid, dimBlock, "Pad");
  cudaStream_t stream = Stream<gpu>::GetStream(grad_out.stream_);
  image_2d_constant_grad_kernel<kBaseThreadBits,
                                DType><<<dimGrid, dimBlock, 0, stream>>>(
      grad_in, grad_out, padT, padL);
}

////////////////////////////////////////////////////////////////////////////////
}  // namespace cuda

template <typename DType>
void pad_image_2d(Tensor<gpu, 4, DType> &dst, const Tensor<gpu, 4, DType> src,
                  const mxnet::TShape pad, int pad_type,
                  DType padding_constant) {
  switch (pad_type) {
    case mxnet::op::pad_enum::kReplicate:
      cuda::image_2d_replicate(dst, src, pad);
      break;
    case mxnet::op::pad_enum::kConstant:
      // single_image_2d_constant(dst[n], src[n], pad, padding_constant);
      break;
  }
};

template <typename DType>
void pad_image_2d_grad(Tensor<gpu, 4, DType> &grad_in,
                       const Tensor<gpu, 4, DType> grad_out,
                       const mxnet::TShape pad, int pad_type,
                       DType padding_constant) {
  switch (pad_type) {
    case mxnet::op::pad_enum::kReplicate:
      cuda::image_2d_replicate_grad(grad_in, grad_out, pad);
      break;
    case mxnet::op::pad_enum::kConstant:
      // single_image_2d_constant(dst[n], src[n], pad, padding_constant);
      break;
  }
};

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
