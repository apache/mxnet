/*!
 * Copyright [2016] <Contributors>
 * \file warp.cu
 * \brief warp operator
 * \author Xu Dong
*/
#include "./warp-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

/* Warp is done in BHWD (coalescing is not obvious in BDHW)
   we assume BHWD format in inputImages
   we assume BHW(YX) format on grids
*/
namespace mshadow {
namespace warp_cuda {

__device__ void getTopLeft(double x, int xOut, int width, int& point, double& weight) {
  double xcoord = x + xOut;
  point = floor(xcoord);
  weight = max(1 - abs(xcoord - point), 0.0f);
}

__device__ bool between(int value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}

__device__ void sumReduceShMem(volatile double s[]) {
  /* obviously only works for 32 elements */
  /* sums up a shared memory array of 32 elements, stores it in s[0] */
  /* whole warp can then read first element (broadcasting) */
  if (threadIdx.x < 16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16]; }
  if (threadIdx.x < 8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
  if (threadIdx.x < 4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
  if (threadIdx.x < 2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
  if (threadIdx.x < 1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
}

// ==Warp forward kernel
template <typename Dtype>
__global__ void bilinearSamplingFromGrid(
  Dtype* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels,
  int inputImages_strideHeight, int inputImages_strideWidth, Dtype* grids_data,
  int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
  Dtype* output_data, int output_strideBatch, int output_strideChannels, int output_strideHeight,
  int output_strideWidth, int inputImages_channels, int inputImages_height,
  int inputImages_width, int output_width) {
  /* 
     each (32,16) block 16 output pixels (for coalescing the grid read)
     x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
     z = batch index
     threadIdx.x : used for features (coalescing is trivial)
  */
  const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
  const bool withinImageBounds = xOut < output_width;
  const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < output_width;
  const int yOut = blockIdx.y;
  const int width = inputImages_width;
  const int height = inputImages_height;
  const int b = blockIdx.z;
  double yf, xf;

  __shared__ double gridData[32];
  if (threadIdx.y == 0 && withinGridBounds) {
    gridData[threadIdx.x] =
    grids_data[b*grids_strideBatch + yOut*grids_strideHeight +
     xOut*grids_strideWidth + threadIdx.x];
  }

  __syncthreads();
  if (!withinImageBounds) return;

  xf = gridData[threadIdx.y*2];
  yf = gridData[threadIdx.y*2+1];

  int yInTopLeft, xInTopLeft;
  double yWeightTopLeft, xWeightTopLeft;

  getTopLeft(xf, xOut, inputImages_width, xInTopLeft, xWeightTopLeft);
  getTopLeft(yf, yOut, inputImages_height, yInTopLeft, yWeightTopLeft);

  const int outAddress = output_strideBatch * b +
  output_strideHeight * yOut + output_strideWidth * xOut;
  const int inTopLeftAddress = inputImages_strideBatch * b +
  inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
  const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
  const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
  const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

  double v = 0;
  double inTopLeft = 0;
  double inTopRight = 0;
  double inBottomLeft = 0;
  double inBottomRight = 0;

  // we are careful with the boundaries
  bool topLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft, 0, height-1);
  bool topRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft, 0, height-1);
  bool bottomLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft+1, 0, height-1);
  bool bottomRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft+1, 0, height-1);

  // interpolation happens here
  for (int t=threadIdx.x; t < inputImages_channels; t+= blockDim.x) {
      if (topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
      if (topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
      if (bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
      if (bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];

      v = xWeightTopLeft * yWeightTopLeft * inTopLeft
        + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
        + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
        + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

      output_data[outAddress + t] = v;
  }
}

// ==warp backward kernel
template<bool onlyGrid, typename Dtype> __global__ void backwardBilinearSampling(
  Dtype* inputImages_data, int inputImages_strideBatch,
  int inputImages_strideChannels, int inputImages_strideHeight,
  int inputImages_strideWidth, Dtype* gradInputImages_data,
  int gradInputImages_strideBatch, int gradInputImages_strideChannels,
  int gradInputImages_strideHeight, int gradInputImages_strideWidth,
  Dtype* grids_data, int grids_strideBatch, int grids_strideYX,
  int grids_strideHeight, int grids_strideWidth,
  Dtype* gradGrids_data, int gradGrids_strideBatch,
  int gradGrids_strideYX, int gradGrids_strideHeight,
  int gradGrids_strideWidth, Dtype* gradOutput_data, int gradOutput_strideBatch,
  int gradOutput_strideChannels, int gradOutput_strideHeight, int gradOutput_strideWidth,
  int inputImages_channels, int inputImages_height,
  int inputImages_width, int gradOutput_width) {
    /* 
       each (32,16) block 16 output pixels (for coalescing the grid read)
       x,y = coordinates
       z = batch index
       threads : used for features
    */
    const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
    const bool withinImageBounds = xOut < gradOutput_width;
    const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < gradOutput_width;
    const int yOut = blockIdx.y;
    const int width = inputImages_width;
    const int height = inputImages_height;
    const int b = blockIdx.z;
    double yf, xf;

    __shared__ double gridData[32];
    if (threadIdx.y == 0 && withinGridBounds) {
      gridData[threadIdx.x] =
      grids_data[b*grids_strideBatch + yOut*grids_strideHeight +
       xOut*grids_strideWidth + threadIdx.x];
    }

    __syncthreads();
    if (withinImageBounds) {
      xf = gridData[threadIdx.y*2];
      yf = gridData[threadIdx.y*2+1];

      int yInTopLeft, xInTopLeft;
      double yWeightTopLeft, xWeightTopLeft;

      getTopLeft(xf, xOut, inputImages_width, xInTopLeft, xWeightTopLeft);
      getTopLeft(yf, yOut, inputImages_height, yInTopLeft, yWeightTopLeft);

      const int inTopLeftAddress = inputImages_strideBatch * b +
       inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
      const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
      const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
      const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

      const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b +
       gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
      const int gradInputImagesTopRightAddress =
       gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
      const int gradInputImagesBottomLeftAddress =
       gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
      const int gradInputImagesBottomRightAddress =
       gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;
      const int gradOutputAddress = gradOutput_strideBatch * b +
       gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

      double topLeftDotProduct = 0;
      double topRightDotProduct = 0;
      double bottomLeftDotProduct = 0;
      double bottomRightDotProduct = 0;

      // we are careful with the boundaries
      bool topLeftIsIn = between(xInTopLeft, 0, width-1) &&
       between(yInTopLeft, 0, height-1);
      bool topRightIsIn = between(xInTopLeft+1, 0, width-1) &&
       between(yInTopLeft, 0, height-1);
      bool bottomLeftIsIn = between(xInTopLeft, 0, width-1) &&
       between(yInTopLeft+1, 0, height-1);
      bool bottomRightIsIn = between(xInTopLeft+1, 0, width-1) &&
       between(yInTopLeft+1, 0, height-1);

      /*
         In that loop we accumulate
         - gradients into the gradInputImages array with atomic adds
         - we compute the dot product that we need for the grid gradient
      */

      for (int t=threadIdx.x; t < inputImages_channels; t+= blockDim.x) {
         double gradOutValue = gradOutput_data[gradOutputAddress + t];

         // bool between(int value, int lowerBound, int upperBound)
         if (topLeftIsIn) {
            double inTopLeft = inputImages_data[inTopLeftAddress + t];
            topLeftDotProduct += inTopLeft * gradOutValue;
            if (!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopLeftAddress + t],
            xWeightTopLeft * yWeightTopLeft * gradOutValue);
         }

         if (topRightIsIn) {
            double inTopRight = inputImages_data[inTopRightAddress + t];
            topRightDotProduct += inTopRight * gradOutValue;
            if (!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopRightAddress + t],
            (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue);
         }

         if (bottomLeftIsIn) {
            double inBottomLeft = inputImages_data[inBottomLeftAddress + t];
            bottomLeftDotProduct += inBottomLeft * gradOutValue;
            if (!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftAddress + t],
            xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue);
         }

         if (bottomRightIsIn) {
            double inBottomRight = inputImages_data[inBottomRightAddress + t];
            bottomRightDotProduct += inBottomRight * gradOutValue;
            if (!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomRightAddress + t],
            (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue);
         }
      }

      /*
         Here we reduce the dot product and compute the grid gradient before writing it.
         could do shuffles and use no shmem at all but cuda arch is 2.0 
      */

      __shared__ volatile double __shmem[16][32];

      __shmem[threadIdx.y][threadIdx.x] = topLeftDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      topLeftDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = topRightDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      topRightDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = bottomLeftDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      bottomLeftDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = bottomRightDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      bottomRightDotProduct = __shmem[threadIdx.y][0];

      yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct -
       (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
      xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct -
       (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

      if (threadIdx.x == 0) {
         gridData[threadIdx.y*2+1] = yf;
         gridData[threadIdx.y*2] = xf;
      }
    }  // must put a big if condition in order not to hang at __syncthreads()...

    __syncthreads();
    if (threadIdx.y == 0 && withinGridBounds)
       gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight +
       xOut*gradGrids_strideWidth + threadIdx.x] = gridData[threadIdx.x];
}
}  // namespace warp_cuda

template <typename Dtype>
void WarpForward(
      const Tensor<gpu, 4, Dtype> &data,
      const Tensor<gpu, 4, Dtype> &grid,
      const Tensor<gpu, 4, Dtype> &out) {
  Dtype *data_bot = data.dptr_;
  Dtype *grid_bot = grid.dptr_;
  Dtype *top = out.dptr_;
  cudaStream_t stream = Stream<gpu>::GetStream(data.stream_);

  /* assume BHWD */
  int data_stride3 = 1;
  int data_stride2 = data.size(3);
  int data_stride1 = data_stride2 * data.size(2);
  int data_stride0 = data_stride1 * data.size(1);

  int grid_stride3 = 1;
  int grid_stride2 = grid.size(3);
  int grid_stride1 = grid_stride2 * grid.size(2);
  int grid_stride0 = grid_stride1 * grid.size(1);

  dim3 blocks((out.size(2)+15)/16, out.size(1), out.size(0));
  dim3 threads(32, 16);

  warp_cuda::bilinearSamplingFromGrid<Dtype><<< blocks, threads, 0, stream >>> (data_bot,
                                                      data_stride0,
                                                      data_stride3,
                                                      data_stride1,
                                                      data_stride2,
                                                      grid_bot,
                                                      grid_stride0,
                                                      grid_stride3,
                                                      grid_stride1,
                                                      grid_stride2,
                                                      top,
                                                      data_stride0,
                                                      data_stride3,
                                                      data_stride1,
                                                      data_stride2,
                                                      data.size(3),
                                                      data.size(1),
                                                      data.size(2),
                                                      out.size(2));
  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Warpop forward: %s\n", cudaGetErrorString(err));
  }
}

template <typename Dtype>
void WarpBackward(const Tensor<gpu, 4, Dtype> &grad_data,
                  const Tensor<gpu, 4, Dtype> &grad_grid,
                  const Tensor<gpu, 4, Dtype> &out_grad,
                  const Tensor<gpu, 4, Dtype> &data,
                  const Tensor<gpu, 4, Dtype> &grid,
                  bool only_grid) {
    Dtype *data_bot = data.dptr_;
    Dtype *grid_bot = grid.dptr_;
    Dtype *out_grad_ = out_grad.dptr_;
    Dtype *grad_data_ = grad_data.dptr_;
    Dtype *grad_grid_ = grad_grid.dptr_;
    cudaStream_t stream = Stream<gpu>::GetStream(data.stream_);

    /* assume BHWD */
    int data_stride3 = 1;
    int data_stride2 = data.size(3);
    int data_stride1 = data_stride2 * data.size(2);
    int data_stride0 = data_stride1 * data.size(1);
    int grid_stride3 = 1;

    int grid_stride2 = grid.size(3);
    int grid_stride1 = grid_stride2 * grid.size(2);
    int grid_stride0 = grid_stride1 * grid.size(1);
    dim3 blocks((out_grad.size(2)+15)/16, out_grad.size(1), out_grad.size(0));
    dim3 threads(32, 16);

    if (only_grid == false)
     // calc grad of grid and data
     warp_cuda::backwardBilinearSampling <false, Dtype> <<< blocks, threads, 0, stream >>> (
                                                        data_bot,
                                                        data_stride0,
                                                        data_stride3,
                                                        data_stride1,
                                                        data_stride2,
                                                        grad_data_,
                                                        data_stride0,
                                                        data_stride3,
                                                        data_stride1,
                                                        data_stride2,
                                                        grid_bot,
                                                        grid_stride0,
                                                        grid_stride3,
                                                        grid_stride1,
                                                        grid_stride2,
                                                        grad_grid_,
                                                        grid_stride0,
                                                        grid_stride3,
                                                        grid_stride1,
                                                        grid_stride2,
                                                        out_grad_,
                                                        data_stride0,
                                                        data_stride3,
                                                        data_stride1,
                                                        data_stride2,
                                                        data.size(3),
                                                        data.size(1),
                                                        data.size(2),
                                                        out_grad.size(2));
    else
      // only calc grad_grid
      warp_cuda::backwardBilinearSampling <true, Dtype> <<< blocks, threads, 0, stream>>> (
                                                        data_bot,
                                                        data_stride0,
                                                        data_stride3,
                                                        data_stride1,
                                                        data_stride2,
                                                        0, 0, 0, 0, 0,
                                                        grid_bot,
                                                        grid_stride0,
                                                        grid_stride3,
                                                        grid_stride1,
                                                        grid_stride2,
                                                        grad_grid_,
                                                        grid_stride0,
                                                        grid_stride3,
                                                        grid_stride1,
                                                        grid_stride2,
                                                        out_grad_,
                                                        data_stride0,
                                                        data_stride3,
                                                        data_stride1,
                                                        data_stride2,
                                                        data.size(3),
                                                        data.size(1),
                                                        data.size(2),
                                                        out_grad.size(2));

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in warpop backward: %s\n", cudaGetErrorString(err));
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(WarpParam param) {
  return new WarpOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
