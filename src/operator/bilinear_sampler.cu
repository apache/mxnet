/*!
 * Copyright (c) 2017 by Contributors
 * \file bilinear_sampler.cu
 * \brief
 * \author Xu Dong
*/

#include "./bilinear_sampler-inl.h"
#include <algorithm>
#include "../common/cuda_utils.h"
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn_bilinear_sampler-inl.h"
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR

namespace mshadow {
namespace cuda {
template<typename DType>
__device__ bool between(DType value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}
template<typename DType>
__global__ void BilinearSamplerForwardKernel(const int i_c, const int i_h,
                                              const int i_w, const DType* data,
                                              const DType* grid, const int o_n,
                                              const int o_c, const int o_h,
                                              const int o_w, DType* out) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < o_n * o_c * o_h * o_w;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, h, w) is the element in out
    int w = index % o_w;
    int h = (index / o_w) % o_h;
    int c = (index / o_w / o_h) % o_c;
    int n = index / o_w / o_h / o_c;
    index_t out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
    index_t grid_index = n * o_h * o_w * 2 + h * o_w + w;
    DType y_real = (*(grid + grid_index + o_h * o_w) + 1) * (i_h - 1) / 2;
    DType x_real = (*(grid + grid_index) + 1) * (i_w - 1) / 2;
    int top_left_y = static_cast<int>(floor(y_real));
    int top_left_x = static_cast<int>(floor(x_real));
    DType top_left_y_w = 1.0 - (y_real - top_left_y);
    DType top_left_x_w = 1.0 - (x_real - top_left_x);
    int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
    DType top_left_v = 0;
    DType top_right_v = 0;
    DType bottom_left_v = 0;
    DType bottom_right_v = 0;
    if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1))
      top_left_v = *(data + data_index);
    if (between(top_left_x + 1, 0, i_w-1) && between(top_left_y, 0, i_h-1))
      top_right_v = *(data + data_index + 1);
    if (between(top_left_x, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
      bottom_left_v = *(data + data_index + i_w);
    if (between(top_left_x+1, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
      bottom_right_v = *(data + data_index + i_w + 1);
    *(out+out_index) = top_left_v * top_left_y_w * top_left_x_w +
                        top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
                        bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
                        bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
  }
}

template<typename DType>
__global__ void BilinearSamplerBackwardKernel(const int i_c, const int i_h,
                                              const int i_w, const DType* grad,
                                              const DType* data, const int o_n,
                                              const int o_c, const int o_h,
                                              const int o_w, DType* g_input,
                                              const DType* grid_src,
                                              DType* grad_grid) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < o_n * o_h * o_w;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, h, w) is the element in grad
    int w = index % o_w;
    int h = (index / o_w) % o_h;
    int n = index / o_w / o_h;
    DType top_left_y_gw = 0.0;
    DType top_left_x_gw = 0.0;
    index_t grid_src_index = n * o_h * o_w * 2 + h * o_w + w;
    DType y_real = (*(grid_src + grid_src_index + o_h * o_w) + 1) * (i_h - 1) / 2;
    DType x_real = (*(grid_src + grid_src_index) + 1) * (i_w - 1) / 2;

    int top_left_y = static_cast<int>(floor(y_real));
    int top_left_x = static_cast<int>(floor(x_real));
    DType top_left_y_w = 1.0 - (y_real - top_left_y);
    DType top_left_x_w = 1.0 - (x_real - top_left_x);
    for (index_t c = 0; c < o_c; ++c) {
      index_t grad_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
      int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
      // calc 4 vertex value in input data
      DType top_left_v = 0;
      DType top_right_v = 0;
      DType bottom_left_v = 0;
      DType bottom_right_v = 0;
      // calc input grad
      if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
        atomicAdd(&g_input[data_index], *(grad + grad_index) * top_left_y_w * top_left_x_w);
        top_left_v = *(data + data_index);
      }
      if (between(top_left_x+1, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
        atomicAdd(&g_input[data_index + 1], *(grad + grad_index) * top_left_y_w
                                        * (1.0 - top_left_x_w));
        top_right_v = *(data + data_index + 1);
      }
      if (between(top_left_x, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
        atomicAdd(&g_input[data_index+ i_w], *(grad + grad_index) * (1.0 - top_left_y_w)
                                        * top_left_x_w);
        bottom_left_v = *(data + data_index + i_w);
      }
      if (between(top_left_x+1, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
        atomicAdd(&g_input[data_index+ i_w + 1], *(grad + grad_index) * (1.0 - top_left_y_w)
                                            * (1.0 - top_left_x_w));
        bottom_right_v = *(data + data_index + i_w + 1);
      }
      // calc weight grad of top_left_w, then multiple -1 is the grad of grid_src
      top_left_y_gw -= *(grad + grad_index) * (top_right_v - bottom_right_v +
                        (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                        * top_left_x_w);
      top_left_x_gw -= *(grad + grad_index) * (bottom_left_v - bottom_right_v +
                        (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                        * top_left_y_w);
    }
    // calc grad of grid
    *(grad_grid + grid_src_index + o_h * o_w) += top_left_y_gw * (i_h - 1) / 2;
    *(grad_grid + grid_src_index) += top_left_x_gw * (i_w - 1) / 2;
  }
}
}  // namespace cuda

template<typename DType>
inline void BilinearSamplerForward(const Tensor<gpu, 4, DType> &output,
                                    const Tensor<gpu, 4, DType> &input,
                                    const Tensor<gpu, 4, DType> &grid_src) {
    DType *out = output.dptr_;
    const DType *data = input.dptr_;
    const DType *grid = grid_src.dptr_;
    int o_n = output.size(0), o_c = output.size(1), o_h = output.size(2), o_w = output.size(3);
    int i_c = input.size(1), i_h = input.size(2), i_w = input.size(3);
    using namespace cuda;
    const int max_block = (output.shape_.Size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    const int grid_dim_x = (max_block > kMaxGridDim) ? kMaxGridDim : max_block;
    const int grid_dim_y =
      (max_block > kMaxGridDim) ? (max_block + kMaxGridDim - 1) / kMaxGridDim : 1;
    dim3 num_blocks(grid_dim_x, grid_dim_y);
    dim3 threads_per_block(kMaxThreadsPerBlock);
    CheckLaunchParam(num_blocks, threads_per_block, "bilinear sampler forward");
    cudaStream_t stream = Stream<gpu>::GetStream(output.stream_);
    cuda::BilinearSamplerForwardKernel<DType> << <num_blocks, threads_per_block, 0, stream >> >(
      i_c, i_h, i_w, data, grid, o_n, o_c, o_h, o_w, out);
    // post kernel check
    cudaError err = cudaPeekAtLastError();
    CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

template<typename DType>
inline void BilinearSamplerBackward(const Tensor<gpu, 4, DType> &input_grad,
                                     const Tensor<gpu, 4, DType> &ggrid,
                                     const Tensor<gpu, 4, DType> &output_grad,
                                     const Tensor<gpu, 4, DType> &input_data,
                                     const Tensor<gpu, 4, DType> &grid) {
  DType *g_input = input_grad.dptr_;
  DType *grad_grid = ggrid.dptr_;
  const DType *grid_src = grid.dptr_;
  const DType *grad = output_grad.dptr_;
  const DType *data = input_data.dptr_;
  int o_n = output_grad.size(0), o_c = output_grad.size(1),
      o_h = output_grad.size(2), o_w = output_grad.size(3);
  int i_c = input_data.size(1), i_h = input_data.size(2), i_w = input_data.size(3);
  using namespace cuda;
  const int max_block = (output_grad.shape_.Size() / o_c + kMaxThreadsPerBlock - 1)
                        / kMaxThreadsPerBlock;
  const int grid_dim_x = (max_block > kMaxGridDim) ? kMaxGridDim : max_block;
  const int grid_dim_y =
    (max_block > kMaxGridDim) ? (max_block + kMaxGridDim - 1) / kMaxGridDim : 1;
  dim3 num_blocks(grid_dim_x, grid_dim_y);
  dim3 threads_per_block(kMaxThreadsPerBlock);
  CheckLaunchParam(num_blocks, threads_per_block, "bilinear sampler backward");
  cudaStream_t stream = Stream<gpu>::GetStream(input_grad.stream_);
  cuda::BilinearSamplerBackwardKernel<DType> << <num_blocks, threads_per_block, 0, stream >> >(
    i_c, i_h, i_w, grad, data, o_n, o_c, o_h, o_w, g_input, grid_src, grad_grid);
  // post kernel check
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(BilinearSamplerParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNBilinearSamplerOp<DType>(param);
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BilinearSamplerOp<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
  return op;
}

}  // namespace op
}  // namespace mxnet
