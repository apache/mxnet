/*!
 * Copyright (c) 2016 by Contributors
 * \file spatial_transformer.cu
 * \brief
 * \author Wei Wu
*/

#include "./spatial_transformer-inl.h"
#include <algorithm>
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn_spatial_transformer-inl.h"
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR

namespace mshadow {
template<typename DType>
__global__ void BilinearSamplingForwardKernel(const int i_c, const int i_h,
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
    index_t top_left_y = min(i_h, max(0, static_cast<int>(floor(y_real))));
    index_t top_left_x = min(i_w, max(0, static_cast<int>(floor(x_real))));
    DType top_left_y_w = 1.0 - (y_real - top_left_y);
    DType top_left_x_w = 1.0 - (x_real - top_left_x);
    index_t data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
    DType top_left_v = *(data + data_index);
    DType top_right_v = *(data + data_index + 1);
    DType bottom_left_v = *(data + data_index + i_w);
    DType bottom_right_v = *(data + data_index + i_w + 1);
    *(out+out_index) = top_left_v * top_left_y_w * top_left_x_w +
                       top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
                       bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
                       bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
    }
}

template<typename DType>
__global__ void BilinearSamplingBackwardKernel(const int i_c, const int i_h,
                                              const int i_w, const DType* grad,
                                              const DType* data, const int o_n,
                                              const int o_c, const int o_h,
                                              const int o_w, DType* g_input,
                                              DType* grid_src) {
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
    index_t top_left_y = min(i_h, max(0, static_cast<int>(floor(y_real))));
    index_t top_left_x = min(i_w, max(0, static_cast<int>(floor(x_real))));
    DType top_left_y_w = 1.0 - (y_real - top_left_y);
    DType top_left_x_w = 1.0 - (x_real - top_left_x);
    for (index_t c = 0; c < o_c; ++c) {
      index_t grad_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
      index_t data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
      // calc 4 vertex value in input data
      DType top_left_v = *(data + data_index);
      DType top_right_v = *(data + data_index + 1);
      DType bottom_left_v = *(data + data_index + i_w);
      DType bottom_right_v = *(data + data_index + i_w + 1);
      // calc input grad
      *(g_input + data_index) += *(grad + grad_index) * top_left_y_w * top_left_x_w;
      *(g_input + data_index + 1) += *(grad + grad_index) * top_left_y_w * (1.0 - top_left_x_w);
      *(g_input + data_index+ i_w) += *(grad + grad_index) * (1.0 - top_left_y_w) * top_left_x_w;
      *(g_input + data_index+ i_w + 1) += *(grad + grad_index) * (1.0 - top_left_y_w) *
                                          (1.0 - top_left_x_w);
      // calc weight grad of top_left_w, then multiple -1 is the grad of grid_src
      top_left_y_gw -= *(grad + grad_index) * (top_right_v - bottom_right_v +
                       (top_left_v - top_right_v - bottom_left_v + bottom_right_v) * top_left_x_w);
      top_left_x_gw -= *(grad + grad_index) * (bottom_left_v - bottom_right_v + (top_left_v -
                       top_right_v - bottom_left_v + bottom_right_v) * top_left_y_w);
    }
    // calc grid_src grad
    *(grid_src + grid_src_index + o_h * o_w) = top_left_y_gw * (i_h - 1) / 2;
    *(grid_src + grid_src_index) = top_left_x_gw * (i_w - 1) / 2;
  }
}

template<typename DType>
inline void BilinearSamplingForward(const Tensor<gpu, 4, DType> &output,
                                    const Tensor<gpu, 4, DType> &input,
                                    const Tensor<gpu, 3, DType> grid_src) {
    DType *out = output.dptr_;
    const DType *data = input.dptr_;
    const DType *grid = grid_src.dptr_;
    int o_n = output.size(0), o_c = output.size(1), o_h = output.size(2), o_w = output.size(3);
    int i_c = input.size(1), i_h = input.size(2), i_w = input.size(3);
    using namespace cuda;
    const int max_block = (output.shape_.Size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    dim3 num_blocks(kMaxGridDim, (max_block + kMaxGridDim - 1) / kMaxGridDim);
    dim3 threads_per_block(kMaxThreadsPerBlock);
    CheckLaunchParam(num_blocks, threads_per_block, "spatial transformer forward");
    cudaStream_t stream = Stream<gpu>::GetStream(output.stream_);
    BilinearSamplingForwardKernel<DType> << <num_blocks, threads_per_block, 0, stream >> >(
      i_c, i_h, i_w, data, grid, o_n, o_c, o_h, o_w, out);
}

template<typename DType>
inline void BilinearSamplingBackward(const Tensor<gpu, 4, DType> &input_grad,
                                     const Tensor<gpu, 3, DType> &grid_src_data,
                                     const Tensor<gpu, 4, DType> &output_grad,
                                     const Tensor<gpu, 4, DType> &input_data) {
  DType *g_input = input_grad.dptr_;
  DType *grid_src = grid_src_data.dptr_;
  const DType *grad = output_grad.dptr_;
  const DType *data = input_data.dptr_;
  int o_n = output_grad.size(0), o_c = output_grad.size(1),
      o_h = output_grad.size(2), o_w = output_grad.size(3);
  int i_c = input_data.size(1), i_h = input_data.size(2), i_w = input_data.size(3);
  using namespace cuda;
  const int max_block = (output_grad.shape_.Size() / o_c + kMaxThreadsPerBlock - 1)
                        / kMaxThreadsPerBlock;
  dim3 num_blocks(kMaxGridDim, (max_block + kMaxGridDim - 1) / kMaxGridDim);
  dim3 threads_per_block(kMaxThreadsPerBlock);
  CheckLaunchParam(num_blocks, threads_per_block, "spatial transformer backward");
  cudaStream_t stream = Stream<gpu>::GetStream(input_grad.stream_);
  BilinearSamplingBackwardKernel<DType> << <num_blocks, threads_per_block, 0, stream >> >(
    i_c, i_h, i_w, grad, data, o_n, o_c, o_h, o_w, g_input, grid_src);
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(SpatialTransformerParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNSpatialTransformerOp<DType>(param);
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SpatialTransformerOp<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
  return op;
}

}  // namespace op
}  // namespace mxnet
