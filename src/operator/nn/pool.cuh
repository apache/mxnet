#ifndef MXNET_OPERATOR_NN_POOL_CUH_
#define MXNET_OPERATOR_NN_POOL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

/*!
 * \brief Get the number of blocks for cuda kernel given N
 */
inline int cuda_get_num_blocks(const int N) {
  using namespace mshadow::cuda;
  return std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
}

template <typename DType>
__global__ void pool_max_2d_gpu_kernel(const int nthreads, const DType* const in_data,
                                       const int channels, const int height, const int width,
                                       const int pooled_height, const int pooled_width,
                                       const int kernel_h, const int kernel_w, const int stride_h,
                                       const int stride_w, const int pad_h, const int pad_w,
                                       OpReqType req_type, DType* const out_data, int32_t* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    const DType* const in_slice =
        in_data + (n * channels + c) * height * width;
    int in_index = hstart * width + wstart;
    DType max_val = in_slice[in_index];
    int max_idx = in_index;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        in_index = h * width + w;
        const DType in_val = in_slice[in_index];
        if (in_val > max_val) {
          max_val = in_val;
          max_idx = in_index;
        }
      }
    }
    mask[index] = max_idx;
    KERNEL_ASSIGN(out_data[index], req_type, max_val);
  }
}

template <typename DType>
__global__ void unpool_max_2d_gpu_kernel(const int nthreads, const DType* const out_data,
                                         const int32_t* const mask, const int channels,
                                         const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         OpReqType req_type, DType* const in_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    DType gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const DType* const out_data_slice = out_data + offset;
    const int* const mask_slice = mask + offset;
    int in_index = h * width + w;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int pooled_index = ph * pooled_width + pw;
        if (mask_slice[pooled_index] == in_index) {
          gradient += out_data_slice[pooled_index];
        }
      }
    }
    KERNEL_ASSIGN(in_data[index], req_type, gradient);
  }
}

template<typename DType>
inline void pool(mshadow::Stream<gpu>* s, const DType* in_data, const TShape& ishape,
                 const TShape& oshape, const TShape& kernel, const TShape& pad,
                 const TShape& stride, const int pool_type, OpReqType req_type,
                 DType* out_data, int32_t* mask = nullptr) {
  if (kernel.ndim() == 2) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_max_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], kernel[0], kernel[1],
                                   stride[0], stride[1], pad[0], pad[1], req_type, out_data, mask);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_max_2d_gpu_kernel);
    }
  }
}

template<typename DType>
inline void unpool(mshadow::Stream<gpu>* s, const DType* out_data, const TShape& ishape,
                   const TShape& oshape, const TShape& kernel, const TShape& pad,
                   const TShape& stride, const int pool_type, OpReqType req_type, DType* in_data,
                   const int32_t* mask = nullptr) {
  if (mxnet::kNullOp == req_type) return;
  if (mxnet::kAddTo != req_type) {
    mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(s, ishape.Size(), in_data);
  }
  if (kernel.ndim() == 2) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_max_2d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_data, mask, ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], kernel[0], kernel[1],
                                     stride[0], stride[1], pad[0], pad[1], req_type, in_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_max_2d_gpu_kernel);
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_POOL_CUH_
