#ifndef MXNET_OPERATOR_NN_POOL_CUH_
#define MXNET_OPERATOR_NN_POOL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "../mxnet_op.h"
#include "../../common/cuda_utils.h"

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
__global__ void pool_max_2d_gpu_kernel(const int nthreads, const DType* in_data,
                                       const int channels, const int height, const int width,
                                       const int pooled_height, const int pooled_width,
                                       const int kernel_h, const int kernel_w, const int stride_h,
                                       const int stride_w, const int pad_h, const int pad_w,
                                       OpReqType req_type, DType* out_data) {
  using mshadow::red::limits::MinValue;
  // index is the output image's pixel index in NCHW
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
    const DType* in_slice =
        in_data + (n * channels + c) * height * width;
    DType max_val = MinValue<DType>();
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int in_index = h * width + w;
        const DType in_val = in_slice[in_index];
        if (in_val > max_val) {
          max_val = in_val;
        }
      }
    }
    KERNEL_ASSIGN(out_data[index], req_type, max_val);
  }
}

template <typename DType>
__global__ void pool_sum_2d_gpu_kernel(const int nthreads, const DType* in_data, const int channels,
                                       const int height, const int width,
                                       const int pooled_height, const int pooled_width,
                                       const int kernel_h, const int kernel_w,
                                       const int stride_h, const int stride_w,
                                       const int pad_h, const int pad_w, OpReqType req_type,
                                       DType* out_data, bool getAvg = false) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int pw = index % pooled_width;
	  const int ph = (index / pooled_width) % pooled_height;
	  const int c = (index / pooled_width / pooled_height) % channels;
	  const int n = index / pooled_width / pooled_height / channels;
	  int hstart = ph * stride_h - pad_h;
	  int wstart = pw * stride_w - pad_w;
	  int hend = min(hstart + kernel_h, height + pad_h);
	  int wend = min(wstart + kernel_w, width + pad_w);
	  const int pool_size = (getAvg? (hend - hstart) * (wend - wstart) : 1);
	  hstart = max(hstart, 0);
	  wstart = max(wstart, 0);
	  hend = min(hend, height);
	  wend = min(wend, width);
	  DType sum = 0;
	  const DType* out_slice =
	 		in_data + (n * channels + c) * height * width;
	  for (int h = hstart; h < hend; ++h) {
		  for (int w = wstart; w < wend; ++w) {
		    sum += out_slice[h * width + w];
		  }
	  }
	  KERNEL_ASSIGN(out_data[index], req_type, sum / pool_size);
  }
}

template <typename DType>
__global__ void unpool_max_2d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const DType* in_data, const DType* out_data,
                                         const int channels, const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         DType* in_grad) {
  // index is the output image's pixel index in NCHW
  // the order has to be consistent with pooling max
  // to avoid adding out_grad to the wrong in_grad
  // in the case where there are multiple max pixels
  // covered by a kernel window
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
    // in data/grad offset batch and channel dims
    int in_offset = (n * channels + c) * height * width;
    const DType* in_data_slice = in_data + in_offset;
    int max_idx = -1;
    DType max_val = out_data[index];
    bool found = false;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        max_idx = h * width + w;
        if (in_data_slice[max_idx] == max_val) {
          found = true;
          break;
        }
      }
      if (found) break;
    }

    // In the case where pad > 0 and kernel = 1, for example,
    // max_idx can be -1 reaching this step.
    if (max_idx >= 0) {
      atomicAdd(&in_grad[in_offset+max_idx], out_grad[index]);
    }
  }
}

template<typename DType>
__global__ void unpool_sum_2d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const int channels, const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         DType* in_grad, bool isAvg = false) {
  // index is the input image index in NCHW
  CUDA_KERNEL_LOOP(index, nthreads) {
	  // find out the local index
	  // find out the local offset
	  const int w = index % width + pad_w;
	  const int h = (index / width) % height + pad_h;
	  const int c = (index / width / height) % channels;
	  const int n = index / width / height / channels;
	  const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
	  const int phend = min(h / stride_h + 1, pooled_height);
	  const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
	  const int pwend = min(w / stride_w + 1, pooled_width);
	  DType gradient = 0;
	  const DType* out_grad_slice =
      out_grad + (n * channels + c) * pooled_height * pooled_width;
	  for (int ph = phstart; ph < phend; ++ph) {
	 	  for (int pw = pwstart; pw < pwend; ++pw) {
		    // figure out the pooling size
			  int hstart = ph * stride_h - pad_h;
			  int wstart = pw * stride_w - pad_w;
			  int hend = min(hstart + kernel_h, height + pad_h);
			  int wend = min(wstart + kernel_w, width + pad_w);
			  int pool_size = (isAvg? (hend - hstart) * (wend - wstart) : 1);
			  gradient += out_grad_slice[ph * pooled_width + pw] / pool_size;
		  }
	  }
    // if req=kWriteTo, in_grad has already been assigned zero values in unpool()
    // use "+=" here instead of "=" to accommodate when req=kAddTo
	  in_grad[index] += gradient;
  }
}

template<typename DType>
inline void pool(mshadow::Stream<gpu>* s, const DType* in_data, const TShape& ishape,
                 const TShape& oshape, const TShape& kernel, const TShape& pad,
                 const TShape& stride, const int pool_type, OpReqType req_type,
                 DType* out_data) {
  if (kernel.ndim() == 2) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_max_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], kernel[0], kernel[1],
                                   stride[0], stride[1], pad[0], pad[1], req_type, out_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_max_2d_gpu_kernel);
    } else if (pool_enum::kAvgPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_sum_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], kernel[0], kernel[1],
                                   stride[0], stride[1], pad[0], pad[1], req_type, out_data, true);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_sum_2d_gpu_kernel);
    } else if (pool_enum::kSumPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_sum_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], kernel[0], kernel[1],
                                   stride[0], stride[1], pad[0], pad[1], req_type, out_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_sum_2d_gpu_kernel);
    } else {
      LOG(FATAL) << "Unknown pooling type " << pool_type;
    }
  }
}

template<typename DType>
inline void unpool(mshadow::Stream<gpu>* s, const DType* out_grad, const DType* in_data,
                   const DType* out_data, const TShape& ishape, const TShape& oshape,
                   const TShape& kernel, const TShape& pad, const TShape& stride,
                   const int pool_type, OpReqType req_type, DType* in_grad) {
  if (mxnet::kNullOp == req_type) return;
  if (mxnet::kAddTo != req_type) {
    mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(s, ishape.Size(), in_grad);
  }
  if (kernel.ndim() == 2) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_max_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     oshape.Size(), out_grad, in_data, out_data,
                                     ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], kernel[0], kernel[1],
                                     stride[0], stride[1], pad[0], pad[1], in_grad);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_max_2d_gpu_kernel);
    } else if (pool_enum::kAvgPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_sum_2d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], kernel[0], kernel[1],
                                     stride[0], stride[1], pad[0], pad[1], in_grad, true);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_sum_2d_gpu_kernel);
    } else if (pool_enum::kSumPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_sum_2d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], kernel[0], kernel[1],
                                     stride[0], stride[1], pad[0], pad[1], in_grad);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_sum_2d_gpu_kernel);
    } else {
      LOG(FATAL) << "Unknown pooling type " << pool_type;
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_POOL_CUH_
