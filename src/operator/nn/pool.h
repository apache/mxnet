#ifndef MXNET_OPERATOR_NN_POOL_H_
#define MXNET_OPERATOR_NN_POOL_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

namespace pool_enum {
enum PoolingOpInputs {kData};
enum PoolingOpOutputs {kOut, kMask};
enum PoolingOpType {kMaxPooling, kAvgPooling, kSumPooling};
enum PoolingOpPadConventionType {kValid, kFull};
}  // namespace pool_enum

template<typename DType>
inline void pool_max_2d_cpu(const DType* in_data, const TShape& ishape, const TShape& oshape,
                            const TShape& kernel, const TShape& pad, const TShape& stride,
                            OpReqType req_type, DType* out_data, int32_t* mask) {
  const index_t height = ishape[2], width = ishape[3];
  const index_t pooled_height = oshape[2], pooled_width = oshape[3];
  const index_t kernel_h = kernel[0], kernel_w = kernel[1];
  const index_t pad_h = pad[0], pad_w = pad[1];
  const index_t stride_h = stride[0], stride_w = stride[1];
  const index_t in_data_offset = ishape[2] * ishape[3];
  const index_t out_data_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (index_t ph = 0; ph < pooled_height; ++ph) {
        for (index_t pw = 0; pw < pooled_width; ++pw) {
          index_t tmp_h = ph * stride_h;
          index_t tmp_w = pw * stride_w;
          index_t hend = std::min(tmp_h + kernel_h - pad_h, height);
          index_t wend = std::min(tmp_w + kernel_w - pad_w, width);
          index_t hstart = (tmp_h > pad_h? tmp_h - pad_h : 0);
          index_t wstart = (tmp_w > pad_w? tmp_w - pad_w : 0);
          const index_t pool_index = ph * pooled_width + pw;
          index_t in_index = hstart * width + wstart;
          DType max_val = in_data[in_index];
          mask[pool_index] = in_index;
          for (index_t h = hstart; h < hend; ++h) {
            for (index_t w = wstart; w < wend; ++w) {
              in_index = h * width + w;
              if (in_data[in_index] > max_val) {
                max_val = in_data[in_index];
                mask[pool_index] = in_index;
              }
            }
          }
          KERNEL_ASSIGN(out_data[pool_index], req_type, max_val);
        }
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
      mask += out_data_offset;
    }
  }
}

template<typename DType>
inline void unpool_max_2d_cpu(mshadow::Stream<cpu>* s, const DType* out_data, const int32_t* mask,
                              const TShape& ishape, const TShape& oshape, DType* in_data) {
  const index_t pooled_height = oshape[2], pooled_width = oshape[3];
  const index_t in_data_offset = ishape[2] * ishape[3];
  const index_t out_data_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (index_t ph = 0; ph < pooled_height; ++ph) {
        for (index_t pw = 0; pw < pooled_width; ++pw) {
          const index_t out_index = ph * pooled_width + pw;
          in_data[mask[out_index]] += out_data[out_index];
        }
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
      mask += out_data_offset;
    }
  }
}

template<typename DType>
inline void pool(mshadow::Stream<cpu>* s, const DType* in_data, const TShape& ishape,
                 const TShape& oshape, const TShape& kernel, const TShape& pad,
                 const TShape& stride, const int pool_type, OpReqType req_type,
                 DType* out_data, int32_t* mask = nullptr) {
  if (kernel.ndim() == 2) {
    if (pool_enum::kMaxPooling == pool_type) {
      pool_max_2d_cpu(in_data, ishape, oshape, kernel, pad, stride, req_type, out_data, mask);
    }
  }
}

template<typename DType>
inline void unpool(mshadow::Stream<cpu>* s, const DType* out_data, const TShape& ishape,
                   const TShape& oshape, const TShape& kernel, const TShape& pad,
                   const TShape& stride, const int pool_type, OpReqType req_type, DType* in_data,
                   const int32_t* mask = nullptr) {
  if (mxnet::kNullOp == req_type) return;
  if (mxnet::kAddTo != req_type) {
    mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, ishape.Size(), in_data);
  }
  if (kernel.ndim() == 2) {
    if (pool_enum::kMaxPooling == pool_type) {
      unpool_max_2d_cpu(s, out_data, mask, ishape, oshape, in_data);
    }
  }
}

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./pool.cuh"
#endif

#endif  // MXNET_OPERATOR_NN_POOL_H_
