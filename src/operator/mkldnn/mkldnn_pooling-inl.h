/*!
 * Copyright (c) 2016 by Contributors
 * \file mkldnn_pooling-inl.h
 * \brief
 * \author yandai
*/

#ifndef MXNET_OPERATOR_MKLDNN_MKLDNN_POOLING_INL_H_
#define MXNET_OPERATOR_MKLDNN_MKLDNN_POOLING_INL_H_
#include <vector>
#include <algorithm>
#include "../pooling-inl.h"
#include "mkldnn_memory-inl.h"

namespace mxnet {
namespace op {

bool UseMKLDNNPooling(PoolingParam param, std::vector<TShape> *in_shape,
                      std::vector<TShape> *out_shape) {
  if (param.kernel.ndim() != 2) {
    return false;
  }
  if (param.pooling_convention == pool_enum::kFull) {
    return true;
  } else {
    TShape input_shape = (*in_shape)[pool_enum::kData];
    TShape output_shape = (*out_shape)[pool_enum::kOut];
    const size_t input_height = input_shape[2];
    const size_t input_width = input_shape[3];
    const size_t output_height = output_shape[2];
    const size_t output_width = output_shape[3];
    size_t full_model_output_height =
        1 + static_cast<int>(ceil(static_cast<float>(
                input_height + 2 * param.pad[0] -
                param.kernel[0]) / param.stride[0]));
    size_t full_model_output_width =
        1 + static_cast<int>(ceil(static_cast<float>(
                input_width + 2 * param.pad[1] -
                param.kernel[1]) / param.stride[1]));
    if ((full_model_output_height == output_height) && (full_model_output_width == output_width)) {
      return true;
    } else {
      return false;
    }
  }
}

template <typename DType>
class MKLDNNPoolingOp : public Operator {
 public:
  explicit MKLDNNPoolingOp(PoolingParam p)
      : param_(p),
        init_mkldnn_(false),
        poolingFwd_(NULL),
        poolingBwd_(NULL),
        fwd_out_data_(new MKLData<DType>()),
        fwd_in_data_(new MKLData<DType>()),
        bwd_in_diff_(new MKLData<DType>()),
        bwd_out_diff_(new MKLData<DType>()) {
    switch (this->param_.pool_type) {
      case pool_enum::kMaxPooling:
        mode_ = dnnAlgorithmPoolingMax;
        break;
      case pool_enum::kAvgPooling:
        mode_ = dnnAlgorithmPoolingAvg;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

  ~MKLDNNPoolingOp() {
    dnnDelete<DType>(poolingFwd_);
    dnnDelete<DType>(poolingBwd_);
    dnnReleaseBuffer<DType>(pooling_res_[dnnResourceWorkspace]);
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    if (this->param_.kernel.ndim() == 2) {
      // 2d pool
      if (!init_mkldnn_) {
        this->Init(s, in_data, out_data);
      }
      Tensor<cpu, 4, DType> data =
          in_data[pool_enum::kData].get<cpu, 4, DType>(s);
      Tensor<cpu, 4, DType> out =
          out_data[pool_enum::kOut].get<cpu, 4, DType>(s);
      pooling_res_[dnnResourceSrc] =
          reinterpret_cast<void *>(fwd_in_data_->get_converted_prv(data.dptr_, false));
      pooling_res_[dnnResourceDst] =
          reinterpret_cast<void *>(fwd_out_data_->get_converted_prv(out.dptr_, false));
      CHECK_EQ(dnnExecute<DType>(poolingFwd_, pooling_res_), E_SUCCESS);
      fwd_out_data_->get_output_ptr(data.dptr_);
    } else {
      LOG(FATAL) << "Only support 2D pooling";
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    if (this->param_.kernel.ndim() == 2) {
      if (!init_mkldnn_) {
        this->Init(s, in_data, out_data);
      }
      Tensor<cpu, 4, DType> ingrad =
          in_grad[pool_enum::kData].get<cpu, 4, DType>(s);
      Tensor<cpu, 4, DType> outgrad =
          out_grad[pool_enum::kOut].get<cpu, 4, DType>(s);
      pooling_res_[dnnResourceDiffDst] =
          reinterpret_cast<void *>(bwd_out_diff_->get_converted_prv(outgrad.dptr_, false));
      pooling_res_[dnnResourceDiffSrc] =
          reinterpret_cast<void *>(bwd_in_diff_->get_converted_prv(ingrad.dptr_, false));
      // single-threaded memset will hurt performance
      memset(pooling_res_[dnnResourceDiffSrc], 0,
             sizeof(DType) * ingrad.shape_.Size());
      CHECK_EQ(dnnExecute<DType>(poolingBwd_, pooling_res_), E_SUCCESS);
      bwd_in_diff_->get_output_ptr(ingrad.dptr_);
    } else {
      LOG(FATAL) << "Only support 2D pooling";
    }
  }

 private:
  inline void Init(mshadow::Stream<cpu> *s, const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    if (!init_mkldnn_) {
      init_mkldnn_ = true;
      if (this->param_.kernel.ndim() == 2) {
        Tensor<cpu, 4, DType> data =
            in_data[pool_enum::kData].get<cpu, 4, DType>(s);
        Tensor<cpu, 4, DType> out =
            out_data[pool_enum::kOut].get<cpu, 4, DType>(s);

        size_t src_sizes[4], src_strides[4];
        size_t dst_sizes[4], dst_strides[4];
        size_t kernel_size[2];
        size_t kernel_stride[4];
        int src_offset[2];
        size_t dim = 4;

        const size_t batch_size = data.size(0);
        const size_t input_channels = data.size(1);
        const size_t input_height = data.size(2);
        const size_t input_width = data.size(3);
        const size_t output_channels = input_channels;
        const size_t output_height = out.size(2);
        const size_t output_width = out.size(3);

        src_sizes[0] = input_width;
        src_sizes[1] = input_height;
        src_sizes[2] = input_channels;
        src_sizes[3] = batch_size;
        src_strides[0] = 1;
        src_strides[1] = src_sizes[0];
        src_strides[2] = src_sizes[0] * src_sizes[1];
        src_strides[3] = src_sizes[0] * src_sizes[1] * src_sizes[2];
        dst_sizes[0] = output_width;
        dst_sizes[1] = output_height;
        dst_sizes[2] = output_channels;
        dst_sizes[3] = batch_size;
        dst_strides[0] = 1;
        dst_strides[1] = dst_sizes[0];
        dst_strides[2] = dst_sizes[0] * dst_sizes[1];
        dst_strides[3] = dst_sizes[0] * dst_sizes[1] * dst_sizes[2];
        kernel_size[0] = this->param_.kernel[1];
        kernel_size[1] = this->param_.kernel[0];
        kernel_stride[0] = this->param_.stride[1];
        kernel_stride[1] = this->param_.stride[0];
        src_offset[0] = -this->param_.pad[1];
        src_offset[1] = -this->param_.pad[0];

        fwd_in_data_->create_user_layout(dim, src_sizes, src_strides);
        fwd_out_data_->create_user_layout(dim, dst_sizes, dst_strides);
        bwd_in_diff_->create_user_layout(dim, src_sizes, src_strides);
        bwd_out_diff_->create_user_layout(dim, dst_sizes, dst_strides);

        CHECK_EQ(dnnPoolingCreateForward<DType>(
                     &poolingFwd_, NULL, mode_, fwd_in_data_->layout_usr,
                     kernel_size, kernel_stride, src_offset, dnnBorderZeros),
                 E_SUCCESS);
        CHECK_EQ(dnnPoolingCreateBackward<DType>(
                     &poolingBwd_, NULL, mode_, fwd_in_data_->layout_usr,
                     kernel_size, kernel_stride, src_offset, dnnBorderZeros),
                 E_SUCCESS);
        CHECK_EQ(dnnLayoutCreateFromPrimitive<DType>(&workspace_, poolingFwd_,
                                                     dnnResourceWorkspace),
                 E_SUCCESS);
        CHECK_EQ(dnnAllocateBuffer<DType>(
                     reinterpret_cast<void **>(&pooling_res_[dnnResourceWorkspace]), workspace_),
                 E_SUCCESS);
        fwd_in_data_->create_internal_layout(poolingFwd_, dnnResourceSrc);
        fwd_out_data_->create_internal_layout(poolingFwd_, dnnResourceDst);
        bwd_out_diff_->create_internal_layout(poolingBwd_, dnnResourceDiffDst);
        bwd_in_diff_->create_internal_layout(poolingBwd_, dnnResourceDiffSrc);
      }
    }
  }

  PoolingParam param_;
  dnnLayout_t workspace_;
  bool init_mkldnn_;
  dnnPrimitive_t poolingFwd_, poolingBwd_;
  std::shared_ptr<MKLData<DType>> fwd_out_data_, fwd_in_data_;
  std::shared_ptr<MKLData<DType>> bwd_in_diff_, bwd_out_diff_;
  dnnAlgorithm_t mode_;
  void *pooling_res_[dnnResourceNumber];
};  // class MKLDNNPoolingOp
}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_MKLDNN_MKLDNN_POOLING_INL_H_
