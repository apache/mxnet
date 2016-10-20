/*!
 * Copyright (c) 2016 by Contributors
 * \file mkldnn_ReLU-inl.h
 * \brief
 * \author Ji Jiang
*/

#ifndef MXNET_OPERATOR_MKLDNN_MKLDNN_RELU_INL_H_
#define MXNET_OPERATOR_MKLDNN_MKLDNN_RELU_INL_H_
#include <algorithm>
#include <vector>
#include "../activation-inl.h"
#include "mkldnn_memory-inl.h"
namespace mxnet {
namespace op {

template <typename DType> class MKLDNNReLUOp : public Operator {
 public:
  explicit MKLDNNReLUOp(ActivationParam p)
      : init_mkldnn_(false),
        reluFwd_(NULL),
        reluBwd_(NULL),
        fwd_out_data_(new MKLData<DType>()),
        fwd_in_data_(new MKLData<DType>()),
        bwd_in_diff_(new MKLData<DType>()),
        bwd_out_diff_(new MKLData<DType>()) {
    param_ = p;
    if (param_.act_type != activation::kReLU)
      LOG(FATAL) << "mkldnn only support this activation type";
  }

  ~MKLDNNReLUOp() {
    dnnDelete<DType>(reluFwd_);
    dnnDelete<DType>(reluBwd_);
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

    if (!init_mkldnn_) {
      this->Init(s, in_data, out_data);
    }

    relu_res_[dnnResourceSrc] =
        reinterpret_cast<void *>(in_data[activation::kData].dptr_);
    relu_res_[dnnResourceDst] =
        reinterpret_cast<void *>(out_data[activation::kOut].dptr_);
    CHECK_EQ(dnnExecute<DType>(reluFwd_, relu_res_), E_SUCCESS);
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);

    relu_res_[dnnResourceDiffDst] =
        reinterpret_cast<void *>(out_grad[activation::kOut].dptr_);
    relu_res_[dnnResourceDiffSrc] =
        reinterpret_cast<void *>(in_grad[activation::kData].dptr_);
    CHECK_EQ(dnnExecute<DType>(reluBwd_, relu_res_), E_SUCCESS);
  }

 private:
  inline void Init(mshadow::Stream<cpu> *s, const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;

    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    if (!init_mkldnn_) {
      init_mkldnn_ = true;

      Tensor<cpu, 4, DType> data;
      Tensor<cpu, 4, DType> out;

      Shape<4> dshape;
      index_t size_left = in_data[activation::kData].Size();
      for (int i = 0; i < 3; ++i) {
        if (i < in_data[activation::kData].ndim()) {
          dshape[i] = in_data[activation::kData].shape_[i];
        } else {
          dshape[i] = 1;
        }
        size_left /= dshape[i];
      }
      dshape[3] = size_left;
      data =
          in_data[activation::kData].get_with_shape<cpu, 4, DType>(dshape, s);
      out = out_data[activation::kOut].get_with_shape<cpu, 4, DType>(dshape, s);

      size_t src_sizes[4], src_strides[4];
      size_t dst_sizes[4], dst_strides[4];

      size_t dim = 4;

      const size_t input_batch_size = data.size(0);
      const size_t input_channels = data.size(1);
      const size_t input_height = data.size(2);
      const size_t input_width = data.size(3);

      const size_t output_batch_size = out.size(0);
      const size_t output_channels = out.size(1);
      const size_t output_height = out.size(2);
      const size_t output_width = out.size(3);

      src_sizes[0] = input_width;
      src_sizes[1] = input_height;
      src_sizes[2] = input_channels;
      src_sizes[3] = input_batch_size;
      src_strides[0] = 1;
      src_strides[1] = src_sizes[0];
      src_strides[2] = src_sizes[0] * src_sizes[1];
      src_strides[3] = src_sizes[0] * src_sizes[1] * src_sizes[2];
      dst_sizes[0] = output_width;
      dst_sizes[1] = output_height;
      dst_sizes[2] = output_channels;
      dst_sizes[3] = output_batch_size;
      dst_strides[0] = 1;
      dst_strides[1] = dst_sizes[0];
      dst_strides[2] = dst_sizes[0] * dst_sizes[1];
      dst_strides[3] = dst_sizes[0] * dst_sizes[1] * dst_sizes[2];

      fwd_in_data_->create_user_layout(dim, src_sizes, src_strides);
      fwd_out_data_->create_user_layout(dim, dst_sizes, dst_strides);
      bwd_in_diff_->create_user_layout(dim, src_sizes, src_strides);
      bwd_out_diff_->create_user_layout(dim, dst_sizes, dst_strides);

      dnnPrimitiveAttributes_t attributes = NULL;
      CHECK_EQ(dnnPrimitiveAttributesCreate<DType>(&attributes), E_SUCCESS);

      CHECK_EQ(dnnReLUCreateForward<DType>(&reluFwd_, attributes,
                                           fwd_in_data_->layout_usr, 0.0f),
               E_SUCCESS);
      CHECK_EQ(dnnReLUCreateBackward<DType>(&reluBwd_, attributes,
                                            bwd_in_diff_->layout_usr,
                                            fwd_in_data_->layout_usr, 0.0f),
               E_SUCCESS);
    }
  }

  bool init_mkldnn_;
  dnnPrimitive_t reluFwd_, reluBwd_;
  std::shared_ptr<MKLData<DType>> fwd_out_data_, fwd_in_data_;
  std::shared_ptr<MKLData<DType>> bwd_in_diff_, bwd_out_diff_;
  ActivationParam param_;
  void *relu_res_[dnnResourceNumber];
};  // class MKLDNNReLUOp
}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_MKLDNN_MKLDNN_RELU_INL_H_
