/*!
 * Copyright (c) 2016 by Contributors
 * \file mkldnn_lrn-inl.h
 * \brief
 * \author Ji Jiang
*/

#ifndef MXNET_OPERATOR_MKLDNN_MKLDNN_LRN_INL_H_
#define MXNET_OPERATOR_MKLDNN_MKLDNN_LRN_INL_H_
#include <algorithm>
#include <vector>
#include "../lrn-inl.h"
#include "mkldnn_memory-inl.h"

namespace mxnet {
namespace op {

template <typename DType> class MKLDNNLocalResponseNormOp : public Operator {
 public:
  explicit MKLDNNLocalResponseNormOp(LRNParam p)
      : init_mkldnn_(false),
        lrnFwd_(NULL),
        lrnBwd_(NULL),
        fwd_out_data_(new MKLData<DType>()),
        fwd_in_data_(new MKLData<DType>()),
        bwd_in_diff_(new MKLData<DType>()),
        bwd_out_diff_(new MKLData<DType>()) {
    param_ = p;
  }

  ~MKLDNNLocalResponseNormOp() {
    dnnDelete<DType>(lrnFwd_);
    dnnDelete<DType>(lrnBwd_);
    dnnReleaseBuffer<DType>(lrn_res_[dnnResourceWorkspace]);
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    // TODO(xxx): Test with gradient chceker
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    // CHECK_EQ(req.size(), 2);
    Stream<cpu> *s = ctx.get_stream<cpu>();

    if (!init_mkldnn_) {
      this->Init(s, in_data, out_data);
    }

    lrn_res_[dnnResourceSrc] =
        reinterpret_cast<void *>(in_data[lrn_enum::kData].dptr_);
    lrn_res_[dnnResourceDst] =
        reinterpret_cast<void *>(out_data[lrn_enum::kOut].dptr_);

    CHECK_EQ(dnnExecute<DType>(lrnFwd_, lrn_res_), E_SUCCESS);
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
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);

    lrn_res_[dnnResourceDiffDst] =
        reinterpret_cast<void *>(out_grad[lrn_enum::kOut].dptr_);
    lrn_res_[dnnResourceDiffSrc] =
        reinterpret_cast<void *>(in_grad[lrn_enum::kData].dptr_);

    CHECK_EQ(dnnExecute<DType>(lrnBwd_, lrn_res_), E_SUCCESS);
  }

 private:
  inline void Init(mshadow::Stream<cpu> *s, const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    if (!init_mkldnn_) {
      init_mkldnn_ = true;

      Tensor<cpu, 4> data = in_data[lrn_enum::kData].get<cpu, 4, DType>(s);
      Tensor<cpu, 4> out = out_data[lrn_enum::kOut].get<cpu, 4, DType>(s);

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

      float alpha = param_.alpha;
      float beta = param_.beta;
      float knorm = param_.knorm;
      size_t nsize = (size_t) param_.nsize;

      fwd_in_data_->create_user_layout(dim, src_sizes, src_strides);
      fwd_out_data_->create_user_layout(dim, dst_sizes, dst_strides);
      bwd_in_diff_->create_user_layout(dim, src_sizes, src_strides);
      bwd_out_diff_->create_user_layout(dim, dst_sizes, dst_strides);

      dnnPrimitiveAttributes_t attributes = NULL;
      CHECK_EQ(dnnPrimitiveAttributesCreate<DType>(&attributes), E_SUCCESS);

      CHECK_EQ(dnnLRNCreateForward<DType>(
                   &lrnFwd_, attributes, fwd_in_data_->layout_usr, nsize, alpha,
                   beta, knorm),
               E_SUCCESS);
      CHECK_EQ(dnnLRNCreateBackward<DType>(
                   &lrnBwd_, attributes, bwd_in_diff_->layout_usr,
                   fwd_in_data_->layout_usr, nsize, alpha, beta, knorm),
               E_SUCCESS);

      CHECK_EQ(dnnLayoutCreateFromPrimitive<DType>(&workspace_, lrnFwd_,
                                                   dnnResourceWorkspace),
               E_SUCCESS);
      CHECK_EQ(dnnAllocateBuffer<DType>(
                   reinterpret_cast<void **>(&lrn_res_[dnnResourceWorkspace]), workspace_),
               E_SUCCESS);
    }
  }

  int count = 0;
  int count1 = 0;
  dnnLayout_t workspace_;
  bool init_mkldnn_;
  dnnPrimitive_t lrnFwd_, lrnBwd_;
  std::shared_ptr<MKLData<DType>> fwd_out_data_, fwd_in_data_;
  std::shared_ptr<MKLData<DType>> bwd_in_diff_, bwd_out_diff_;
  LRNParam param_;
  void *lrn_res_[dnnResourceNumber];
};  // class MKLDNNLocalResponseNormOp
}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_MKLDNN_MKLDNN_LRN_INL_H_
