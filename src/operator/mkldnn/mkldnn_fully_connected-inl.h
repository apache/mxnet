/*!
 * Copyright (c) 2016 by Contributors
 * \file mkldnn_fully_connected-inl.h
 * \brief
 * \author Ji Jiang
*/

#ifndef MXNET_OPERATOR_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_
#include <algorithm>
#include <vector>
#include "../activation-inl.h"
#include "mkldnn_memory-inl.h"
namespace mxnet {
namespace op {

template <typename DType> class MKLDNNFullyConnectedOp : public Operator {
 public:
  explicit MKLDNNFullyConnectedOp(FullyConnectedParam p)
      : init_mkldnn_(false),
        fully_connected_Fwd_(NULL),
        fully_connected_BwdData_(NULL),
        fully_connected_BwdFilter_(NULL),
        fully_connected_BwdBias_(NULL),
        fwd_out_data_(new MKLData<DType>()),
        fwd_in_data_(new MKLData<DType>()),
        bwd_in_diff_(new MKLData<DType>()),
        bwd_out_diff_(new MKLData<DType>()) {
    param_ = p;
  }

  ~MKLDNNFullyConnectedOp() {
    dnnDelete<DType>(fully_connected_Fwd_);
    dnnDelete<DType>(fully_connected_BwdData_);
    dnnDelete<DType>(fully_connected_BwdFilter_);
    dnnDelete<DType>(fully_connected_BwdBias_);
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);

    Stream<cpu> *s = ctx.get_stream<cpu>();

    if (!init_mkldnn_) {
      this->Init(s, in_data, out_data);
    }

    fully_connected_res_[dnnResourceSrc] =
        reinterpret_cast<void *>(in_data[fullc::kData].dptr_);
    fully_connected_res_[dnnResourceDst] =
        reinterpret_cast<void *>(out_data[fullc::kOut].dptr_);
    fully_connected_res_[dnnResourceFilter] =
        reinterpret_cast<void *>(in_data[fullc::kWeight].dptr_);
    if (!param_.no_bias) {
      fully_connected_res_[dnnResourceBias] =
          reinterpret_cast<void *>(in_data[fullc::kBias].dptr_);
    }

    CHECK_EQ(dnnExecute<DType>(fully_connected_Fwd_, fully_connected_res_),
             E_SUCCESS);
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
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);

    fully_connected_res_[dnnResourceSrc] =
        reinterpret_cast<void *>(in_data[fullc::kData].dptr_);
    fully_connected_res_[dnnResourceFilter] =
        reinterpret_cast<void *>(in_data[fullc::kWeight].dptr_);

    fully_connected_res_[dnnResourceDiffDst] =
        reinterpret_cast<void *>(out_grad[fullc::kOut].dptr_);
    fully_connected_res_[dnnResourceDiffSrc] =
        reinterpret_cast<void *>(in_grad[fullc::kData].dptr_);
    fully_connected_res_[dnnResourceDiffFilter] =
        reinterpret_cast<void *>(in_grad[fullc::kWeight].dptr_);

    if (!param_.no_bias) {
      fully_connected_res_[dnnResourceDiffBias] =
          reinterpret_cast<void *>(in_grad[fullc::kBias].dptr_);
    }
    CHECK_EQ(
        dnnExecute<DType>(fully_connected_BwdFilter_, fully_connected_res_),
        E_SUCCESS);
    if (!param_.no_bias) {
      CHECK_EQ(
          dnnExecute<DType>(fully_connected_BwdBias_, fully_connected_res_),
          E_SUCCESS);
    }
    CHECK_EQ(dnnExecute<DType>(fully_connected_BwdData_, fully_connected_res_),
             E_SUCCESS);
  }

 private:
  inline void Init(mshadow::Stream<cpu> *s, const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    if (!init_mkldnn_) {
      init_mkldnn_ = true;

      const TShape &ishape = in_data[fullc::kData].shape_;
      const TShape &oshape = out_data[fullc::kOut].shape_;

      Tensor<cpu, 4, DType> data;
      Tensor<cpu, 4, DType> out;

      Shape4(in_data[fullc::kData].shape_[0], in_data[fullc::kData].shape_[1],
             1, 1);

      Shape<4> dshape =
          Shape4(ishape[0], ishape.ProdShape(1, ishape.ndim()), 1, 1);
      Shape<4> odshape =
          Shape4(oshape[0], oshape.ProdShape(1, oshape.ndim()), 1, 1);

      data = in_data[fullc::kData].get_with_shape<cpu, 4, DType>(dshape, s);
      out = out_data[fullc::kOut].get_with_shape<cpu, 4, DType>(odshape, s);

      size_t src_sizes[4];
      size_t dst_sizes[2];

      size_t dim = 4;

      const size_t input_batch_size = data.size(0);
      const size_t input_channels = data.size(1);
      const size_t input_height = data.size(2);
      const size_t input_width = data.size(3);

      const size_t output_batch_size = out.size(0);
      const size_t output_channels = out.size(1);

      src_sizes[0] = input_width;
      src_sizes[1] = input_height;
      src_sizes[2] = input_channels;
      src_sizes[3] = input_batch_size;

      dst_sizes[0] = output_channels;
      dst_sizes[1] = output_batch_size;

      dnnPrimitiveAttributes_t attributes = NULL;
      CHECK_EQ(dnnPrimitiveAttributesCreate<DType>(&attributes), E_SUCCESS);
      if (!param_.no_bias) {
        CHECK_EQ(dnnInnerProductCreateForwardBias<DType>(
                     &fully_connected_Fwd_, attributes, dim, src_sizes,
                     output_channels),
                 E_SUCCESS);
      } else {
        CHECK_EQ(dnnInnerProductCreateForward<DType>(&fully_connected_Fwd_,
                                                     attributes, dim, src_sizes,
                                                     output_channels),
                 E_SUCCESS);
      }
      CHECK_EQ(dnnInnerProductCreateBackwardData<DType>(
                   &fully_connected_BwdData_, attributes, dim, src_sizes,
                   output_channels),
               E_SUCCESS);
      CHECK_EQ(dnnInnerProductCreateBackwardFilter<DType>(
                   &fully_connected_BwdFilter_, attributes, dim, src_sizes,
                   output_channels),
               E_SUCCESS);
      if (!param_.no_bias) {
        CHECK_EQ(dnnInnerProductCreateBackwardBias<DType>(
                     &fully_connected_BwdBias_, attributes, 2, dst_sizes),
                 E_SUCCESS);
      }
    }
  }

  bool init_mkldnn_;
  dnnPrimitive_t fully_connected_Fwd_, fully_connected_BwdData_,
      fully_connected_BwdFilter_, fully_connected_BwdBias_;
  std::shared_ptr<MKLData<DType>> fwd_out_data_, fwd_in_data_;
  std::shared_ptr<MKLData<DType>> bwd_in_diff_, bwd_out_diff_;
  FullyConnectedParam param_;
  void *fully_connected_res_[dnnResourceNumber];
};  // class MKLDNNFullyConnectedOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_
