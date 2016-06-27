/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_convolution-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CUDNN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CUDNN_CONVOLUTION_INL_H_

#include <algorithm>
#include <vector>
#include "./convolution-inl.h"
namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
template<typename DType>
class CuDNNConvolutionOp : public Operator {
 public:
  CuDNNConvolutionOp(ConvolutionParam param,
                              std::vector<TShape> *in_shape,
                              std::vector<TShape> *out_shape,
                              Context ctx);

  ~CuDNNConvolutionOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(bias_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyFilterDescriptor(filter_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyConvolutionDescriptor(conv_desc_), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> data = in_data[conv::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> wmat = in_data[conv::kWeight].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> out = out_data[conv::kOut].get<gpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(wmat.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    Tensor<gpu, 1, DType> workspace =
        ctx.requested[conv::kTempSpace].get_space_typed<gpu, 1, DType>(
                                 mshadow::Shape1(forward_workspace_), s);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      CHECK_EQ(cudnnConvolutionForward(s->dnn_handle_,
                                       &alpha,
                                       in_desc_,
                                       data.dptr_ + data_offset_ * g,
                                       filter_desc_,
                                       wmat.dptr_ + weight_offset_ * g,
                                       conv_desc_,
                                       algo_,
                                       workspace.dptr_,
                                       forward_workspace_byte_,
                                       &beta,
                                       out_desc_,
                                       out.dptr_ + out_offset_ * g), CUDNN_STATUS_SUCCESS);
      if (!param_.no_bias) {
        beta = 1.0f;
        Tensor<gpu, 1, DType> bias = in_data[conv::kBias].get<gpu, 1, DType>(s);
#if CUDNN_MAJOR >= 4
        CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                                &alpha,
                                bias_desc_,
                                bias.dptr_ + bias_offset_ * g,
                                &beta,
                                out_desc_,
                                out.dptr_ + out_offset_ * g), CUDNN_STATUS_SUCCESS);
#endif
#if CUDNN_MAJOR == 3
        CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                                CUDNN_ADD_SAME_C,
                                &alpha,
                                bias_desc_,
                                bias.dptr_ + bias_offset_ * g,
                                &beta,
                                out_desc_,
                                out.dptr_ + out_offset_ * g), CUDNN_STATUS_SUCCESS);
#endif
      }
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
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> grad = out_grad[conv::kOut].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> wmat = in_data[conv::kWeight].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> gwmat = in_grad[conv::kWeight].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> data = in_data[conv::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> gdata = in_grad[conv::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 1, DType> workspace =
      ctx.requested[conv::kTempSpace].get_space_typed<gpu, 1, DType>(
      mshadow::Shape1(backward_workspace_), s);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      if (!param_.no_bias) {
        Tensor<gpu, 1, DType> gbias = in_grad[conv::kBias].get<gpu, 1, DType>(s);
        CHECK_EQ(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                              &alpha,
                                              out_desc_,
                                              grad.dptr_ + out_offset_ * g,
                                              req[conv::kBias] == kWriteTo ? &beta : &beta_add,
                                              bias_desc_,
                                              gbias.dptr_ + bias_offset_ * g),
                 CUDNN_STATUS_SUCCESS);
      }
      #if CUDNN_MAJOR <= 4
      CHECK_EQ(cudnnConvolutionBackwardFilter_v3(s->dnn_handle_,
               &alpha,
               in_desc_,
               data.dptr_ + data_offset_ * g,
               out_desc_,
               grad.dptr_ + out_offset_ * g,
               conv_desc_,
               back_algo_w_,
               workspace.dptr_,
               backward_workspace_byte_,
               req[conv::kWeight] == kWriteTo? &beta : &beta_add,
               filter_desc_,
               gwmat.dptr_ + weight_offset_ * g), CUDNN_STATUS_SUCCESS);
      #elif CUDNN_MAJOR == 5
      CHECK_EQ(cudnnConvolutionBackwardFilter(s->dnn_handle_,
               &alpha,
               in_desc_,
               data.dptr_ + data_offset_ * g,
               out_desc_,
               grad.dptr_ + out_offset_ * g,
               conv_desc_,
               back_algo_w_,
               workspace.dptr_,
               backward_workspace_byte_,
               req[conv::kWeight] == kWriteTo? &beta : &beta_add,
               filter_desc_,
               gwmat.dptr_ + weight_offset_ * g), CUDNN_STATUS_SUCCESS);
      #endif
      #if CUDNN_MAJOR <= 4
      CHECK_EQ(cudnnConvolutionBackwardData_v3(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat.dptr_ + weight_offset_ * g,
               out_desc_,
               grad.dptr_ + out_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               in_desc_,
               gdata.dptr_ + data_offset_ * g), CUDNN_STATUS_SUCCESS);
      #elif CUDNN_MAJOR == 5
      CHECK_EQ(cudnnConvolutionBackwardData(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat.dptr_ + weight_offset_ * g,
               out_desc_,
               grad.dptr_ + out_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               in_desc_,
               gdata.dptr_ + data_offset_ * g), CUDNN_STATUS_SUCCESS);
      #endif
    }
  }

 private:
  bool init_cudnn_;
  size_t forward_workspace_;
  size_t backward_workspace_;
  size_t forward_workspace_byte_;
  size_t backward_workspace_byte_;
  size_t data_offset_;
  size_t out_offset_;
  size_t weight_offset_;
  size_t bias_offset_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t algo_;
  cudnnConvolutionBwdDataAlgo_t back_algo_;
  cudnnConvolutionBwdFilterAlgo_t back_algo_w_;
  #if CUDNN_MAJOR == 5
  cudnnTensorFormat_t format_;
  #endif
  ConvolutionParam param_;
};
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_CONVOLUTION_INL_H_
