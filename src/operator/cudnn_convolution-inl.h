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
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1
class CuDNNConvolutionOp : public Operator {
 public:
  explicit CuDNNConvolutionOp(ConvolutionParam param) {
    this->param_ = param;
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(real_t);
    init_cudnn_ = false;
    // TODO(xxx): fp16
    dtype_ = CUDNN_DATA_FLOAT;
  }

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
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> data = in_data[kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> wmat = in_data[kWeight].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> out = out_data[kOut].get<gpu, 4, real_t>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(wmat.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    if (!init_cudnn_) {
      Init(s, in_data, out_data);
    }
    Tensor<gpu, 1> workspace = ctx.requested[kTempSpace].get_space<gpu>(
      mshadow::Shape1(forward_workspace_), s);
    CHECK_EQ(cudnnConvolutionForward(s->dnn_handle_,
                                     &alpha,
                                     in_desc_,
                                     data.dptr_,
                                     filter_desc_,
                                     wmat.dptr_,
                                     conv_desc_,
                                     algo_,
                                     workspace.dptr_,
                                     forward_workspace_byte_,
                                     &beta,
                                     out_desc_,
                                     out.dptr_), CUDNN_STATUS_SUCCESS);
    if (!param_.no_bias) {
      beta = 1.0f;
      Tensor<gpu, 1> bias = in_data[kBias].get<gpu, 1, real_t>(s);
      CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                              CUDNN_ADD_SAME_C,
                              &alpha,
                              bias_desc_,
                              bias.dptr_,
                              &beta,
                              out_desc_,
                              out.dptr_), CUDNN_STATUS_SUCCESS);
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
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    // TODO(bing): think about how to support add to
    CHECK_EQ(req[kWeight], kWriteTo);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> grad = out_grad[kOut].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> wmat = in_data[kWeight].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> gwmat = in_grad[kWeight].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> data = in_data[kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> gdata = in_grad[kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 1> workspace = ctx.requested[kTempSpace].get_space<gpu>(
      mshadow::Shape1(backward_workspace_), s);
    if (!param_.no_bias) {
      Tensor<gpu, 1> gbias = in_grad[kBias].get<gpu, 1, real_t>(s);
      CHECK_EQ(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                            &alpha,
                                            out_desc_,
                                            grad.dptr_,
                                            &beta,
                                            bias_desc_,
                                            gbias.dptr_), CUDNN_STATUS_SUCCESS);
    }
    CHECK_EQ(cudnnConvolutionBackwardFilter_v3(s->dnn_handle_,
             &alpha,
             in_desc_,
             data.dptr_,
             out_desc_,
             grad.dptr_,
             conv_desc_,
             back_algo_w_,
             workspace.dptr_,
             backward_workspace_byte_,
             &beta,
             filter_desc_,
             gwmat.dptr_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnConvolutionBackwardData_v3(s->dnn_handle_,
             &alpha,
             filter_desc_,
             wmat.dptr_,
             out_desc_,
             grad.dptr_,
             conv_desc_,
             back_algo_,
             workspace.dptr_,
             backward_workspace_byte_,
             &beta,
             in_desc_,
             gdata.dptr_), CUDNN_STATUS_SUCCESS);
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(real_t));
      size_t back_size = 0;
      size_t back_size_w = 0;
      Tensor<gpu, 4> data = in_data[kData].get<gpu, 4, real_t>(s);
      Tensor<gpu, 4> out = out_data[kOut].get<gpu, 4, real_t>(s);
      CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&bias_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateFilterDescriptor(&filter_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateConvolutionDescriptor(&conv_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                          dtype_,
                                          param_.num_filter,
                                          data.shape_[1],
                                          param_.kernel[0],
                                          param_.kernel[1]), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetConvolution2dDescriptor(conv_desc_,
               param_.pad[0],
               param_.pad[1],
               param_.stride[0],
               param_.stride[1],
               1,
               1,
               CUDNN_CROSS_CORRELATION), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(in_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          data.shape_[0],
                                          data.shape_[1],
                                          data.shape_[2],
                                          data.shape_[3]), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(out_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          out.shape_[0],
                                          out.shape_[1],
                                          out.shape_[2],
                                          out.shape_[3]), CUDNN_STATUS_SUCCESS);
      if (!param_.no_bias) {
        Tensor<gpu, 1> bias = in_data[kBias].get<gpu, 1, real_t>(s);
        CHECK_EQ(cudnnSetTensor4dDescriptor(bias_desc_,
                                            CUDNN_TENSOR_NCHW,
                                            dtype_,
                                            1,
                                            bias.shape_[0],
                                            1,
                                            1), CUDNN_STATUS_SUCCESS);
      }
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
      CHECK_EQ(cudnnGetConvolutionForwardAlgorithm(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               conv_desc_,
               out_desc_,
               CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
               workspace_byte,
               &algo_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
               in_desc_,
               out_desc_,
               conv_desc_,
               filter_desc_,
               CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
               workspace_byte,
               &back_algo_w_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardDataAlgorithm(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               conv_desc_,
               in_desc_,
               CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
               workspace_byte,
               &back_algo_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               conv_desc_,
               in_desc_,
               back_algo_,
               &back_size), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               in_desc_,
               out_desc_,
               conv_desc_,
               filter_desc_,
               back_algo_w_,
               &back_size_w), CUDNN_STATUS_SUCCESS);
      backward_workspace_byte_ = std::max(back_size, back_size_w);
      CHECK_EQ(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               conv_desc_,
               out_desc_,
               algo_,
               &forward_workspace_byte_), CUDNN_STATUS_SUCCESS);
      forward_workspace_ = forward_workspace_byte_ / sizeof(real_t) + 1;
      backward_workspace_ = backward_workspace_byte_ / sizeof(real_t) + 1;
    }
  }

  bool init_cudnn_;
  size_t forward_workspace_;
  size_t backward_workspace_;
  size_t forward_workspace_byte_;
  size_t backward_workspace_byte_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t algo_;
  cudnnConvolutionBwdDataAlgo_t back_algo_;
  cudnnConvolutionBwdFilterAlgo_t back_algo_w_;
  ConvolutionParam param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_CONVOLUTION_INL_H_
