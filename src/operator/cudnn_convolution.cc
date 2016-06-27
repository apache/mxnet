/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_convolution.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./cudnn_convolution-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace op {
template<typename DType>
CuDNNConvolutionOp<DType>::CuDNNConvolutionOp(ConvolutionParam param,
                                       std::vector<TShape> *in_shape,
                                       std::vector<TShape> *out_shape,
                                       Context ctx) {
  using namespace mshadow;
  this->param_ = param;
  // convert MB to words
  param_.workspace = (param_.workspace << 20) / sizeof(DType);
  init_cudnn_ = false;
  // TODO(xxx): fp16
  dtype_ = mshadow::DataType<DType>::kCudnnFlag;

  size_t expected = param_.no_bias ? 2 : 3;
#if CUDNN_MAJOR == 5
  format_ = CUDNN_TENSOR_NCHW;
#endif
  CHECK_EQ(in_shape->size(), expected);
  CHECK_EQ(out_shape->size(), 1);

  size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));
  size_t back_size = 0;
  size_t back_size_w = 0;
  TShape &x_shape = (*in_shape)[conv::kData];
  TShape &y_shape = (*out_shape)[conv::kOut];
  data_offset_ = x_shape[1] / param_.num_group * x_shape[2] * x_shape[3];
  out_offset_ = y_shape[1] /param_.num_group * y_shape[2] * y_shape[3];
  weight_offset_ = param_.num_filter / param_.num_group * x_shape[1] / param_.num_group
                   * param_.kernel[0] * param_.kernel[1];
  CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateTensorDescriptor(&bias_desc_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateFilterDescriptor(&filter_desc_), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateConvolutionDescriptor(&conv_desc_), CUDNN_STATUS_SUCCESS);
#if CUDNN_MAJOR == 5
  CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                      dtype_,
                                      format_,
                                      param_.num_filter / param_.num_group,
                                      x_shape[1] / param_.num_group,
                                      param_.kernel[0],
                                      param_.kernel[1]), CUDNN_STATUS_SUCCESS);
#else
  CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                      dtype_,
                                      param_.num_filter / param_.num_group,
                                      x_shape[1] / param_.num_group,
                                      param_.kernel[0],
                                      param_.kernel[1]), CUDNN_STATUS_SUCCESS);
#endif
  CHECK_EQ(cudnnSetConvolution2dDescriptor(conv_desc_,
                                           param_.pad[0],
                                           param_.pad[1],
                                           param_.stride[0],
                                           param_.stride[1],
                                           1,
                                           1,
                                           CUDNN_CROSS_CORRELATION), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnSetTensor4dDescriptorEx(in_desc_,
                                        dtype_,
                                        x_shape[0],
                                        x_shape[1] / param_.num_group,
                                        x_shape[2],
                                        x_shape[3],
                                        x_shape[1] * x_shape[2] * x_shape[3],
                                        x_shape[2] * x_shape[3],
                                        x_shape[3],
                                        1), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnSetTensor4dDescriptorEx(out_desc_,
                                        dtype_,
                                        y_shape[0],
                                        y_shape[1] / param_.num_group,
                                        y_shape[2],
                                        y_shape[3],
                                        y_shape[1] * y_shape[2] * y_shape[3],
                                        y_shape[2] * y_shape[3],
                                        y_shape[3],
                                        1), CUDNN_STATUS_SUCCESS);
  if (!param_.no_bias) {
    TShape bias_shape = (*in_shape)[conv::kBias];
    bias_offset_ = bias_shape[0] / param_.num_group;
    CHECK_EQ(cudnnSetTensor4dDescriptor(bias_desc_,
                                        CUDNN_TENSOR_NCHW,
                                        dtype_,
                                        1,
                                        bias_shape[0] / param_.num_group,
                                        1,
                                        1), CUDNN_STATUS_SUCCESS);
  }

  Engine::VarHandle var = Engine::Get()->NewVariable();
  if (param.cudnn_tune == conv::kOff) {
    Engine::Get()->PushSync([&](RunContext rctx) {
      Stream<gpu> *s = rctx.get_stream<gpu>();
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
      CHECK_EQ(cudnnGetConvolutionForwardAlgorithm(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               conv_desc_,
               out_desc_,
               CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
               workspace_byte,
               &algo_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
               in_desc_,
               out_desc_,
               conv_desc_,
               filter_desc_,
               CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
               workspace_byte,
               &back_algo_w_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardDataAlgorithm(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               conv_desc_,
               in_desc_,
               CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
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
      forward_workspace_ = forward_workspace_byte_ / sizeof(DType) + 1;
      backward_workspace_ = backward_workspace_byte_ / sizeof(DType) + 1;
    }, ctx, {}, {var});
  } else {
    Engine::Get()->PushSync([&](RunContext rctx) {
      Stream<gpu> *s = rctx.get_stream<gpu>();
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
      const int max_nalgo = 10;
      int nalgo = max_nalgo;
      int i;

      cudnnConvolutionFwdAlgoPerf_t fwd_algo[max_nalgo];
      CHECK_EQ(cudnnFindConvolutionForwardAlgorithm(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               conv_desc_,
               out_desc_,
               max_nalgo,
               &nalgo,
               fwd_algo), CUDNN_STATUS_SUCCESS);
      i = 0;
      while (i < nalgo
             && fwd_algo[i].status != CUDNN_STATUS_SUCCESS
             && param.cudnn_tune == conv::kLimited
             && fwd_algo[i].memory > workspace_byte) ++i;
      if (i == nalgo) {
        LOG(FATAL) << "Failed to find an convolution algorithm.";
      } else {
        forward_workspace_byte_ = fwd_algo[i].memory;
        algo_ = fwd_algo[i].algo;
        LOG(INFO) << "Selecting " << static_cast<int>(algo_) << " for forward.";
      }

      cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo[max_nalgo];
      CHECK_EQ(cudnnFindConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
               in_desc_,
               out_desc_,
               conv_desc_,
               filter_desc_,
               max_nalgo,
               &nalgo,
               bwd_filter_algo), CUDNN_STATUS_SUCCESS);
      i = 0;
      while (i < nalgo
             && bwd_filter_algo[i].status != CUDNN_STATUS_SUCCESS
             && param.cudnn_tune == conv::kLimited
             && bwd_filter_algo[i].memory > workspace_byte) ++i;
      if (i == nalgo) {
        LOG(FATAL) << "Failed to find an convolution algorithm.";
      } else {
        backward_workspace_byte_ = bwd_filter_algo[i].memory;
        back_algo_w_ = bwd_filter_algo[i].algo;
        LOG(INFO) << "Selecting " << static_cast<int>(back_algo_w_) << " for backward filter.";
      }

      cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo[max_nalgo];
      CHECK_EQ(cudnnFindConvolutionBackwardDataAlgorithm(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               conv_desc_,
               in_desc_,
               max_nalgo,
               &nalgo,
               bwd_data_algo), CUDNN_STATUS_SUCCESS);
      i = 0;
      while (i < nalgo
             && bwd_data_algo[i].status != CUDNN_STATUS_SUCCESS
             && param.cudnn_tune == conv::kLimited
             && bwd_data_algo[i].memory > workspace_byte) ++i;
      if (i == nalgo) {
        LOG(FATAL) << "Failed to find an convolution algorithm.";
      } else {
        backward_workspace_byte_ = std::max(backward_workspace_byte_, bwd_data_algo[i].memory);
        back_algo_ = bwd_data_algo[i].algo;
        LOG(INFO) << "Selecting " << static_cast<int>(back_algo_) << " for backward data.";
      }
      
      forward_workspace_ = forward_workspace_byte_ / sizeof(DType) + 1;
      backward_workspace_ = backward_workspace_byte_ / sizeof(DType) + 1;
    }, ctx, {}, {var});
  }
  Engine::Get()->WaitForVar(var);
  Engine::Get()->DeleteVariable([](RunContext s) {}, ctx, var);
  init_cudnn_ = true;
}

template<>
CuDNNConvolutionOp<float>::CuDNNConvolutionOp(ConvolutionParam,
                                                    std::vector<TShape>*,
                                                    std::vector<TShape>*,
                                                    Context);
template<>
CuDNNConvolutionOp<double>::CuDNNConvolutionOp(ConvolutionParam,
                                                     std::vector<TShape>*,
                                                     std::vector<TShape>*,
                                                     Context);
template<>
CuDNNConvolutionOp<mshadow::half::half_t>::CuDNNConvolutionOp(ConvolutionParam,
                                                                    std::vector<TShape>*,
                                                                    std::vector<TShape>*,
                                                                    Context);

}  // namespace op
}  // namespace mxnet