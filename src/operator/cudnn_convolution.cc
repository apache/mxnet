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
#if MXNET_USE_CUDNN == 1
void TuneCudnnConvolution(ConvolutionParam param,
                          std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          Context ctx,
                          cudnnDataType_t dtype,
                          cudnnConvolutionFwdAlgo_t *algo,
                          cudnnConvolutionBwdDataAlgo_t *back_algo,
                          cudnnConvolutionBwdFilterAlgo_t *back_algo_w,
                          size_t *forward_workspace_byte,
                          size_t *backward_workspace_byte) {
  using namespace mshadow;
  // convert MB to bytes

  size_t expected = param.no_bias ? 2 : 3;
#if CUDNN_MAJOR == 5
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
#endif
  CHECK_EQ(in_shape->size(), expected);
  CHECK_EQ(out_shape->size(), 1);
  TShape &x_shape = (*in_shape)[conv::kData];
  TShape &y_shape = (*out_shape)[conv::kOut];


  size_t workspace_byte = param.workspace << 20;
  cudnnTensorDescriptor_t in_desc;
  cudnnTensorDescriptor_t out_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateTensorDescriptor(&bias_desc), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateFilterDescriptor(&filter_desc), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnCreateConvolutionDescriptor(&conv_desc), CUDNN_STATUS_SUCCESS);
#if CUDNN_MAJOR == 5
  CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc,
                                      dtype,
                                      format,
                                      param.num_filter / param.num_group,
                                      x_shape[1] / param.num_group,
                                      param.kernel[0],
                                      param.kernel[1]), CUDNN_STATUS_SUCCESS);
#else
  CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc,
                                      dtype,
                                      param.num_filter / param.num_group,
                                      x_shape[1] / param.num_group,
                                      param.kernel[0],
                                      param.kernel[1]), CUDNN_STATUS_SUCCESS);
#endif
  CHECK_EQ(cudnnSetConvolution2dDescriptor(conv_desc,
                                           param.pad[0],
                                           param.pad[1],
                                           param.stride[0],
                                           param.stride[1],
                                           1,
                                           1,
                                           CUDNN_CROSS_CORRELATION), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnSetTensor4dDescriptorEx(in_desc,
                                        dtype,
                                        x_shape[0],
                                        x_shape[1] / param.num_group,
                                        x_shape[2],
                                        x_shape[3],
                                        x_shape[1] * x_shape[2] * x_shape[3],
                                        x_shape[2] * x_shape[3],
                                        x_shape[3],
                                        1), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnSetTensor4dDescriptorEx(out_desc,
                                        dtype,
                                        y_shape[0],
                                        y_shape[1] / param.num_group,
                                        y_shape[2],
                                        y_shape[3],
                                        y_shape[1] * y_shape[2] * y_shape[3],
                                        y_shape[2] * y_shape[3],
                                        y_shape[3],
                                        1), CUDNN_STATUS_SUCCESS);
  if (!param.no_bias) {
    TShape bias_shape = (*in_shape)[conv::kBias];
    CHECK_EQ(cudnnSetTensor4dDescriptor(bias_desc,
                                        CUDNN_TENSOR_NCHW,
                                        dtype,
                                        1,
                                        bias_shape[0] / param.num_group,
                                        1,
                                        1), CUDNN_STATUS_SUCCESS);
  }

  Engine::VarHandle var = Engine::Get()->NewVariable();
  Engine::Get()->PushSync([=](RunContext rctx) {
    Stream<gpu> *s = rctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    const int kMaxAlgos = 10;
    int nalgo = kMaxAlgos;
    int i;

    cudnnConvolutionFwdAlgoPerf_t fwd_algo[kMaxAlgos];
    CHECK_EQ(cudnnFindConvolutionForwardAlgorithm(s->dnn_handle_,
             in_desc,
             filter_desc,
             conv_desc,
             out_desc,
             kMaxAlgos,
             &nalgo,
             fwd_algo), CUDNN_STATUS_SUCCESS);
    i = 0;
    while (i < nalgo
           && (fwd_algo[i].status != CUDNN_STATUS_SUCCESS
           || (param.cudnn_tune == conv::kLimited
           && fwd_algo[i].memory > workspace_byte))) ++i;
    if (i == nalgo) {
      LOG(FATAL) << "Failed to find an convolution algorithm.";
    } else {
      *forward_workspace_byte = fwd_algo[i].memory;
      *algo = fwd_algo[i].algo;
    }

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo[kMaxAlgos];
    CHECK_EQ(cudnnFindConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
             in_desc,
             out_desc,
             conv_desc,
             filter_desc,
             kMaxAlgos,
             &nalgo,
             bwd_filter_algo), CUDNN_STATUS_SUCCESS);
    i = 0;
    while (i < nalgo
           && (bwd_filter_algo[i].status != CUDNN_STATUS_SUCCESS
           || (param.cudnn_tune == conv::kLimited
           && bwd_filter_algo[i].memory > workspace_byte))) ++i;
    if (i == nalgo) {
      LOG(FATAL) << "Failed to find an convolution algorithm.";
    } else {
      *backward_workspace_byte = bwd_filter_algo[i].memory;
      *back_algo_w = bwd_filter_algo[i].algo;
    }

    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo[kMaxAlgos];
    CHECK_EQ(cudnnFindConvolutionBackwardDataAlgorithm(s->dnn_handle_,
             filter_desc,
             out_desc,
             conv_desc,
             in_desc,
             kMaxAlgos,
             &nalgo,
             bwd_data_algo), CUDNN_STATUS_SUCCESS);
    i = 0;
    while (i < nalgo
           && (bwd_data_algo[i].status != CUDNN_STATUS_SUCCESS
           || (param.cudnn_tune == conv::kLimited
           && bwd_data_algo[i].memory > workspace_byte))) ++i;
    if (i == nalgo) {
      LOG(FATAL) << "Failed to find an convolution algorithm.";
    } else {
      *backward_workspace_byte = std::max(*backward_workspace_byte, bwd_data_algo[i].memory);
      *back_algo = bwd_data_algo[i].algo;
    }
  }, ctx, {}, {var});
  Engine::Get()->WaitForVar(var);
  Engine::Get()->DeleteVariable([](RunContext s) {}, ctx, var);

  CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnDestroyTensorDescriptor(bias_desc), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnDestroyFilterDescriptor(filter_desc), CUDNN_STATUS_SUCCESS);
  CHECK_EQ(cudnnDestroyConvolutionDescriptor(conv_desc), CUDNN_STATUS_SUCCESS);
}
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet
