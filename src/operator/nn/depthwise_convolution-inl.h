/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file depthwise_convolution-inl.h
 * \brief CUDA depthwise convolution code
 * \author shuqian.qu@hobot.cc
*/
#ifndef MXNET_OPERATOR_NN_DEPTHWISE_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_DEPTHWISE_CONVOLUTION_INL_H_
#include <algorithm>
#include <vector>
#include "./convolution-inl.h"
#include "../../common/cuda_utils.h"

#if MXNET_USE_CUDA
#include <cub/cub.cuh>
#include "./depthwise_convolution_tf.cuh"

#define ROUND_TO_MULTIPLE(x, m) ((((x) + (m) - 1) / (m)) * (m))

namespace mxnet {
namespace op {
using namespace tf::depthwise_conv;
template<typename DType>
class DepthwiseConvolutionOp {
 public:
  void Init(const ConvolutionParam& param,
            const std::vector<TShape>& in_shape,
            const std::vector<TShape>& out_shape) {
    args_.batch = in_shape[conv::kData][0];
    args_.in_channel = in_shape[conv::kData][1];
    args_.in_height = in_shape[conv::kData][2];
    args_.in_width = in_shape[conv::kData][3];
    args_.filter_height = in_shape[conv::kWeight][2];
    args_.filter_width = in_shape[conv::kWeight][3];
    args_.stride_height = param.stride[0];
    args_.stride_width = param.stride[1];
    args_.pad_height = param.pad[0];
    args_.pad_width = param.pad[1];
    args_.out_channel = out_shape[conv::kOut][1];
    args_.out_height = out_shape[conv::kOut][2];
    args_.out_width = out_shape[conv::kOut][3];
    bias_term_ = !param.no_bias;
  }

  ~DepthwiseConvolutionOp() {}

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data);

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad);

 private:
  DepthwiseArgs args_;
  bool bias_term_;
};  // class DepthwiseConvolutionOp

namespace depthwise_conv {

template<typename DType>
void DepthwiseConv2dForwardGpu(mshadow::Stream<gpu> *stream,
                               const DepthwiseArgs& args,
                               const std::vector<TBlob> &in_data,
                               const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  using namespace tf::depthwise_conv::cuda;
  Tensor<gpu, 4, DType> data = in_data[conv::kData].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight = in_data[conv::kWeight].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> out = out_data[conv::kOut].get<gpu, 4, DType>(stream);

  // select kernel
  if (CanLaunchDepthwiseConv2dGPUSmall(args)) {
    LaunchDepthwiseConv2dGPUSmall<DType, DIRECTION_FORWARD>(
        stream,
        args,
        data.dptr_,
        weight.dptr_,
        out.dptr_);
  } else {
    int num_output = out_data[conv::kOut].shape_.Size();
    int block_num = std::min(num_output/mshadow::cuda::kBaseThreadNum + 1,
                             mshadow::cuda::kMaxGridNum);
    auto s = mshadow::Stream<gpu>::GetStream(stream);
    if (args.filter_height == 3 && args.filter_width == 3) {
      DepthwiseConv2dForwardKernel<DType, 3, 3>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(data.dptr_,
                                                               weight.dptr_,
                                                               args,
                                                               num_output,
                                                               out.dptr_);
    } else {
      DepthwiseConv2dForwardKernel<DType, -1, -1>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(data.dptr_,
                                                               weight.dptr_,
                                                               args,
                                                               num_output,
                                                               out.dptr_);
    }
    MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dForwardKernel);
  }
}

template<typename DType>
void DepthwiseConv2dBackwardDataGpu(mshadow::Stream<gpu> *stream,
                                    const DepthwiseArgs& args,
                                    const std::vector<TBlob> &out_grad,
                                    const std::vector<TBlob> &in_data,
                                    const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  using namespace tf::depthwise_conv::cuda;
  Tensor<gpu, 4, DType> out_g = out_grad[conv::kOut].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight = in_data[conv::kWeight].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> in_data_g = in_grad[conv::kData].get<gpu, 4, DType>(stream);
  // select kernel
  if (CanLaunchDepthwiseConv2dGPUSmall(args)) {
    LaunchDepthwiseConv2dGPUSmall<DType, DIRECTION_BACKWARD>(
        stream,
        args,
        out_g.dptr_,
        weight.dptr_,
        in_data_g.dptr_);
  } else {
    int num_in_grad = in_grad[conv::kData].shape_.Size();
    auto s = mshadow::Stream<gpu>::GetStream(stream);
    int block_num = std::min(num_in_grad/mshadow::cuda::kBaseThreadNum + 1,
                             mshadow::cuda::kMaxGridNum);
    DepthwiseConv2dBackwardDataKernel<DType>
        <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(args,
                                                             out_g.dptr_,
                                                             weight.dptr_,
                                                             in_data_g.dptr_,
                                                             num_in_grad);
    MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardDataKernel);
  }
}

template<typename DType>
void DepthwiseConv2dBackwardFilterGpu(mshadow::Stream<gpu> *stream,
                                      const DepthwiseArgs& args,
                                      const std::vector<TBlob> &out_grad,
                                      const std::vector<TBlob> &in_data,
                                      const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  using namespace tf::depthwise_conv::cuda;
  Tensor<gpu, 4, DType> out_g = out_grad[conv::kOut].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> in_d = in_data[conv::kData].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight_grad = in_grad[conv::kWeight].get<gpu, 4, DType>(stream);
  // select kernel
  if (TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<DType>(stream, args,
                                                            out_g.dptr_,
                                                            in_d.dptr_,
                                                            weight_grad.dptr_)) {
    return;
  } else {
    int num_out_grad = out_grad[conv::kOut].shape_.Size();
    auto s = mshadow::Stream<gpu>::GetStream(stream);
    int block_num = std::min(args.out_channel * args.batch, mshadow::cuda::kMaxGridNum);
    if (args.filter_width == 3 && args.filter_height == 3) {
      DepthwiseConv2dBackwardFilterKernel<DType, 3, 3>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(args,
                                                               out_g.dptr_,
                                                               in_d.dptr_,
                                                               weight_grad.dptr_,
                                                               num_out_grad);
    } else {
      DepthwiseConv2dBackwardFilterKernel<DType, -1, -1>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(args,
                                                               out_g.dptr_,
                                                               in_d.dptr_,
                                                               weight_grad.dptr_,
                                                               num_out_grad);
    }
    MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardFilterKernel);
  }
}
}  // namespace depthwise_conv

template<typename DType>
void DepthwiseConvolutionOp<DType>::Forward(const OpContext &ctx,
                                            const std::vector<TBlob> &in_data,
                                            const std::vector<OpReqType> &req,
                                            const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  auto stream = ctx.get_stream<gpu>();
  CHECK_EQ(req[conv::kOut], kWriteTo);
  // output forward
  depthwise_conv::DepthwiseConv2dForwardGpu<DType>(stream, args_, in_data, out_data);

  // bias forward
  if (bias_term_) {
    Tensor<gpu, 1, DType> bias = in_data[conv::kBias].get<gpu, 1, DType>(stream);
    Tensor<gpu, 3, DType> output_3d = out_data[conv::kOut].get_with_shape<gpu, 3, DType>(
        Shape3(args_.batch, args_.out_channel, args_.out_height * args_.out_width), stream);
    // has bias term, broadcast it to the same shape of output_3d in channel dim
    output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
  }
}

template<typename DType>
void DepthwiseConvolutionOp<DType>::Backward(const OpContext &ctx,
                                             const std::vector<TBlob> &out_grad,
                                             const std::vector<TBlob> &in_data,
                                             const std::vector<OpReqType> &req,
                                             const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  auto stream = ctx.get_stream<gpu>();
  // backward data
  if (req[conv::kData] != kNullOp) {
    if (req[conv::kData] != kAddTo) {
      mshadow::Tensor<gpu, 4, DType> igrad = in_grad[conv::kData].get<gpu, 4, DType>(stream);
      igrad = 0.0f;
    }
    depthwise_conv::DepthwiseConv2dBackwardDataGpu<DType>(stream,
                                                          args_,
                                                          out_grad,
                                                          in_data,
                                                          in_grad);
  }

  // backward filter
  if (req[conv::kWeight] != kNullOp) {
    if (req[conv::kWeight] != kAddTo) {
      mshadow::Tensor<gpu, 4, DType> wgrad = in_grad[conv::kWeight].get<gpu, 4, DType>(stream);
      wgrad = 0.0f;
    }
    depthwise_conv::DepthwiseConv2dBackwardFilterGpu<DType>(stream,
                                                            args_,
                                                            out_grad,
                                                            in_data,
                                                            in_grad);
  }

  // backward bias
  if (bias_term_) {
    Tensor<gpu, 1, DType> dbias = in_grad[conv::kBias].get<gpu, 1, DType>(stream);
    Tensor<gpu, 3, DType> dout = out_grad[conv::kOut].get_with_shape<gpu, 3, DType>(
        Shape3(args_.batch, args_.out_channel, args_.out_height * args_.out_width), stream);
    ASSIGN_DISPATCH(dbias, req[conv::kBias], sumall_except_dim<1>(dout));
  }
}
}  // namespace op
}  // namespace mxnet
#endif

#endif  // MXNET_OPERATOR_NN_DEPTHWISE_CONVOLUTION_INL_H_
