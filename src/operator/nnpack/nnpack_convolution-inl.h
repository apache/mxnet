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
 * Copyright (c) 2016 by Contributors
 * \file nnpack_convolution-inl.h
 * \brief
 * \author Carwin
*/
#ifndef MXNET_OPERATOR_NNPACK_NNPACK_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NNPACK_NNPACK_CONVOLUTION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../convolution-inl.h"
#include "nnpack.h"
#include "nnpack_util.h"

namespace mxnet {
namespace op {

template <typename xpu, typename DType>
class NNPACKConvolutionOp : public ConvolutionOp<xpu, DType> {
 private:
  ConvolutionParam param_;

 public:
  explicit NNPACKConvolutionOp(ConvolutionParam p)
      : ConvolutionOp<xpu, DType>(p) {
    this->param_ = p;
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[conv::kData].get<xpu, 4, DType>(s);
    const size_t batch_size = data.shape_[0];
    const size_t input_c = data.shape_[1];
    const size_t input_h = data.shape_[2];
    const size_t input_w = data.shape_[3];
    Shape<3> wmat_shape =
        Shape3(param_.num_group, param_.num_filter / param_.num_group,
               input_c / param_.num_group * param_.kernel[0] *
                   param_.kernel[1]);
    Tensor<xpu, 3, DType> wmat =
        in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    Tensor<xpu, 4, DType> out = out_data[conv::kOut].get<xpu, 4, DType>(s);
    nnp_size input_size = {input_w, input_h};
    nnp_padding input_padding = {param_.pad[0], param_.pad[1], param_.pad[0],
                               param_.pad[1]};
    nnp_size kernel_size = {param_.kernel[1], param_.kernel[0]};
    nnp_size output_subsampling = {param_.stride[1], param_.stride[0]};
    Tensor<xpu, 1, DType> bias = in_data[conv::kBias].get<xpu, 1, DType>(s);

    nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;
    nnp_convolution_transform_strategy kts = nnp_convolution_transform_strategy_tuple_based;
    nnp_status status = nnp_status_success;
    if (batch_size == 1) {
      status = nnp_convolution_inference(
      algorithm,                    // enum nnp_convolution_algorithm,
      kts,                          // enum nnp_convolution_transform_strategy,
      input_c,                      // size_t input_channels,
      param_.num_filter,            // size_t output_channels,
      input_size,                   // struct nnp_size input_size,
      input_padding,                // struct nnp_padding input_padding,
      kernel_size,                  // struct nnp_size kernel_size,
      output_subsampling,           // struct nnp_size output_subsampling,
      data.dptr_,                   // const float input[],
      wmat.dptr_,                   // const float kernel[],
      bias.dptr_,                   // const float bias[],
      out.dptr_,                    // float output[],
      nnpackinitialize.threadpool,  // pthreadpool_t threadpool,
      nullptr);
    } else {
      status = nnp_convolution_output(
      algorithm,                    // enum nnp_convolution_algorithm algorithm,
      batch_size,                   // size_t batch size of input tensor
      input_c,                      // size_t input_channels,
      param_.num_filter,            // size_t output_channels,
      input_size,                   // struct nnp_size input_size,
      input_padding,                // struct nnp_padding input_padding,
      kernel_size,                  // struct nnp_size kernel_size,
      data.dptr_,                   // const float input[],
      wmat.dptr_,                   // const float kernel[],
      bias.dptr_,                   // const float bias[],
      out.dptr_,                    // float output[],
      nnpackinitialize.threadpool,  // pthreadpool_t threadpool,
      nullptr);
    }
    if (nnp_status_success != status) {
      LOG(FATAL) << "nnpack convolution feedforward failed status=" << status;
    }
  }
};  // class NNPACKConvolutionOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NNPACK_NNPACK_CONVOLUTION_INL_H_
