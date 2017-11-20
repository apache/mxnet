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
 * \file nnpack_fully_connected-inl.h
 * \brief
 * \author Wei Wu
*/
#ifndef MXNET_OPERATOR_NNPACK_NNPACK_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_NNPACK_NNPACK_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../fully_connected-inl.h"
#include "nnpack.h"
#include "nnpack_util.h"

namespace mxnet {
namespace op {

template <typename xpu, typename DType>
class NNPACKFullyConnectedOp : public FullyConnectedOp<xpu, DType> {
 private:
  FullyConnectedParam param_;

 public:
  explicit NNPACKFullyConnectedOp(FullyConnectedParam p)
      : FullyConnectedOp<xpu, DType>(p) {
    this->param_ = p;
  }

 public:
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
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_data[fullc::kOut].shape_;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    const size_t batch_size = data.shape_[0];
    const size_t input_c = data.shape_[1];
    nnp_status status = nnp_status_success;
    if (batch_size == 1) {
      status = nnp_fully_connected_inference(
      input_c,                       // size_t input_channels,
      param_.num_hidden,             // size_t output_channels,
      data.dptr_,                    // const float input[],
      wmat.dptr_,                    // const float kernel[],
      out.dptr_,                     // float output[],
      nnpackinitialize.threadpool);  // pthreadpool_t threadpool,
    } else {
      status = nnp_fully_connected_output(
      batch_size,                    // size_t batch size of input tensor
      input_c,                       // size_t input_channels,
      param_.num_hidden,             // size_t output_channels,
      data.dptr_,                    // const float input[],
      wmat.dptr_,                    // const float kernel[],
      out.dptr_,                     // float output[],
      nnpackinitialize.threadpool,   // pthreadpool_t threadpool,
      nullptr);
    }
    if (nnp_status_success != status) {
      LOG(FATAL) << "nnpack fully conneted feedforward failed status=" << status;
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get<xpu, 1, DType>(s);
      out += repmat(bias, data.size(0));
    }
  }
};  // class NNPACKFullyConnectedOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NNPACK_NNPACK_FULLY_CONNECTED_INL_H_
