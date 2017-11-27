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
 * Copyright (c) 2015 by Contributors
 * \file softmax_activation-inl.h
 * \brief SoftmaxActivation operator
 * \author Junyuan Xie, Da Zheng
*/
#ifndef MXNET_OPERATOR_NN_SOFTMAX_ACTIVATION_INL_H_
#define MXNET_OPERATOR_NN_SOFTMAX_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace softmax_activation {
enum SoftmaxActivationOpInputs {kData};
enum SoftmaxActivationOpOutputs {kOut};
enum SoftmaxActivationOpType {kInstance, kChannel};
enum SoftmaxActivationOpResource {kTempSpace};
}  // softmax_activation

struct SoftmaxActivationParam : public dmlc::Parameter<SoftmaxActivationParam> {
  // use int for enumeration
  int mode;
  DMLC_DECLARE_PARAMETER(SoftmaxActivationParam) {
    DMLC_DECLARE_FIELD(mode)
    .add_enum("instance", softmax_activation::kInstance)
    .add_enum("channel", softmax_activation::kChannel)
    .set_default(softmax_activation::kInstance)
    .describe("Specifies how to compute the softmax. If set to ``instance``, "
              "it computes softmax for each instance. If set to ``channel``, "
              "It computes cross channel softmax for each position of each instance.");
  }
};

/**
 * \brief This is the implementation of softmax_activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class SoftmaxActivationOp {
 public:
  void Init(SoftmaxActivationParam p) {
    this->param_ = p;
  }

  void Forward(const OpContext &ctx, const TBlob &in_data,
      const OpReqType &req, const TBlob &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.mode == softmax_activation::kInstance) {
      Tensor<xpu, 2> data = in_data.FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> out = out_data.FlatTo2D<xpu, real_t>(s);
      Softmax(out, data);
    } else {
      CHECK_GE(in_data.ndim(), 3)
        << "Input need to have a least 3 dimensions when mode=channel";
      int n = in_data.size(0);
      int k = in_data.size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data.Size()/n/k));
      Tensor<xpu, 3, real_t> data = in_data.get_with_shape<xpu, 3, real_t>(s3, s);
      Tensor<xpu, 3, real_t> out = out_data.get_with_shape<xpu, 3, real_t>(s3, s);
      Softmax(out, data);
    }
  }

  void Backward(const OpContext &ctx, const TBlob &out_grad,
      const TBlob &out_data, const OpReqType &req, const TBlob &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // Use 3d tensor for both mode -> {instance, channel}. Get shapes
    int total_size = in_grad.Size();
    int batch_size = in_grad.shape_[0];
    int channel_num = in_grad.shape_[1];
    int rest_size = total_size / (batch_size * channel_num);
    const Shape<3> data_shape = Shape3(batch_size, channel_num, rest_size);
    // Get tensors
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3> m_out_grad =
      out_grad.get_with_shape<xpu, 3, real_t>(data_shape, s);
    Tensor<xpu, 3> m_out_data =
      out_data.get_with_shape<xpu, 3, real_t>(data_shape, s);
    Tensor<xpu, 3> m_in_grad =
      in_grad.get_with_shape<xpu, 3, real_t>(data_shape, s);
    // get requested temp space
    Tensor<xpu, 2> workspace = ctx.requested[softmax_activation::kTempSpace].get_space<xpu>(
        Shape2(batch_size, rest_size), s);
    workspace = reduce_with_axis<red::sum, false>(m_out_grad * m_out_data, 1);
    Assign(m_in_grad, req,
        m_out_data * (m_out_grad - broadcast_with_axis(workspace, 0, channel_num)));
  }

 private:
  SoftmaxActivationParam param_;
};  // class SoftmaxActivationOp


template<typename xpu>
void SoftmaxActivationCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  const SoftmaxActivationParam& param = nnvm::get<SoftmaxActivationParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  static thread_local SoftmaxActivationOp<xpu> op;
  op.Init(param);
  op.Forward(ctx, inputs[0], req[0], outputs[0]);
}

template<typename xpu>
void SoftmaxActivationGradCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  const SoftmaxActivationParam& param = nnvm::get<SoftmaxActivationParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);

  static thread_local SoftmaxActivationOp<xpu> op;
  op.Init(param);
  op.Backward(ctx, inputs[0], inputs[1], req[0], outputs[0]);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_SOFTMAX_ACTIVATION_INL_H_
