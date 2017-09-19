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
 * \file activation-inl.h
 * \brief Activation operator
*/

#ifndef MXNET_OPERATOR_NN_ACTIVATION_INL_H_
#define MXNET_OPERATOR_NN_ACTIVATION_INL_H_

#include <vector>

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {
namespace mxnet_op {

template<typename Op>
struct grad_op {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType out_data, DType out_grad) {
    return Op::Map(out_data) * out_grad;
  }
};

template<typename xpu, typename Op>
void activation_forward(const OpContext& ctx, const NDArray &_in_data,
    const NDArray &_out_data, const OpReqType &req) {
  const TBlob &in_data = _in_data.data();
  const TBlob &out_data = _out_data.data();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
      Kernel<op_with_req<Op, req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>());
      });
    });
}

template<typename xpu, typename Op>
void activation_backward(const OpContext& ctx, const NDArray &_in_grad,
    const NDArray &_out_data, const NDArray &_out_grad, const OpReqType &req) {
  const TBlob &in_grad = _in_grad.data();
  const TBlob &out_data = _out_data.data();
  const TBlob &out_grad = _out_grad.data();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
      Kernel<op_with_req<grad_op<Op>, req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>(), out_data.dptr<DType>(),
          out_grad.dptr<DType>());
      });
  });
}

}  // namespace mxnet_op

namespace activation {
enum ActivationOpInputs {kData};
enum ActivationOpOutputs {kOut};
enum ActivationOpType {kReLU, kSigmoid, kTanh, kSoftReLU};
}  // activation

struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .describe("Activation function to be applied.");
  }
};

template<typename xpu>
void ActivationCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const NDArray& in_data = inputs[0];
  const NDArray& out_data = outputs[0];
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  switch (param.act_type) {
    case activation::kReLU:
      mxnet_op::activation_forward<xpu, mshadow_op::relu>(ctx, in_data,
          out_data, req[0]);
      break;
    case activation::kSigmoid:
      mxnet_op::activation_forward<xpu, mshadow_op::sigmoid>(ctx, in_data,
          out_data, req[0]);
      break;
    case activation::kTanh:
      mxnet_op::activation_forward<xpu, mshadow_op::tanh>(ctx, in_data,
          out_data, req[0]);
      break;
    case activation::kSoftReLU:
      mxnet_op::activation_forward<xpu, mshadow_op::softrelu>(ctx, in_data,
          out_data, req[0]);
      break;
    default:
      LOG(FATAL) << "unknown activation type";
  }
}


template<typename xpu>
void ActivationGradCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const NDArray& out_grad = inputs[0];
  const NDArray& out_data = inputs[1];
  const NDArray& in_grad = outputs[0];
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  switch (param.act_type) {
    case activation::kReLU:
      mxnet_op::activation_backward<xpu, mshadow_op::relu_grad>(ctx, in_grad,
          out_data, out_grad, req[0]);
      break;
    case activation::kSigmoid:
      mxnet_op::activation_backward<xpu, mshadow_op::sigmoid_grad>(ctx, in_grad,
          out_data, out_grad, req[0]);
      break;
    case activation::kTanh:
      mxnet_op::activation_backward<xpu, mshadow_op::tanh_grad>(ctx, in_grad,
          out_data, out_grad, req[0]);
      break;
    case activation::kSoftReLU:
      mxnet_op::activation_backward<xpu, mshadow_op::softrelu_grad>(ctx, in_grad,
          out_data, out_grad, req[0]);
      break;
    default:
      LOG(FATAL) << "unknown activation type";
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_ACTIVATION_INL_H_
